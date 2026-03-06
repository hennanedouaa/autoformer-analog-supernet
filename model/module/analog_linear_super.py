from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import re

try:
    from aihwkit.nn import AnalogLinear
except Exception:
    AnalogLinear = None


# ----------------------------------------------------------------------
# Helper functions for analog linear operations
# ----------------------------------------------------------------------

def _get_linear_weight_bias(op: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Retrieve weight (and bias) from an analog linear layer."""
    if hasattr(op, "get_weights") and callable(getattr(op, "get_weights")):
        wb = op.get_weights()
        if isinstance(wb, (tuple, list)):
            W = wb[0]
            b = wb[1] if len(wb) > 1 else None
        else:
            W, b = wb, None
        return W, b
    raise AttributeError("Cannot retrieve analog linear weights.")


def _set_linear_weight(op: nn.Module, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
    """Set weight (and bias) into an analog linear layer."""
    W = weight.detach().contiguous()
    b = bias.detach().contiguous() if bias is not None else None
    if hasattr(op, "set_weights") and callable(getattr(op, "set_weights")):
        if b is not None:
            op.set_weights(W, b)
        else:
            op.set_weights(W)
        return
    raise AttributeError("Cannot set analog linear weights.")


def _get_digital_weight_for_config(digital_module: nn.Module, in_dim: int, out_dim: int, role: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract fused weight and bias from a digital module for a specific (in_dim, out_dim) configuration.
    Handles:
      - Standard nn.Linear or LinearSuper (simple slicing)
      - qkv_super with separate q_proj, k_proj, v_proj (concatenates along output dimension)
      - qkv_super with fused interleaved weight (special stride-3 sampling)
    """
    # Special handling for attn_qkv
    if role == "attn_qkv":
        # Case 1: separate projections (used in some implementations)
        if hasattr(digital_module, 'q_proj') and hasattr(digital_module, 'k_proj') and hasattr(digital_module, 'v_proj'):
            qkv_dim = out_dim // 3
            embed_dim = in_dim

            def get_proj_weight(proj):
                if hasattr(proj, 'samples') and 'weight' in proj.samples:
                    return proj.samples['weight'], proj.samples.get('bias', None)
                else:
                    w = proj.weight[:qkv_dim, :embed_dim]
                    b = proj.bias[:qkv_dim] if proj.bias is not None else None
                    return w, b

            w_q, b_q = get_proj_weight(digital_module.q_proj)
            w_k, b_k = get_proj_weight(digital_module.k_proj)
            w_v, b_v = get_proj_weight(digital_module.v_proj)

            w_fused = torch.cat([w_q, w_k, w_v], dim=0)
            b_fused = torch.cat([b_q, b_k, b_v], dim=0) if b_q is not None else None
            return w_fused, b_fused

        # Case 2: fused interleaved qkv_super (like the digital class provided)
        else:
            full_weight = digital_module.weight  # shape (super_out_dim, super_in_dim)
            full_bias = digital_module.bias if hasattr(digital_module, 'bias') and digital_module.bias is not None else None

            # 1. Truncate input dimension: take first in_dim columns
            weight_trunc = full_weight[:, :in_dim]  # (super_out_dim, in_dim)

            # 2. Determine qkv_dim (each head group has out_dim/3 rows)
            qkv_dim = out_dim // 3

            # 3. Extract rows with stride 3 for each part (Q, K, V)
            # Rows are interleaved as: Q0, K0, V0, Q1, K1, V1, ...
            q_weight = weight_trunc[0:3*qkv_dim:3, :]   # rows 0,3,6,...
            k_weight = weight_trunc[1:3*qkv_dim:3, :]   # rows 1,4,7,...
            v_weight = weight_trunc[2:3*qkv_dim:3, :]   # rows 2,5,8,...

            # 4. Concatenate in order: all Q, then all K, then all V
            fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # (3*qkv_dim, in_dim)

            # 5. Bias: simple slice (bias is not interleaved)
            fused_bias = full_bias[:out_dim].clone().detach() if full_bias is not None else None

            # Sanity check (optional, can be removed)
            assert fused_weight.shape[0] == out_dim and fused_weight.shape[1] == in_dim, \
                f"Fused weight shape {fused_weight.shape} != ({out_dim}, {in_dim})"
            return fused_weight, fused_bias

    # For all other roles (fc1, fc2, attn_proj, head), use standard slicing
    else:
        full_weight = digital_module.weight
        full_bias = digital_module.bias if hasattr(digital_module, 'bias') and digital_module.bias is not None else None

        # If the digital module is a super module with sampled weights, use those if they match
        if hasattr(digital_module, 'samples') and 'weight' in digital_module.samples:
            w_sampled = digital_module.samples['weight']
            if w_sampled.shape[0] == out_dim and w_sampled.shape[1] == in_dim:
                w = w_sampled
                b = digital_module.samples.get('bias', None)
                return w.clone().detach(), b.clone().detach() if b is not None else None

        # Otherwise slice the full weight
        w_slice = full_weight[:out_dim, :in_dim]
        b_slice = full_bias[:out_dim] if full_bias is not None else None
        return w_slice.clone().detach(), b_slice.clone().detach() if b_slice is not None else None


# ----------------------------------------------------------------------
# Main class: AnalogLinearSuper
# ----------------------------------------------------------------------

class AnalogLinearSuper(nn.Module):
    def __init__(self, super_config: Dict, *, role: str, rpu_config, AnalogLinearCls=None, bias=True, scale=False):
        super().__init__()
        if AnalogLinearCls is None:
            if AnalogLinear is None:
                raise ImportError("AIHWKit not installed.")
            AnalogLinearCls = AnalogLinear

        self.role = role
        self.scale = scale
        pairs_raw = super_config["linear_super_ops"][role]
        self.super_out_dim = max(out_dim for _, out_dim in pairs_raw)
        self.ops = nn.ModuleDict()
        for in_dim, out_dim in pairs_raw:
            key = f"in{in_dim}_out{out_dim}"
            self.ops[key] = AnalogLinearCls(in_features=in_dim, out_features=out_dim, bias=bias, rpu_config=rpu_config)

        self.current_in_dim: Optional[int] = None
        self.current_out_dim: Optional[int] = None
        self.current_op: Optional[nn.Module] = None
        self.sample_scale: float = 1.0

    def _key(self, in_dim: int, out_dim: int) -> str:
        return f"in{in_dim}_out{out_dim}"

    def set_sample_config(self, sample_in_dim: int, sample_out_dim: int):
        key = self._key(sample_in_dim, sample_out_dim)
        if key not in self.ops:
            raise ValueError(f"Key {key} not instantiated for role {self.role}")
        self.current_in_dim = sample_in_dim
        self.current_out_dim = sample_out_dim
        self.current_op = self.ops[key]
        self.sample_scale = (self.super_out_dim / sample_out_dim) if self.scale else 1.0
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        y = self.current_op(x)
        if self.scale:
            y = y * self.sample_scale
        return y

    def program_analog_weights(self):
        """Program all analog tiles with the currently stored digital weights."""
        for key, op in self.ops.items():
            if hasattr(op, "get_weights") and hasattr(op, "set_weights"):
                W, b = _get_linear_weight_bias(op)
                _set_linear_weight(op, W, b)
            else:
                print(f"Warning: op {key} does not support programming analog weights")

    def get_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        weight_dict: Dict[str, torch.Tensor] = {}
        bias_dict: Dict[str, Optional[torch.Tensor]] = {}

        for key, op in self.ops.items():
            W, b = _get_linear_weight_bias(op)
            weight_dict[key] = W
            bias_dict[key] = b

        return weight_dict, bias_dict

    def set_weights(self,
                    weight_dict: Dict[str, torch.Tensor],
                    bias_dict: Optional[Dict[str, Optional[torch.Tensor]]] = None) -> "AnalogLinearSuper":
        for key, op in self.ops.items():
            if key not in weight_dict:
                raise KeyError(f"No weight entry for key {key}")

            W = weight_dict[key]
            b = None if bias_dict is None else bias_dict.get(key, None)

            _set_linear_weight(op, W, b)

        return self

    def get_current_weights(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        return _get_linear_weight_bias(self.current_op)

    def set_current_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        if self.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        _set_linear_weight(self.current_op, weight, bias)

    # ----------------------------------------------------------------------
    # Method: copy weights for ALL operators from a digital module
    # ----------------------------------------------------------------------
    def copy_all_ops_from_digital(self, digital_module: nn.Module) -> "AnalogLinearSuper":
        """
        Copy weights from a digital module to every analog operator in this supermodule.
        The digital module must provide the full weight (or separate projections for qkv).
        This method does NOT rely on the current sampled config; it iterates over all ops.
        """
        for key, op in self.ops.items():
            m = re.match(r"in(\d+)_out(\d+)", key)
            if not m:
                raise ValueError(f"Invalid op key: {key}")
            in_dim = int(m.group(1))
            out_dim = int(m.group(2))

            w, b = _get_digital_weight_for_config(digital_module, in_dim, out_dim, self.role)
            op.set_weights(w, b)

        return self