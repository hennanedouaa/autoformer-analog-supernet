from __future__ import annotations
from typing import Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn

try:
    from aihwkit.nn import AnalogConv2d
except Exception:
    AnalogConv2d = None  # type: ignore


def _to_2tuple(x: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError(f"Invalid size specification: {x}")


def _get_conv_weight_bias(conv: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(conv, "get_weights") and callable(getattr(conv, "get_weights")):
        wb = conv.get_weights()
        if isinstance(wb, (tuple, list)):
            w = wb[0]
            b = wb[1] if len(wb) > 1 else None
        else:
            w, b = wb, None
        if b is None and getattr(conv, "bias", None) is not None:
            b = conv.bias
        return w, b
    raise AttributeError("Cannot retrieve analog conv weights.")


def _set_conv_weight(conv: nn.Module, weight: torch.Tensor) -> None:
    W = weight.detach().contiguous()
    if hasattr(conv, "set_weights") and callable(getattr(conv, "set_weights")):
        conv.set_weights(W)
        return
    raise AttributeError("Cannot set analog conv weights.")


class AnalogPatchembedSuper(nn.Module):
    def __init__(self, super_config: Dict, *, rpu_config, AnalogConv2dCls=None):
        super().__init__()
        if AnalogConv2dCls is None:
            if AnalogConv2d is None:
                raise ImportError("AIHWKit not installed.")
            AnalogConv2dCls = AnalogConv2d

        pe_cfg = super_config["patch_embed"]
        self.img_size = _to_2tuple(pe_cfg["img_size"])
        self.patch_size = _to_2tuple(pe_cfg["patch_size"])
        self.in_chans = int(pe_cfg.get("in_chans", 3))
        self.scale = bool(pe_cfg.get("scale", False))

        self.embed_dim_choices = sorted(int(d) for d in pe_cfg["embed_dim_choices"])
        self.super_embed_dim = max(self.embed_dim_choices)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        self.ops = nn.ModuleDict()
        for embed_dim in self.embed_dim_choices:
            self.ops[str(embed_dim)] = AnalogConv2dCls(
                in_channels=self.in_chans,
                out_channels=embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding=0,
                dilation=1,
                rpu_config=rpu_config,
            )

        self.sample_embed_dim: Optional[int] = None
        self.current_op: Optional[nn.Module] = None
        self.sample_scale: float = 1.0

    def set_sample_config(self, sample_embed_dim: int):
        sample_embed_dim = int(sample_embed_dim)
        if sample_embed_dim not in self.embed_dim_choices:
            raise ValueError(f"Embedding dimension {sample_embed_dim} not in {self.embed_dim_choices}")
        self.sample_embed_dim = sample_embed_dim
        self.current_op = self.ops[str(sample_embed_dim)]
        self.sample_scale = (self.super_embed_dim / sample_embed_dim) if self.scale else 1.0
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        _, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Input size {(H, W)} does not match {self.img_size}")
        y = self.current_op(x).flatten(2).transpose(1, 2)
        if self.scale:
            y = y * self.sample_scale
        return y

    def get_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get weights/biases from this supermodule.
        Returns two dicts {embed_dim: weight} and {embed_dim: bias}.
        """
        weight_dict: Dict[str, torch.Tensor] = {}
        bias_dict: Dict[str, torch.Tensor] = {}

        for key, conv in self.ops.items():
            # Each AnalogConv2d supports get_weights()
            w, b = _get_conv_weight_bias(conv)
            weight_dict[key] = w
            bias_dict[key] = b

        return weight_dict, bias_dict

    def set_weights(self,
                    weight_dict: Dict[str, torch.Tensor],
                    bias_dict: Optional[Dict[str, torch.Tensor]] = None) -> "AnalogPatchembedSuper":
        """
        Set weights/biases into the analog conv ops.
        If bias_dict is None, only weights are set.
        """
        for key, conv in self.ops.items():
            if key not in weight_dict:
                raise KeyError(f"No weight entry for embed_dim {key}")

            w = weight_dict[key]
            b = None if bias_dict is None else bias_dict.get(key, None)

            # Write into the analog layer
            _set_conv_weight(conv, w)

            if b is not None:
                # AIHWKit supports set_weights(weight, bias) at the layer level
                conv.set_weights(w, b)

        return self

    # ----------------------------------------------------------------------
    # New helper method to copy weights from a digital Conv2d
    # ----------------------------------------------------------------------
    def copy_from_digital_conv(self, digital_conv: nn.Conv2d) -> "AnalogPatchembedSuper":
        """
        Copy weights from a digital Conv2d (with out_channels = self.super_embed_dim)
        to all analog operators in this supermodule.
        The digital conv's weight and bias are sliced to match each embed_dim choice.

        Args:
            digital_conv: A torch.nn.Conv2d layer with out_channels == self.super_embed_dim
                          (the largest embedding dimension). Its weight and bias tensors
                          will be sliced along the output channel dimension.

        Returns:
            self (for chaining)

        Example usage:
            analog_patch = AnalogPatchembedSuper(super_config, rpu_config=rpu)
            analog_patch.copy_from_digital_conv(digital_patch.proj)
        """
        weight_dict = {}
        bias_dict = {}
        for embed_dim_str in self.ops.keys():
            embed_dim = int(embed_dim_str)
            # Slice along the output channel dimension (first axis of weight)
            w_slice = digital_conv.weight[:embed_dim].clone().detach().contiguous()
            b_slice = digital_conv.bias[:embed_dim].clone().detach().contiguous() if digital_conv.bias is not None else None
            weight_dict[embed_dim_str] = w_slice
            if b_slice is not None:
                bias_dict[embed_dim_str] = b_slice
        self.set_weights(weight_dict, bias_dict if bias_dict else None)
        return self