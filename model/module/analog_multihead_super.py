from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .analog_linear_super import AnalogLinearSuper
from ..utils import trunc_normal_


def softmax(x: torch.Tensor, dim: int, onnx_trace: bool = False) -> torch.Tensor:
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    return F.softmax(x, dim=dim, dtype=torch.float32)


class RelativePosition2D_super(nn.Module):
    def __init__(self, num_units: int, max_relative_position: int):
        super().__init__()
        self.num_units = int(num_units)
        self.max_relative_position = int(max_relative_position)

        self.embeddings_table_v = nn.Parameter(
            torch.randn(self.max_relative_position * 2 + 2, self.num_units)
        )
        self.embeddings_table_h = nn.Parameter(
            torch.randn(self.max_relative_position * 2 + 2, self.num_units)
        )

        trunc_normal_(self.embeddings_table_v, std=0.02)
        trunc_normal_(self.embeddings_table_h, std=0.02)

        self.sample_head_dim: Optional[int] = None
        self.sample_embeddings_table_h: Optional[torch.Tensor] = None
        self.sample_embeddings_table_v: Optional[torch.Tensor] = None

    def set_sample_config(self, sample_head_dim: int) -> None:
        self.sample_head_dim = int(sample_head_dim)
        self.sample_embeddings_table_h = self.embeddings_table_h[:, : self.sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:, : self.sample_head_dim]

    def calc_sampled_param_num(self) -> int:
        if self.sample_embeddings_table_h is None or self.sample_embeddings_table_v is None:
            raise RuntimeError("Call set_sample_config() first.")
        return int(self.sample_embeddings_table_h.numel() + self.sample_embeddings_table_v.numel())

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        if self.sample_embeddings_table_h is None or self.sample_embeddings_table_v is None:
            raise RuntimeError("Call set_sample_config() first.")

        length_q = length_q - 1
        length_k = length_k - 1

        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)

        side = int(length_q ** 0.5)

        distance_mat_v = (range_vec_k[None, :] // side) - (range_vec_q[:, None] // side)
        distance_mat_h = (range_vec_k[None, :] % side) - (range_vec_q[:, None] % side)

        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.max_relative_position, self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.max_relative_position, self.max_relative_position)

        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1

        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0).long()
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0).long()

        embeddings = self.sample_embeddings_table_v[final_mat_v] + self.sample_embeddings_table_h[final_mat_h]
        return embeddings


class AnalogAttentionSuper(nn.Module):
    def __init__(
        self,
        super_config: dict,
        *,
        rpu_config,
        super_embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        relative_position: bool = False,
        max_relative_position: int = 14,
        scale: bool = False,
        change_qkv: bool = False,
    ):
        super().__init__()

        self.super_embed_dim = int(super_embed_dim)
        self.num_heads = int(num_heads)
        self.fc_scale = bool(scale)
        self.change_qkv = bool(change_qkv)

        head_dim_super = self.super_embed_dim // self.num_heads
        self.base_scale = float(qk_scale) if qk_scale is not None else (head_dim_super ** -0.5)

        # Analog linear layers
        self.qkv = AnalogLinearSuper(
            super_config,
            role="attn_qkv",
            rpu_config=rpu_config,
            bias=qkv_bias,
            scale=False,
        )
        self.proj = AnalogLinearSuper(
            super_config,
            role="attn_proj",
            rpu_config=rpu_config,
            bias=True,
            scale=False,
        )

        self.relative_position = bool(relative_position)
        self.max_relative_position = int(max_relative_position)

        # Relative position embedding tables
        self.max_head_dim = None
        if self.relative_position:
            pe = super_config.get("patch_embed", {})
            embed_choices = pe.get("embed_dim_choices", [self.super_embed_dim])
            sn = super_config.get("supernet", {})
            head_choices = sn.get("num_heads_choices", [self.num_heads])

            max_head_dim = 0
            for e in embed_choices:
                for h in head_choices:
                    e = int(e)
                    h = int(h)
                    if h > 0:
                        max_head_dim = max(max_head_dim, e // h)
            max_head_dim = max(max_head_dim, 64)
            self.max_head_dim = int(max_head_dim)

            self.rel_pos_embed_k = RelativePosition2D_super(num_units=self.max_head_dim, max_relative_position=self.max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(num_units=self.max_head_dim, max_relative_position=self.max_relative_position)

        self.sample_qk_embed_dim: Optional[int] = None
        self.sample_num_heads: Optional[int] = None
        self.sample_scale: Optional[float] = None
        self.sample_in_embed_dim: Optional[int] = None
        self.sample_head_dim: Optional[int] = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(
        self,
        *,
        sample_in_embed_dim: int,
        sample_num_heads: int,
        sample_q_embed_dim: Optional[int] = None,
    ) -> None:
        self.sample_in_embed_dim = int(sample_in_embed_dim)
        self.sample_num_heads = int(sample_num_heads)

        if not self.change_qkv:
            qkv_dim = self.sample_in_embed_dim
        else:
            if sample_q_embed_dim is None:
                raise ValueError("sample_q_embed_dim must be provided when change_qkv=True.")
            qkv_dim = int(sample_q_embed_dim)
        self.sample_qk_embed_dim = int(qkv_dim)

        self.sample_head_dim = int(qkv_dim // self.sample_num_heads)
        self.sample_scale = float(self.sample_head_dim ** -0.5)

        # Configure qkv
        try:
            self.qkv.set_sample_config(self.sample_in_embed_dim, 3 * qkv_dim)
        except ValueError:
            # fallback to super_embed_dim if requested out not instantiated
            qkv_dim = self.super_embed_dim
            self.sample_qk_embed_dim = qkv_dim
            self.sample_head_dim = qkv_dim // self.sample_num_heads
            self.sample_scale = float(self.sample_head_dim ** -0.5)
            self.qkv.set_sample_config(self.sample_in_embed_dim, 3 * qkv_dim)

        # Configure proj
        try:
            self.proj.set_sample_config(qkv_dim, self.sample_in_embed_dim)
        except ValueError:
            self.proj.set_sample_config(self.super_embed_dim, self.sample_in_embed_dim)

        # Slice rel-pos tables
        if self.relative_position:
            if self.sample_head_dim > self.max_head_dim:
                raise RuntimeError(f"sample_head_dim={self.sample_head_dim} > max_head_dim={self.max_head_dim}")
            self.rel_pos_embed_k.set_sample_config(self.sample_head_dim)
            self.rel_pos_embed_v.set_sample_config(self.sample_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.sample_num_heads
        qkv_dim = self.sample_qk_embed_dim
        head_dim = self.sample_head_dim

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.sample_scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (
                (q.permute(2, 0, 1, 3).reshape(N, H * B, head_dim) @ r_p_k.transpose(2, 1))
                .transpose(1, 0)
                .reshape(B, H, N, N)
                * self.sample_scale
            )

        attn = softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, qkv_dim)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * H, N)
            x_out = x_out + (
                (attn_1 @ r_p_v).transpose(1, 0).reshape(B, H, N, head_dim).transpose(2, 1).reshape(B, N, qkv_dim)
            )

        if self.fc_scale:
            x_out = x_out * (self.super_embed_dim / qkv_dim)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


    def get_weights(self) -> dict:
        """
        Retrieve weights/biases of all analog linear ops in this attention layer.
        Returns a nested dict:
        {
            "qkv": {"in{in}_out{out}": (weight, bias), ...},
            "proj": {"in{in}_out{out}": (weight, bias), ...}
        }
        """
        weights = {
            "qkv": {},
            "proj": {}
        }

        # qkv weights
        w_dict, b_dict = self.qkv.get_weights()
        for k in w_dict.keys():
            weights["qkv"][k] = (w_dict[k], b_dict[k])

        # proj weights
        w_dict, b_dict = self.proj.get_weights()
        for k in w_dict.keys():
            weights["proj"][k] = (w_dict[k], b_dict[k])

        return weights

    def set_weights(self, weights: dict) -> None:
        """
        Set weights/biases of all analog linear ops in this attention layer.
        `weights` should be in the same format as returned by get_weights().
        """
        if "qkv" in weights:
            w_dict = {k: v[0] for k, v in weights["qkv"].items()}
            b_dict = {k: v[1] for k, v in weights["qkv"].items()}
            self.qkv.set_weights(w_dict, b_dict)

        if "proj" in weights:
            w_dict = {k: v[0] for k, v in weights["proj"].items()}
            b_dict = {k: v[1] for k, v in weights["proj"].items()}
            self.proj.set_weights(w_dict, b_dict)

    def get_current_weights(self) -> dict:
        """
        Retrieve weights/bias of the currently sampled ops in qkv and proj.
        Returns:
            {"qkv": (weight, bias), "proj": (weight, bias)}
        """
        if self.qkv.current_op is None or self.proj.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        return {
            "qkv": self.qkv.get_current_weights(),
            "proj": self.proj.get_current_weights()
        }

    def set_current_weights(self, qkv_wb: tuple, proj_wb: tuple) -> None:
        """
        Set weights/bias of the currently sampled ops.
        Args:
            qkv_wb: (weight, bias) for the sampled qkv op
            proj_wb: (weight, bias) for the sampled proj op
        """
        if self.qkv.current_op is None or self.proj.current_op is None:
            raise RuntimeError("Call set_sample_config() first.")
        self.qkv.set_current_weights(*qkv_wb)
        self.proj.set_current_weights(*proj_wb)