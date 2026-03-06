# model/analog_supernet_transformer.py

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import trunc_normal_, DropPath
from model.module.layernorm_super import LayerNormSuper
from model.module.analog_embedding_super import AnalogPatchembedSuper
from model.module.analog_linear_super import AnalogLinearSuper
from model.module.analog_multihead_super import AnalogAttentionSuper


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, "gelu"):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def calc_dropout(dropout: float, sample_embed_dim: int, super_embed_dim: int) -> float:
    return float(dropout) * float(sample_embed_dim) / float(super_embed_dim)


def _is_tensor_like(x) -> bool:
    return isinstance(x, (torch.Tensor, torch.nn.Parameter))


class Vision_TransformerSuper(nn.Module):
    """
    Analog AutoFormer-ViT supernet.

    The YAML config must be provided via the `super_config` argument.
    """

    def __init__(
        self,
        *,
        super_config: Dict,  # now required, no default
        rpu_config,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pre_norm: bool = True,
        scale: bool = False,
        gp: bool = False,
        relative_position: bool = True,
        change_qkv: bool = True,
        abs_pos: bool = True,
        max_relative_position: int = 14,
    ):
        super().__init__()

        self.super_config = super_config

        self.super_embed_dim = int(embed_dim)
        self.super_mlp_ratio = float(mlp_ratio)
        self.super_layer_num = int(depth)
        self.super_num_heads = int(num_heads)
        self.super_dropout = float(drop_rate)
        self.super_attn_dropout = float(attn_drop_rate)

        self.num_classes = int(num_classes)
        self.pre_norm = bool(pre_norm)
        self.scale = bool(scale)
        self.gp = bool(gp)

        # instantiate patch embedding using the provided config
        self.patch_embed_super = AnalogPatchembedSuper(
            super_config,
            rpu_config=rpu_config,
        )
        # override size parameters if user provided them
        if img_size is not None:
            new_img = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
            self.patch_embed_super.img_size = new_img
        if patch_size is not None:
            new_patch = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
            self.patch_embed_super.patch_size = new_patch
        # recompute num_patches after any overrides
        self.patch_embed_super.num_patches = (
            (self.patch_embed_super.img_size[0] // self.patch_embed_super.patch_size[0]) *
            (self.patch_embed_super.img_size[1] // self.patch_embed_super.patch_size[1])
        )

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                TransformerEncoderLayer(
                    super_config=super_config,
                    rpu_config=rpu_config,
                    dim=self.super_embed_dim,
                    num_heads=self.super_num_heads,
                    mlp_ratio=self.super_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    relative_position=relative_position,
                    change_qkv=change_qkv,
                    max_relative_position=max_relative_position,
                )
            )

        num_patches = self.patch_embed_super.num_patches

        self.abs_pos = bool(abs_pos)
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.super_embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=self.super_embed_dim)

        # classifier head
        self.head = (
            AnalogLinearSuper(super_config, role="head", rpu_config=rpu_config, bias=True, scale=False)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        # ... (unchanged) ...
        if isinstance(m, nn.Linear):
            w = getattr(m, "weight", None)
            if _is_tensor_like(w):
                trunc_normal_(w, std=0.02)
            b = getattr(m, "bias", None)
            if _is_tensor_like(b):
                nn.init.constant_(b, 0)
            return
        if isinstance(m, nn.LayerNorm):
            b = getattr(m, "bias", None)
            w = getattr(m, "weight", None)
            if _is_tensor_like(b):
                nn.init.constant_(b, 0)
            if _is_tensor_like(w):
                nn.init.constant_(w, 1.0)
            return
        am = getattr(m, "analog_module", None)
        if am is not None:
            b = getattr(am, "bias", None)
            if _is_tensor_like(b):
                nn.init.constant_(b, 0)
        b = getattr(m, "bias", None)
        if _is_tensor_like(b):
            nn.init.constant_(b, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "rel_pos_embed"}

    def set_sample_config(self, config: dict) -> None:
        # ... (unchanged) ...
        self.sample_embed_dim = config["embed_dim"]
        self.sample_mlp_ratio = config["mlp_ratio"]
        self.sample_layer_num = config["layer_num"]
        self.sample_num_heads = config["num_heads"]

        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)

        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])

        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]

        for i, blk in enumerate(self.blocks):
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(
                    self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim
                )
                blk.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout,
                    sample_out_dim=self.sample_output_dim[i],
                    sample_attn_dropout=sample_attn_dropout,
                )
            else:
                blk.set_sample_config(is_identity_layer=True)

        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])

        if isinstance(self.head, AnalogLinearSuper):
            self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # ... (unchanged) ...
        B = x.shape[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., : self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., : self.sample_embed_dim[0]]
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        for blk in self.blocks:
            x = blk(x)
        if self.pre_norm:
            x = self.norm(x)
        if self.gp:
            return torch.mean(x[:, 1:], dim=1)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


class TransformerEncoderLayer(nn.Module):
    # ... (unchanged, but ensure it receives super_config) ...
    def __init__(
        self,
        *,
        super_config: Dict,
        rpu_config,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_norm: bool = True,
        scale: bool = False,
        relative_position: bool = False,
        change_qkv: bool = False,
        max_relative_position: int = 14,
    ):
        super().__init__()
        self.super_embed_dim = int(dim)
        self.super_mlp_ratio = float(mlp_ratio)
        self.super_ffn_embed_dim_this_layer = int(self.super_mlp_ratio * self.super_embed_dim)
        self.super_num_heads = int(num_heads)
        self.normalize_before = bool(pre_norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale = bool(scale)
        self.relative_position = bool(relative_position)
        self.change_qkv = bool(change_qkv)

        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.sample_out_dim = None
        self.is_identity_layer = None

        self.attn = AnalogAttentionSuper(
            super_config,
            rpu_config=rpu_config,
            super_embed_dim=self.super_embed_dim,
            num_heads=self.super_num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
            scale=self.scale,
            relative_position=self.relative_position,
            change_qkv=self.change_qkv,
            max_relative_position=max_relative_position,
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.activation_fn = gelu

        self.fc1 = AnalogLinearSuper(super_config, role="mlp_fc1", rpu_config=rpu_config, bias=True, scale=False)
        self.fc2 = AnalogLinearSuper(super_config, role="mlp_fc2", rpu_config=rpu_config, bias=True, scale=False)

    def set_sample_config(self, **kwargs):
        # ... (unchanged) ...
        if kwargs.get("is_identity_layer"):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = int(kwargs["sample_embed_dim"])
        self.sample_out_dim = int(kwargs["sample_out_dim"])
        self.sample_mlp_ratio = float(kwargs["sample_mlp_ratio"])
        self.sample_ffn_embed_dim_this_layer = int(self.sample_embed_dim * self.sample_mlp_ratio)
        self.sample_num_heads_this_layer = int(kwargs["sample_num_heads"])
        self.sample_dropout = float(kwargs["sample_dropout"])
        self.sample_attn_dropout = float(kwargs["sample_attn_dropout"])
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        sample_q_embed_dim = (
            int(self.sample_num_heads_this_layer * 64) if self.change_qkv else int(self.sample_embed_dim)
        )
        self.attn.set_sample_config(
            sample_in_embed_dim=self.sample_embed_dim,
            sample_num_heads=self.sample_num_heads_this_layer,
            sample_q_embed_dim=sample_q_embed_dim,
        )
        self.fc1.set_sample_config(self.sample_embed_dim, self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(self.sample_ffn_embed_dim_this_layer, self.sample_out_dim)
        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def maybe_layer_norm(self, layer_norm, x, *, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x

    def forward(self, x):
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x