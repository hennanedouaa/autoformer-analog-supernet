import math
import warnings
from itertools import repeat
import collections.abc as container_abcs
import argparse
import random
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from AutoFormer.lib.config import cfg, update_config_from_file
from AutoFormer.model.supernet_transformer import (
    Vision_TransformerSuper as DigitalVision_TransformerSuper,
)
from AutoFormer.supernet_engine import sample_configs

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from hardwareConfig.rpu_config import gen_rpu_config


# -------------------------------------------------
# Utils
# -------------------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, (str, bytes)):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -------------------------------------------------
# Patch Embedding
# -------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert (h, w) == self.img_size, (
            f"Input image size ({h}x{w}) doesn't match model "
            f"({self.img_size[0]}x{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -------------------------------------------------
# Relative Position
# -------------------------------------------------
class RelativePosition2D(nn.Module):
    def __init__(self, head_dim, max_relative_position):
        super().__init__()
        self.head_dim = head_dim
        self.max_relative_position = max_relative_position

        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dim)
        )
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dim)
        )

        trunc_normal_(self.embeddings_table_v, std=0.02)
        trunc_normal_(self.embeddings_table_h, std=0.02)

    def forward(self, length_q, length_k):
        length_q_wo_cls = length_q - 1
        length_k_wo_cls = length_k - 1

        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q_wo_cls, device=device)
        range_vec_k = torch.arange(length_k_wo_cls, device=device)

        side_q = int(length_q_wo_cls ** 0.5)

        distance_mat_v = (
            range_vec_k[None, :] // side_q
            - range_vec_q[:, None] // side_q
        )
        distance_mat_h = (
            range_vec_k[None, :] % side_q
            - range_vec_q[:, None] % side_q
        )

        distance_mat_clipped_v = torch.clamp(
            distance_mat_v, -self.max_relative_position, self.max_relative_position
        )
        distance_mat_clipped_h = torch.clamp(
            distance_mat_h, -self.max_relative_position, self.max_relative_position
        )

        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1

        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0).long()
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0).long()

        embeddings = (
            self.embeddings_table_v[final_mat_v]
            + self.embeddings_table_h[final_mat_h]
        )
        return embeddings


# -------------------------------------------------
# MLP
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------------------------
# Attention
# -------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        relative_position=False,
        max_relative_position=14,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim if qkv_dim is not None else dim
        assert self.qkv_dim % num_heads == 0, "qkv_dim must be divisible by num_heads"

        self.head_dim = self.qkv_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.relative_position = relative_position

        self.qkv = nn.Linear(dim, self.qkv_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.qkv_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D(self.head_dim, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D(self.head_dim, max_relative_position)

    def forward(self, x):
        b, n, _ = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(n, n)
            attn = attn + (
                q.permute(2, 0, 1, 3).reshape(n, b * self.num_heads, self.head_dim)
                @ r_p_k.transpose(2, 1)
            ).transpose(1, 0).reshape(b, self.num_heads, n, n) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, self.qkv_dim)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(n, n)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(n, b * self.num_heads, n)
            out = out + (
                attn_1 @ r_p_v
            ).transpose(1, 0).reshape(b, self.num_heads, n, self.head_dim) \
             .transpose(2, 1).reshape(b, n, self.qkv_dim)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------------------------------
# Transformer Block
# -------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_dim=None,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        relative_position=False,
        max_relative_position=14,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
            relative_position=relative_position,
            max_relative_position=max_relative_position,
        )

        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.pre_norm:
            return layer_norm(x)
        return x

    def forward(self, x):
        residual = x
        x = self.maybe_layer_norm(self.attn_norm, x, before=True)
        x = self.attn(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_norm, x, before=True)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_norm, x, after=True)

        return x


# -------------------------------------------------
# Heterogeneous Standalone Vision Transformer
# -------------------------------------------------
class VisionTransformerHetero(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        mlp_ratio_list=None,
        num_heads_list=None,
        change_qkv=False,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pre_norm=True,
        gp=False,
        relative_position=False,
        abs_pos=True,
        max_relative_position=14,
    ):
        super().__init__()

        assert mlp_ratio_list is not None and len(mlp_ratio_list) == depth
        assert num_heads_list is not None and len(num_heads_list) == depth

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.gp = gp
        self.abs_pos = abs_pos
        self.pre_norm = pre_norm
        self.change_qkv = change_qkv

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList()

        for i in range(depth):
            qkv_dim = num_heads_list[i] * 64 if change_qkv else embed_dim
            self.blocks.append(
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads_list[i],
                    mlp_ratio=mlp_ratio_list[i],
                    qkv_dim=qkv_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    relative_position=relative_position,
                    max_relative_position=max_relative_position,
                )
            )

        self.norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        b = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:], dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------
def resolve_split_dir(root, split_name):
    root = Path(root)
    candidates = [
        root / split_name,
        root / "imagenet-mini" / split_name,
        root,
    ]

    for c in candidates:
        if c.exists():
            class_dirs = [p for p in c.iterdir() if p.is_dir()]
            if len(class_dirs) > 0:
                return str(c)

    raise FileNotFoundError(f"Could not find a valid '{split_name}' split under: {root}")


def build_imagenetmini_loaders(data_path, input_size=224, batch_size=64, num_workers=4):
    train_dir = resolve_split_dir(data_path, "train")
    val_dir = resolve_split_dir(data_path, "val")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, val_dataset, train_loader, val_loader


# ---------------------------------------------------------
# Sampling / copy helpers
# ---------------------------------------------------------
def normalize_subnet_config(subnet_cfg):
    subnet_cfg = dict(subnet_cfg)

    if "layer_num" not in subnet_cfg:
        if "depth" in subnet_cfg:
            subnet_cfg["layer_num"] = subnet_cfg["depth"]
        else:
            raise KeyError("Sampled subnet config has neither 'layer_num' nor 'depth'.")

    depth = int(subnet_cfg["layer_num"])

    if not isinstance(subnet_cfg.get("embed_dim"), list):
        subnet_cfg["embed_dim"] = [subnet_cfg["embed_dim"]] * depth
    if not isinstance(subnet_cfg.get("mlp_ratio"), list):
        subnet_cfg["mlp_ratio"] = [subnet_cfg["mlp_ratio"]] * depth
    if not isinstance(subnet_cfg.get("num_heads"), list):
        subnet_cfg["num_heads"] = [subnet_cfg["num_heads"]] * depth

    return subnet_cfg


def sample_linear_weight(weight, in_dim, out_dim):
    return weight[:out_dim, :in_dim].contiguous()


def sample_linear_bias(bias, out_dim):
    if bias is None:
        return None
    return bias[:out_dim].contiguous()


def sample_qkv_weight_change_qkv(weight, sample_in_dim, sample_out_dim):
    # same as qkv_super weight slicing
    w = weight[:, :sample_in_dim]
    w = torch.cat([w[i:sample_out_dim:3, :] for i in range(3)], dim=0)
    return w.contiguous()


def copy_ln(dst_ln, src_ln_super, dim):
    dst_ln.weight.data.copy_(src_ln_super.weight[:dim])
    dst_ln.bias.data.copy_(src_ln_super.bias[:dim])


def copy_rel_pos(dst_rel, src_rel_super, head_dim):
    dst_rel.embeddings_table_v.data.copy_(src_rel_super.embeddings_table_v[:, :head_dim])
    dst_rel.embeddings_table_h.data.copy_(src_rel_super.embeddings_table_h[:, :head_dim])


def build_standalone_from_config(args, subnet_cfg):
    depth = subnet_cfg["layer_num"]
    embed_dim = subnet_cfg["embed_dim"][0]

    if len(set(subnet_cfg["embed_dim"])) != 1:
        raise ValueError("This standalone extraction assumes one shared embed_dim across blocks.")

    model = VisionTransformerHetero(
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=args.num_classes,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio_list=subnet_cfg["mlp_ratio"],
        num_heads_list=subnet_cfg["num_heads"],
        change_qkv=args.change_qkv,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path,
        pre_norm=True,
        gp=args.gp,
        relative_position=args.relative_position,
        abs_pos=not args.no_abs_pos,
        max_relative_position=args.max_relative_position,
    )
    return model


def copy_subnet_weights(supernet, standalone, subnet_cfg, change_qkv=False, relative_position=False):
    depth = subnet_cfg["layer_num"]
    embed_dim = subnet_cfg["embed_dim"][0]

    # patch embedding
    standalone.patch_embed.proj.weight.data.copy_(
        supernet.patch_embed_super.proj.weight.data[:embed_dim]
    )
    standalone.patch_embed.proj.bias.data.copy_(
        supernet.patch_embed_super.proj.bias.data[:embed_dim]
    )

    # cls token and pos embed
    standalone.cls_token.data.copy_(supernet.cls_token.data[..., :embed_dim])
    if standalone.pos_embed is not None:
        standalone.pos_embed.data.copy_(supernet.pos_embed.data[..., :embed_dim])

    # blocks
    for i in range(depth):
        sb = supernet.blocks[i]
        db = standalone.blocks[i]

        mlp_ratio = subnet_cfg["mlp_ratio"][i]
        num_heads = subnet_cfg["num_heads"][i]
        ffn_dim = int(embed_dim * mlp_ratio)
        qkv_dim = num_heads * 64 if change_qkv else embed_dim

        copy_ln(db.attn_norm, sb.attn_layer_norm, embed_dim)
        copy_ln(db.ffn_norm, sb.ffn_layer_norm, embed_dim)

        # qkv
        if change_qkv:
            db.attn.qkv.weight.data.copy_(
                sample_qkv_weight_change_qkv(
                    sb.attn.qkv.weight.data,
                    sample_in_dim=embed_dim,
                    sample_out_dim=3 * qkv_dim,
                )
            )
        else:
            db.attn.qkv.weight.data.copy_(
                sample_linear_weight(sb.attn.qkv.weight.data, embed_dim, 3 * qkv_dim)
            )

        # qkv bias is prefix slicing
        if db.attn.qkv.bias is not None and sb.attn.qkv.bias is not None:
            db.attn.qkv.bias.data.copy_(sample_linear_bias(sb.attn.qkv.bias.data, 3 * qkv_dim))

        # proj
        db.attn.proj.weight.data.copy_(
            sample_linear_weight(sb.attn.proj.weight.data, qkv_dim, embed_dim)
        )
        if db.attn.proj.bias is not None and sb.attn.proj.bias is not None:
            db.attn.proj.bias.data.copy_(sample_linear_bias(sb.attn.proj.bias.data, embed_dim))

        if relative_position:
            head_dim = qkv_dim // num_heads
            copy_rel_pos(db.attn.rel_pos_embed_k, sb.attn.rel_pos_embed_k, head_dim)
            copy_rel_pos(db.attn.rel_pos_embed_v, sb.attn.rel_pos_embed_v, head_dim)

        # mlp
        db.mlp.fc1.weight.data.copy_(
            sample_linear_weight(sb.fc1.weight.data, embed_dim, ffn_dim)
        )
        if db.mlp.fc1.bias is not None and sb.fc1.bias is not None:
            db.mlp.fc1.bias.data.copy_(sample_linear_bias(sb.fc1.bias.data, ffn_dim))

        db.mlp.fc2.weight.data.copy_(
            sample_linear_weight(sb.fc2.weight.data, ffn_dim, embed_dim)
        )
        if db.mlp.fc2.bias is not None and sb.fc2.bias is not None:
            db.mlp.fc2.bias.data.copy_(sample_linear_bias(sb.fc2.bias.data, embed_dim))

    # final norm
    if isinstance(standalone.norm, nn.LayerNorm):
        copy_ln(standalone.norm, supernet.norm, embed_dim)

    # classifier head
    standalone.head.weight.data.copy_(
        sample_linear_weight(supernet.head.weight.data, embed_dim, standalone.head.out_features)
    )
    if standalone.head.bias is not None and supernet.head.bias is not None:
        standalone.head.bias.data.copy_(sample_linear_bias(supernet.head.bias.data, standalone.head.out_features))


# ---------------------------------------------------------
# Eval / train helpers
# ---------------------------------------------------------
def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    sum_top1 = 0.0
    sum_top5 = 0.0
    sum_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = topk_accuracy(outputs, targets, topk=(1, 5))

        bs = images.size(0)
        total += bs
        sum_loss += loss.item() * bs
        sum_top1 += acc1.item() * bs / 100.0
        sum_top5 += acc5.item() * bs / 100.0

    return {
        "loss": sum_loss / total,
        "top1": 100.0 * sum_top1 / total,
        "top5": 100.0 * sum_top5 / total,
        "n": total,
    }


def train_one_epoch_fixed(model, loader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def make_analog_standalone(digital_model, device):
    rpu_config = gen_rpu_config()
    analog_model = convert_to_analog(
        copy.deepcopy(digital_model).cpu(),
        rpu_config=rpu_config,
    )
    analog_model = analog_model.to(device)
    return analog_model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--digital-ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)

    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="./standalone_analog_ckpts")

    parser.add_argument("--gp", action="store_true")
    parser.add_argument("--relative_position", action="store_true")
    parser.add_argument("--change_qkv", action="store_true")
    parser.add_argument("--no_abs_pos", action="store_true")
    parser.add_argument("--max_relative_position", type=int, default=14)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    update_config_from_file(args.cfg)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Building digital supernet...")
    supernet = DigitalVision_TransformerSuper(
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.num_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
    )

    ckpt_path = Path(args.digital_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}\n"
            f"Current working directory: {Path.cwd()}"
        )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = supernet.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded: {args.digital_ckpt}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    # legit AutoFormer search space
    choices = {
        "embed_dim": list(cfg.SEARCH_SPACE.EMBED_DIM),
        "mlp_ratio": list(cfg.SEARCH_SPACE.MLP_RATIO),
        "num_heads": list(cfg.SEARCH_SPACE.NUM_HEADS),
        "depth": list(cfg.SEARCH_SPACE.DEPTH),
    }

    print("\nSearch space used for sampling:")
    print(choices)

    subnet_cfg = sample_configs(choices)
    subnet_cfg = normalize_subnet_config(subnet_cfg)

    print("\nSampled subnet config:")
    print(subnet_cfg)

    # activate subnet in supernet
    supernet.set_sample_config(subnet_cfg)

    # build standalone and copy weights
    standalone = build_standalone_from_config(args, subnet_cfg)
    copy_subnet_weights(
        supernet=supernet,
        standalone=standalone,
        subnet_cfg=subnet_cfg,
        change_qkv=args.change_qkv,
        relative_position=args.relative_position,
    )
    standalone = standalone.to(device)

    # data
    train_dataset, val_dataset, train_loader, val_loader = build_imagenetmini_loaders(
        data_path=args.data_path,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ---------------------------------------------------------
    # 1) Evaluate DIGITAL standalone
    # ---------------------------------------------------------
    print("\n=== Evaluating DIGITAL standalone subnet ===")
    digital_metrics = evaluate(standalone, val_loader, device)
    print(f"[Digital] Loss : {digital_metrics['loss']:.4f}")
    print(f"[Digital] Top1 : {digital_metrics['top1']:.2f}%")
    print(f"[Digital] Top5 : {digital_metrics['top5']:.2f}%")

    # ---------------------------------------------------------
    # 2) Convert standalone to ANALOG and evaluate before training
    # ---------------------------------------------------------
    print("\n=== Converting standalone model to ANALOG ===")
    analog_standalone = make_analog_standalone(standalone, device)

    print("\n=== Evaluating ANALOG standalone before HW-aware training ===")
    analog_before_metrics = evaluate(analog_standalone, val_loader, device)
    print(f"[Analog before] Loss : {analog_before_metrics['loss']:.4f}")
    print(f"[Analog before] Top1 : {analog_before_metrics['top1']:.2f}%")
    print(f"[Analog before] Top5 : {analog_before_metrics['top5']:.2f}%")

    # ---------------------------------------------------------
    # 3) HW-aware fine-tuning of the ANALOG standalone
    # ---------------------------------------------------------
    print("\n=== Starting HW-aware fine-tuning for ANALOG standalone ===")
    criterion = nn.CrossEntropyLoss()

    optimizer = AnalogAdam(
        analog_standalone.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    if hasattr(optimizer, "regroup_param_groups"):
        optimizer.regroup_param_groups(analog_standalone)

    best_top1 = -1.0
    best_path = Path(args.output_dir) / "best_analog_standalone.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_fixed(
            analog_standalone,
            train_loader,
            optimizer,
            criterion,
            device=device,
            clip_grad=1.0,
        )

        val_metrics = evaluate(analog_standalone, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Top1 {val_metrics['top1']:.2f}% | "
            f"Val Top5 {val_metrics['top5']:.2f}%"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(analog_standalone.state_dict(), best_path)

    # ---------------------------------------------------------
    # 4) Final evaluation after HW-aware training
    # ---------------------------------------------------------
    print("\n=== Evaluating ANALOG standalone after HW-aware training ===")
    analog_after_metrics = evaluate(analog_standalone, val_loader, device)
    print(f"[Analog after] Loss : {analog_after_metrics['loss']:.4f}")
    print(f"[Analog after] Top1 : {analog_after_metrics['top1']:.2f}%")
    print(f"[Analog after] Top5 : {analog_after_metrics['top5']:.2f}%")

    final_path = Path(args.output_dir) / "final_analog_standalone.pth"
    torch.save(analog_standalone.state_dict(), final_path)

    print(f"\nSaved best analog standalone to:  {best_path}")
    print(f"Saved final analog standalone to: {final_path}")


if __name__ == "__main__":
    main()