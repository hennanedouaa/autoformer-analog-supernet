# ----------------------------------------------------------------------
# Full Hardware-Aware Fine-Tuning Script with Fair Sampling
# Safe analog initialization version
# Modified to support BALANCED DEPTH SAMPLING
# ----------------------------------------------------------------------
"""
To execute it:

python training_script.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --digital-ckpt digital_ckpts/supernet-tiny.pth \
    --analog-config configs/analog-supernet-T.yaml \
    --data-path /home/douaa/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini \
    --batch-size 64 \
    --epochs 100 \
    --output_dir ./analog-ckpts \
    --gp --change_qkv --relative_position
"""

import inspect
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# AutoFormer imports
from AutoFormer.lib.config import cfg, update_config_from_file
from AutoFormer.lib.datasets import build_dataset
from AutoFormer.model.supernet_transformer import Vision_TransformerSuper as DigitalVision_TransformerSuper
from AutoFormer.supernet_train import get_args_parser

# AIHWKit and analog model imports
from aihwkit.optim import AnalogAdam
from hardwareConfig.rpu_config import gen_rpu_config
from model.analog_supernet_transformer import Vision_TransformerSuper as AnalogVision_TransformerSuper
from supernet_engine import FairSampler, train_one_epoch


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def safe_torch_load(path, map_location="cpu"):
    """
    Use weights_only=True when supported by the installed torch version.
    Falls back safely on older versions.
    """
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        pass
    return torch.load(path, map_location=map_location)


def get_param_groups(model, weight_decay=0.01):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "layernorm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def initialize_analog_supernet_safe(digital_model, analog_model, verbose=True):
    """
    Safe CPU-side copy from digital AutoFormer supernet to analog AutoFormer supernet.
    """

    def log(msg):
        if verbose:
            print(msg, flush=True)

    digital_model.eval()
    analog_model.eval()

    with torch.no_grad():
        log("=" * 60)
        log("Copying ALL weights from digital to analog supernet")
        log("=" * 60)

        # 1. Patch embedding
        log("STEP 1: patch embedding")
        digital_patch_conv = digital_model.patch_embed_super.proj
        analog_model.patch_embed_super.copy_from_digital_conv(digital_patch_conv)
        log("DONE 1: patch embedding")

        # 2. Positional embeddings
        log("STEP 2: cls_token")
        analog_model.cls_token.copy_(digital_model.cls_token)
        log("DONE 2: cls_token")

        if getattr(analog_model, "abs_pos", False):
            log("STEP 3: pos_embed")
            analog_model.pos_embed.copy_(digital_model.pos_embed)
            log("DONE 3: pos_embed")
        else:
            log("STEP 3: pos_embed skipped (abs_pos=False)")

        # 3. Transformer blocks
        for i, (digital_blk, analog_blk) in enumerate(zip(digital_model.blocks, analog_model.blocks)):
            log(f"BLOCK {i}: start")

            log(f"BLOCK {i}: layer norms")
            analog_blk.attn_layer_norm.weight.copy_(digital_blk.attn_layer_norm.weight)
            analog_blk.attn_layer_norm.bias.copy_(digital_blk.attn_layer_norm.bias)
            analog_blk.ffn_layer_norm.weight.copy_(digital_blk.ffn_layer_norm.weight)
            analog_blk.ffn_layer_norm.bias.copy_(digital_blk.ffn_layer_norm.bias)
            log(f"BLOCK {i}: layer norms copied")

            log(f"BLOCK {i}: attn.qkv")
            analog_blk.attn.qkv.copy_all_ops_from_digital(digital_blk.attn.qkv)
            log(f"BLOCK {i}: attn.qkv copied")

            log(f"BLOCK {i}: attn.proj")
            analog_blk.attn.proj.copy_all_ops_from_digital(digital_blk.attn.proj)
            log(f"BLOCK {i}: attn.proj copied")

            log(f"BLOCK {i}: fc1")
            analog_blk.fc1.copy_all_ops_from_digital(digital_blk.fc1)
            log(f"BLOCK {i}: fc1 copied")

            log(f"BLOCK {i}: fc2")
            analog_blk.fc2.copy_all_ops_from_digital(digital_blk.fc2)
            log(f"BLOCK {i}: fc2 copied")

            has_rel_k = hasattr(digital_blk.attn, "rel_pos_embed_k") and hasattr(analog_blk.attn, "rel_pos_embed_k")
            has_rel_v = hasattr(digital_blk.attn, "rel_pos_embed_v") and hasattr(analog_blk.attn, "rel_pos_embed_v")

            if has_rel_k and has_rel_v:
                log(f"BLOCK {i}: relative position embeddings")
                analog_blk.attn.rel_pos_embed_k.embeddings_table_v.copy_(
                    digital_blk.attn.rel_pos_embed_k.embeddings_table_v
                )
                analog_blk.attn.rel_pos_embed_k.embeddings_table_h.copy_(
                    digital_blk.attn.rel_pos_embed_k.embeddings_table_h
                )
                analog_blk.attn.rel_pos_embed_v.embeddings_table_v.copy_(
                    digital_blk.attn.rel_pos_embed_v.embeddings_table_v
                )
                analog_blk.attn.rel_pos_embed_v.embeddings_table_h.copy_(
                    digital_blk.attn.rel_pos_embed_v.embeddings_table_h
                )
                log(f"BLOCK {i}: relative position embeddings copied")
            else:
                log(f"BLOCK {i}: relative position embeddings skipped")

            log(f"BLOCK {i}: done")

        # 4. Final norm
        if hasattr(digital_model, "norm") and digital_model.norm is not None:
            log("STEP 4: final norm")
            analog_model.norm.weight.copy_(digital_model.norm.weight)
            analog_model.norm.bias.copy_(digital_model.norm.bias)
            log("DONE 4: final norm")
        else:
            log("STEP 4: final norm skipped")

        # 5. Head classifier
        if isinstance(analog_model.head, nn.Module) and not isinstance(analog_model.head, nn.Identity):
            log("STEP 5: head")
            if hasattr(analog_model.head, "copy_all_ops_from_digital"):
                analog_model.head.copy_all_ops_from_digital(digital_model.head)
            else:
                analog_model.head.weight.copy_(digital_model.head.weight)
                if getattr(digital_model.head, "bias", None) is not None:
                    analog_model.head.bias.copy_(digital_model.head.bias)
            log("DONE 5: head")
        else:
            log("STEP 5: head skipped (Identity)")

        log("=" * 60)
        log("All analog operators initialized from digital model.")
        log("=" * 60)


# ----------------------------------------------------------------------
# Fairness tests adapted for BALANCED DEPTH SAMPLING
# ----------------------------------------------------------------------
def test_depth_fairness(sampler, verbose=True):
    """
    Over one cycle, each depth should appear exactly:
        len(embed_choices) * len(rh_combos)
    times.
    """
    cycle_len = len(sampler.depth_embed_cycle)
    counts = {int(L): 0 for L in sampler.depth_choices}

    for _ in range(cycle_len):
        layer_num, _, _, _ = sampler.sample_subnet()
        counts[int(layer_num)] += 1

    expected = len(sampler.embed_choices) * len(sampler.rh_combos)
    all_ok = True

    for L in sorted(counts.keys()):
        cnt = counts[L]
        if cnt != expected:
            print(f"❌ depth {L}: count={cnt}, expected={expected}", flush=True)
            all_ok = False
        elif verbose:
            print(f"✅ depth {L}: count={cnt}", flush=True)

    if all_ok:
        print("\n✅ Depth fairness test passed.", flush=True)
    else:
        print("\n⚠️ Depth fairness test failed.", flush=True)

    return all_ok


def test_operator_counts(sampler, verbose=True):
    """
    Simulate one full cycle and count how many times each operator
    (fc1, fc2, qkv, proj) is used in each ACTIVE block.

    Important:
    - fc1/fc2 keys collapse over head choice, so their expected count is
      active_depth_count * num_head_choices
    - qkv/proj keys collapse over mlp ratio choice, so their expected count is
      active_depth_count * num_mlp_ratio_choices
    """
    cycle_len = len(sampler.depth_embed_cycle)

    counts = {
        b: {
            "fc1": {},
            "fc2": {},
            "qkv": {},
            "proj": {}
        } for b in range(sampler.max_L)
    }

    num_mlp = len(set(x[0] for x in sampler.rh_combos))
    num_heads = len(set(x[1] for x in sampler.rh_combos)) if sampler.change_qkv else 1

    for _ in range(cycle_len):
        layer_num, embed_dim, mlp_ratio, num_heads_list = sampler.sample_subnet()
        d = embed_dim[0]

        for b in range(layer_num):
            r = mlp_ratio[b]
            h = num_heads_list[b] if sampler.change_qkv else None
            d_q = h * 64 if sampler.change_qkv else d

            fc1_key = f"in{d}_out{int(d * r)}"
            counts[b]["fc1"][fc1_key] = counts[b]["fc1"].get(fc1_key, 0) + 1

            fc2_key = f"in{int(d * r)}_out{d}"
            counts[b]["fc2"][fc2_key] = counts[b]["fc2"].get(fc2_key, 0) + 1

            qkv_key = f"in{d}_out{3 * d_q}"
            counts[b]["qkv"][qkv_key] = counts[b]["qkv"].get(qkv_key, 0) + 1

            proj_key = f"in{d_q}_out{d}"
            counts[b]["proj"][proj_key] = counts[b]["proj"].get(proj_key, 0) + 1

    all_ok = True
    for b in range(sampler.max_L):
        active_depth_count = sum(1 for L in sampler.depth_choices if L > b)

        expected_fc = active_depth_count * num_heads
        expected_qkv = active_depth_count * num_mlp

        if verbose:
            print(f"\nBlock {b}:", flush=True)

        for role in ["fc1", "fc2"]:
            op_counts = counts[b][role]
            bad = [k for k, cnt in op_counts.items() if cnt != expected_fc]
            if bad:
                print(f"  ❌ {role}: {len(bad)} operators have count != {expected_fc}: {bad}", flush=True)
                all_ok = False
            elif verbose:
                print(f"  ✅ {role}: all {len(op_counts)} operators appear exactly {expected_fc} times", flush=True)

        for role in ["qkv", "proj"]:
            op_counts = counts[b][role]
            bad = [k for k, cnt in op_counts.items() if cnt != expected_qkv]
            if bad:
                print(f"  ❌ {role}: {len(bad)} operators have count != {expected_qkv}: {bad}", flush=True)
                all_ok = False
            elif verbose:
                print(f"  ✅ {role}: all {len(op_counts)} operators appear exactly {expected_qkv} times", flush=True)

    if all_ok:
        print("\n✅ Operator count test passed.", flush=True)
    else:
        print("\n⚠️ Operator count test failed.", flush=True)

    return all_ok


def test_sampler_short(sampler, verbose=True):
    """
    For each active block b, over one full cycle:
    each (depth, embed_dim, mlp_ratio, num_heads) combo that activates block b
    should appear exactly once.
    """
    cycle_len = len(sampler.depth_embed_cycle)
    counts = {b: {} for b in range(sampler.max_L)}

    for _ in range(cycle_len):
        layer_num, embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        d = embed_dim[0]

        for b in range(layer_num):
            if sampler.change_qkv:
                key = (layer_num, d, mlp_ratio[b], num_heads[b])
            else:
                key = (layer_num, d, mlp_ratio[b])

            counts[b][key] = counts[b].get(key, 0) + 1

    all_ok = True

    for b in range(sampler.max_L):
        active_depths = [L for L in sampler.depth_choices if L > b]
        total_expected = len(active_depths) * len(sampler.embed_choices) * len(sampler.rh_combos)

        if len(counts[b]) != total_expected:
            if verbose:
                print(f"Block {b}: {len(counts[b])} distinct combos out of {total_expected}", flush=True)
            all_ok = False
        else:
            repeated = any(cnt > 1 for cnt in counts[b].values())
            if repeated:
                if verbose:
                    print(f"Block {b}: some combos appear multiple times", flush=True)
                all_ok = False
            else:
                if verbose:
                    print(f"Block {b}: all active combos appear exactly once", flush=True)

    if all_ok:
        print("✅ Short fairness test passed!", flush=True)
    else:
        print("⚠️ Short fairness test failed.", flush=True)

    return all_ok


def test_sampler_long(sampler, num_steps=10000, tolerance=0.05, verbose=True):
    """
    Long-run approximate fairness test.

    For each active block b, we count keys:
        (depth, embed_dim, mlp_ratio, num_heads)
    among configs where block b is active.

    Since each active key appears once per full global cycle,
    the expected count is:
        num_steps / len(sampler.depth_embed_cycle)
    """
    counts = {b: {} for b in range(sampler.max_L)}
    all_ok = True

    cycle_len = len(sampler.depth_embed_cycle)
    expected = num_steps / cycle_len
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)

    for _ in range(num_steps):
        layer_num, embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        d = embed_dim[0]

        for b in range(layer_num):
            if sampler.change_qkv:
                key = (layer_num, d, mlp_ratio[b], num_heads[b])
            else:
                key = (layer_num, d, mlp_ratio[b])

            counts[b][key] = counts[b].get(key, 0) + 1

    for b in range(sampler.max_L):
        active_depths = [L for L in sampler.depth_choices if L > b]
        total_keys = len(active_depths) * len(sampler.embed_choices) * len(sampler.rh_combos)

        block_counts = list(counts[b].values())

        if len(block_counts) != total_keys:
            print(f"Block {b}: only {len(block_counts)} combos out of {total_keys} seen", flush=True)
            all_ok = False
            continue

        min_c = min(block_counts)
        max_c = max(block_counts)
        mean_c = sum(block_counts) / len(block_counts)
        std_c = (sum((c - mean_c) ** 2 for c in block_counts) / len(block_counts)) ** 0.5

        if verbose:
            print(f"Block {b}: min={min_c}, max={max_c}, mean={mean_c:.2f}, std={std_c:.2f}", flush=True)

        outliers = [c for c in block_counts if c < lower or c > upper]
        if outliers:
            print(
                f"  ❌ Block {b} has {len(outliers)} outliers "
                f"(expected {expected:.2f} ± {tolerance * 100:.0f}%)",
                flush=True
            )
            all_ok = False
        else:
            if verbose:
                print(f"  ✅ Block {b} all counts within tolerance.", flush=True)

    if all_ok:
        print(f"\n✅ Long-term fairness test passed ({num_steps} steps, tolerance {tolerance * 100:.0f}%).", flush=True)
    else:
        print("\n⚠️ Long-term fairness test failed.", flush=True)

    return all_ok


def main():
    # ------------------------
    # Parse arguments
    # ------------------------
    parser = get_args_parser()
    parser.add_argument(
        '--digital-ckpt',
        type=str,
        required=True,
        help='Path to digital checkpoint (.pth)'
    )
    parser.add_argument(
        '--analog-config',
        type=str,
        required=True,
        help='Path to analog supernet YAML'
    )
    args = parser.parse_args()

    update_config_from_file(args.cfg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = args.epochs

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------
    # Create Digital Model
    # ------------------------
    print("Creating digital model...", flush=True)
    digital_model = DigitalVision_TransformerSuper(
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=1000,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos
    )

    digital_state_dict = None
    if Path(args.digital_ckpt).is_file():
        checkpoint = safe_torch_load(args.digital_ckpt, map_location="cpu")
        digital_state_dict = checkpoint.get("model", checkpoint)

        missing, unexpected = digital_model.load_state_dict(digital_state_dict, strict=False)
        print(f"Digital checkpoint loaded from {args.digital_ckpt}", flush=True)
        print(f"Digital load_state_dict -> missing: {len(missing)}, unexpected: {len(unexpected)}", flush=True)
    else:
        print("Warning: digital checkpoint not found, using random weights.", flush=True)

    # ------------------------
    # Create Analog Supernet
    # ------------------------
    print("Creating analog supernet...", flush=True)
    with open(args.analog_config, "r") as f:
        SUPER_CONFIG = yaml.safe_load(f)

    rpu_config = gen_rpu_config()

    analog_model = AnalogVision_TransformerSuper(
        super_config=SUPER_CONFIG,
        rpu_config=rpu_config,
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=1000,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        max_relative_position=args.max_relative_position
    )

    if digital_state_dict is None:
        raise FileNotFoundError(f"Digital checkpoint not found: {args.digital_ckpt}")

    initialize_analog_supernet_safe(digital_model, analog_model, verbose=True)
    print("Analog weight initialization finished successfully.", flush=True)

    digital_model = digital_model.to(args.device)
    analog_model = analog_model.to(args.device)
    print(f"Digital and analog models moved to {args.device}", flush=True)

    # ------------------------
    # Data loaders
    # ------------------------
    print("Building datasets...", flush=True)
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}", flush=True)

    # ------------------------
    # Search space
    # ------------------------
    embed_choices = list(cfg.SEARCH_SPACE.EMBED_DIM)
    mlp_ratio_choices = list(cfg.SEARCH_SPACE.MLP_RATIO)
    num_heads_choices = list(cfg.SEARCH_SPACE.NUM_HEADS)
    depth_choices = list(cfg.SEARCH_SPACE.DEPTH)

    L = max(depth_choices)
    change_qkv = args.change_qkv

    print(f"embed_choices: {embed_choices}", flush=True)
    print(f"mlp_ratio_choices: {mlp_ratio_choices}", flush=True)
    print(f"num_heads_choices: {num_heads_choices}", flush=True)
    print(f"depth_choices: {depth_choices}", flush=True)

    # ------------------------
    # Fair sampler with DEPTH SAMPLING
    # ------------------------
    sampler = FairSampler(
        depth_choices=depth_choices,
        change_qkv=change_qkv,
        embed_choices=embed_choices,
        mlp_ratio_choices=mlp_ratio_choices,
        num_heads_choices=num_heads_choices
    )

    print("\n--- Testing sampler fairness (depth counts over one cycle) ---", flush=True)
    test_depth_fairness(sampler, verbose=True)

    print("\n--- Testing sampler fairness (short cycle) ---", flush=True)
    test_sampler_short(sampler, verbose=True)

    print("\n--- Testing sampler fairness (operator counts over one cycle) ---", flush=True)
    test_operator_counts(sampler, verbose=True)

    print("\n--- Testing sampler fairness (long-term, 10000 steps) ---", flush=True)
    test_sampler_long(sampler, num_steps=10000, tolerance=0.05, verbose=True)

    # Reset sampler after tests so training starts from a fresh cycle
    sampler = FairSampler(
        depth_choices=depth_choices,
        change_qkv=change_qkv,
        embed_choices=embed_choices,
        mlp_ratio_choices=mlp_ratio_choices,
        num_heads_choices=num_heads_choices
    )

    # ------------------------
    # Optimizer and loss
    # ------------------------
    param_groups = get_param_groups(analog_model, weight_decay=0.01)

    optimizer = AnalogAdam(
        param_groups,
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer.regroup_param_groups(analog_model)

    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # ------------------------
    # Validation config
    # NOTE: kept as full-config validation
    # ------------------------
    full_config = {
        "layer_num": L,
        "embed_dim": [cfg.SUPERNET.EMBED_DIM] * L,
        "mlp_ratio": [cfg.SUPERNET.MLP_RATIO] * L,
    }
    if change_qkv:
        full_config["num_heads"] = [cfg.SUPERNET.NUM_HEADS] * L

    # ------------------------
    # Training loop
    # ------------------------
    print("\nStarting hardware-aware fine-tuning...", flush=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            analog_model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            sampler,
            device=args.device,
            clip_grad=1.0
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%",
            flush=True
        )

        # Validation on full config
        analog_model.eval()
        analog_model.set_sample_config(full_config)

        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)

                outputs = analog_model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"Validation Acc (full config): {val_acc:.2f}%", flush=True)

        analog_model.train()

    # ------------------------
    # Save fine-tuned model
    # ------------------------
    save_path = Path(args.output_dir) / "analog_supernet_finetuned.pth"
    torch.save(analog_model.state_dict(), save_path)
    print(f"Fine-tuning finished. Model saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()