# ----------------------------------------------------------------------
# Full Hardware‑Aware Fine‑Tuning Script with Fair Sampling
# Accepts paths as arguments for reusability
# ----------------------------------------------------------------------
'''
To execute it:
python training_script.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --digital-ckpt checkpoints/supernet-tiny.pth \
    --analog-config configs/analog-supernet-T.yaml \
    --data-path dataset/imagenet-mini \
    --batch-size 64 \
    --epochs 100 \
    --output_dir ./analog-ckpts \
    --gp --change_qkv --relative_position
'''

import torch
import torch.nn as nn
import random
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# AutoFormer imports
from AutoFormer.lib.config import cfg, update_config_from_file
from AutoFormer.lib.datasets import build_dataset
from AutoFormer.model.supernet_transformer import Vision_TransformerSuper as DigitalVision_TransformerSuper
from AutoFormer.supernet_train import get_args_parser
from AutoFormer.lib import utils

# AIHWKit and analog model imports
from aihwkit.optim import AnalogAdam
from hardwareConfig.rpu_config import gen_rpu_config
from model.analog_supernet_transformer import Vision_TransformerSuper as AnalogVision_TransformerSuper
from supernet_engine import initialize_analog_supernet, FairSampler, train_one_epoch

# ----------------------------------------------------------------------
# Fairness test: operator counts over one cycle (each operator should appear twice)
# ----------------------------------------------------------------------
def test_operator_counts(sampler, verbose=True):
    """
    Simulate one full cycle (16 batches) and count how many times each operator
    (fc1, fc2, qkv, proj) is used in each block. Each operator should appear exactly twice.
    """
    cycle_len = len(sampler.d_cycle)  # should be 16
    # We'll store counts per block per role. Each role has a dict mapping operator key to count.
    counts = {
        b: {
            'fc1': {},
            'fc2': {},
            'qkv': {},
            'proj': {}
        } for b in range(sampler.L)
    }

    for step in range(cycle_len):
        embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        d = embed_dim[0]  # all blocks share the same d
        for b in range(sampler.L):
            r = mlp_ratio[b]
            h = num_heads[b] if sampler.change_qkv else None
            d_q = h * 64 if sampler.change_qkv else d  # when change_qkv=True, d_q = h*64
            # fc1 operator key
            fc1_key = f"in{d}_out{int(d*r)}"
            counts[b]['fc1'][fc1_key] = counts[b]['fc1'].get(fc1_key, 0) + 1
            # fc2 operator key
            fc2_key = f"in{int(d*r)}_out{d}"
            counts[b]['fc2'][fc2_key] = counts[b]['fc2'].get(fc2_key, 0) + 1
            # qkv operator key
            qkv_key = f"in{d}_out{3*d_q}"
            counts[b]['qkv'][qkv_key] = counts[b]['qkv'].get(qkv_key, 0) + 1
            # proj operator key
            proj_key = f"in{d_q}_out{d}"
            counts[b]['proj'][proj_key] = counts[b]['proj'].get(proj_key, 0) + 1

    all_ok = True
    for b in range(sampler.L):
        if verbose:
            print(f"\nBlock {b}:")
        for role in ['fc1', 'fc2', 'qkv', 'proj']:
            op_counts = counts[b][role]
            # Check that each operator appears exactly twice
            bad = [k for k, cnt in op_counts.items() if cnt != 2]
            if bad:
                print(f"  ❌ {role}: {len(bad)} operators have count != 2: {bad}")
                all_ok = False
            else:
                if verbose:
                    print(f"  ✅ {role}: all {len(op_counts)} operators appear exactly twice")
    if all_ok:
        print("\n✅ Operator count test passed: each operator used exactly twice in 16 batches.")
    else:
        print("\n⚠️ Operator count test failed.")
    return all_ok

# ----------------------------------------------------------------------
# Fairness test: short exact cycle (should now pass)
# ----------------------------------------------------------------------
def test_sampler_short(sampler, verbose=True):
    """Test that over one cycle each block sees all (d, r, h) combos exactly once."""
    cycle_len = len(sampler.d_cycle)  # should be 16
    counts = {b: {} for b in range(sampler.L)}
    for _ in range(cycle_len):
        embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        d = embed_dim[0]  # all blocks share the same d
        for b in range(sampler.L):
            if sampler.change_qkv:
                key = (d, mlp_ratio[b], num_heads[b])
            else:
                key = (d, mlp_ratio[b])
            counts[b][key] = counts[b].get(key, 0) + 1

    total_combos = len(sampler.embed_choices) * len(sampler.rh_combos)
    all_ok = True
    for b in range(sampler.L):
        if len(counts[b]) != total_combos:
            if verbose:
                print(f"Block {b}: {len(counts[b])} distinct combos out of {total_combos} (expected {total_combos})")
            all_ok = False
        else:
            if any(cnt > 1 for cnt in counts[b].values()):
                if verbose:
                    print(f"Block {b}: some combos appear multiple times")
                all_ok = False
            else:
                if verbose:
                    print(f"Block {b}: all combos appear exactly once")
    if all_ok:
        print("✅ Short fairness test passed!")
    else:
        print("⚠️ Short fairness test failed (unexpected).")
    return all_ok

# ----------------------------------------------------------------------
# Long‑term fairness test (tolerance)
# ----------------------------------------------------------------------
def test_sampler_long(sampler, num_steps=10000, tolerance=0.05, verbose=True):
    d_count = len(sampler.embed_choices)
    rh_count = len(sampler.rh_combos)
    total_combos = d_count * rh_count
    expected = num_steps / total_combos
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)

    counts = {b: {} for b in range(sampler.L)}
    for _ in range(num_steps):
        embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        d = embed_dim[0]
        for b in range(sampler.L):
            if sampler.change_qkv:
                key = (d, mlp_ratio[b], num_heads[b])
            else:
                key = (d, mlp_ratio[b])
            counts[b][key] = counts[b].get(key, 0) + 1

    all_ok = True
    for b in range(sampler.L):
        block_counts = list(counts[b].values())
        if len(block_counts) != total_combos:
            print(f"Block {b}: only {len(block_counts)} combos out of {total_combos} seen")
            all_ok = False
            continue
        min_c = min(block_counts)
        max_c = max(block_counts)
        mean_c = sum(block_counts) / len(block_counts)
        std_c = (sum((c - mean_c)**2 for c in block_counts) / len(block_counts)) ** 0.5
        if verbose:
            print(f"Block {b}: min={min_c}, max={max_c}, mean={mean_c:.2f}, std={std_c:.2f}")
        outliers = [c for c in block_counts if c < lower or c > upper]
        if outliers:
            print(f"  ❌ Block {b} has {len(outliers)} outliers (expected {expected:.2f} ± {tolerance*100:.0f}%)")
            all_ok = False
        else:
            if verbose:
                print(f"  ✅ Block {b} all counts within tolerance.")
    if all_ok:
        print(f"\n✅ Long‑term fairness test passed ({num_steps} steps, tolerance {tolerance*100:.0f}%).")
    else:
        print(f"\n⚠️ Long‑term fairness test failed.")
    return all_ok

def main():
    # ------------------------
    # Parse arguments
    # ------------------------
    parser = get_args_parser()
    parser.add_argument('--digital-ckpt', type=str, required=True,
                        help='Path to digital checkpoint (.pth)')
    parser.add_argument('--analog-config', type=str, required=True,
                        help='Path to analog supernet YAML (e.g., tiny_analog.yaml)')
    args = parser.parse_args()
    
    # Update config with the digital config file
    update_config_from_file(args.cfg)
    
    # Make sure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = args.epochs
    # ------------------------
    # Create Digital Model and Load Pretrained Weights
    # ------------------------
    print("Creating digital model...")
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
    
    if Path(args.digital_ckpt).is_file():
        checkpoint = torch.load(args.digital_ckpt, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        digital_model.load_state_dict(state_dict, strict=False)
        print(f"Digital checkpoint loaded from {args.digital_ckpt}")
    else:
        print("Warning: digital checkpoint not found, using random weights.")
    
    # ------------------------
    # Create Analog Supernet with Hardware Noise
    # ------------------------
    print("Creating analog supernet...")
    with open(args.analog_config, "r") as f:
        SUPER_CONFIG = yaml.safe_load(f)
    
    rpu_config = gen_rpu_config()  # your hardware non‑idealities config
    
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
        drop_path_rate=0.05, #made it smaller 
        gp=args.gp,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        max_relative_position=args.max_relative_position
    ).to(args.device)
    
    # Load digital state dict into analog model (initial weight transfer)
    ckpt = torch.load(args.digital_ckpt, map_location="cpu")
    digital_sd = ckpt.get("model", ckpt)
    analog_model.load_state_dict(digital_sd, strict=False)
    
    # Copy weights from digital to all analog operators
    initialize_analog_supernet(digital_model, analog_model, verbose=True)
    
    # ------------------------
    # Data Loaders (using AutoFormer's build_dataset)
    # ------------------------
    print("Building datasets...")
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
    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")
    
    # ------------------------
    # Get search space from digital config (cfg.SEARCH_SPACE)
    # ------------------------
    embed_choices = cfg.SEARCH_SPACE.EMBED_DIM
    mlp_ratio_choices = cfg.SEARCH_SPACE.MLP_RATIO
    num_heads_choices = cfg.SEARCH_SPACE.NUM_HEADS
    
    # Depth choices - use maximum depth for full training
    depth_choices = cfg.SEARCH_SPACE.DEPTH
    L = max(depth_choices)  # e.g., 14 for tiny
    
    change_qkv = args.change_qkv  # should be True
    
    # ------------------------
    # Fair Sampling Setup (exact per‑cycle fairness)
    # ------------------------
    sampler = FairSampler(
        L=L,
        change_qkv=change_qkv,
        embed_choices=embed_choices,
        mlp_ratio_choices=mlp_ratio_choices,
        num_heads_choices=num_heads_choices
    )
    
    # ------------------------------------------------------------------
    # Test the sampler fairness before starting training
    # ------------------------------------------------------------------
    print("\n--- Testing sampler fairness (short cycle) ---")
    test_sampler_short(sampler, verbose=True)
    
    print("\n--- Testing sampler fairness (operator counts over one cycle) ---")
    test_operator_counts(sampler, verbose=True)
    
    print("\n--- Testing sampler fairness (long‑term, 10000 steps) ---")
    test_sampler_long(sampler, num_steps=10000, tolerance=0.05, verbose=True)
    
    # ------------------------
    # Optimizer and Loss
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
    # Training Loop
    # ------------------------
    
    print("\nStarting hardware‑aware fine‑tuning...")
    for epoch in range(EPOCHS):

        train_loss, train_acc, global_step = train_one_epoch(
        analog_model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        sampler,
        device=args.device,
        clip_grad=1.0
         )
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%")
        
        # Validation
        analog_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = analog_model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        analog_model.train()
    
    # Save fine‑tuned model
    save_path = Path(args.output_dir) / "analog_supernet_finetuned.pth"
    torch.save(analog_model.state_dict(), save_path)
    print(f"Fine‑tuning finished. Model saved to {save_path}")
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
if __name__ == "__main__":
    main()