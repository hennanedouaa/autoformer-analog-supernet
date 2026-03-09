"""
supernet_engine.py

Utilities for initializing an analog AutoFormer supernet from a digital one,
plus a fair subnet sampler and one-epoch training helper.

Modified version:
- adds BALANCED DEPTH SAMPLING
- returns sampled layer_num instead of forcing fixed max depth
"""

import random
import torch
import torch.nn as nn
from tqdm import tqdm


def initialize_analog_supernet(digital_model, analog_model, verbose=True):
    """
    Copy all weights from a digital AutoFormer supernet to an analog AutoFormer supernet.

    IMPORTANT:
    - This should be called while BOTH models are still on CPU.
    - Move the analog model to CUDA only AFTER this function finishes.
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

        # ------------------------------------------------------------------
        # 1. Patch embedding
        # ------------------------------------------------------------------
        log("STEP 1: patch embedding...")
        digital_patch_conv = digital_model.patch_embed_super.proj
        analog_model.patch_embed_super.copy_from_digital_conv(digital_patch_conv)
        log("DONE 1: patch embedding copied.")

        # ------------------------------------------------------------------
        # 2. cls_token and positional embeddings
        # ------------------------------------------------------------------
        log("STEP 2: cls_token...")
        analog_model.cls_token.copy_(digital_model.cls_token)
        log("DONE 2: cls_token copied.")

        if getattr(analog_model, "abs_pos", False):
            log("STEP 3: pos_embed...")
            analog_model.pos_embed.copy_(digital_model.pos_embed)
            log("DONE 3: pos_embed copied.")
        else:
            log("STEP 3: pos_embed skipped (abs_pos=False).")

        # ------------------------------------------------------------------
        # 3. Transformer blocks
        # ------------------------------------------------------------------
        log("STEP 4: transformer blocks...")
        for i, (digital_blk, analog_blk) in enumerate(zip(digital_model.blocks, analog_model.blocks)):
            log(f"\nBLOCK {i}: start")

            # LayerNorms
            log(f"BLOCK {i}: attn_layer_norm...")
            analog_blk.attn_layer_norm.weight.copy_(digital_blk.attn_layer_norm.weight)
            analog_blk.attn_layer_norm.bias.copy_(digital_blk.attn_layer_norm.bias)

            log(f"BLOCK {i}: ffn_layer_norm...")
            analog_blk.ffn_layer_norm.weight.copy_(digital_blk.ffn_layer_norm.weight)
            analog_blk.ffn_layer_norm.bias.copy_(digital_blk.ffn_layer_norm.bias)
            log(f"BLOCK {i}: LayerNorms copied.")

            # Attention QKV
            log(f"BLOCK {i}: attn.qkv...")
            analog_blk.attn.qkv.copy_all_ops_from_digital(digital_blk.attn.qkv)
            log(f"BLOCK {i}: attn.qkv copied.")

            # Attention projection
            log(f"BLOCK {i}: attn.proj...")
            analog_blk.attn.proj.copy_all_ops_from_digital(digital_blk.attn.proj)
            log(f"BLOCK {i}: attn.proj copied.")

            # MLP fc1
            log(f"BLOCK {i}: fc1...")
            analog_blk.fc1.copy_all_ops_from_digital(digital_blk.fc1)
            log(f"BLOCK {i}: fc1 copied.")

            # MLP fc2
            log(f"BLOCK {i}: fc2...")
            analog_blk.fc2.copy_all_ops_from_digital(digital_blk.fc2)
            log(f"BLOCK {i}: fc2 copied.")

            # Relative position embeddings
            has_rel_k = hasattr(digital_blk.attn, "rel_pos_embed_k") and hasattr(analog_blk.attn, "rel_pos_embed_k")
            has_rel_v = hasattr(digital_blk.attn, "rel_pos_embed_v") and hasattr(analog_blk.attn, "rel_pos_embed_v")

            if has_rel_k and has_rel_v:
                log(f"BLOCK {i}: relative position embeddings...")
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
                log(f"BLOCK {i}: relative position embeddings copied.")
            else:
                log(f"BLOCK {i}: relative position embeddings skipped.")

            log(f"BLOCK {i}: done")

        # ------------------------------------------------------------------
        # 4. Final norm
        # ------------------------------------------------------------------
        if hasattr(digital_model, "norm") and digital_model.norm is not None:
            log("\nSTEP 5: final norm...")
            analog_model.norm.weight.copy_(digital_model.norm.weight)
            analog_model.norm.bias.copy_(digital_model.norm.bias)
            log("DONE 5: final norm copied.")
        else:
            log("\nSTEP 5: final norm skipped.")

        # ------------------------------------------------------------------
        # 5. Head classifier
        # ------------------------------------------------------------------
        log("STEP 6: head...")
        if isinstance(analog_model.head, nn.Module) and not isinstance(analog_model.head, nn.Identity):
            if hasattr(analog_model.head, "copy_all_ops_from_digital"):
                analog_model.head.copy_all_ops_from_digital(digital_model.head)
                log("DONE 6: head copied with copy_all_ops_from_digital.")
            else:
                analog_model.head.weight.copy_(digital_model.head.weight)
                if getattr(digital_model.head, "bias", None) is not None:
                    analog_model.head.bias.copy_(digital_model.head.bias)
                log("DONE 6: head copied directly.")
        else:
            log("STEP 6: head skipped (Identity).")

        log("\n" + "=" * 60)
        log("All analog operators initialized from digital model.")
        log("=" * 60)


class FairSampler:
    """
    Fair round-robin sampler with balanced DEPTH sampling.

    What is balanced over one cycle:
    - each (depth, embed_dim) pair appears exactly len(rh_combos) times
    - for each active block b < depth, and for each (depth, embed_dim) pair,
      every (mlp_ratio, num_heads) combo appears exactly once over the cycle

    Notes:
    - max_L = max(depth_choices)
    - blocks with index >= sampled depth are inactive and therefore ignored
    - returned lists have length = sampled layer_num
    """

    def __init__(self, depth_choices, change_qkv, embed_choices, mlp_ratio_choices, num_heads_choices):
        self.depth_choices = list(depth_choices)
        self.max_L = max(self.depth_choices)
        self.change_qkv = change_qkv
        self.embed_choices = list(embed_choices)

        if self.change_qkv:
            self.rh_combos = [(r, h) for r in mlp_ratio_choices for h in num_heads_choices]
        else:
            self.rh_combos = [(r,) for r in mlp_ratio_choices]

        self.step = 0
        self._new_cycle()

    def _new_cycle(self):
        """
        Create a fresh cycle.

        Cycle length:
            len(depth_choices) * len(embed_choices) * len(rh_combos)

        Construction:
        - we create all (depth, embed_dim) pairs
        - shuffle their order
        - repeat each pair len(rh_combos) times
        - for each active block b and each pair (L, d), assign a shuffled
          permutation of rh_combos so every combo is used exactly once
        """
        combos_per_pair = len(self.rh_combos)

        depth_embed_pairs = [(L, d) for L in self.depth_choices for d in self.embed_choices]
        random.shuffle(depth_embed_pairs)

        self.depth_embed_cycle = []
        for pair in depth_embed_pairs:
            self.depth_embed_cycle.extend([pair] * combos_per_pair)

        # For each block index b in [0, max_L), we will build a cycle of RH choices.
        # For inactive blocks (b >= sampled depth), the stored value is unused.
        self.rh_cycles = [[] for _ in range(self.max_L)]

        # For every block and every (L, d) pair, create a shuffled permutation
        # of rh_combos so active blocks see exact fairness over the cycle.
        perms_by_block = []
        for b in range(self.max_L):
            per_pair = {}
            for pair in depth_embed_pairs:
                L, d = pair
                if b < L:
                    perm = self.rh_combos.copy()
                    random.shuffle(perm)
                    per_pair[pair] = perm
            perms_by_block.append(per_pair)

        # Track how many times we have already used each pair for each block.
        used_count_by_block = []
        for b in range(self.max_L):
            counts = {}
            for pair in depth_embed_pairs:
                L, _ = pair
                if b < L:
                    counts[pair] = 0
            used_count_by_block.append(counts)

        for pair in self.depth_embed_cycle:
            L, _ = pair
            for b in range(self.max_L):
                if b < L:
                    idx = used_count_by_block[b][pair]
                    self.rh_cycles[b].append(perms_by_block[b][pair][idx])
                    used_count_by_block[b][pair] += 1
                else:
                    self.rh_cycles[b].append(None)

        self.step = 0

    def sample_subnet(self):
        """
        Return:
            layer_num, embed_dim, mlp_ratio, num_heads

        where:
            - layer_num is the sampled depth
            - embed_dim has length layer_num
            - mlp_ratio has length layer_num
            - num_heads has length layer_num (or [] if change_qkv=False)
        """
        layer_num, d = self.depth_embed_cycle[self.step]

        embed_dim = [d] * layer_num
        mlp_ratio = []
        num_heads = []

        for b in range(layer_num):
            choice = self.rh_cycles[b][self.step]

            if self.change_qkv:
                r, h = choice
                mlp_ratio.append(r)
                num_heads.append(h)
            else:
                r = choice[0]
                mlp_ratio.append(r)

        self.step += 1
        if self.step == len(self.depth_embed_cycle):
            self._new_cycle()

        return layer_num, embed_dim, mlp_ratio, num_heads


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    sampler,
    device,
    clip_grad=1.0
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Optional stats so you can verify balanced depth exposure during training
    depth_counts = {int(L): 0 for L in sampler.depth_choices}

    loop = tqdm(loader, desc="Training")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # -------------------------
        # Sample subnet (NOW includes balanced depth)
        # -------------------------
        layer_num, embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()
        depth_counts[int(layer_num)] += 1

        config_dict = {
            "layer_num": layer_num,
            "embed_dim": embed_dim,
            "mlp_ratio": mlp_ratio,
        }

        if sampler.change_qkv:
            config_dict["num_heads"] = num_heads

        model.set_sample_config(config_dict)

        # -------------------------
        # Forward
        # -------------------------
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # -------------------------
        # Backward
        # -------------------------
        loss.backward()

        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # -------------------------
        # Scheduler step
        # -------------------------
        if scheduler is not None:
            scheduler.step()

        # -------------------------
        # Metrics
        # -------------------------
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(
            loss=float(loss.item()),
            acc=100.0 * correct / max(total, 1),
            lr=optimizer.param_groups[0]["lr"],
            depth=layer_num
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)

    print(f"Depth usage this epoch: {depth_counts}", flush=True)

    return epoch_loss, epoch_acc