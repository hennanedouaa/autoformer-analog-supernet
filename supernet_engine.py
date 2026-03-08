"""
supernet_engine.py

Utilities for initializing an analog AutoFormer supernet from a digital one.
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
    Round-robin sampler that guarantees exact fairness over one cycle.

    For each cycle:
        - Embedding dimensions are shuffled and each repeated len(rh_combos) times.
        - For each block and each embedding dimension, a shuffled permutation of
          the available (r, h) combinations is created.
        - Over a full cycle, each block sees every combination exactly once.
    """

    def __init__(self, L, change_qkv, embed_choices, mlp_ratio_choices, num_heads_choices):
        self.L = L
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
        Create a fresh cycle with exact fairness.
        Cycle length = len(embed_choices) * len(rh_combos)
        """
        combos_per_d = len(self.rh_combos)

        d_order = self.embed_choices.copy()
        random.shuffle(d_order)

        self.d_cycle = []
        for d in d_order:
            self.d_cycle.extend([d] * combos_per_d)

        self.rh_cycles = []
        for _ in range(self.L):
            perms = {}
            for d in self.embed_choices:
                perm = self.rh_combos.copy()
                random.shuffle(perm)
                perms[d] = perm

            used_count = {d: 0 for d in self.embed_choices}
            rh_cycle = []

            for d in self.d_cycle:
                idx = used_count[d]
                rh_cycle.append(perms[d][idx])
                used_count[d] += 1

            self.rh_cycles.append(rh_cycle)

        self.step = 0

    def sample_subnet(self):
        """
        Return (embed_dim, mlp_ratio, num_heads) for the next batch.
        """
        d = self.d_cycle[self.step]
        embed_dim = [d] * self.L
        mlp_ratio = []
        num_heads = []

        for b in range(self.L):
            if self.change_qkv:
                r, h = self.rh_cycles[b][self.step]
                mlp_ratio.append(r)
                num_heads.append(h)
            else:
                r = self.rh_cycles[b][self.step][0]
                mlp_ratio.append(r)

        self.step += 1
        if self.step == len(self.d_cycle):
            self._new_cycle()

        return embed_dim, mlp_ratio, num_heads


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

    loop = tqdm(loader, desc="Training")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # -------------------------
        # Sample subnet
        # -------------------------
        embed_dim, mlp_ratio, num_heads = sampler.sample_subnet()

        config_dict = {
            "layer_num": sampler.L,
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
            lr=optimizer.param_groups[0]["lr"]
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)

    return epoch_loss, epoch_acc