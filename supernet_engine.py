"""
supernet_engine.py

Utilities for initializing an analog AutoFormer supernet from a digital one.
"""

import torch
import torch.nn as nn
import random
from tqdm import tqdm

def initialize_analog_supernet(digital_model, analog_model, verbose=True):
    """
    Copy all weights from a digital AutoFormer supernet to an analog AutoFormer supernet.
    """
    # Put both models in eval mode to disable dropout/stochastic depth
    digital_model.eval()
    analog_model.eval()

    if verbose:
        print("=" * 60)
        print("Copying ALL weights from digital to analog supernet")
        print("=" * 60)

    # 1. Patch embedding (special case – not a linear super)
    digital_patch_conv = digital_model.patch_embed_super.proj
    analog_model.patch_embed_super.copy_from_digital_conv(digital_patch_conv)
    if verbose:
        print("Patch embedding copied (all embed dims).")

    # 2. Positional embeddings (direct copy)
    analog_model.cls_token.data.copy_(digital_model.cls_token.data)
    if verbose:
        print("cls_token copied.")
    if analog_model.abs_pos:
        analog_model.pos_embed.data.copy_(digital_model.pos_embed.data)
        if verbose:
            print("pos_embed copied.")

    # 3. Transformer blocks
    for i, (digital_blk, analog_blk) in enumerate(zip(digital_model.blocks, analog_model.blocks)):
        if verbose:
            print(f"\n  Processing block {i}...")

        # LayerNorms (direct tensor copy)
        analog_blk.attn_layer_norm.weight.data.copy_(digital_blk.attn_layer_norm.weight.data)
        analog_blk.attn_layer_norm.bias.data.copy_(digital_blk.attn_layer_norm.bias.data)
        analog_blk.ffn_layer_norm.weight.data.copy_(digital_blk.ffn_layer_norm.weight.data)
        analog_blk.ffn_layer_norm.bias.data.copy_(digital_blk.ffn_layer_norm.bias.data)
        if verbose:
            print("    LayerNorms copied.")

        # Attention QKV (uses special handling inside copy_all_ops_from_digital)
        analog_blk.attn.qkv.copy_all_ops_from_digital(digital_blk.attn.qkv)
        if verbose:
            print("    attn.qkv: all ops copied.")

        # Attention projection
        analog_blk.attn.proj.copy_all_ops_from_digital(digital_blk.attn.proj)
        if verbose:
            print("    attn.proj: all ops copied.")

        # MLP fc1
        analog_blk.fc1.copy_all_ops_from_digital(digital_blk.fc1)
        if verbose:
            print("    fc1: all ops copied.")

        # MLP fc2
        analog_blk.fc2.copy_all_ops_from_digital(digital_blk.fc2)
        if verbose:
            print("    fc2: all ops copied.")

        # Relative position embeddings (if used)
        if hasattr(digital_blk.attn, 'rel_pos_embed_k') and hasattr(analog_blk.attn, 'rel_pos_embed_k'):
            analog_blk.attn.rel_pos_embed_k.embeddings_table_v.data.copy_(
                digital_blk.attn.rel_pos_embed_k.embeddings_table_v.data)
            analog_blk.attn.rel_pos_embed_k.embeddings_table_h.data.copy_(
                digital_blk.attn.rel_pos_embed_k.embeddings_table_h.data)
            analog_blk.attn.rel_pos_embed_v.embeddings_table_v.data.copy_(
                digital_blk.attn.rel_pos_embed_v.embeddings_table_v.data)
            analog_blk.attn.rel_pos_embed_v.embeddings_table_h.data.copy_(
                digital_blk.attn.rel_pos_embed_v.embeddings_table_h.data)
            if verbose:
                print("    Relative position embeddings copied.")

    # 4. Final layer norm
    if hasattr(digital_model, 'norm') and digital_model.norm is not None:
        analog_model.norm.weight.data.copy_(digital_model.norm.weight.data)
        analog_model.norm.bias.data.copy_(digital_model.norm.bias.data)
        if verbose:
            print("\nFinal norm copied.")

    # 5. Head classifier
    if isinstance(analog_model.head, nn.Module) and not isinstance(analog_model.head, nn.Identity):
        if hasattr(analog_model.head, 'copy_all_ops_from_digital'):
            analog_model.head.copy_all_ops_from_digital(digital_model.head)
            if verbose:
                print("Head: all ops copied.")
        else:
            # Fallback (shouldn't happen for AnalogLinearSuper)
            analog_model.head.weight.data.copy_(digital_model.head.weight.data)
            if digital_model.head.bias is not None:
                analog_model.head.bias.data.copy_(digital_model.head.bias.data)
            if verbose:
                print("Head copied (direct).")
    else:
        if verbose:
            print("Head is identity, skipping.")

    if verbose:
        print("\n" + "=" * 60)
        print("All analog operators initialized from digital model.")
        print("=" * 60)


class FairSampler:
    """
    Round‑robin sampler that guarantees exact fairness over a cycle of 16 batches.
    
    For each cycle:
        - The 4 embedding dimensions are taken in a random order, each used for 4 consecutive batches.
        - For each block, a random permutation of the 4 (r, h) pairs is created for each embedding dimension.
        - The (r, h) for a block in a batch is the next element from the permutation corresponding to the current d.
    After 16 batches, every block has seen every combination (d, r, h) exactly once.
    At the end of the cycle, the order of d values and each block's per‑d permutations are reshuffled.
    """

    def __init__(self, L, change_qkv, embed_choices, mlp_ratio_choices, num_heads_choices):
        self.L = L
        self.change_qkv = change_qkv
        self.embed_choices = embed_choices

        # Prepare (r, h) combos
        if change_qkv:
            self.rh_combos = [(r, h) for r in mlp_ratio_choices for h in num_heads_choices]  # 4 combos
        else:
            self.rh_combos = [(r,) for r in mlp_ratio_choices]  # 2 combos

        # Generate the first cycle
        self._new_cycle()

        # Pointers for current position in the cycle
        self.step = 0

    def _new_cycle(self):
        """Create a new cycle of 16 batches with guaranteed fairness."""
        # Random order of the 4 embedding dimensions, each repeated 4 times
        d_order = self.embed_choices.copy()
        random.shuffle(d_order)
        self.d_cycle = []
        for d in d_order:
            self.d_cycle.extend([d] * 4)

        # For each block, build a list of 16 (r, h) values that pairs with the d_cycle
        self.rh_cycles = []  # list of lists, one per block
        for b in range(self.L):
            # For each d, create a shuffled permutation of the 4 (r,h) combos
            perms = {}
            for d in self.embed_choices:
                perm = self.rh_combos.copy()
                random.shuffle(perm)
                perms[d] = perm
            # Build the cycle by concatenating the permutations in the order of d_cycle
            rh_cycle = []
            for d in self.d_cycle:
                # pop from the front to avoid reuse
                rh_cycle.append(perms[d].pop(0))
            self.rh_cycles.append(rh_cycle)

    def sample_subnet(self):
        """Return (embed_dim, mlp_ratio, num_heads) for the next batch."""
        # Get current values from the cycle
        d = self.d_cycle[self.step]
        embed_dim = [d] * self.L
        mlp_ratio = []
        num_heads = []
        for b in range(self.L):
            if self.change_qkv:
                r, h = self.rh_cycles[b][self.step]
                num_heads.append(h)
            else:
                r = self.rh_cycles[b][self.step][0]
            mlp_ratio.append(r)

        # Advance step
        self.step += 1
        if self.step == len(self.d_cycle):
            self.step = 0
            self._new_cycle()  # start a new cycle with reshuffled orders

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

        images = images.to(device)
        labels = labels.to(device)

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
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        # -------------------------
        # Backward
        # -------------------------
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # -------------------------
        # Cosine scheduler step
        # -------------------------
        scheduler.step()

        # -------------------------
        # Metrics
        # -------------------------
        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(
            loss=loss.item(),
            acc=100. * correct / total,
            lr=optimizer.param_groups[0]["lr"]
        )

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc