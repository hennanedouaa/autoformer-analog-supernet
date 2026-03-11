'''
python generate_validation_pool.py \
    --cfg AutoFormer/experiments/supernet/supernet-T.yaml \
    --output validation_pool_T.json
'''

import json
import random
import argparse

from AutoFormer.lib.config import cfg, update_config_from_file


# ------------------------------------------------------------
# LHS sampler
# ------------------------------------------------------------
def lhs_unit(n_samples: int, dim: int, seed: int = 42):

    rng = random.Random(seed)
    result = [[0.0] * dim for _ in range(n_samples)]

    for d in range(dim):
        vals = [(i + rng.random()) / n_samples for i in range(n_samples)]
        rng.shuffle(vals)

        for i in range(n_samples):
            result[i][d] = vals[i]

    return result


def map_to_choice(x, choices):

    k = len(choices)
    idx = min(int(x * k), k - 1)
    return choices[idx]


def subnet_to_key(cfg):
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------
# LHS subnet generator
# ------------------------------------------------------------
def generate_lhs_subnet_configs(
        depth_choices,
        embed_choices,
        mlp_choices,
        head_choices,
        n_configs=40,
        seed=42,
):

    max_depth = max(depth_choices)
    dim = 2 + max_depth * 2

    unique = {}
    tries = 0

    while len(unique) < n_configs:

        points = lhs_unit(n_configs * 2, dim, seed + tries)

        for p in points:

            depth = map_to_choice(p[0], depth_choices)
            embed = map_to_choice(p[1], embed_choices)

            mlp_ratio = []
            num_heads = []

            for b in range(max_depth):

                r = map_to_choice(p[2 + 2*b], mlp_choices)
                h = map_to_choice(p[2 + 2*b + 1], head_choices)

                if b < depth:
                    mlp_ratio.append(r)
                    num_heads.append(h)

            cfg = {
                "layer_num": depth,
                "embed_dim": [embed] * depth,
                "mlp_ratio": mlp_ratio,
                "num_heads": num_heads,
            }

            unique[subnet_to_key(cfg)] = cfg

            if len(unique) >= n_configs:
                break

        tries += 1

    return list(unique.values())[:n_configs]


# ------------------------------------------------------------
# Anchor configs
# ------------------------------------------------------------
def alternating_pattern(length, a, b):
    return [a if i % 2 == 0 else b for i in range(length)]


def first_half_second_half(length, first, second):

    split = length // 2
    return [first] * split + [second] * (length - split)


def build_anchor_configs(depth_choices, embed_choices, mlp_choices, head_choices):

    anchors = []

    min_d = min(depth_choices)
    mid_d = depth_choices[len(depth_choices)//2]
    max_d = max(depth_choices)

    min_e = min(embed_choices)
    mid_e = embed_choices[len(embed_choices)//2]
    max_e = max(embed_choices)

    min_m = min(mlp_choices)
    max_m = max(mlp_choices)

    min_h = min(head_choices)
    max_h = max(head_choices)

    def make_cfg(depth, embed, mlp_ratio, num_heads):

        return {
            "layer_num": depth,
            "embed_dim": [embed] * depth,
            "mlp_ratio": mlp_ratio,
            "num_heads": num_heads,
        }

    anchors.append(make_cfg(min_d, min_e, [min_m]*min_d, [min_h]*min_d))
    anchors.append(make_cfg(mid_d, mid_e, [min_m]*mid_d, [min_h]*mid_d))
    anchors.append(make_cfg(max_d, max_e, [max_m]*max_d, [max_h]*max_d))

    anchors.append(make_cfg(min_d, max_e, [max_m]*min_d, [max_h]*min_d))
    anchors.append(make_cfg(max_d, min_e, [min_m]*max_d, [min_h]*max_d))

    anchors.append(make_cfg(
        max_d,
        mid_e,
        alternating_pattern(max_d, min_m, max_m),
        alternating_pattern(max_d, min_h, max_h)
    ))

    anchors.append(make_cfg(
        max_d,
        mid_e,
        first_half_second_half(max_d, min_m, max_m),
        first_half_second_half(max_d, min_h, max_h)
    ))

    anchors.append(make_cfg(
        mid_d,
        max_e,
        alternating_pattern(mid_d, min_m, max_m),
        [max_h]*mid_d
    ))

    anchors.append(make_cfg(
        mid_d,
        min_e,
        [max_m]*mid_d,
        alternating_pattern(mid_d, min_h, max_h)
    ))

    anchors.append(make_cfg(
        max_d,
        max_e,
        [max_m]*max_d,
        [max_h]*max_d
    ))

    return anchors


# ------------------------------------------------------------
# Full config
# ------------------------------------------------------------
def build_full_config(depth_choices, embed_choices, mlp_choices, head_choices):

    depth = max(depth_choices)
    embed = max(embed_choices)
    mlp = max(mlp_choices)
    head = max(head_choices)

    return {
        "layer_num": depth,
        "embed_dim": [embed] * depth,
        "mlp_ratio": [mlp] * depth,
        "num_heads": [head] * depth,
    }


# ------------------------------------------------------------
# Merge configs
# ------------------------------------------------------------
def merge_configs(lhs, anchors, full_config, target=50):

    merged = {}
    ordered = []

    for cfg in anchors + lhs:

        key = subnet_to_key(cfg)

        if key not in merged:
            merged[key] = cfg
            ordered.append(cfg)

    ordered = ordered[:target]

    if subnet_to_key(full_config) not in {subnet_to_key(c) for c in ordered}:
        ordered.append(full_config)

    return ordered


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", default="validation_pool.json")
    parser.add_argument("--pool-size", type=int, default=50)

    args = parser.parse_args()

    update_config_from_file(args.cfg)

    depth_choices = list(cfg.SEARCH_SPACE.DEPTH)
    embed_choices = list(cfg.SEARCH_SPACE.EMBED_DIM)
    mlp_choices = list(cfg.SEARCH_SPACE.MLP_RATIO)
    head_choices = list(cfg.SEARCH_SPACE.NUM_HEADS)

    print("Depth:", depth_choices)
    print("Embed:", embed_choices)
    print("MLP:", mlp_choices)
    print("Heads:", head_choices)

    lhs = generate_lhs_subnet_configs(
        depth_choices,
        embed_choices,
        mlp_choices,
        head_choices,
        n_configs=args.pool_size - 10
    )

    anchors = build_anchor_configs(
        depth_choices,
        embed_choices,
        mlp_choices,
        head_choices
    )

    full_config = build_full_config(
        depth_choices,
        embed_choices,
        mlp_choices,
        head_choices
    )

    final_pool = merge_configs(lhs, anchors, full_config, args.pool_size)

    with open(args.output, "w") as f:
        json.dump(final_pool, f, indent=2)

    print(f"\nSaved {len(final_pool)} configs to {args.output}")


if __name__ == "__main__":
    main()