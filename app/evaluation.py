#!/usr/bin/env python3
import sys
import math
import torch
import csv
import os
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# --- Poincaré Distance Function ---
# ============================================================
def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    eps = 1e-6
    u_norm_sq = torch.sum(u * u).item()
    v_norm_sq = torch.sum(v * v).item()
    diff = u - v
    diff_norm_sq = torch.sum(diff * diff).item()
    num = 2.0 * diff_norm_sq
    denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    safe_denom = denom + eps
    x = 1.0 + (num / safe_denom)
    if x < 1.0 + eps:
        x = 1.0 + eps
    return math.acosh(x)

# ============================================================
# --- Embeddings Reader ---
# ============================================================
def read_embeddings_csv(path: str) -> Dict[str, torch.Tensor]:
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            word = row[0].strip()
            try:
                vec = torch.tensor([float(x) for x in row[1:]], dtype=torch.float32)
                embeddings[word] = vec
            except ValueError:
                continue
    return embeddings

# ============================================================
# --- Read hyper/hypo pairs ---
# ============================================================
def read_pairs_from_csv(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            pairs.append((row[0].strip(), row[1].strip()))
    return pairs

# ============================================================
# --- Group by hypernym ---
# ============================================================
def group_by_hypernym(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    grouped = defaultdict(list)
    for hyper, hypo in pairs:
        grouped[hyper].append(hypo)
    return grouped

# ============================================================
# --- Average Precision ---
# ============================================================
def avg_precision(gold_set: Set[str], ranked_list: List[str]) -> float:
    if not gold_set:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, w in enumerate(ranked_list, start=1):
        if w in gold_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(gold_set)

# ============================================================
# --- Plotting Function ---
# ============================================================
def plot_histograms(results: List[Tuple[float, float]], output_base_path: str):
    if not results:
        print("[WARNING] No results to plot.")
        return
    ranks = np.array([r for r, _ in results])
    maps = np.array([ap for _, ap in results])
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=50, log=True, color='#1f77b4', edgecolor='black')
    plt.axvline(ranks.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean Rank: {ranks.mean():.2f}')
    plt.title('Distribution of Minimum Ranks (Lower is Better)', fontsize=16)
    plt.xlabel('Minimum Rank', fontsize=14)
    plt.ylabel('Frequency (Log Scale)', fontsize=14)
    plt.legend()
    plt.savefig(f"{output_base_path}_rank_hist.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(maps, bins=20, range=(0, 1), color='#ff7f0e', edgecolor='black')
    plt.axvline(maps.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean AP (MAP): {maps.mean():.4f}')
    plt.title('Distribution of Mean Average Precision (MAP)', fontsize=16)
    plt.xlabel('Average Precision (AP)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.savefig(f"{output_base_path}_map_hist.png", bbox_inches='tight')
    plt.close()

# ============================================================
# --- Main Evaluation ---
# ============================================================
def main():
    if len(sys.argv) < 5:
        print("Usage: python evaluation.py <embeddings.csv> <eval.csv> <train.csv> <output.txt>")
        sys.exit(1)

    trained_emb_path = sys.argv[1]
    eval_data_path = sys.argv[2]
    train_data_path = sys.argv[3]
    result_file = sys.argv[4]

    start_time = time.time()

    embeddings = read_embeddings_csv(trained_emb_path)
    eval_pairs = read_pairs_from_csv(eval_data_path)
    train_pairs = read_pairs_from_csv(train_data_path)

    emb_keys = set(embeddings.keys())
    train_hypers = set(h for h, _ in train_pairs)  # ← ② train に存在する hypernym のみを評価対象に

    # --- ② hyper が train に存在しないペアを除外 ---
    filtered_eval_pairs = [
        (h, y) for (h, y) in eval_pairs
        if h in emb_keys and y in emb_keys and h in train_hypers
    ]

    grouped_eval = group_by_hypernym(filtered_eval_pairs)
    grouped_train = group_by_hypernym(train_pairs)
    all_words = list(embeddings.keys())

    results = []
    debug_texts = []
    debug_limit = 200

    for idx, (hyper, hypos_eval) in enumerate(grouped_eval.items(), start=1):
        known_hypos = set(grouped_train.get(hyper, []))

        # --- ① train に既出の hyponym を gold から除外 ---
        hypos_eval = [h for h in hypos_eval if h not in known_hypos]
        if not hypos_eval:
            continue  # goldが全てtrainに含まれる場合は評価対象外

        candidate_words = [w for w in all_words if w != hyper and w not in known_hypos]
        if hyper not in embeddings:
            continue

        distances = []
        for w in candidate_words:
            if w not in embeddings:
                continue
            d = poincare_distance(embeddings[hyper], embeddings[w])
            distances.append((w, d))

        ranked = [w for w, _ in sorted(distances, key=lambda x: x[1])]
        ranks_found = [ranked.index(h) for h in hypos_eval if h in ranked]
        rank_value = float(min(ranks_found) + 1) if ranks_found else float(len(ranked))
        ap_value = avg_precision(set(hypos_eval), ranked)
        results.append((rank_value, ap_value))

        if idx <= debug_limit:
            debug_texts.append(
                f"[DEBUG] Anchor: {hyper}\n"
                f"  Eval hyponyms (gold): {' '.join(hypos_eval)}\n"
                f"  Filtered (train) hypos: {' '.join(sorted(known_hypos))}\n"
                f"  Candidates after filtering: {len(candidate_words)}\n"
                f"  Top 10 nearest words: {' '.join(ranked[:10])}\n"
                f"  Rank: {rank_value}\n"
                f"  MAP: {ap_value}\n"
            )

    if not results:
        print("[ERROR] No valid evaluation pairs after filtering.")
        sys.exit(0)

    mean_rank = sum(r for r, _ in results) / len(results)
    mean_ap = sum(ap for _, ap in results) / len(results)

    output_base_path = os.path.splitext(result_file)[0]
    plot_histograms(results, output_base_path)

    summary = "\n".join([
        "--- Filtered Evaluation Results (Link Prediction) ---",
        f"Total Hypernyms (Evaluated): {len(results)}",
        f"Mean Rank: {mean_rank}",
        f"Mean Average Precision (MAP): {mean_ap}",
        "",
        "--- DEBUG (First {debug_limit} Anchors) ---",
        "\n".join(debug_texts)
    ])

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[INFO] Saved summary text to: {result_file}")

if __name__ == "__main__":
    main()
