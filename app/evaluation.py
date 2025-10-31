#!/usr/bin/env python3
import sys
import math
import torch
import csv
import os
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# ============================================================
# --- Poincaré Distance Function (Haskell版準拠) ---
# ============================================================
def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    HaskellのpoincareDistanceを正確に再現
    """
    eps = 1e-6

    u_norm_sq = torch.sum(u * u).item()
    v_norm_sq = torch.sum(v * v).item()
    diff = u - v
    diff_norm_sq = torch.sum(diff * diff).item()

    num = 2.0 * diff_norm_sq
    denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    safe_denom = denom + eps

    x = 1.0 + (num / safe_denom)

    # x のクリッピング
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
        next(reader, None)  # skip header
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
        next(reader, None)  # skip header
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

    print("======================================")
    print("  Hyperbolic Embedding Evaluation (Filtered)")
    print("======================================")
    print(f"Embeddings : {trained_emb_path}")
    print(f"Eval data  : {eval_data_path}")
    print(f"Train data : {train_data_path}")
    print(f"Result out : {result_file}")
    print("======================================\n")

    start_time = time.time()

    # --- Load data ---
    print("[INFO] Loading embeddings...")
    embeddings = read_embeddings_csv(trained_emb_path)
    print(f"[INFO] Loaded {len(embeddings)} embeddings")

    print("[INFO] Loading eval/train pairs...")
    eval_pairs = read_pairs_from_csv(eval_data_path)
    train_pairs = read_pairs_from_csv(train_data_path)
    print(f"[INFO] Loaded {len(eval_pairs)} eval pairs (before filtering)")
    print(f"[INFO] Loaded {len(train_pairs)} train pairs\n")

    # --- Filter pairs where both hyper & hypo exist ---
    emb_keys = set(embeddings.keys())
    filtered_eval_pairs = [(h, y) for (h, y) in eval_pairs if h in emb_keys and y in emb_keys]
    print(f"[INFO] Filtered eval pairs: {len(filtered_eval_pairs)} (both hyper/hypo exist)\n")

    grouped_eval = group_by_hypernym(filtered_eval_pairs)
    grouped_train = group_by_hypernym(train_pairs)
    all_words = list(embeddings.keys())

    results = []
    debug_texts = []
    debug_limit = 200
    total = len(grouped_eval)
    last_log_time = time.time()

    # --- Evaluation loop ---
    for idx, (hyper, hypos_eval) in enumerate(grouped_eval.items(), start=1):
        known_hypos = set(grouped_train.get(hyper, []))
        candidate_words = [w for w in all_words if w != hyper and w not in known_hypos]

        distances = []
        for w in candidate_words:
            d = poincare_distance(embeddings[hyper], embeddings[w])
            distances.append((w, d))
        ranked = [w for w, _ in sorted(distances, key=lambda x: x[1])]

        ranks_found = [ranked.index(h) for h in hypos_eval if h in ranked]
        rank_value = float(min(ranks_found) + 1) if ranks_found else float(len(ranked))
        ap_value = avg_precision(set(hypos_eval), ranked)
        results.append((rank_value, ap_value))

        # --- Debug logging for first N anchors ---
        if idx <= debug_limit:
            dbg = "\n".join([
                f"[DEBUG] Anchor: {hyper}",
                f"  Eval hyponyms (gold): {' '.join(hypos_eval)}",
                f"  Filtered (train) hypos: {' '.join(sorted(known_hypos))}",
                f"  Candidates after filtering: {len(candidate_words)}",
                f"  Top 10 nearest words: {' '.join(ranked[:10])}",
                f"  Rank: {rank_value}",
                f"  Mean Average Precision (MAP): {ap_value}",
                ""
            ])
            debug_texts.append(dbg)

        # --- Progress log every ~2秒 ---
        if time.time() - last_log_time > 2:
            percent = idx / total * 100
            elapsed = time.time() - start_time
            print(f"[PROGRESS] {idx}/{total} ({percent:.1f}%) done | elapsed: {elapsed:.1f}s")
            last_log_time = time.time()

    # --- Results summary ---
    if not results:
        print("[ERROR] No valid evaluation pairs after filtering.")
        sys.exit(0)

    mean_rank = sum(r for r, _ in results) / len(results)
    mean_ap = sum(ap for _, ap in results) / len(results)

    elapsed = time.time() - start_time
    summary = "\n".join([
        "--- Filtered Evaluation Results (Link Prediction) ---",
        f"Total Hypernyms (Evaluated): {len(results)}",
        f"Mean Rank: {mean_rank}",
        f"Mean Average Precision (MAP): {mean_ap}",
        f"Elapsed Time: {elapsed:.2f} sec",
        "",
        "--- DEBUG (first 5 anchors) ---",
        "\n".join(debug_texts[:5])
    ])

    print("\n" + summary)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n✅ Saved summary to: {result_file}")
    print("======================================")


if __name__ == "__main__":
    main()
