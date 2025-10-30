#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import csv
import tempfile
from gensim.models.poincare import PoincareModel

def convert_to_tsv_if_needed(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.readline()
        delimiter = "," if "," in sample else "\t"
    if delimiter == "\t":
        return path
    tmp_tsv = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv").name
    with open(path, "r", encoding="utf-8") as fin, open(tmp_tsv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter=",")
        writer = csv.writer(fout, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                writer.writerow(row[:2])
    print(f"[INFO] Converted {path} → {tmp_tsv} (TSV)")
    return tmp_tsv

def read_relations_tsv(path):
    relations = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                relations.append((row[0], row[1]))
    return relations

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 poincare.py <train_csv> <output_dir>")
        sys.exit(1)

    train_csv = sys.argv[1]
    output_dir = sys.argv[2]

    print("======================================")
    print("   Poincaré Embeddings   ")
    print("======================================")
    print(f"Train file : {train_csv}")
    print(f"Output dir : {output_dir}")
    print("======================================\n")

    train_tsv = convert_to_tsv_if_needed(train_csv)

    print("=== Step 1: Reading training relations ===")
    train_pairs = read_relations_tsv(train_tsv)
    print(f"[INFO] Loaded {len(train_pairs)} training pairs")

    print("\n=== Step 2: Training Poincaré Embeddings ===")
    model = PoincareModel(train_pairs, size=3, negative=10)
    model.train(epochs=200)
    print("[INFO] Training completed.")

    print("\n=== Step 3: Saving embeddings to CSV ===")
    embedding_csv_path = f"{output_dir}/embedding_python.csv"
    with open(embedding_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word"] + [f"dim{i+1}" for i in range(model.kv.vector_size)])
        for word in model.kv.index_to_key:
            vector = model.kv[word]
            writer.writerow([word] + vector.tolist())

    print(f"[INFO] Saved embeddings to {embedding_csv_path}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
