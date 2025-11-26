#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-config/poincare_steps3_5.yaml}

echo "======================================"
echo " Training and Evaluating in Python"
echo " Using config: $CONFIG_FILE"
echo "======================================"

train_csv_abs=$(yq -r '.train_csv_abs' "$CONFIG_FILE")
eval_csv_abs=$(yq -r '.eval_csv_abs' "$CONFIG_FILE")
base_output_dir=$(yq -r '.output_dir_abs' "$CONFIG_FILE")
evaluation_output=$(yq -r '.evaluation_output' "$CONFIG_FILE")
use_transitive=$(yq -r '.use_transitive_closure' "$CONFIG_FILE")

if [ "$use_transitive" = "true" ]; then
    python_output_dir="${base_output_dir}_python_transitiveTrue"
else
    python_output_dir="${base_output_dir}_python_transitiveFalse"
fi
mkdir -p "$python_output_dir"

echo "[INFO] Python output directory: $python_output_dir"

poincare_script="/Users/honokakobayashi/dev/Univ/Research/app/poincare.py"
evaluation_script="/Users/honokakobayashi/dev/Univ/Research/app/evaluation.py"

echo ">>> Step1: Running Python Poincaré embedding (Gensim)"
python3 "$poincare_script" "$train_csv_abs" "$python_output_dir"

echo ">>> Step2: Python embeddings evaluation"
python3 "$evaluation_script" \
  "$python_output_dir/embedding_python.csv" \
  "$eval_csv_abs" \
  "$train_csv_abs" \
  "$python_output_dir/$evaluation_output"

echo "======================================"
echo "✅ Pipeline completed successfully."
echo "Results saved in: $python_output_dir/$evaluation_output"
echo "======================================"
