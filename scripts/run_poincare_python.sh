#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-config/poincare.yaml}

echo "======================================"
echo "Running Poincare pipeline using config: $CONFIG_FILE"
echo "======================================"

n_lines=$(yq '.n_lines' "$CONFIG_FILE")
filter_nouns=$(yq '.filter_nouns' "$CONFIG_FILE")
use_transitive_closure=$(yq '.use_transitive_closure' "$CONFIG_FILE")
head=$(yq '.head' "$CONFIG_FILE") # true/false を保持

timestamp=$(date +"%Y%m%d_%H%M%S")

# run_nameの基本部分を定義
run_name="rows${n_lines}_${timestamp}_transitive${use_transitive_closure}"

# headの値に基づいてサフィックスを追加
if [ "$head" = "true" ]; then
    run_name="${run_name}_head"
else
    # headがtrueでない場合 (_random)
    run_name="${run_name}_random"
fi

# output_dirを定義し、ディレクトリを作成
output_dir="output/Python/${run_name}"
mkdir -p "$output_dir"

echo "run_name: $run_name"
echo "output_dir: $output_dir"
echo "======================================"

echo ">>> Step1: Generating random train/eval data"
# head変数の値に基づいて実行するPythonスクリプトを決定
if [ "$head" = "true" ]; then
    script_to_run="/Users/honokakobayashi/dev/Univ/Research/app/datageneration.py"
    echo "Running dataGeneration.py (Head/Ordered mode)"
else
    script_to_run="/Users/honokakobayashi/dev/Univ/Research/app/dataGenerationRandom.py"
    echo "Running dataGenerationRandom.py (Random mode)"
fi

# 決定したスクリプトを実行。引数は同じ。
python3 "$script_to_run" \
  "$n_lines" "$filter_nouns" "$output_dir"
train_csv="${output_dir}/train.csv"
eval_csv="${output_dir}/eval.csv"

if [ "$use_transitive_closure" = "true" ]; then
  echo ">>> Step1.5: Computing transitive closure"
  python3 /Users/honokakobayashi/dev/Univ/Research/app/transitive_closure.py \
    "$train_csv" "${output_dir}/train_closure.csv"
  train_csv="${output_dir}/train_closure.csv"
  echo "Using transitive closure for training"
else
  echo ">>> Step0.5: Skipping transitive closure"
fi

echo ">>> Step2: Running Python Poincaré embedding (Gensim)"
python3 /Users/honokakobayashi/dev/Univ/Research/app/poincare.py "$train_csv" "$output_dir"

echo ">>> Step3: Python embeddings evaluation"
python3 /Users/honokakobayashi/dev/Univ/Research/app/evaluation.py \
  "$output_dir/embedding_python.csv" \
  "$eval_csv" \
  "$train_csv" \
  "$output_dir/evaluation.txt"

echo ">>> Step4: Visualizing embeddings"
python3 /Users/honokakobayashi/dev/Univ/Research/app/visualize.py \
  "$output_dir/embedding_python.csv" "$output_dir/poincare_disk.pdf"

summary_file="$output_dir/run_summary.txt"
{
  echo "======================================"
  echo " POINCARÉ PIPELINE RUN SUMMARY"
  echo "======================================"
  echo "Date: $(date)"
  echo ""
  echo "Config file: $CONFIG_FILE"
  echo "Output dir : $output_dir"
  echo ""
  echo "--- Parameters ---"
  echo "n_lines      : $n_lines"
  echo "filter_nouns: $filter_nouns"
  echo "use_transitive_closure: $use_transitive_closure"
  echo ""
  echo "--- Files in this run ---"
  ls -1 "$output_dir"
  echo ""
  echo "======================================"
} > "$summary_file"

echo "✅ All results saved in: $output_dir"
