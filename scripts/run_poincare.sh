#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-config/poincare.yaml}

echo "======================================"
echo "Running Poincare pipeline using config: $CONFIG_FILE"
echo "======================================"

# --- YAMLから設定読み込み ---
dim=$(yq '.dim' "$CONFIG_FILE")
epochs=$(yq '.epochs' "$CONFIG_FILE")
baseLR=$(yq '.baseLR' "$CONFIG_FILE")
negK=$(yq '.negK' "$CONFIG_FILE")
burnC=$(yq '.burnC' "$CONFIG_FILE")
burnEpochs=$(yq '.burnEpochs' "$CONFIG_FILE")
batchSize=$(yq '.batchSize' "$CONFIG_FILE")
train_csv=$(yq -r '.trainCSV' "$CONFIG_FILE")
eval_csv=$(yq -r '.evalCSV' "$CONFIG_FILE")

# --- 出力ディレクトリ作成 ---
row_count=$(docker-compose exec -T hasktorch bash -c "tail -n +2 /home/ubuntu/Research/$train_csv | wc -l | tr -d ' '")
run_name="dim${dim}_ep${epochs}_lr${baseLR}_neg${negK}_rows${row_count}"
output_dir="output/${run_name}"
mkdir -p "$output_dir"

echo "Run name    : $run_name"
echo "Output dir  : $output_dir"
echo "======================================"

# --- Step1: PoincareBatch.hs ---
echo ">>> Step1: Running PoincareBatch.hs"
docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run PoincareBatch \
    $dim $epochs $baseLR $negK $burnC $burnEpochs $batchSize \
    $train_csv $output_dir/embeddings.csv $output_dir/learning_curve.png
"

# --- Step2: Evaluation.hs ---
echo ">>> Step2: Running Evaluation.hs"
eval_output="$output_dir/evaluation_results.txt"

docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run Evaluation $output_dir/embeddings.csv $eval_csv $eval_output
"

echo ""
echo "======================================"
echo "Pipeline finished."
echo "All outputs saved in: $output_dir"
echo "Evaluation results:   $eval_output"
echo "======================================"
