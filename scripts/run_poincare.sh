#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-config/poincare.yaml}

echo "======================================"
echo "Running Poincare pipeline using config: $CONFIG_FILE"
echo "======================================"

dim=$(yq '.dim' "$CONFIG_FILE")
epochs=$(yq '.epochs' "$CONFIG_FILE")
baseLR=$(yq '.baseLR' "$CONFIG_FILE")
negK=$(yq '.negK' "$CONFIG_FILE")
burnC=$(yq '.burnC' "$CONFIG_FILE")
burnEpochs=$(yq '.burnEpochs' "$CONFIG_FILE")
batchSize=$(yq '.batchSize' "$CONFIG_FILE")
n_lines=$(yq '.n_lines' "$CONFIG_FILE")
filter_nouns=$(yq '.filter_nouns' "$CONFIG_FILE")
use_transitive_closure=$(yq '.use_transitive_closure' "$CONFIG_FILE")
timestamp=$(date +"%Y%m%d_%H%M%S")
run_name="dim${dim}_ep${epochs}_lr${baseLR}_neg${negK}_rows${n_lines}_${timestamp}"
output_dir="output/${run_name}"
mkdir -p "$output_dir"

echo "Run name   : $run_name"
echo "Output dir : $output_dir"
echo "======================================"

echo ">>> Step1: Generating random train/eval data"
python3 /Users/honokakobayashi/dev/Univ/Research/app/dataGenerationRandom.py \
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

echo ">>> Step2: Running PoincareBatch.hs"
docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run PoincareBatch \
    $dim $epochs $baseLR $negK $burnC $burnEpochs $batchSize \
    $train_csv $output_dir/embeddings.csv $output_dir/learning_curve.png
"
echo ">>> Step3: Running Python Poincaré embedding (Gensim)"
python3 /Users/honokakobayashi/dev/Univ/Research/app/poincare.py "$train_csv" "$output_dir"

echo ">>> Step4: Haskell embeddings evaluation"
eval_output="$output_dir/evaluation_results.txt"

docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run Evaluation $output_dir/embeddings.csv $eval_csv $train_csv $eval_output
"
echo ">>> Step5: Python embeddings evaluation"
docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run Evaluation $output_dir/embedding_python.csv $eval_csv $train_csv $output_dir/evaluation_python.txt
"

echo ">>> Step6: Visualizing embeddings"
python3 /Users/honokakobayashi/dev/Univ/Research/app/visualize.py \
  "$output_dir/embeddings.csv" "$output_dir/poincare_disk.pdf"

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
  echo "dim         : $dim"
  echo "epochs      : $epochs"
  echo "baseLR      : $baseLR"
  echo "negK        : $negK"
  echo "burnC       : $burnC"
  echo "burnEpochs  : $burnEpochs"
  echo "batchSize   : $batchSize"
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
