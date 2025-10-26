# 実行方法について
## 一括実行
```
bash scripts/run_poincare.sh config/config.yaml 
```
yamlは以下。
```yaml
# --- dataset generation ---
n_lines: 150
filter_nouns: true
use_transitive_closure: true

# --- training settings ---
dim: 5
epochs: 100
baseLR: 0.05
negK: 10
burnC: 0.1
burnEpochs: 10
batchSize: 256
```
一括実行すると、yamlの設定とタイムスタンプを元に一意のディレクトリが作成され、その中に全てのデータが出力される。実行のまとめの情報が、/run_summary.txtとして作成される。
```zsh
output/
└── dim5_ep100_lr0.05_neg10_rows150_20251026_182333/
    ├── train.csv
    ├── train_closure.csv
    ├── eval.csv
    ├── embeddings.csv
    ├── learning_curve.png
    ├── evaluation_results.txt
    ├── poincare_disk.pdf
    └── run_summary.txt
```
---
### STEP1: データセット生成  
WordNetのデータベースからtrain, evalデータセットを作成。　　  
何行のデータを使用するか、名詞に絞るかどうかをyamlで指定できる。  
```yaml
n_lines: 150
filter_nouns: true
```
個別実行する際は以下のコマンド。
```zsh
dataGenerationRandom.py <n_lines> <filter_nouns(True/False)> <output_dir>
```

### STEP1.5: 推移閉包の作成
trainデータについて、推移閉包を作成。   
yamlで実行するかスキップするかを、use_transitive_closureで指定可能。
```yaml
use_transitive_closure: true 
```
個別実行する際は以下のコマンド。
```zsh
transitive_closure.py <input_csv> <output_csv>
```
---
### STEP2: Hyperbolic Embedding による学習
作成したtrainデータに対して、学習を実行。    
yamlで各パラメータを指定する。
```yaml
dim: 5
epochs: 100
baseLR: 0.05
negK: 10
burnC: 0.1
burnEpochs: 10
batchSize: 256
```
個別実行する際は以下のコマンド。
```zsh
docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run PoincareBatch \
    <dim> <epochs> <baseLR> <negK> <burnC> <burnEpochs> <batchSize> \
    <path_to_train_csv> <path_to_output_embedding_csv> <path_to_output_lerningCurve.png>
```
---
### STEP3: 学習後のEmbeddingの評価
評価結果がテキストファイルとして出力される。RankとMAPが計算される。
個別実行する際は以下のコマンド。
```zsh
docker-compose exec hasktorch /bin/bash -c "
  cd /home/ubuntu/Research && \
  stack run Evaluation <path_to_trained_embedding_csv> <path_to_eval_csv> <path_to_train_csv> <path_to_eval_output_txt>
"
```
---
### STEP4: Embeddingの可視化
Embeddingを２次元で可視化した結果を出力する。
個別実行する際は以下のコマンド。
```zsh
python3 visualize.py <path_to_embeddings_csv>  <path_to_visualize_output_pdf>

```