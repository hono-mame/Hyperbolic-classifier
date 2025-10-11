import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split # 分割にはsklearnの利用を推奨

# 名詞だけに絞るかどうか
filter_nouns = True  # True: 名詞のみ, False: 全品詞

# ランダムに抽出する行数
n_head = 1000 

conn = sqlite3.connect("/Users/honokakobayashi/dev/Univ/Research/data/wnjpn.db")

query = """
SELECT 
    w2.lemma AS hyper,
    w1.lemma AS hypo
FROM synlink AS sl
INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1
INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2
INNER JOIN sense AS se1 ON se1.synset = sy1.synset
INNER JOIN sense AS se2 ON se2.synset = sy2.synset
INNER JOIN word AS w1 ON w1.wordid = se1.wordid
INNER JOIN word AS w2 ON w2.wordid = se2.wordid
WHERE sl.link = 'hypo'
  AND se1.lang = 'jpn' AND se2.lang = 'jpn'
  AND w1.lang = 'jpn' AND w2.lang = 'jpn'
"""

if filter_nouns:
    query += " AND sy1.pos = 'n' AND sy2.pos = 'n'"

df = pd.read_sql_query(query, conn)
base_path = "/Users/honokakobayashi/dev/Univ/Research/data/Hyperbolic/"
file_suffix = "_nouns" if filter_nouns else ""

# 【変更箇所】ファイル名に "_random" を追加
n_head_suffix = f"_random_{n_head}" if n_head else "" 

# 全データの保存（このファイル名は変更なし）
output_file = base_path + f"hypernym_relations_jpn{file_suffix}.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"抽出完了。 {len(df)} 行を {output_file} に保存しました。")

# ランダムに n_head 行の抽出
if n_head > 0 and n_head < len(df):
    # ランダムサンプリング
    df_head = df.sample(n=n_head, replace=False, random_state=42).copy()
    print(f"ランダムに {n_head} 行を抽出したデータフレームを作成しました。")
else:
    df_head = df.copy() # n_headを指定しない、または全行の場合
    print("全行を使用します。")

# train_test_splitを使ってランダムに80%を訓練、20%をテストに分割
df_train, df_test = train_test_split(df_head, test_size=0.2, random_state=42) 


# 保存
# 訓練データ (80%)
# 【変更箇所】ファイル名に "_random_n" が含まれます
train_file = base_path + f"hypernym_relations_jpn{file_suffix}_train{n_head_suffix}.csv"
df_train.to_csv(train_file, index=False, encoding="utf-8")
print(f"訓練データ: {len(df_train)} 行を {train_file} に保存しました。")

# テストデータ (20%)
# 【変更箇所】ファイル名に "_random_n" が含まれます
test_file = base_path + f"hypernym_relations_jpn{file_suffix}_eval{n_head_suffix}.csv"
df_test.to_csv(test_file, index=False, encoding="utf-8")
print(f"テストデータ: {len(df_test)} 行を {test_file} に保存しました。")

conn.close()