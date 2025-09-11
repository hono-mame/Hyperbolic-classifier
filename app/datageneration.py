import sqlite3
import pandas as pd

# 名詞だけに絞るかどうか
filter_nouns = False  # True: 名詞のみ, False: 全品詞

# 先頭 n 行だけを保存
n_head = 10000

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
base_file = "/Users/honokakobayashi/dev/Univ/Research/data/Hyperbolic/hypernym_relations_jpn"
output_file = base_file + ("_nouns.csv" if filter_nouns else ".csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"抽出完了。 {len(df)} 行を {output_file} に保存しました。")

head_file = base_file + ("_nouns_head" if filter_nouns else "_head") + f"_{n_head}.csv"
df.head(n_head).to_csv(head_file, index=False, encoding="utf-8-sig")
print(f"先頭 {n_head} 行を {head_file} に保存しました。")

conn.close()
