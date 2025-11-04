#!/usr/bin/env python3
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

if len(sys.argv) < 4:
    print("Usage: dataGenerationRandom.py <n_lines> <filter_nouns> <output_dir>")
    sys.exit(1)
n_lines = int(sys.argv[1])
filter_nouns = sys.argv[2].lower() == "true"
output_dir = sys.argv[3]
os.makedirs(output_dir, exist_ok=True)

conn = sqlite3.connect("/Users/honokakobayashi/dev/Univ/Research/data/wnjpn.db")

query = """
SELECT 
    w1.lemma AS hyper,
    w2.lemma AS hypo
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
print(f"Total rows before sampling: {len(df)}")
if 0 < n_lines < len(df):
    df = df.sample(n=n_lines, replace=False, random_state=42)
    print(f"Randomly sampled {n_lines} rows.")
else:
    print("Using all rows.")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

train_file = os.path.join(output_dir, "train.csv")
eval_file = os.path.join(output_dir, "eval.csv")
df_train.to_csv(train_file, index=False, header=False, encoding="utf-8")
df_test.to_csv(eval_file, index=False, header=False, encoding="utf-8")

print(f"Train saved to: {train_file}")
print(f"Eval saved to : {eval_file}")

conn.close()