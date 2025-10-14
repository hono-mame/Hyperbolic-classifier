import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareKeyedVectors
import matplotlib

if len(sys.argv) < 3:
    print("Usage: python3 visualize.py <embeddings.csv> <output.pdf>")
    sys.exit(1)

csv_path = sys.argv[1]
output_pdf = sys.argv[2]

# ===== フォント設定 =====
plt.rcParams['font.family'] = 'Hiragino Sans'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ===== CSVを読み込み =====
df = pd.read_csv(csv_path)
words = df.iloc[:, 0].values
vecs = df.iloc[:, 1:].values.astype(float)

# ===== Poincaréベクトルオブジェクト作成 =====
kv = PoincareKeyedVectors(vector_size=vecs.shape[1], vector_count=len(words))
kv.add_vectors(words, vecs)

# ===== ノルムを確認 =====
norms = np.linalg.norm(vecs, axis=1)
print(f"平均ノルム: {norms.mean():.3f}, 最大ノルム: {norms.max():.3f}")

# ===== 可視化 =====
fig, ax = plt.subplots(figsize=(8, 8))
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linewidth=1.5)
ax.add_artist(circle)

x, y = vecs[:, 0], vecs[:, 1]
ax.scatter(x, y, c='royalblue', s=30)

for i, word in enumerate(words):
    ax.text(x[i], y[i], word, fontsize=10, ha='center', va='center')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title("Poincaré Disk Visualization (2D projection)", fontsize=14)
ax.axis('off')

plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"✅ 保存しました: {output_pdf}")
