import pandas as pd
import networkx as nx

df = pd.read_csv('/Users/honokakobayashi/dev/Univ/Research/data/Hyperbolic/hypernym_relations_jpn_nouns_train_random_1000.csv', header=None)
# --- 有向グラフを作成 ---
G = nx.DiGraph()
G.add_edges_from(df.values)
# --- 推移閉包を計算 ---
closure = nx.transitive_closure(G)

closure_edges = list(closure.edges())
closure_df = pd.DataFrame(closure_edges)
closure_df.to_csv('/Users/honokakobayashi/dev/Univ/Research/data/Hyperbolic/transitive_hypernym_relations_jpn_nouns_train_random_1000.csv', index=False, header=False)
