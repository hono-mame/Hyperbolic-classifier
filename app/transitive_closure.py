#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import sys

if len(sys.argv) < 3:
    print("Usage: transitive_closure.py <input_csv> <output_csv>")
    sys.exit(1)
input_csv = sys.argv[1]
output_csv = sys.argv[2]

df = pd.read_csv(input_csv, header=None)
G = nx.DiGraph()
G.add_edges_from(df.values)
closure = nx.transitive_closure(G)
closure_df = pd.DataFrame(list(closure.edges()))
closure_df.to_csv(output_csv, index=False, header=False)
print(f"Transitive closure saved to {output_csv}")
