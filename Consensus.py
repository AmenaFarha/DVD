import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
import random

def gen_data(num_vectors, dimensions):
    data = np.random.rand(num_vectors, dimensions)
    column_names = [f"Dim_{i+1}" for i in range(dimensions)]
    return pd.DataFrame(data, columns=column_names)

def hash_val(data):
    data_str = ''.join(map(str, data))
    return hashlib.sha256(data_str.encode()).hexdigest()

def get_coordinators(V, D, N, top_k=2):
    shard_map = defaultdict(list)
    coord_map = {}

    for d in D:
        relevance = random.random()
        top_nodes = sorted(N, key=lambda x: relevance * random.random(), reverse=True)[:top_k]
        shard_map[d] = top_nodes
        coord_map[d] = top_nodes

    return shard_map, coord_map

def gen_hash_table(V):
    return {dim: hash_val(data) for dim, data in V.items()}

def check_consensus(dimension, hash_table, nodes):
    ref_hash = hash_table[dimension]
    votes = sum(1 for _ in nodes if hash_val(np.random.rand(10)) == ref_hash)
    return votes >= (len(nodes) // 2) + 1  # 51% consensus threshold

def commit_protocol(V, D, N, sub_cluster_size=5):
    storage = {}
    shard_map, coord_map = get_coordinators(V, D, N, top_k=2)
    hash_table = gen_hash_table(V)

    commits = {}

    for dim, coord_nodes in coord_map.items():
        sub_clusters = {node: random.sample(N, sub_cluster_size) for node in coord_nodes}

        for coord in coord_nodes:
            if hash_table[dim] not in storage:
                storage[hash_table[dim]] = dim

            # Consensus check within sub-cluster
            if check_consensus(dim, hash_table, sub_clusters[coord]):
                commits[dim] = coord
                print(f"Dimension {dim} committed by {coord} with consensus.")

    return commits, shard_map, hash_table


num_vectors = 10
dimensions = 5
data = gen_data(num_vectors, dimensions)

vector_data = {col: data[col].values for col in data.columns}
dims = list(vector_data.keys())
nodes = ["Node1", "Node2", "Node3", "Node4", "Node5", "Node6", "Node7", "Node8", "Node9", "Node10"]

commits, shard_map, hash_table = commit_protocol(vector_data, dims, nodes)

print("Commits:", commits)
print("Shard Map:", dict(shard_map))
print("Hash Table:", hash_table)
