import numpy as np
import pandas as pd
from collections import defaultdict
import random

def create_synthetic_vector_dataset(num_vectors, dimensions):
    data = np.random.rand(num_vectors, dimensions)
    column_names = [f"Dimension_{i+1}" for i in range(dimensions)]
    vector_dataset = pd.DataFrame(data, columns=column_names)
    return vector_dataset

def compute_random_relevance_score():
    return random.random()

def update_index(dimension, node, index_structure):
    if dimension not in index_structure:
        index_structure[dimension] = []
    index_structure[dimension].append(node)
    return index_structure

def sync_dimension(dimension, node, shared_storage, ledger):
    shared_storage[dimension] = node
    ledger[dimension] = node

def partition_vector_data(V, D, N, S, M, top_k=2):
    T = defaultdict(list)
    I = {}

    for d in D:
        relevance_score = compute_random_relevance_score()
        top_k_nodes = sorted(N, key=lambda x: relevance_score * random.random(), reverse=True)[:top_k]

        for node in top_k_nodes:
            if d not in T[node]:
                T[node].append(d)
                S[d] = V[d]
                M[d] = node
                I = update_index(d, node, I)
                sync_dimension(d, node, S, M)

    return T, I

num_vectors = 10
dimensions = 5
synthetic_dataset = create_synthetic_vector_dataset(num_vectors, dimensions)

vector_data = {col: synthetic_dataset[col].values for col in synthetic_dataset.columns}
dimensions_list = list(vector_data.keys())
nodes = ["Node1", "Node2", "Node3"]
shared_storage = {}
ledger = {}

shard_table, index_structure = partition_vector_data(vector_data, dimensions_list, nodes, shared_storage, ledger, top_k=2)
print("Shard Table:", dict(shard_table))
print("Index Structure:", index_structure)
