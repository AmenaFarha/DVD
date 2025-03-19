from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import hashlib
import json


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(['ID', 'Diagnosis'], axis=1) 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices, :]

def assign_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def compute_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            centroids[i] = points.mean(axis=0)
    return centroids

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def hash_data(data):
    data_string = json.dumps(data.tolist())
    return hashlib.sha256(data_string.encode()).hexdigest()

def simulate_blockchain_node(data, hash_value):
    return hash_data(data) == hash_value

def two_phase_commit(data, comm, sub_cluster_size=10):
    hash_value = hash_data(data)
    agreement_count = sum(simulate_blockchain_node(data, hash_value) for _ in range(sub_cluster_size))

    return agreement_count >= (sub_cluster_size * 0.51)

def parallel_kmeans(data, k, num_steps=100, n_components=None, local_steps=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    if rank == 0 and n_components is not None:
        data = apply_pca(data, n_components)
    data = comm.bcast(data, root=0)

    data_split = np.array_split(data, size, axis=0)
    local_data = data_split[rank]

    local_data_filename = f'local_data_rank_{rank}.csv'
    np.savetxt(local_data_filename, local_data, delimiter=",")

    centroids = initialize_centroids(data, k) if rank == 0 else None
    centroids = comm.bcast(centroids, root=0)

    for i in range(num_steps):
        local_labels = assign_clusters(local_data, centroids)

        local_centroids = compute_centroids(local_data, local_labels, k)

        if (i + 1) % local_steps == 0 or i == num_steps - 1:
            all_centroids = np.zeros_like(local_centroids)
            comm.Allreduce(local_centroids, all_centroids, op=MPI.SUM)
            centroids = all_centroids / size

    if two_phase_commit(local_data, comm):
        print(f"Rank {rank}: Commit approved and data stored.")
    else:
        print(f"Rank {rank}: Commit rejected due to insufficient consensus.")

    comm.Barrier()

    if rank == 0:
        end_time = MPI.Wtime()
        print("All data processing and verification completed.")
        print("Total time taken: {:.4f} seconds".format(end_time - start_time))
        print("Centroids:\n", centroids)

if __name__ == "__main__":
    filepath = 'C:/Users/afarha/Study/Project DVD/wdbc.csv'  
    data = load_and_preprocess_data(filepath)
    k = 5
    n_components = 10
    parallel_kmeans(data, k, n_components=n_components, local_steps=5)
