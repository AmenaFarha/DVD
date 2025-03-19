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
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def distribute_columns(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_cols = data.shape[1]
    cols_per_rank = n_cols // size
    remainder = n_cols % size
    start = rank * cols_per_rank + min(rank, remainder)
    end = start + cols_per_rank + (1 if rank < remainder else 0)
    return data[:, start:end]

def hash_data(data):
    data_string = json.dumps(data.tolist())
    return hashlib.sha256(data_string.encode()).hexdigest()

def simulate_blockchain_node(data, hash_value):

    return hash_data(data) == hash_value

def two_phase_commit(data, comm, sub_cluster_size=10):
    hash_value = hash_data(data)
    agreement_count = 0


    for _ in range(sub_cluster_size):
        if simulate_blockchain_node(data, hash_value):
            agreement_count += 1

    return agreement_count >= (sub_cluster_size * 0.51)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    filepath = 'C:/Users/afarha/Study/Project DVD/wdbc2.csv' 
    data = load_and_preprocess_data(filepath)
    data = apply_pca(data, n_components=10) 

    local_data = distribute_columns(data, comm)

    np.savetxt(f'local_data_rank_{rank}.csv', local_data, delimiter=",")

    if two_phase_commit(local_data, comm):
        print(f"Rank {rank}: Commit approved and data stored.")
    else:
        print(f"Rank {rank}: Commit rejected due to insufficient consensus.")

    comm.Barrier()
    if rank == 0:
        end_time = MPI.Wtime()
        print("All data processing and verification completed.")
        print("Total time taken: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
