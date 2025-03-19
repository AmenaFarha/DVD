from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import hashlib
import json

def load_and_preprocess_data(filepath):
    """ Load and standardize the dataset. """
    data = pd.read_csv(filepath)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def apply_pca(data, n_components):
    """ Reduce dimensions of data using PCA. """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def distribute_columns(data, comm):
    """ Distribute columns of the data to each MPI process (physical node). """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_cols = data.shape[1]
    cols_per_rank = n_cols // size
    remainder = n_cols % size
    start = rank * cols_per_rank + min(rank, remainder)
    end = start + cols_per_rank + (1 if rank < remainder else 0)
    return data[:, start:end]

def hash_data(data):
    """ Hash the data using SHA-256 for integrity check. """
    data_string = json.dumps(data.tolist())
    return hashlib.sha256(data_string.encode()).hexdigest()

def simulate_blockchain_node(data, hash_value):
    """ Simulate a blockchain node verifying data by comparing hash values. """
    return hash_data(data) == hash_value

def virtual_node_task(args):
    """ Task executed by each virtual node (core) within a physical node. """
    data, hash_value = args
    return simulate_blockchain_node(data, hash_value)

def two_phase_commit(data, comm, sub_cluster_size=10):
    """ Perform a two-phase commit using 10 cores per physical node. """
    hash_value = hash_data(data)

    # Use MPI-based pool instead of multiprocessing
    with MPIPoolExecutor(max_workers=sub_cluster_size) as executor:
        results = list(executor.map(virtual_node_task, [(data, hash_value)] * sub_cluster_size))

    agreement_count = sum(results)
    
    # Check if the majority agrees (51% consensus)
    return agreement_count >= (sub_cluster_size * 0.51)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    # Load and preprocess data
    filepath = 'C:/Users/afarha/Study/Project DVD/wdbc2.csv' 
    data = load_and_preprocess_data(filepath)
    data = apply_pca(data, n_components=10)

    # Distribute data across MPI nodes (physical nodes)
    local_data = distribute_columns(data, comm)

    # Save local data for each physical node
    np.savetxt(f'reduced_data_rank_{rank}.csv', local_data, delimiter=",")

    # Parallel verification using 10 MPI-based cores per physical node
    if two_phase_commit(local_data, comm, sub_cluster_size=10):
        print(f"Rank {rank}: Commit approved and data stored.")
    else:
        print(f"Rank {rank}: Commit rejected due to insufficient consensus.")

    # Synchronize MPI processes
    comm.Barrier()

    if rank == 0:
        end_time = MPI.Wtime()
        print("All data processing and verification completed.")
        print("Total time taken: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
