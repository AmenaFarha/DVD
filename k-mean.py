from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import os
import subprocess
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

file_path = "wdbc.csv"
df = pd.read_csv(file_path)

ids = df['ID'].values
df = df.drop(columns=['ID', 'Diagnosis'])
features = df.values
scaler = StandardScaler()
features = scaler.fit_transform(features)
n_clusters = 5
batch_size = 50

num_samples = features.shape[0]
samples_per_process = num_samples // size
start_index = rank * samples_per_process
end_index = (rank + 1) * samples_per_process if rank != size - 1 else num_samples
data_slice = features[start_index:end_index]
id_slice = ids[start_index:end_index] 

kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
kmeans.partial_fit(data_slice)
all_centroids = comm.gather(kmeans.cluster_centers_, root=0)
if rank == 0:
    all_centroids = np.vstack(all_centroids)
    final_centroids = np.zeros((n_clusters, features.shape[1]))
    for i in range(n_clusters):
        final_centroids[i] = np.mean(all_centroids[i::n_clusters], axis=0)

    global_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=final_centroids, n_init=1, random_state=42)
    global_kmeans.fit(features)
    labels = global_kmeans.labels_

    df['ID'] = ids
    df['Cluster'] = labels
    output_dir = "partition"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        cluster_file = os.path.join(output_dir, f'cluster_{cluster}.csv')
        cluster_data.to_csv(cluster_file, index=False)
        print(f"Cluster {cluster} data saved to {cluster_file}")

    print("Clustering completed and data stored separately.")

print("\nRunning Step 2 (MPI script)...")
subprocess.run(["mpiexec", "-n", "2", "python", "Creating hash.py"], check=True)

print("\nRunning Step 3 (MPI script)...")
subprocess.run(["mpiexec", "-n", "2", "python", "Consensus.py"], check=True)

end_time = time.time()
total_time = end_time - start_time
num_objects = num_samples 
throughput = num_objects / total_time if total_time > 0 else 0

print(f"\nAll steps completed. Total time taken: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} objects per second")

