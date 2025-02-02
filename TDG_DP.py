from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import networkx as nx
import pandas as pd
import os
import subprocess
import time

spark = SparkSession.builder \
    .appName("Feature Dependency Graph with GraphX") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

start_time = time.time()
file_path = "transformed_data.csv"
output_path = "partition"
num_objects = 570

df = spark.read.csv(file_path, header=True, inferSchema=True)
pandas_df = df.toPandas()

if 'ID' not in pandas_df.columns or 'Diagnosis' not in pandas_df.columns:
    print("Required columns ('ID' or 'Diagnosis') are missing.")
    spark.stop()
    exit(1)


id_column = pandas_df['ID']
diagnosis_column = pandas_df['Diagnosis']
feature_columns = [col for col in pandas_df.columns if col not in ['ID', 'Diagnosis']]
correlation_matrix = pandas_df[feature_columns].corr()

threshold = 0.7
G = nx.Graph()
for i, col1 in enumerate(feature_columns):
    for j, col2 in enumerate(feature_columns):
        if i < j and abs(correlation_matrix.iloc[i, j]) > threshold:
            G.add_edge(col1, col2, weight=correlation_matrix.iloc[i, j])

clusters = list(nx.algorithms.community.greedy_modularity_communities(G))
print(f"Identified {len(clusters)} clusters.")

all_features_in_clusters = set().union(*clusters)
isolated_features = list(set(feature_columns) - all_features_in_clusters)
os.makedirs(output_path, exist_ok=True)

pandas_df['Cluster'] = -1
for idx, cluster in enumerate(clusters):
    cluster_features = list(cluster)
    cluster_df = pandas_df.dropna(subset=cluster_features)
    cluster_df = cluster_df[cluster_features + ['ID', 'Diagnosis']]
    output_file = os.path.join(output_path, f'cluster_{idx}.csv')
    cluster_df.to_csv(output_file, index=False)
    print(f"Cluster {idx} data saved to {output_file}")

if isolated_features:
    isolated_df = pandas_df[isolated_features + ['ID', 'Diagnosis']].dropna()
    isolated_file = os.path.join(output_path, 'isolated_features.csv')
    isolated_df.to_csv(isolated_file, index=False)
    print(f"Isolated features saved to {isolated_file}")

print("\nRunning Step 2 (MPI script)...")
subprocess.run(["mpiexec", "-n", "2", "python", "Creating hash.py"], check=True)

print("\nRunning Step 3 (MPI script)...")
subprocess.run(["mpiexec", "-n", "2", "python", "Consensus.py"], check=True)
end_time = time.time()
total_time = end_time - start_time
throughput = num_objects / total_time if total_time > 0 else 0

print(f"\nAll steps completed. Total time taken: {total_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} objects per second")
spark.stop()
