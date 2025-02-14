import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import threading

def load_data():
    file_path = r'C:\Users\afarha\Study\Project DVD\wdbc.csv'
    data = pd.read_csv(file_path)
    data = data.select_dtypes(include=[np.number]) 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

data_scaled = load_data()

n_components_test = min(len(data_scaled), data_scaled.shape[1])  
incremental_pca_test = IncrementalPCA(n_components=n_components_test)

incremental_pca_test.fit(data_scaled)
explained_var_ratio = incremental_pca_test.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_var_ratio)

n_components = np.where(cumulative_explained_variance >= 0.999)[0][0] + 1
print(f"Number of components to retain 50% variance: {n_components}")

incremental_pca = IncrementalPCA(n_components=n_components)

def partial_fit(start, end):
    incremental_pca.partial_fit(data_scaled[start:end])

num_threads = 4 
threads = []
batch_size = len(data_scaled) // num_threads
for i in range(num_threads):
    start = i * batch_size
    end = start + batch_size if i != num_threads - 1 else len(data_scaled)
    thread = threading.Thread(target=partial_fit, args=(start, end))
    threads.append(thread)
    thread.start()
    
for thread in threads:
    thread.join()

transformed_data = incremental_pca.transform(data_scaled)
print("Explained Variance Ratio:", incremental_pca.explained_variance_ratio_)
print("Transformed Data Shape:", transformed_data.shape)

output_path = r'C:\Users\afarha\Study\Project DVD\transformed_data.csv'
pd.DataFrame(transformed_data).to_csv(output_path, index=False)
print(f"Transformed data saved to {output_path}")

