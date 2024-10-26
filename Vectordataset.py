import numpy as np
import pandas as pd

def create_synthetic_vector_dataset(num_vectors, dimensions):
    data = np.random.rand(num_vectors, dimensions)
    column_names = [f"Dimension_{i+1}" for i in range(dimensions)]
    vector_dataset = pd.DataFrame(data, columns=column_names)
    
    return vector_dataset

num_vectors = 10 
dimensions = 5 
synthetic_dataset = create_synthetic_vector_dataset(num_vectors, dimensions)
print(synthetic_dataset)
