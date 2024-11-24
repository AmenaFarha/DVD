from mpi4py import MPI
import pandas as pd
import hashlib
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

input_folder = "E:/Study/Ph.D/Research Dissertation/breast+cancer+wisconsin+diagnostic/partition"
output_folder = "E:/Study/Ph.D/Research Dissertation/breast+cancer+wisconsin+diagnostic/hash"

if rank == 0:
    try:
        files = os.listdir(input_folder)
        files = [f for f in files if f.endswith('.csv')]
    except Exception as e:
        print(f"Error accessing folder '{input_folder}': {e}")
        files = []
else:
    files = None

files = comm.bcast(files, root=0)

if not files:
    print(f"Processor {rank}: No CSV files found in the directory.")
    MPI.Finalize()
    exit()

assigned_files = [files[i] for i in range(rank, len(files), size)]

if len(assigned_files) == 0:
    print(f"Processor {rank}: No files assigned.")
else:
    print(f"Processor {rank} assigned files: {assigned_files}")

for file_name in assigned_files:
    file_path = os.path.join(input_folder, file_name)
    
    try:
        df = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')
    except Exception as e:
        print(f"Processor {rank}: Error reading the file {file_path}: {e}")
        continue

    if 'ID' not in df.columns or 'Diagnosis' not in df.columns:
        print(f"Processor {rank}: Error - Required columns 'ID' or 'Diagnosis' are missing in {file_path}")
        continue
    
    hash_df = pd.DataFrame()
    hash_df['ID'] = df['ID']
    hash_df['Diagnosis'] = df['Diagnosis']

    for column in df.columns:
        if column not in ['ID', 'Diagnosis']:
            try:
                hash_df[column] = df[column].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
            except Exception as e:
                print(f"Processor {rank}: Error hashing column {column} in file {file_name}: {e}")
                hash_df[column] = None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hash_file_path = os.path.join(output_folder, f'hash_{file_name}')
    try:
        hash_df.to_csv(hash_file_path, index=False)
        print(f"Processor {rank} saved hash values to {hash_file_path}")
    except Exception as e:
        print(f"Processor {rank}: Error saving file {hash_file_path}: {e}")

MPI.Finalize()
