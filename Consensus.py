from mpi4py import MPI
import pandas as pd
import hashlib
import os
import threading
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

original_input_folder = "partition"
hash_input_folder = "hash"
output_folder = "output"
main_folder = "main"

if rank == 0:
    for folder in [output_folder, main_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

comm.Barrier()  

if rank == 0:
    try:
        files = os.listdir(original_input_folder)
        files = [f for f in files if f.endswith('.csv')]
    except Exception as e:
        print(f"Error accessing folder '{original_input_folder}': {e}")
        files = []
else:
    files = None

files = comm.bcast(files, root=0)

if not files:
    if rank == 0:
        print("No CSV files found in the directory.")
    MPI.Finalize()
    sys.exit()

assigned_files = [files[i] for i in range(rank, len(files), size)]
if len(assigned_files) == 0:
    print(f"Processor {rank}: No files assigned.")
    MPI.Finalize()
    sys.exit()
else:
    print(f"Processor {rank} assigned files: {assigned_files}")

lock = threading.Lock()

def verify_file(original_df, hash_df, thread_id, votes):
    """Function for each thread to verify the entire file and save the result."""
    verification_passed = True

    for column in original_df.columns:
        if column not in ['ID', 'Diagnosis']:
            new_hashes = original_df[column].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()).tolist()
            existing_hashes = hash_df[column].tolist()
            for i, (new_hash, existing_hash) in enumerate(zip(new_hashes, existing_hashes)):
                if new_hash != existing_hash:
                    print(f"Thread {thread_id} (Processor {rank}): Hash mismatch in column '{column}' at row {i + 1}")
                    verification_passed = False
                    break

    output_file_path = os.path.join(output_folder, f'verified_{file_name}_thread_{thread_id}.csv')
    original_df.to_csv(output_file_path, index=False)
    print(f"Thread {thread_id} (Processor {rank}): Data stored in {output_file_path}")
    if verification_passed:
        with lock:
            votes.append(1)
    return verification_passed

def process_file(file_name):
    """Coordinator process to manage verification and commit."""
    original_file_path = os.path.join(original_input_folder, file_name)
    hash_file_path = os.path.join(hash_input_folder, f'hash_{file_name}')

    try:
        original_df = pd.read_csv(original_file_path, delimiter=',', on_bad_lines='skip')
        hash_df = pd.read_csv(hash_file_path, delimiter=',', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading files {original_file_path} or {hash_file_path}: {e}")
        return

    if 'ID' not in original_df.columns or 'Diagnosis' not in original_df.columns:
        print(f"Processor {rank}: Error - Required columns 'ID' or 'Diagnosis' are missing.")
        return

    if original_df.shape[0] != hash_df.shape[0]:
        print(f"Processor {rank}: Error - Mismatch in the number of rows.")
        return

    num_threads = 10
    threads = []
    votes = []

    for i in range(num_threads):
        thread = threading.Thread(target=verify_file, args=(original_df, hash_df, i, votes))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    if len(votes) > num_threads // 2:
        print(f"Processor {rank}: Commit request approved for {file_name}.")
        comm.bcast("COMMIT", root=rank)

        confirmation_votes = []
        for _ in range(num_threads):
            confirmation_votes.append(1)

        if len(confirmation_votes) > num_threads // 2:
            main_file_path = os.path.join(main_folder, f'final_{file_name}')
            original_df.to_csv(main_file_path, index=False)
            print(f"Processor {rank}: Final data stored in {main_file_path}")
    else:
        print(f"Processor {rank}: Commit request denied for {file_name}.")

for file_name in assigned_files:
    print(f"Processor {rank} processing file: {file_name}")
    process_file(file_name)

MPI.Finalize()
