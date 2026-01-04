import os
import sys
import glob
import time
import torch
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from mpi4py import MPI

def convert_subject_to_hdf5(subject_filename, input_dir, output_dir):
    """
    Converts a subject's gzipped hdf5 file into a single HDF5 file.
    """
    subject_id = subject_filename.split('.')[0]
    h5_path = os.path.join(output_dir, f"{subject_id}.h5")

    if os.path.exists(h5_path):
        return

    try:
        # Load the gzipped hdf5 file
        gzipped_hdf5_path = os.path.join(input_dir, subject_filename)
        if not os.path.exists(gzipped_hdf5_path):
            print(f"[Rank {MPI.COMM_WORLD.Get_rank():02d}] WARNING: Gzipped HDF5 file not found for {subject_id}. Skipping.", file=sys.stderr)
            return
        
        with h5py.File(gzipped_hdf5_path, 'r') as source_f, h5py.File(h5_path, 'w', libver='latest') as dest_f:
            # --- Memory-Efficient Data Copy ---
            # Directly copy the dataset without loading it into a NumPy array.
            # h5py handles this chunk-by-chunk internally.
            source_dataset = source_f['fmri_data']
            dest_f.create_dataset(
                'fmri_data',
                data=source_dataset,  # Pass the dataset object directly
                compression=None
            )

            # --- Attribute Copy ---
            # Copy all attributes in one go.
            for key, value in source_f.attrs.items():
                dest_f.attrs[key] = value

    except Exception as e:
        # Use rank variable defined in the global scope
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[Rank {rank:02d}] ERROR processing {subject_id}: {e}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert subject gzipped HDF5 to non-compressed HDF5 format.')
    parser.add_argument('--dataset_path', type=str, nargs='+', required=True,
                        help='List of relative dataset directory names (e.g., S1200 ABCD).')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    IMAGE_HOME = '/tmp/NeuroX/swift/'
    
    # REVISED: Correctly join IMAGE_HOME with relative paths from argparse
    BASE_DATASET_PATH_LIST = [os.path.join(IMAGE_HOME, name) for name in args.dataset_path]
    INPUT_HDF5_PATH_LIST = [os.path.join(path, 'hdf5') for path in BASE_DATASET_PATH_LIST]
    OUTPUT_HDF5_PATH_LIST = [os.path.join(path, 'hdf5_v2') for path in BASE_DATASET_PATH_LIST]
    
    for BASE_DATASET_PATH, INPUT_HDF5_PATH, OUTPUT_HDF5_PATH in zip(BASE_DATASET_PATH_LIST, INPUT_HDF5_PATH_LIST, OUTPUT_HDF5_PATH_LIST):
        subjects = None

        if rank == 0:
            print(f"--- Processing Dataset: {os.path.basename(BASE_DATASET_PATH)} ---")
            print(f"Starting HDF5 conversion with {size} MPI processes.")
            if not os.path.isdir(BASE_DATASET_PATH):
                print(f"Error: Base dataset path not found at '{BASE_DATASET_PATH}'", file=sys.stderr)
                comm.Abort(1)

            os.makedirs(OUTPUT_HDF5_PATH, exist_ok=True)
            subjects=sorted(os.listdir(INPUT_HDF5_PATH))
            print(f"Found {len(subjects)} total subjects to process.")
            sys.stdout.flush()

        subjects = comm.bcast(subjects, root=0)
        comm.barrier()

        if subjects is None and rank != 0:
            print(f"[Rank {rank:02d}] Did not receive subject list. Aborting.", file=sys.stderr)
            comm.Abort(1)

        subjects_for_this_rank = subjects[rank::size]

        # REVISED: Disable tqdm for all ranks except 0 to keep logs clean
        progress_bar = tqdm(subjects_for_this_rank,
                                desc=f"Rank {rank:02d} Converting",
                                position=rank,
                                dynamic_ncols=True,
                                disable=(rank != 0))
        
        for subject_filename in progress_bar:
            convert_subject_to_hdf5(subject_filename, INPUT_HDF5_PATH, OUTPUT_HDF5_PATH)

        comm.barrier()

        if rank == 0:
            print(f"--- Finished Dataset: {os.path.basename(BASE_DATASET_PATH)} ---\n")

    if rank == 0:
        print("All processes have completed the conversion successfully.")

def load_fmri():
    for i in range(100):
        if i == 20:
            start_time = time.time()
        with h5py.File(h5path, 'r') as f:
            mri_data = f['fmri_data'][...,:2]
    end_time = time.time()
    print(f"Average time per load: {(end_time - start_time):.4f} seconds")