import os
import sys
import glob
import time
import torch
import numpy as np
import h5py
from tqdm import tqdm
import argparse  # NEW: Import argparse

from mpi4py import MPI

def convert_subject_to_hdf5(subject_img_path, output_dir):
    """
    Converts a subject's directory of .pt frames into a single HDF5 file.
    This function is executed by each worker process on its assigned subjects.

    Args:
        subject_img_path (str): Path to the directory containing a subject's frame_xxx.pt files.
        output_dir (str): Directory where the HDF5 file will be saved.
    """
    subject_id = os.path.basename(subject_img_path)
    h5_path = os.path.join(output_dir, f"{subject_id}.h5")

    # To prevent re-computing already converted files
    if os.path.exists(h5_path):
        return # Silently skip if the file already exists

    try:
        # Find all frame files and sort them numerically to ensure correct temporal order
        frame_files = sorted(
            glob.glob(os.path.join(subject_img_path, 'frame_*.pt')),
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )

        if not frame_files:
            return # No frames found for this subject

        # Load all frames and stack them into a single 4D numpy array (X, Y, Z, T)
        all_frames = [torch.load(f, map_location='cpu').numpy() for f in frame_files]
        full_4d_data = np.concatenate(all_frames, axis=-1)

        with h5py.File(h5_path, 'w', libver='latest') as f:
            # Create dataset with chunking for efficient slicing later on.
            # Chunking along the time dimension is most effective for your use case.
            f.create_dataset('fmri_data', data=full_4d_data, chunks=(*full_4d_data.shape[:-1], 1), compression="gzip")

            # Load scaling stats and save them as attributes of the HDF5 file
            stats_path = os.path.join(subject_img_path, 'global_stats.pt')
            if os.path.exists(stats_path):
                stats_dict = torch.load(stats_path, map_location='cpu')
                for key, value in stats_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    f.attrs[key] = value

    except Exception as e:
        print(f"[Rank {rank:02d}] ERROR processing {subject_id}: {e}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert subject images to HDF5 format.')
    parser.add_argument('--dataset_path', type=str, nargs='+', default=[], 
                        help='Paths to the root of your dataset (e.g., ./S1200)')
    args = parser.parse_args()  

    # Initialize MPI to get rank (process ID) and size (total number of processes)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Configuration ---
    # NOTE: This path must be accessible from all processes/nodes.
    IMAGE_HOME = '/flare/NeuroX/swift/' # aurora image path
    BASE_DATASET_PATH_LIST = args.dataset_path  # UPDATED: Use dataset_path from arguments
    OUTPUT_HDF5_PATH_LIST = [ os.path.join(BASE_DATASET_PATH, 'hdf5') for BASE_DATASET_PATH in BASE_DATASET_PATH_LIST ] # Directory to save HDF5 files
    # -------------------

    for BASE_DATASET_PATH, OUTPUT_HDF5_PATH in zip(BASE_DATASET_PATH_LIST, OUTPUT_HDF5_PATH_LIST):
        # Ensure paths are absolute
        BASE_DATASET_PATH = os.path.abspath(BASE_DATASET_PATH)
        OUTPUT_HDF5_PATH = os.path.abspath(OUTPUT_HDF5_PATH)
        all_subject_folders = None

        if rank == 0:
            print(f"Starting HDF5 conversion with {size} MPI processes.")
            if not os.path.isdir(BASE_DATASET_PATH):
                print(f"Error: Base dataset path not found at '{BASE_DATASET_PATH}'", file=sys.stderr)
                comm.Abort(1)

            # Rank 0 is the "master" that finds all the work to be done.
            os.makedirs(OUTPUT_HDF5_PATH, exist_ok=True)
            subjects_dir = os.path.join(BASE_DATASET_PATH, 'img')
            all_subject_folders = sorted([
                os.path.join(subjects_dir, d) for d in os.listdir(subjects_dir)
                if os.path.isdir(os.path.join(subjects_dir, d))
            ])
            print(f"Found {len(all_subject_folders)} total subjects to process.")
            sys.stdout.flush() # Ensure message is printed before broadcasting

        # Broadcast the list of subjects from Rank 0 to all other ranks.
        all_subject_folders = comm.bcast(all_subject_folders, root=0)
        
        # Wait for all processes to receive the broadcast before continuing
        comm.barrier()

        if all_subject_folders is None and rank != 0:
            print(f"[Rank {rank:02d}] Did not receive subject list. Aborting.", file=sys.stderr)
            comm.Abort(1)

        # Each rank determines its slice of the work using array slicing.
        # This is a very clean and efficient way to distribute the list.
        subjects_for_this_rank = all_subject_folders[rank::size]

        # Use tqdm for a nice progress bar. `position=rank` helps keep the bars
        # from overwriting each other in the terminal.
        # The progress bar will only show for subjects processed by THIS rank.
        for subject_path in tqdm(subjects_for_this_rank,
                                desc=f"Rank {rank:02d}",
                                position=rank,
                                dynamic_ncols=True):
            convert_subject_to_hdf5(subject_path, OUTPUT_HDF5_PATH)

        # Final barrier to ensure all processes are done before the script exits.
        print(f"[Rank {rank:02d}] Finished processing all {len(subjects_for_this_rank)} assigned subjects.")
        sys.stdout.flush()
        comm.barrier()

        if rank == 0:
            print("\nAll processes have completed the conversion successfully.")