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

def convert_subject_to_hdf5(subject_img_path, output_dir):
    """
    Converts a subject's directory of .pt frames into a single HDF5 file.
    """
    subject_id = os.path.basename(subject_img_path)
    h5_path = os.path.join(output_dir, f"{subject_id}.h5")

    if os.path.exists(h5_path):
        return

    try:
        frame_files = sorted(
            glob.glob(os.path.join(subject_img_path, 'frame_*.pt')),
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )

        if not frame_files:
            return

        all_frames = [torch.load(f, map_location='cpu').squeeze().numpy() for f in frame_files]
        full_4d_data = np.stack(all_frames, axis=-1)

        with h5py.File(h5_path, 'w', libver='latest') as f:
            f.create_dataset('fmri_data', data=full_4d_data, chunks=(*full_4d_data.shape[:-1], 1))
            
            stats_path = os.path.join(subject_img_path, 'global_stats.pt')
            if os.path.exists(stats_path):
                stats_dict = torch.load(stats_path, map_location='cpu')
                for key, value in stats_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    f.attrs[key] = value

    except Exception as e:
        # Use rank variable defined in the global scope
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[Rank {rank:02d}] ERROR processing {subject_id}: {e}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert subject .pt frames to HDF5 format.')
    parser.add_argument('--dataset_path', type=str, nargs='+', required=True,
                        help='List of relative dataset directory names (e.g., S1200 ABCD).')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    IMAGE_HOME = '/flare/NeuroX/swift/'
    
    # REVISED: Correctly join IMAGE_HOME with relative paths from argparse
    BASE_DATASET_PATH_LIST = [os.path.join(IMAGE_HOME, name) for name in args.dataset_path]
    OUTPUT_HDF5_PATH_LIST = [os.path.join(args.output_path, name, 'hdf5_v2') for name in args.dataset_path]
    
    for BASE_DATASET_PATH, OUTPUT_HDF5_PATH in zip(BASE_DATASET_PATH_LIST, OUTPUT_HDF5_PATH_LIST):
        all_subject_folders = None

        if rank == 0:
            print(f"--- Processing Dataset: {os.path.basename(BASE_DATASET_PATH)} ---")
            print(f"Starting HDF5 conversion with {size} MPI processes.")
            if not os.path.isdir(BASE_DATASET_PATH):
                print(f"Error: Base dataset path not found at '{BASE_DATASET_PATH}'", file=sys.stderr)
                comm.Abort(1)

            os.makedirs(OUTPUT_HDF5_PATH, exist_ok=True)
            subjects_dir = os.path.join(BASE_DATASET_PATH, 'img')
            all_subject_folders = sorted([
                os.path.join(subjects_dir, d) for d in os.listdir(subjects_dir)
                if os.path.isdir(os.path.join(subjects_dir, d))
            ])
            print(f"Found {len(all_subject_folders)} total subjects to process.")
            sys.stdout.flush()

        all_subject_folders = comm.bcast(all_subject_folders, root=0)
        comm.barrier()

        if all_subject_folders is None and rank != 0:
            print(f"[Rank {rank:02d}] Did not receive subject list. Aborting.", file=sys.stderr)
            comm.Abort(1)

        subjects_for_this_rank = all_subject_folders[rank::size]

        # REVISED: Disable tqdm for all ranks except 0 to keep logs clean
        progress_bar = tqdm(subjects_for_this_rank,
                                desc=f"Rank {rank:02d} Converting",
                                position=rank,
                                dynamic_ncols=True,
                                disable=(rank != 0))
        
        for subject_path in progress_bar:
            convert_subject_to_hdf5(subject_path, OUTPUT_HDF5_PATH)

        comm.barrier()

        if rank == 0:
            print(f"--- Finished Dataset: {os.path.basename(BASE_DATASET_PATH)} ---\n")

    if rank == 0:
        print("All processes have completed the conversion successfully.")