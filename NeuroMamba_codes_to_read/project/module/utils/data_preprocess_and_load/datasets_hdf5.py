# 4D_fMRI_Transformer
import glob
import os
import pdb, time
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from mpi4py import MPI

# --- MPI Initialization ---
# Initialize MPI communicator once at the module level
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# -------------------------

class BaseDatasetHDF5(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.img_root = os.path.join(self.root, 'hdf5')
        self.file_handles = {}
        self.file_metadata_cache = {}  # NEW: Cache metadata
        self.data = self._set_data(self.root, self.subject_dict)

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def scale_input(self, y, h5_file_handle):
        """ Scales input tensor y using stats from HDF5 file attributes. """
        if self.input_scaling_method == 'none':
            pass
        else:
            stats_dict = dict(h5_file_handle.attrs)
            if self.input_scaling_method == 'minmax':
                y = y / stats_dict['global_max']
            elif self.input_scaling_method == 'znorm_zeroback':
                background = y == 0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
                y[background] = 0
            elif self.input_scaling_method == 'znorm_minback':
                background = y == 0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
            elif self.input_scaling_method == 'robust':
                y = (y - stats_dict['median']) / stats_dict['iqr']
        return torch.from_numpy(y).float()

    def load_sequence_hdf5(self, h5_file_handle, start_frame, sample_duration):
        """
        REVISED: Loads a sequence of frames by slicing an HDF5 file using an existing handle.
        """
        fmri_data = h5_file_handle['fmri_data']
        num_frames = fmri_data.shape[-1]

        # Define the indices for the primary sequence
        indices = np.arange(start_frame, start_frame + sample_duration, self.stride_within_seq)
        y = fmri_data[..., indices]
        y = self.scale_input(y, h5_file_handle) # Pass the file handle
        y = y.unsqueeze(0) # Add batch dimension

        if self.contrastive:
            # Select a random, non-overlapping sequence for the contrastive pair
            full_range = np.arange(0, num_frames - sample_duration + 1)
            exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            
            if len(available_choices) == 0: # Handle cases with very short sequences
                available_choices = full_range

            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            random_indices = np.arange(random_start_frame, random_start_frame + sample_duration, self.stride_within_seq)
            
            random_y = fmri_data[..., random_indices]
            random_y = self.scale_input(random_y, f)
            random_y = random_y.unsqueeze(0) # Add batch dimension

            return (y, random_y)
        else:
            # Logic for shuffling time sequence if needed
            if self.shuffle_time_sequence:
                shuffled_indices = random.sample(list(range(y.shape[-1])), y.shape[-1])
                y = y[..., shuffled_indices]  # Shuffle only the time dimension
                y = y.unsqueeze(0)
            
            return y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def __del__(self):
        """
        Destructor to close all open file handles when the dataset object is destroyed.
        """
        for path in self.file_handles:
            self.file_handles[path].close()

    def _set_data(self, root, subject_dict):
        """
        REVISED: This method now orchestrates the data setup in a distributed-safe way.
        Only Rank 0 builds the data index from files; it's then broadcast to all other ranks.
        """
        data = None
        # --- Rank 0: The Leader ---
        # Only rank 0 handles file I/O to build the data index.
        if rank == 0:
            # First, check if a pre-computed index file (CSV) exists.
            image_name, skip_flag, loaded_data, save_flag = self._check_dataset_csv(root, subject_dict)
            
            if skip_flag:
                # If CSV exists and is valid, use its data.
                data = loaded_data
            else:
                # If no valid CSV, build the index from scratch.
                data = []
                print("Building data index from HDF5 files...")
                for i, subject_name in enumerate(tqdm(subject_dict.keys(), desc="Scanning HDF5s")):
                    sex, target = subject_dict[subject_name]
                    
                    # This helper gets the specific path for the subject, maybe implemented by child classes
                    subject_path = self._get_subject_path(subject_name)
                    if os.path.exists(subject_path):
                        data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
                        data.extend(data_tuple_list)
                    else:
                        print(f"Warning: HDF5 file not found for {subject_name} at {subject_path}")
                        pass
                if save_flag:
                    self._save_dataset_csv(data, image_name)

        # --- Broadcast to All Ranks ---
        # Rank 0 sends the 'data' list it created.
        # All other ranks receive it. This is a blocking call.
        data = comm.bcast(data, root=0)
        comm.barrier() # Add a barrier to ensure all processes are in sync

        if self.train and data:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
            
        return data

    def _get_subject_path(self, subject_name):
        # It should be checked per each dataset and be overrided
        return os.path.join(self.img_root, f"{subject_name}.h5")
    
    def _make_data_tuple_list(self, subject_path, i, subject_name, target, sex):
        # Helper function adapted for HDF5.
        with h5py.File(subject_path, 'r') as f:
            num_frames = f['fmri_data'].shape[-1]
            
        session_duration = num_frames - self.sample_duration + 1
        if self.train and self.num_train_fMRI_segments is not None:
            assert self.num_train_fMRI_segments < (session_duration // self.stride)
            session_duration = self.num_train_fMRI_segments * self.stride
            
        start_frames = []
        for start_i in range(self.stride_within_seq):
            start_frames.extend(range(start_i, session_duration, self.stride))
            
        # The subject_path now points directly to the .h5 file
        data_tuple_list = [(i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex) for start_frame in start_frames]
        return data_tuple_list


    def _check_dataset_csv(self, root, subject_dict):
        # This function can remain largely the same, caching is still useful
        image_name = os.path.basename(os.path.normpath(root))
        data = []
        skip_flag = False
        save_flag = False
        if self.use_subj_dict:
            self.dataset_csv = f"./data/data_tuple/{self.downstream_task}_{image_name}_{self.split}_seqlen{self.sequence_length}_withinseq_{self.stride_within_seq}_betweenseq{self.stride_between_seq}_hdf5.csv"
            if root.startswith('/tmp'): # use DAOS storage
                self.dataset_csv = self.dataset_csv.replace('.csv', '_daos.csv')
            if self.split == 'train' and self.num_train_fMRI_segments is not None:
                self.dataset_csv = self.dataset_csv.replace('_hdf5.csv', f'_num_train_fMRI_segments{self.num_train_fMRI_segments}_hdf5.csv')
            if self.limit_samples:
                self.dataset_csv = self.dataset_csv.replace('.csv', f'_limit_samples{self.limit_samples}.csv')
            if os.path.exists(self.dataset_csv):
                print("Use saved csv file in _set_data() in datasets_hdf5.py", self.dataset_csv)
                data = pd.read_csv(self.dataset_csv).values.tolist()
                skip_flag = True
            else:
                print("Could not find a saved csv file for _set_data() in datasets_hdf5.py")
                save_flag = True
        return image_name, skip_flag, data, save_flag

    def _save_dataset_csv(self, data, image_name):
        # This function remains the same
        if rank == 0:
            column_names = ['i', 'subject', 'subject_path', 'start_frame', 'sample_duration', 'num_frames', 'target', 'sex']
            if not os.path.exists(self.dataset_csv):
                os.makedirs("./data/data_tuple/", exist_ok=True)
                df = pd.DataFrame(data, columns=column_names)
                df.to_csv(self.dataset_csv, index=False)
                print(f"[RANK:0] Save data_tuple to {self.dataset_csv}")
            else:
                time.sleep(1) # a small wait just in case

    def _get_file_handle(self, path):
        """
        Lazily opens and returns a file handle. Each worker process will
        maintain its own dictionary of open file handles.
        """
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(
            path,
            'r',
            rdcc_nbytes=1024**2 * 32,      # 32 MB chunk cache (default is 1MB)
            rdcc_w0=0.75,                   # Preemption policy (0.75 = balanced)
            rdcc_nslots=10007,              # Number of chunk slots (prime number)
            swmr=False,                     # Not needed for read-only
            libver='latest'                 # Use latest HDF5 format for better performance
        )
        return self.file_handles[path]
    
class S1200(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # Use the new HDF5 loading function
        h5_file_handle = self._get_file_handle(subject_path)
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)

        if self.contrastive:
            y, rand_y = loaded_data
            
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value)
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (3, 9, 0, 0, 10, 8), value=background_value)
            rand_y = rand_y.permute(0,2,3,4,1)
            
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            y = loaded_data
            
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value)
            y = y.permute(0,2,3,4,1) 

            return {
                "fmri_sequence": y,
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

class ABCD(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # Use the new HDF5 loading function
        h5_file_handle = self._get_file_handle(subject_path)
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)

        if self.contrastive:
            y, rand_y = loaded_data
            
            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value)
            y = y.permute(0, 2, 3, 4, 1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = torch.nn.functional.pad(rand_y, (0, 1, 0, 0, 0, 0), value=background_value)
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 
        else:
            y = loaded_data
            
            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value)
            y = y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": y,
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

class UKB(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_subject_path(self, subject_name):
        # UKB-specific logic with its unique naming convention
        subject_filename = f"{subject_name}_20227_2_0"
        return os.path.join(self.img_root, f"{subject_filename}.h5")
    
    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # Get a persistent file handle instead of opening the file here
        h5_file_handle = self._get_file_handle(subject_path)

        # Use the new HDF5 loading function
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)

        if self.contrastive:
            y, rand_y = loaded_data
            
            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3) 
            rand_y = torch.nn.functional.pad(rand_y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            rand_y = rand_y.permute(0,2,3,4,1) 

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            y = loaded_data

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 

            return {
                "fmri_sequence": y,
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
    
class Dummy(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()
        
        return {
                "fmri_sequence": y,
                "subject_name": str(subj),
                "target": target,
                "TR": 0,
                "sex": sex
            } 

class ABIDE(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        image_name, skip_flag, data, save_flag = self._check_dataset_csv(root, subject_dict)
        for i, subject_name in enumerate(subject_dict):
            sex, target, site_id, data_type = subject_dict[subject_name]
            subject_path = os.path.join(root,'img', data_type, subject_name)
            data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
            data.extend(data_tuple_list)
        assert self.input_scaling_method == 'none'
        print('Assume that normalization already done and global_stats.pt does not exist (preprocessing v1)')

        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        
        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        h5_file_handle = self._get_file_handle(subject_path)
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)
        y = loaded_data
        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (0, -1, -10, -9, -1, 0), value=background_value) # (97,115,97) -> (96, 96, 96)
        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject_name": str(subject_name),
            "target": target,
            "TR": start_frame,
            "sex": sex
        } 
    
class HBN(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
            data.extend(data_tuple_list)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1) #### 이건 왜 하는거지?

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        h5_file_handle = self._get_file_handle(subject_path)
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)

        #contrastive learning
        if self.contrastive:
            y, rand_y = loaded_data

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = loaded_data

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # HBN rest image shape: 81, 95, 81
                y = torch.nn.functional.pad(y, (7, 8, 1, 0, 7, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
                # pad(y)는 뭘까나
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": str(subject_name),
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

class YooAttn(BaseDatasetHDF5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject_path = os.path.join(img_root, subject_name)
            data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
            data.extend(data_tuple_list)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        
        h5_file_handle = self._get_file_handle(subject_path)
        loaded_data = self.load_sequence_hdf5(h5_file_handle, start_frame, sequence_length)
        y = loaded_data

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (2, 3, 0, 0, 2, 3), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 
        
        return {
            "fmri_sequence": y,
            "subject_name": str(subject_name),
            "target": target,
            "TR": start_frame,
            "sex": sex,
            "study_name": 'Yoo_Attn'
            } 


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes(datasets)
        self.total_len = self.cumulative_sizes[-1]
        # check all datasets has target values and set self.target_values
        if all(hasattr(dataset, "target_values") for dataset in datasets):
            self.target_values = np.concatenate([dataset.target_values for dataset in datasets], axis=0)

    def _compute_cumulative_sizes(self, datasets):
        sizes = []
        total_size = 0
        for dataset in datasets:
            total_size += len(dataset)
            sizes.append(total_size)
        return sizes

    def _find_dataset_idx(self, idx):
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                return i
        raise IndexError("Index out of range")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > self.total_len:
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = self.total_len + idx

        dataset_idx = self._find_dataset_idx(idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
