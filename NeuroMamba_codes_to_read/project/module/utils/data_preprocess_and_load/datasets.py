# 4D_fMRI_Transformer
import glob
import os
import pdb, time
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import nibabel as nb
import nilearn
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def scale_input(self, y, subject_path): # 250414 jubin added
        if self.input_scaling_method == 'none':
            #print('Assume that normalization already done and global_stats.pt does not exist (preprocessing v1)')
            pass
        else:
            stats_path = os.path.join(subject_path,'global_stats.pt')
            stats_dict = torch.load(stats_path) # ex) {'valid_voxels': 172349844, 'global_mean': tensor(7895.4902), 'global_std': tensor(5594.5850), 'global_max': tensor(37244.4766)}
            if self.input_scaling_method == 'minmax':
                y = y / stats_dict['global_max'] # assume that min value is zero and in background  
            elif self.input_scaling_method == 'znorm_zeroback':
                background = y==0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
                y[background] = 0
            elif self.input_scaling_method == 'znorm_minback':
                background = y==0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
            elif self.input_scaling_method == 'robust': 
                y = (y - stats_dict['median']) / stats_dict['iqr']

        return y

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive:
            num_frames = len(glob.glob(os.path.join(subject_path, 'frame_*.pt')))
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)
            y = self.scale_input(y, subject_path)

            random_y = []
            
            full_range = np.arange(0, num_frames-sample_duration+1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration,self.stride_within_seq)]
            # if self.with_voxel_norm:
            #     load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            random_y = self.scale_input(random_y, subject_path) # 250414 jubin added

            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
            # if self.with_voxel_norm:
            #     load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            y = self.scale_input(y, subject_path) # 250414 jubin added
            
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")
    
    def _make_data_tuple_list(self, subject_path, i, subject_name, target, sex):
        # helper function for creating data tuples. Should be used in each Dataset's _set_data()
        num_frames = len(glob.glob(os.path.join(subject_path,'frame_*')))
        session_duration = num_frames - self.sample_duration + 1
        if self.train and self.num_train_fMRI_segments is not None:
            assert self.num_train_fMRI_segments < (session_duration // self.stride)
            session_duration = self.num_train_fMRI_segments * self.stride
        start_frames = []
        for start_i in range(self.stride_within_seq): # when using stride_within_seq > 1, we should add start index other than 0 < start_i <= stride_within_seq
            start_frames.extend(range(start_i, session_duration, self.stride))
        data_tuple_list = [ (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex) for start_frame in start_frames ]
        return data_tuple_list

    def _check_dataset_csv(self, root, subject_dict):
        image_name = [*filter(lambda x: 'MNI_to_TRs' in x, root.split("/"))][0] # 250304 jubin change to extract dataset name in a more robust way
        data = []
        skip_flag = False
        save_flag = False
        if self.use_subj_dict:
            self.dataset_csv = f"./data/data_tuple/{self.downstream_task}_{image_name}_{self.split}_seqlen{self.sequence_length}_withinseq_{self.stride_within_seq}_betweenseq{self.stride_between_seq}.csv"
            if self.split == 'train' and self.num_train_fMRI_segments is not None:
                self.dataset_csv = self.dataset_csv.replace('.csv', f'_num_train_fMRI_segments{self.num_train_fMRI_segments}.csv')
            if self.limit_samples:
                self.dataset_csv = self.dataset_csv.replace('.csv', f'_limit_samples{self.limit_samples}.csv')
            if os.path.exists(self.dataset_csv):
                imported_dataset = pd.read_csv(self.dataset_csv)
                imported_dataset.subject = imported_dataset.subject.astype(str)
                if set(imported_dataset.subject) == set(subject_dict.keys()):
                    print("Use saved csv file in _set_data() in datasets.py")
                    data = imported_dataset.values.tolist()
                    skip_flag = True    
                else:
                    print("Saved csv file is different from subject_dict in _set_data() in datasets.py")
            else:
                print("Could not find a saved csv file for _set_data() in datasets.py")
                if self.limit_samples == None:
                    save_flag = True # should deal with saving csv files with limit samples configurations later
        return image_name, skip_flag, data, save_flag

    def _save_dataset_csv(self, data, image_name):
        if os.environ.get("RANK") == '0':
            column_names = ['i', 'subject', 'subject_path', 'start_frame', 'sample_duration', 'num_frames', 'target', 'sex']
            if self.split == 'train' and self.num_train_fMRI_segments is not None:
                self.dataset_csv = self.dataset_csv.replace('.csv', f'_num_train_fMRI_segments{self.num_train_fMRI_segments}.csv')
            if not os.path.exists(self.dataset_csv):
                os.makedirs("./data/data_tuple/", exist_ok=True)
                df = pd.DataFrame(data, columns=column_names)
                df.to_csv(self.dataset_csv, index=False)
                print(f"[RANK:0] Save data_tuple to {self.dataset_csv}")
        else:
            import time
            time.sleep(10) # wait for rank 0 to finish saving the csv file

class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        image_name, skip_flag, data, save_flag = self._check_dataset_csv(root, subject_dict)
        if not skip_flag:
            for i, subject_name in enumerate(subject_dict):
                sex,target = subject_dict[subject_name]
                subject_path = os.path.join(img_root, subject_name)
                data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
                data.extend(data_tuple_list)
            if save_flag:
                self._save_dataset_csv(data, image_name)
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

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
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        image_name, skip_flag, data, save_flag = self._check_dataset_csv(root, subject_dict)
        if not skip_flag:
            for i, subject_name in enumerate(subject_dict):
                sex, target = subject_dict[subject_name]
                # subject_name = subject[4:]
                subject_path = os.path.join(img_root, subject_name)
                data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
                data.extend(data_tuple_list)
            if save_flag:
                self._save_dataset_csv(data, image_name)

        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            #if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
            #    y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            #elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
            #    y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 
        

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        image_name, skip_flag, data, save_flag = self._check_dataset_csv(root, subject_dict)
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2
        if not skip_flag:
            for i, subject_name in enumerate(subject_dict):
                sex, target = subject_dict[subject_name]
                subject20227 = str(subject_name)+'_20227_2_0'
                subject_path = os.path.join(img_root, subject20227)
                data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
                data.extend(data_tuple_list)
            if save_flag:
                self._save_dataset_csv(data, image_name)
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

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
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex
                    } 
    
class Dummy(BaseDataset):
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
                "subject_name": subj,
                "target": target,
                "TR": 0,
                "sex": sex
            } 

class ABIDE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []

        for i, subject_name in enumerate(subject_dict):
            sex,target,site_id,data_type = subject_dict[subject_name]
            #subject_path = os.path.join(root, data_type, 'sub-'+subjecta)
            subject_path = os.path.join(root,'img',data_type, subject_name)
            #num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
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

        y = self.load_sequence(subject_path, start_frame, sequence_length,num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (0, -1, -10, -9, -1, 0), value=background_value) # (97,115,97) -> (96, 96, 96)
        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject_name": subject_name,
            "target": target,
            "TR": start_frame,
            "sex": sex
        } 
    
class HBN(BaseDataset):
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
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

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
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

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
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

class YooAttn(BaseDataset):
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
        
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (2, 3, 0, 0, 2, 3), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 
        
        return {
            "fmri_sequence": y,
            "subject_name": subject_name,
            "target": target,
            "TR": start_frame,
            "sex": sex,
            "study_name": 'Yoo_Attn'
            } 


### 20250625 MSJ
class ADNI(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        image_name, skip_flag, data, save_flag = self._check_dataset_csv(root, subject_dict)

        if not skip_flag:
            for i, subject_name in enumerate(subject_dict):
                sex, target = subject_dict[subject_name]
                subject_path = os.path.join(img_root, subject_name)
                data_tuple_list = self._make_data_tuple_list(subject_path, i, subject_name, target, sex)
                data.extend(data_tuple_list)

            if save_flag:
                self._save_dataset_csv(data, image_name)

        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive: 
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            ## padding 순서 : z, y, x (자른 후의 shape가 96, 96, 96이 되도록 양쪽에 padding)
            #x : [5:53] y : [7:64] z : [4:51]
            y = torch.nn.functional.pad(y, (24, 25, 19, 20, 24, 24), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3) 
            rand_y = torch.nn.functional.pad(rand_y, (24, 25, 19, 20, 24, 24), value=background_value) # adjust this padding level according to your data 
            rand_y = rand_y.permute(0,2,3,4,1) 

            return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex,
                    # "study_name": 'ToPS'
            } 
        
        else: ### 여기만 사용
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3) 
            y = torch.nn.functional.pad(y, (24, 25, 19, 20, 24, 24), value=background_value) # adjust this padding level according to your data 
            y = y.permute(0,2,3,4,1) 
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex
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
