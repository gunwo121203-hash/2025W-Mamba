import os  
import glob
import pickle  
import random  
import numpy as np  
import pandas as pd  
import torch
import pytorch_lightning as pl  
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder 
from torch.utils.data import DataLoader, Subset 

from .data_preprocess_and_load.datasets import (  
    S1200, ABCD, UKB, Dummy, HBN, ABIDE, ConcatDataset
)
from .data_utils import _stratified_undersampling, _psm_undersampling
from .parser import str2bool  

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.setup()
        pl.seed_everything(seed=self.hparams.seed)
        self.valid_only = kwargs.get("valid_only", False)
        
    def define_split_file_path(self, dataset_name, dataset_split_num, pretraining):
        # generate splits folder
        if pretraining:
            split_dir_path = f'./data/splits/{dataset_name}/pretraining/'
        else:
            split_dir_path = f'./data/splits/{dataset_name}/downstream/'
        os.makedirs(split_dir_path, exist_ok=True)
        split_file_path = os.path.join(split_dir_path, f"split_fixed_{dataset_split_num}.txt")
        return split_file_path
        
    def get_dataset(self, dataset_name):
        if self.hparams.use_hdf5:
            from .data_preprocess_and_load.datasets_hdf5 import S1200, ABCD, UKB, Dummy, HBN, ABIDE, ConcatDataset
        else:
            from .data_preprocess_and_load.debug_datasets import S1200, ABCD, UKB, Dummy, HBN, ABIDE, ConcatDataset
        if dataset_name == "Dummy":
            return Dummy
        elif dataset_name == "HBN":
            return HBN
        elif dataset_name == "S1200":
            return S1200
        elif dataset_name == "ABCD":
            return ABCD
        elif 'UKB' in dataset_name:
            return UKB
        elif dataset_name == 'ABIDE':
            return ABIDE
        else:
            raise NotImplementedError
    
    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict, split_file_path):
        with open(split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")
    
    def determine_split_stratified(self, S, idx, split_file_path):
        print('making stratified split')
        site_dict = {x:S[x][idx] for x in S} # index 2: site_id, idex 3: data type (ABIDE1/ABIDE2)
        site_ids = np.array(list(site_dict.values()))
        print('site_ids:',site_ids)
        
        #remove sites that has only one valid samples
        one_value_sites=[]
        values, counts = np.unique(site_ids, return_counts=True)
        # Print the value counts
        for value, count in zip(values, counts):
            # print(f"{value}: {count}") # 20,40 has one level
            if count == 1:
                one_value_sites.append(value)
                
        filtered_site_dict = {subj:site for subj,site in site_dict.items() if site not in one_value_sites}
        filtered_subjects = np.array(list(filtered_site_dict.keys()))
        filtered_site_ids = np.array(list(filtered_site_dict.values()))

        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=1-self.hparams.train_split-self.hparams.val_split, random_state=self.hparams.dataset_split_num)
        trainval_indices, test_indices = next(strat_split.split(filtered_subjects, filtered_site_ids)) # 0.
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.val_split, random_state=self.hparams.dataset_split_num)
        train_indices, valid_indices = next(strat_split.split(filtered_subjects[trainval_indices], filtered_site_ids[trainval_indices]))
        S_train, S_val, S_test = filtered_subjects[trainval_indices][train_indices], filtered_subjects[trainval_indices][valid_indices], filtered_subjects[test_indices]
        
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test}, split_file_path)
        return S_train, S_val, S_test
    
    def determine_split_randomly(self, S, split_file_path):
        S = list(S.keys())
        S_train = int(len(S) * self.hparams.train_split)
        S_val = int(len(S) * self.hparams.val_split)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
        # train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(S_train, S_val, S_test, self.subject_list)
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test}, split_file_path)
        return S_train, S_val, S_test
    
    def load_split(self, split_file_path):
        subject_order = open(split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self, dataset_name, image_path):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        if self.hparams.use_subj_dict: # jub change.
            print("Use subj dict for make_subject_dict")
            image_name = [*filter(lambda x: 'MNI_to_TRs' in x, image_path.split("/"))][0] # 250304 jubin change to extract dataset name in a more robust way
            subj_dict_path = f"./data/subj_dict/{dataset_name}_{self.hparams.downstream_task}_{image_name}.pickle"
            # if self.hparams.use_hdf5:
            #     subj_dict_path = subj_dict_path.replace('.pickle', '_hdf5.pickle')
            if os.path.exists(subj_dict_path):
                with open(subj_dict_path, 'rb') as f:
                    final_dict = pickle.load(f)
                return final_dict

        if self.hparams.use_hdf5:
            # For HDF5, subjects are individual .h5 files in a directory.
            img_root = os.path.join(image_path, 'hdf5')
            if not os.path.exists(img_root):
                raise FileNotFoundError(f"HDF5 directory not found at {img_root}. Please run the conversion script.")
            subject_files = glob.glob(os.path.join(img_root, '*.h5'))
            subject_list = [os.path.basename(f).replace('.h5', '') for f in subject_files]
        else:
            img_root = os.path.join(image_path, 'img')
            subject_list = os.listdir(img_root)

        final_dict = dict()
        if dataset_name == "S1200":
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_precise_age.csv"))
            meta_data_all = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_all.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'Gender'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'CogTotalComp_AgeAdj'
            else: raise NotImplementedError()

            if self.hparams.downstream_task == 'sex':
                meta_task = meta_data[['Subject',task_name]].dropna()
            elif self.hparams.downstream_task == 'age':
                meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                #rename column subject to Subject
                meta_task = meta_task.rename(columns={'subject': 'Subject'})
            elif self.hparams.downstream_task == 'int_total':
                meta_task = meta_data[['Subject',task_name,'Gender']].dropna()  
            
            for subject in subject_list:
                if int(subject) in meta_task['Subject'].values:
                    if self.hparams.downstream_task == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        target = 1 if target == "M" else 0
                        sex = target
                    elif self.hparams.downstream_task == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                        sex = 1 if sex == "M" else 0
                    elif self.hparams.downstream_task == 'int_total':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                        sex = 1 if sex == "M" else 0
                    final_dict[subject]=[sex,target]

        elif dataset_name == "HBN": 
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HBN_metadata_231212.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'Sex'
            elif self.hparams.downstream_task == 'age': task_name = 'Age'
            elif self.hparams.downstream_task == 'Dx.ndd_HC': task_name = 'Dx.ndd_HC'
            elif self.hparams.downstream_task == 'Dx.all_HC': task_name = 'Dx.all_HC'
            elif self.hparams.downstream_task == 'Dx.adhd_HC': task_name = 'Dx.adhd_HC'
            elif self.hparams.downstream_task == 'Dx.asd_HC': task_name = 'Dx.asd_HC'
            elif self.hparams.downstream_task == 'Dx.adhd_asd': task_name = 'Dx.adhd_asd'
            else: raise ValueError('downstream task not supported')
           
            if self.hparams.downstream_task == 'Sex':
                meta_task = meta_data[['SUBJECT_ID',task_name]].dropna()
            else:
                meta_task = meta_data[['SUBJECT_ID',task_name,'Sex']].dropna() # 왜 이렇게 하는거지?: sex를 왜 포함?
            
            for subject in subject_list:
                if subject in meta_task['SUBJECT_ID'].values:
                    target = meta_task[meta_task["SUBJECT_ID"]==subject][task_name].values[0]
                    sex = meta_task[meta_task["SUBJECT_ID"]==subject]["Sex"].values[0]
                    final_dict[subject]=[sex,target]

        elif dataset_name == "ABCD":            
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "ABCD_phenotype_total.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
            else: raise ValueError('downstream task not supported')
           
            if self.hparams.downstream_task == 'sex':
                meta_task = meta_data[['subjectkey',task_name]].dropna()
            else:
                meta_task = meta_data[['subjectkey',task_name,'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subjectkey'].values:
                    target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    final_dict[subject]=[sex,target]
            
        elif "UKB" in dataset_name:
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_fluid' : task_name = 'fluid'
            elif self.hparams.downstream_task == 'depression_current' : task_name = 'Depressed.Current'
            else: raise ValueError('downstream task not supported')
                
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "UKB_phenotype_depression_included.csv"))
            if task_name == 'sex':
                meta_task = meta_data[['eid',task_name]].dropna()
            else:
                meta_task = meta_data[['eid',task_name,'sex']].dropna()

            for subject in subject_list:
                if subject.endswith('20227_2_0') and (int(subject[:7]) in meta_task['eid'].values):
                    target = meta_task[meta_task["eid"]==int(subject[:7])][task_name].values[0]
                    sex = meta_task[meta_task["eid"]==int(subject[:7])]["sex"].values[0]
                    final_dict[str(subject[:7])] = [sex,target]
                else:
                    continue 

        elif dataset_name == "ABIDE":
            if self.hparams.downstream_task == 'sex': task_name = 'SEX'
            elif self.hparams.downstream_task == 'age': task_name = 'AGE_AT_SCAN'
            elif self.hparams.downstream_task == 'ASD': task_name = 'DX_GROUP'
            else: raise NotImplementedError()
            
            abide1=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE1_pheno.csv"))
            abide2=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE2_pheno_total.csv"),encoding= 'unicode_escape')
            total=pd.concat([abide1,abide2])
            # only leave version2
            meta_data = total.loc[~total.duplicated('SUB_ID',keep='last'),:].reset_index(drop=True)

            img_abide1 = os.path.join(image_path, 'img', 'ABIDE1')
            img_abide2 = os.path.join(image_path, 'img', 'ABIDE2')
            subj_list1 = os.listdir(img_abide1)
            subj_list2 = os.listdir(img_abide2)
            subj_list1 = [subj for subj in subj_list1 if subj not in subj_list2]
            img_root = os.path.join(image_path, 'ABIDE1')

            if self.hparams.downstream_task  == 'sex':
                meta_task = meta_data[['SUB_ID',task_name,'SITE_ID']].dropna()
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})
            elif self.hparams.downstream_task  == 'age':
                meta_task = meta_data[['SUB_ID',task_name,'SEX','SITE_ID']].dropna()
                #rename column subject to Subject
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})
            elif self.hparams.downstream_task  == 'ASD':
                meta_task = meta_data[['SUB_ID',task_name,'SEX','SITE_ID']].dropna()
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})

            le = LabelEncoder()
            meta_task['SITE_ID'] = le.fit_transform(meta_task['SITE_ID'])
            
            for i, subject in enumerate(subj_list1):
                #subject = subject[4:]
                if int(subject) in meta_task['Subject'].values:
                    if self.hparams.downstream_task == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0] -1
                        sex = target-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.hparams.downstream_task == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.hparams.downstream_task == 'ASD':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]-1
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0] 
                    final_dict[subject] = [sex, target, site_id, 'ABIDE1']
            for i, subject in enumerate(subj_list2):
                #subject = subject[4:]
                if int(subject) in meta_task['Subject'].values:
                    if self.hparams.downstream_task == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0] -1
                        sex = target-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.hparams.downstream_task == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.hparams.downstream_task == 'ASD':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]-1
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0] 
                    final_dict[subject] = [sex, target, site_id, 'ABIDE2']

        elif dataset_name == "ADNI":
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "ADNI_final_filtered_metadata.csv"))
            subjectkey = 'SubID'
            meta_data = meta_data.rename(columns={'PTGENDER': 'sex', 'age_at_scan': 'age'})
            meta_data.sex = meta_data.sex.map(lambda x: 1 if 'M' in str(x).strip().upper() else 0)
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'AD': task_name = 'DIAGNOSIS'
            elif self.hparams.downstream_task == 'conversion': task_name = 'CONVERSION'
            elif self.hparams.downstream_task == 'NextDiagnosis': task_name = 'NextDiagnosis'
            else: raise ValueError(f'downstream task {self.hparams.downstream_task} not supported')

            if self.hparams.downstream_task == 'sex':
                meta_task = meta_data[[subjectkey,task_name]].dropna()
            else:
                meta_task = meta_data[[subjectkey,task_name,'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task[subjectkey].values:
                    target = meta_task[meta_task[subjectkey]==subject][task_name].values[0]
                    sex = meta_task[meta_task[subjectkey]==subject]["sex"].values[0]
                    final_dict[subject]=[sex,target]

        if self.hparams.use_subj_dict: # jub change. 
            os.makedirs("./data/subj_dict/", exist_ok=True)
            with open(subj_dict_path, 'wb') as f:
                pickle.dump(final_dict, f)
            print(f"Save subject_dict to {subj_dict_path}")
            
        return final_dict

    def _undersampling(self, dataset:dict, num_samples=None, binary=True, randomize=True):
        assert binary == True
        keys = list(dataset.keys())
        ctrl_keys = []
        case_keys = [] 
        for _key in keys: 
            if dataset[_key][1] == 0: 
                ctrl_keys.append(_key)
            elif dataset[_key][1] == 1: 
                case_keys.append(_key)
                
        if num_samples is not None: 
            num_ctrl = int(num_samples / 2)
            num_case = int(num_samples / 2)
        else: 
            if len(ctrl_keys) >= len(case_keys): 
                num_ctrl = len(case_keys)
                num_case = len(case_keys)
            elif len(ctrl_keys) < len(case_keys):
                num_ctrl = len(ctrl_keys)
                num_case = len(ctrl_keys)
        sampled_ctrl_keys = random.sample(ctrl_keys, num_ctrl)
        sampled_case_keys = random.sample(case_keys, num_case)
        sampled_keys = sampled_ctrl_keys + sampled_case_keys
        undersampled_dataset = {key: dataset[key] for key in sampled_keys}
                
        if randomize: 
            undersampled_keys = list(undersampled_dataset.keys())
            randomized_undersampled_keys = random.sample(undersampled_keys, len(undersampled_keys))
            undersampled_dataset = {key: undersampled_dataset[key] for key in randomized_undersampled_keys}
        return undersampled_dataset, num_ctrl, num_case  

    def setup(self, stage=None):
        # this function will be called at each devices        
        params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "contrastive":self.hparams.use_contrastive,
                "contrastive_type":self.hparams.contrastive_type,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task": self.hparams.downstream_task,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "input_scaling_method" : self.hparams.input_scaling_method,
                "label_scaling_method" : self.hparams.label_scaling_method,
                "num_train_fMRI_segments": self.hparams.num_train_fMRI_segments,
                "dtype":'float16'
                }
        
        train_Dataset_objects = []
        val_Dataset_objects = []
        test_Dataset_objects = []
        for dataset_name, image_path in zip(self.hparams.dataset_name, self.hparams.image_path):
            split_file_path = self.define_split_file_path(dataset_name, self.hparams.dataset_split_num, self.hparams.pretraining)
            Dataset = self.get_dataset(dataset_name)
            subject_dict = self.make_subject_dict(dataset_name, image_path)

            if os.path.exists(split_file_path):
                train_names, val_names, test_names = self.load_split(split_file_path)
            elif dataset_name == 'ABIDE':
                #stratified split for ABIDE dataset
                idx = 2 # idx = 2 for site_id
                train_names, val_names, test_names = self.determine_split_stratified(subject_dict, idx, split_file_path)
            else:
                train_names, val_names, test_names = self.determine_split_randomly(subject_dict, split_file_path)

            if self.hparams.bad_subj_path:
                bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
                for bad_subj in bad_subjects:
                    bad_subj = bad_subj.strip()
                    if bad_subj in list(subject_dict.keys()):
                        print(f'removing bad subject: {bad_subj}')
                        del subject_dict[bad_subj]

            train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
            val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
            test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}

            def _sample_dict(dataset_name: str, image_path: str, data_dict: dict,
                             limit_samples: float, balanced_samples: bool, split: str):
                """
                Controller function to sample a subset from a data dictionary based on specified method.

                Args:
                    dataset_name (str): The name of the dataset (e.g., 'UKB', 'S1200').
                    image_path (str): The path to the dataset images.
                    data_dict (dict): The input dictionary of subjects for the current split (train/val/test).
                    limit_samples (float): The proportion or absolute number of samples to limit to.
                    balanced_samples (bool): Flag indicating if balanced sampling should be used for classification.
                    split (str): The name of the current data split ('train', 'val', or 'test').

                Returns:
                    data_dict: The new, sampled dictionary of subjects.
                """
                sampling_is_active = limit_samples is not None or balanced_samples
                if not sampling_is_active: # If no limiting or balancing is requested, return the original dictionary
                    return data_dict

                print(f"\n--- Applying Sampling to '{split}' set iwth sampling method {self.hparams.subject_sampling_method} ---")
                # Determine the number of samples to select, if limited
                n_samples_to_select = len(data_dict) # Default to all samples
                if limit_samples:
                    if limit_samples < 1.0:
                        n_samples_to_select = int(len(data_dict) * limit_samples)
                    else:
                        n_samples_to_select = int(limit_samples)
                    n_samples_to_select = max(1, n_samples_to_select) # Ensure at least one sample is selected if a limit is set
                
                # For PSM: only applicable for binary classification and when balancing is requested
                if self.hparams.subject_sampling_method == 'psm':
                    if balanced_samples and self.hparams.downstream_task_type == 'classification':
                        data_dict, num_class_0, num_class_1 = _psm_undersampling(
                            hparams=self.hparams,
                            dataset_name=dataset_name,
                            image_path=image_path,
                            subject_dict=data_dict,
                            confound_keys=['sex', 'age'], # Keys to balance for
                            n_samples_to_select=n_samples_to_select,
                            print_effect_size=True
                        )
                        print(f"PSM Sampling Complete for '{split}': {num_class_0} controls, {num_class_1} cases.")
                    else:
                        print(f"Warning: PSM is only for balanced binary classification. "
                            f"Falling back to random sampling for '{split}' set.")
                        if not limit_samples:
                            n_samples_to_select = None
                        data_dict, num_ctrl, num_case = self._undersampling(dataset=data_dict, num_samples=n_samples_to_select)

                # For Stratified Sampling
                elif self.hparams.subject_sampling_method == 'stratified':
                    data_dict, num_class_0, num_class_1 = _stratified_undersampling(
                        hparams=self.hparams,
                        dataset_name=dataset_name,
                        image_path=image_path,
                        subject_dict=data_dict,
                        stratify_on_keys=['sex', 'age', self.hparams.downstream_task],
                        n_samples_to_select=n_samples_to_select,
                        balanced=balanced_samples,
                        print_effect_size=True
                    )
                    if balanced_samples:
                        print(f"Stratified Balanced Sampling Complete for '{split}': {num_class_0} Class 0, {num_class_1} Class 1.")
                    else:
                        print(f"Stratified Proportional Sampling Complete for '{split}'.")
                
                # For simple balanced undersampling (if specified) or as a fallback
                elif balanced_samples:
                    if not limit_samples:
                        n_samples_to_select = None
                    data_dict, num_ctrl, num_case = self._undersampling(dataset=data_dict, num_samples=n_samples_to_select)
                    print(f"Default Balanced Undersampling Complete for '{split}': {num_ctrl} controls, {num_case} cases.")

                # For simple random sampling (if specified) or as a fallback for non-balanced limiting
                elif n_samples_to_select < len(data_dict):
                    print(f"Applying Random Sampling: selecting {n_samples_to_select} subjects.")
                    keys = list(data_dict.keys())
                    sampled_keys = random.sample(keys, n_samples_to_select)
                    data_dict = {key: data_dict[key] for key in sampled_keys}
                return data_dict

            for split, limit_key, balanced_key, data_dict in [
                ('train', 'limit_training_samples', 'balanced_training_samples', train_dict),
                ('val', 'limit_validation_samples', 'balanced_validation_samples', val_dict),
                ('test', 'limit_test_samples', 'balanced_test_samples', test_dict)
            ]:
                limit_samples = getattr(self.hparams, limit_key)
                balanced_samples = getattr(self.hparams, balanced_key)
                if split == 'train':
                    train_dict = _sample_dict(dataset_name, image_path, data_dict, limit_samples, balanced_samples, split)
                elif split == 'val':
                    val_dict = _sample_dict(dataset_name, image_path, data_dict, limit_samples, balanced_samples, split)
                elif split == 'test':
                    test_dict = _sample_dict(dataset_name, image_path, data_dict, limit_samples, balanced_samples, split)
                    
            params.update({"root":image_path})
            def create_dataset_object(dataset_type, subject_dict, split):
                return Dataset(
                    **params,
                    subject_dict=subject_dict,
                    train=(dataset_type == 'training'),
                    use_subj_dict=self.hparams.use_subj_dict,
                    limit_samples=getattr(self.hparams, f'limit_{dataset_type}_samples'),
                    split=split
                )
            train_Dataset_objects.append(create_dataset_object('training', train_dict, "train"))
            val_Dataset_objects.append(create_dataset_object('validation', val_dict, "val"))
            test_Dataset_objects.append(create_dataset_object('test', test_dict, "test"))

            if self.hparams.include_valid_to_train:
                print(f"Include validation set in training for {dataset_name}.")
                print(f"number of train_subj in {dataset_name} (including val):", len(train_dict) + len(val_dict))
            else:
                print(f"number of train_subj in {dataset_name}:", len(train_dict))
                print(f"number of val_subj in {dataset_name}:", len(val_dict))
            print(f"number of test_subj in {dataset_name}:", len(test_dict))

        if self.hparams.include_valid_to_train:
            merged_train_Dataset_objects = train_Dataset_objects + val_Dataset_objects
            self.train_dataset = ConcatDataset(merged_train_Dataset_objects)
        else:
            self.train_dataset = ConcatDataset(train_Dataset_objects)
        self.val_dataset = ConcatDataset(val_Dataset_objects)
        self.test_dataset = ConcatDataset(test_Dataset_objects)


        if self.hparams.include_valid_to_train:
            print("length of train_idx (including val):", len(self.train_dataset))
        else:
            print("length of train_idx:", len(self.train_dataset))
            print("length of val_idx:", len(self.val_dataset))  
        print("length of test_idx:", len(self.test_dataset))

        
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": self.hparams.pin_memory,
                "persistent_workers": False if self.hparams.dataset_name == 'Dummy' else self.hparams.persistent_workers,
                "shuffle": train
            }
        
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        if self.hparams.include_valid_to_train:
            return [self.test_loader]
        elif self.valid_only:
            return [self.val_loader]
        else:
            return [self.val_loader, self.test_loader] 

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--input_scaling_method", type=str, default='minmax',choices=["minmax","znorm_zeroback","znorm_minback","robust", 'none'],
                          help="normalization strategy for input sub-sequences (added in SwiFT v2), specify none if your preprocessing pipeline is based on v1")
        group.add_argument("--dataset_name", nargs="+", type=str, choices=["S1200", "ABCD", "UKB", "UKB_v1","UKB_v2","UKB_v3","UKB_v4","UKB_v5","UKB_v6", "Dummy", "HBN", "ABIDE"], default="S1200",help="dataset name(s) to be used")
        group.add_argument("--image_path", nargs="+", default=None, help="path(s) to image datasets preprocessed for SwiFT (should correspond to dataset_name)")
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--train_split", default=0.7, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=float, default=1.0, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--pin_memory", action='store_true')
        group.add_argument("--persistent_workers", action='store_true')
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--limit_training_samples", type=float, default=None, help="use if you want to limit training samples")
        group.add_argument("--limit_validation_samples", type=float, default=None, help="use if you want to limit validation samples")
        group.add_argument("--limit_test_samples", type=float, default=None, help="use if you want to limit test samples")
        group.add_argument("--balanced_training_samples",  action='store_true', help="use if you want to limit training samples")
        group.set_defaults(balanced_training_samples=False)
        group.add_argument("--balanced_validation_samples",  action='store_true', help="use if you want to limit training samples")
        group.set_defaults(balanced_training_samples=False)
        group.add_argument("--balanced_test_samples",  action='store_true', help="use if you want to limit training samples")
        group.set_defaults(balanced_training_samples=False)
        group.add_argument("--subject_sampling_method", type=str, default='random', choices=['random', 'balanced_random', 'stratified', 'psm'])
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--num_train_fMRI_segments", type=int, default=None, help="the number of fMRI segments to use")
        group.add_argument("--use_subj_dict", action='store_true') # 250203 jub change for faster dataset setup
        group.add_argument("--use_hdf5", action='store_true') # 250203 jub change for faster dataset setup
        group.add_argument("--include_valid_to_train", action='store_true') # 250812 jub change for larger training sample

        return parser