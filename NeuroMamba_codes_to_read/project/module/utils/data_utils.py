import os
import random

import numpy as np
import pandas as pd  
from psmpy import PsmPy
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer


def _stratified_undersampling(hparams: object, dataset_name: str, image_path: str, subject_dict: dict, stratify_on_keys: list,
                              n_samples_to_select: int, balanced: bool = False, randomize: bool = True, print_effect_size: bool = False):
    """
    Performs stratified sampling, with an option for balanced undersampling for classification tasks.

    If `balanced` is True and the downstream task is classification, this function
    creates a subset where each class has an equal number of subjects. Within each
    class, it further stratifies by other keys (e.g., age, sex).

    If `balanced` is False or the task is regression, it performs standard proportional
    stratified sampling based on all provided keys.

    Args:
        hparams (object): The hyperparameters object containing dataset and task info.
        dataset_name (str): Name of the dataset (e.g., 'UKB', 'HBN').
        image_path (str): Path to the dataset images and metadata.
        subject_dict (dict): The input dictionary of subjects.
        stratify_on_keys (list): Keys for stratification (e.g., ['sex', 'age', 'intelligence']).
        n_samples_to_select (int): The total number of subjects to select.
        balanced (bool): If True, forces equal samples per class for classification tasks.
        randomize (bool): Shuffle the final sampled subjects.
        print_effect_size (bool): If True, prints statistical comparison of confounders.

    Returns:
        dict: A new dictionary containing the sampled subjects.
        int: Number of subjects from the first class (or total if not balanced).
        int: Number of subjects from the second class (or 0 if not balanced/binary).
    """      
    # --- 1. Prepare DataFrame with Full Metadata for Stratification ---
    full_meta_df = _get_metadata(dataset_name, image_path)
    subject_col = next((col for col in full_meta_df.columns if 'subj' in col.lower() or 'id' in col.lower()), None)
    age_col =  next((col for col in full_meta_df.columns if 'age' == col.lower()), None)
    if subject_col and age_col:
        renamed_df = full_meta_df[[subject_col, age_col]].rename(columns={subject_col: 'subject', age_col: 'age'})
        renamed_df['subject'] = renamed_df['subject'].astype(str)
        combined_ages_df = renamed_df.drop_duplicates(subset=['subject'])
    else: # Handle case where no metadata files could be loaded
        raise ValueError(f"No metadata files could be loaded for dataset {dataset_name}. Cannot proceed with stratified sampling.")

    df = pd.DataFrame.from_dict(subject_dict, orient='index', columns=['sex', hparams.downstream_task])
    df.index.name = 'subject'
    df.reset_index(inplace=True)
    df['subject'] = df['subject'].astype(str)
    df = pd.merge(df, combined_ages_df, on='subject', how='left', suffixes=('', '_duplicate'))
    df = df.loc[:,~df.columns.duplicated()].copy()
    df.dropna(inplace=True)
    stratify_on_keys = list(set(stratify_on_keys))
    print(f"\n--- Stratified Sampling for {n_samples_to_select} samples ---")
    
    target_col = hparams.downstream_task
    is_classification = hparams.downstream_task_type == 'classification'
    
    # --- 2. Decide Sampling Strategy: Balanced or Proportional ---
    use_balanced_sampling = balanced and is_classification

    if use_balanced_sampling:
        # --- BALANCED STRATIFIED SAMPLING (for Classification) ---
        print(f"Strategy: Balanced Stratified Undersampling for target '{target_col}'")
        print(f"Stratifying within classes on: {[k for k in stratify_on_keys if k != target_col]}")

        if df[target_col].isnull().any():
            print(f"Target column '{target_col}' contains NaN values. These subjects will be dropped before sampling.")
            df.dropna(subset=[target_col], inplace=True)

        classes = df[target_col].unique()
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError(f"Balanced sampling requires at least 2 classes, but found {n_classes} in '{target_col}'.")
        
        n_per_class = n_samples_to_select // n_classes
        class_dfs = [df[df[target_col] == c].copy() for c in classes]
        
        # Check if sampling is possible and adjust if needed
        min_class_size = min(len(cdf) for cdf in class_dfs)
        if n_per_class > min_class_size:
            print(f"Requested {n_per_class} samples per class, but smallest class has only {min_class_size} members. "
                            f"Adjusting to sample {min_class_size} from each of the {n_classes} classes. Total samples will be {n_classes * min_class_size}.")
            n_per_class = min_class_size

        sampled_dfs = []
        for df_class in class_dfs:
            if len(df_class) <= n_per_class:
                sampled_dfs.append(df_class)
            else:
                # Stratify within this class using other keys
                df_class_strat = df_class.copy()
                confound_keys = [k for k in stratify_on_keys if k != target_col and k in df_class_strat.columns]
                
                for col in confound_keys:
                    if df_class_strat[col].nunique() > 10: # Bin continuous confounds
                        n_bins = min(5, df_class_strat[col].nunique())
                        if n_bins < 2: continue
                        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                        df_class_strat[f'{col}_bin'] = discretizer.fit_transform(df_class_strat[[col]]).astype(int)
                        confound_keys[confound_keys.index(col)] = f'{col}_bin'
                
                valid_confound_keys = [k for k in confound_keys if k in df_class_strat.columns]
                
                if not valid_confound_keys or df_class_strat[valid_confound_keys].nunique().min() < 2:
                    print(f"  Warning: Not enough variation in confound keys for a class. Using random sampling instead.")
                    sampled_dfs.append(df_class_strat.sample(n=n_per_class, random_state=hparams.seed))
                    continue

                df_class_strat['stratify_key'] = df_class_strat[valid_confound_keys].astype(str).agg('_'.join, axis=1)
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_per_class, random_state=hparams.seed)
                try:
                    train_indices, _ = next(splitter.split(df_class_strat, df_class_strat['stratify_key']))
                    sampled_dfs.append(df_class_strat.iloc[train_indices])
                except ValueError:
                    print(f"  Warning: Stratified sampling failed within a class. Falling back to random sampling.")
                    sampled_dfs.append(df_class_strat.sample(n=n_per_class, random_state=hparams.seed))

        sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    else:
        # --- PROPORTIONAL STRATIFIED SAMPLING (for Regression or unbalanced Classification) ---
        print(f"Strategy: Proportional Stratified Sampling on keys: {stratify_on_keys}")
        
        df_stratify = df.copy()
        processed_strat_keys = list(stratify_on_keys)
        for col in stratify_on_keys:
            if col not in df_stratify.columns: processed_strat_keys.remove(col); continue
            if df_stratify[col].nunique() > 10:
                n_bins = min(5, df_stratify[col].nunique())
                if n_bins < 2: continue
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile');
                df_stratify[f'{col}_bin'] = discretizer.fit_transform(df_stratify[[col]]).astype(int)
                processed_strat_keys[processed_strat_keys.index(col)] = f'{col}_bin'
        
        valid_stratify_keys = [k for k in processed_strat_keys if k in df_stratify.columns]
        if not valid_stratify_keys:
            sampled_df = df.sample(n=n_samples_to_select, random_state=hparams.seed)
        else:
            df_stratify['stratify_key'] = df_stratify[valid_stratify_keys].astype(str).agg('_'.join, axis=1)
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples_to_select, random_state=hparams.seed)
            try:
                train_indices, _ = next(splitter.split(df_stratify, df_stratify['stratify_key']))
                sampled_df = df_stratify.iloc[train_indices]
            except ValueError:
                sampled_df = df.sample(n=n_samples_to_select, random_state=hparams.seed)

    print(f"Successfully sampled {len(sampled_df)} subjects.")
    print("Final sample distribution:\n", sampled_df[target_col].value_counts())

    # --- (Optional) Print effect sizes/distribution differences on the final sampled data ---
    if print_effect_size:
        print_target_col = 'sex' if hparams.downstream_task_type == 'regression' else target_col
        confound_keys_for_stats = [key for key in stratify_on_keys if key != print_target_col]
        _calculate_and_print_effect_sizes(
            hparams=hparams,
            df=sampled_df,
            target_col=print_target_col,
            confound_cols=confound_keys_for_stats
        )
    
    # --- 4. Convert back to dictionary format ---
    sampled_keys = sampled_df['subject'].tolist()
    final_sampled_dict = {key: subject_dict[key] for key in sampled_keys}

    if randomize:
        shuffled_keys = random.sample(list(final_sampled_dict.keys()), len(final_sampled_dict))
        final_sampled_dict = {key: final_sampled_dict[key] for key in shuffled_keys}

    return final_sampled_dict, len(sampled_df[sampled_df[target_col]==0]), len(sampled_df[sampled_df[target_col]==1])


def _psm_undersampling(hparams: object, dataset_name: str, image_path: str, subject_dict: dict,
                       confound_keys: list, n_samples_to_select: int, print_effect_size: bool = False):
    """
    Performs balanced undersampling using Propensity Score Matching (PSM).

    This method is for binary classification tasks. It models the probability (propensity score)
    of a subject being in the 'treatment' group based on confounding variables (e.g., age, sex).
    It then matches subjects from the treatment and control groups with similar propensity
    scores to create a balanced dataset.

    Args:
        hparams (object): The hyperparameters object containing dataset and task info.
        dataset_name (str): Name of the dataset (e.g., 'UKB', 'HBN').
        image_path (str): Path to the dataset images and metadata.
        subject_dict (dict): The input dictionary of subjects.
        confound_keys (list): List of keys to use as confounders for matching (e.g., ['sex', 'age']).
                              The main target variable should NOT be in this list.
        n_samples_to_select (int): The total number of subjects to select.
        print_effect_size (bool): If True, prints statistical comparison of confounders
                                  between the two target classes in the final matched set.

    Returns:
        dict: A new dictionary containing the matched (sampled) subjects.
        int: Number of subjects selected from the control group (class 0).
        int: Number of subjects selected from the case group (class 1).
    """   
    # --- 1. Prepare DataFrame with Full Metadata ---
    full_meta_df = _get_metadata(dataset_name, image_path)
    subject_col = next((col for col in full_meta_df.columns if 'subj' in col.lower() or 'id' in col.lower()), None)
    age_col = next((col for col in full_meta_df.columns if 'age' == col.lower()), None)
    if subject_col and age_col:
        renamed_df = full_meta_df[[subject_col, age_col]].rename(columns={subject_col: 'subject', age_col: 'age'})
        renamed_df['subject'] = renamed_df['subject'].astype(str)
        combined_ages_df = renamed_df.drop_duplicates(subset=['subject'])
    else:
        raise ValueError(f"No metadata files could be loaded for dataset {dataset_name}. Cannot proceed with stratified sampling.")

    target_col = hparams.downstream_task
    df = pd.DataFrame.from_dict(subject_dict, orient='index', columns=['sex', target_col])
    df.index.name = 'subject'
    df.reset_index(inplace=True)
    df['subject'] = df['subject'].astype(str)
    df = pd.merge(df, combined_ages_df, on='subject', how='inner', suffixes=('', '_duplicate'))
    df = df.loc[:,~df.columns.duplicated()].copy()
    
    if df[target_col].nunique() != 2:
        raise ValueError(f"Propensity Score Matching is only for binary tasks, but target '{target_col}' has {df[target_col].nunique()} classes.")

    print(f"\n--- Propensity Score Matching for target '{target_col}' ---")
    print(f"Matching based on confounders: {confound_keys}")

    # --- 2. Perform Propensity Score Matching using psmpy ---
    df_psm = df.copy()
    df_psm[target_col] = df_psm[target_col] == 1 # psmpy requires the target column to be boolean (True for treatment/case, False for control) 
    df_psm.dropna(subset=[target_col], inplace=True) # remove subjects with no target value
    # Instantiate PsmPy class
    psm = PsmPy(
        df_psm,
        treatment=target_col,
        indx='subject',
    )
    
    # Calculate propensity scores using logistic regression
    if len(df_psm[df_psm[target_col] == 1]) >= 20: # balance=True does not work when minor class includes less than 20 subjects in the current PSM module..
        psm.logistic_ps(balance=True)
    else:
        psm.logistic_ps(balance=False)
    
    # Perform matching (e.g., 1-to-1 matching without replacement)
    # caliper=0.05 means matches are only made if propensity scores are within 5% of each other
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
    psm.effect_size_plot()
    print(psm.effect_size)
    
    # --- 3. Select a subset of matched data based on n_samples_to_select ---
    # Get the matched dataset
    matched_df = psm.df_matched
    if n_samples_to_select > matched_df.shape[0]:
        print(f"Requested {n_samples_to_select} samples, but only {matched_df.shape[0]} matched subjects available. Adjusting to use all matched subjects.")
    else:
        sampled_df = matched_df[~matched_df.matched_ID.isna()].sample(n=n_samples_to_select // 2, random_state=hparams.seed)
        matched_df = matched_df[matched_df['subject'].isin(sampled_df['subject']) | matched_df['subject'].isin(sampled_df['matched_ID'])].copy()
        print(f"Successfully matched {len(matched_df)} subjects ({len(matched_df)//2} pairs).")

    print("Final sample distribution:\n", matched_df[target_col].value_counts())
    
    # --- 4. (Optional) Print effect sizes on the matched data ---
    if print_effect_size:
        _calculate_and_print_effect_sizes(
            hparams=hparams,
            df=matched_df,
            target_col=target_col,
            confound_cols=confound_keys
        )

    # --- 5. Convert back to dictionary format ---
    sampled_keys = matched_df['subject'].tolist()
    final_sampled_dict = {key: subject_dict[key] for key in sampled_keys}

    n_class_1 = int(matched_df[target_col].sum())
    n_class_0 = len(matched_df) - n_class_1
    
    return final_sampled_dict, n_class_0, n_class_1


def _get_metadata(dataset_name, image_path):
    if dataset_name == "S1200":
        meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_gender.csv"))
        meta_data.rename(columns={'Gender': 'sex'}, inplace=True)
        meta_data['sex'] = meta_data['sex'].map(lambda x: 1 if x == "M" else 0)
        meta_data.pop('Age')
        meta_data_age = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_precise_age.csv"))
        meta_data_age.rename(columns={'subject': 'Subject'}, inplace=True)
        meta_data_age.dropna(subset=['age'], inplace=True)
        meta_data_age.Subject = meta_data_age.Subject.map(lambda x: int(x))
        meta_data['age'] = np.nan
        meta_data.loc[meta_data.Subject.isin(meta_data_age.Subject),'age'] = meta_data_age['age']
        meta_data.dropna(subset=['sex','age'], inplace=True)
    elif dataset_name == "HBN": 
        meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HBN_metadata_231212.csv"))
    elif dataset_name == "ABCD":           
        meta_data = pd.read_csv(os.path.join(image_path, "metadata", "ABCD_phenotype_total.csv"))           
    elif "UKB" in dataset_name:
        meta_data = pd.read_csv(os.path.join(image_path, "metadata", "UKB_phenotype_depression_included.csv"))
        meta_data.rename(columns={'eid': 'subject'}, inplace=True) # rename eid to Subject for consistency
        meta_data.subject = meta_data.subject.map(lambda x: str(x))
    elif dataset_name == "ABIDE":
        abide1=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE1_pheno.csv"))
        abide2=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE2_pheno_total.csv"), encoding= 'unicode_escape')
        meta_data=pd.concat([abide1,abide2])
    return meta_data
    

def _calculate_and_print_effect_sizes(hparams: object, df: pd.DataFrame, target_col: str, confound_cols: list):
    """
    Calculates and prints effect sizes/distribution differences for confounding
    variables between the two classes of a binary target variable.

    Args:
        df (pd.DataFrame): The DataFrame containing the sampled data.
        target_col (str): The name of the binary target variable column (e.g., 'depression_current').
        confound_cols (list): A list of column names for variables to check (e.g., ['sex', 'age']).
    """
    print("\n--- Calculating Effect Sizes/Distribution Differences on Sampled Data ---")
    
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found. Cannot calculate effect sizes.")
        return
        
    target_labels = df[target_col].dropna()
    if target_labels.nunique() != 2:
        print(f"Warning: Target column '{target_col}' is not binary ({target_labels.nunique()} unique values found). Skipping effect size calculation.")
        return
        
    group_labels = target_labels.unique()
    group0_mask = (df[target_col] == group_labels[0])
    group1_mask = (df[target_col] == group_labels[1])

    for confound_col in confound_cols:
        if confound_col == target_col or confound_col not in df.columns:
            continue

        print(f"\n  Checking distribution of '{confound_col}' across '{target_col}' groups:")
        
        # --- For Categorical Confounding Variables (e.g., 'sex') ---
        if confound_col == 'sex' or (confound_col == hparams.downstream_task and hparams.downstream_task_type == 'classification'):
            contingency_table = pd.crosstab(df[target_col], df[confound_col])
            print("    Contingency Table:")
            print(contingency_table.to_string(header=True, index=True))
            
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                print("    Skipping Chi-squared test (table is not 2x2 or larger).")
                continue
            try:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(f"    Chi-squared test: chi2 = {chi2:.3f}, p-value = {p:.3f}")
                if p < 0.05:
                    print("    -> NOTE: Statistically significant association detected (p < 0.05). The groups differ on this variable.")
                else:
                    print("    -> OK: No statistically significant association detected (p >= 0.05). The groups are balanced on this variable.")
            except ValueError as e:
                print(f"    Could not perform Chi-squared test: {e}")

        # --- For Continuous Confounding Variables (e.g., 'age') ---
        else:
            if df[confound_col].dtype.name == 'object':
                print(f"    Warning: '{confound_col}' is not numeric. Skipping t-test.")
                # use regex to extract first number. Since some HCP metadata has age like 31-33 or 36+
                df[confound_col] = df[confound_col].str.extract(r'(\d+)').astype(float)
            if df[confound_col].dtype.name not in ['float64', 'int64']:
                print(f"    Warning: '{confound_col}' is not numeric. Skipping t-test.")
                continue
            group0_data = df.loc[group0_mask, confound_col].dropna()
            group1_data = df.loc[group1_mask, confound_col].dropna()
            
            if len(group0_data) < 2 or len(group1_data) < 2:
                print("    Skipping t-test (one or both groups have fewer than 2 samples).")
                continue

            t_stat, p_val = ttest_ind(group0_data, group1_data, equal_var=False) # Welch's t-test for unequal variances
            
            # Calculate Cohen's d for effect size
            mean0, mean1 = group0_data.mean(), group1_data.mean()
            std0, std1 = group0_data.std(ddof=1), group1_data.std(ddof=1)
            n0, n1 = len(group0_data), len(group1_data)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2))
            cohens_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else 0

            print(f"    Group 0 (target={group_labels[0]}): N={n0}, Mean={mean0:.2f}, Std={std0:.2f}")
            print(f"    Group 1 (target={group_labels[1]}): N={n1}, Mean={mean1:.2f}, Std={std1:.2f}")
            print(f"    T-test: t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
            print(f"    Effect Size (Cohen's d): {cohens_d:.3f}")
            if p_val < 0.05:
                print(f"    -> NOTE: Statistically significant difference in means detected (p < 0.05). The groups differ on '{confound_col}'.")
            else:
                print(f"    -> OK: No statistically significant difference in means detected (p >= 0.05). The groups are balanced on '{confound_col}'.")