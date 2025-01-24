import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from data.label_shift_utils import prepare_label_shift_data
from data.confounder_utils import prepare_confounder_data

root_dir = '/path/to/your/project' # Change this to your root directory

dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA'
    },
    'CUB': {
        'root_dir': 'cub'
    },
    'CMNIST': {
        "root_dir": 'CMNIST'
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']


def prepare_data(args, train, return_full_dataset=False):

    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    
    if args.dataset == 'CUB':
        if args.shift:
            # shift=True => force mislabel + distribution shift
            args.edited_mislabel = True
            print("[INFO] (CUB) shift=True -> enforcing edited_mislabel=True.")
            create_mislabel_edited_metadata(args)
            apply_distribution_shift(args)  # no replicate
        else:
            # shift=False
            if args.edited_mislabel:
                print("[INFO] (CUB) shift=False, edited_mislabel=True -> create mislabel only.")
                create_mislabel_edited_metadata(args)
            else:
                print("[INFO] (CUB) shift=False, edited_mislabel=False -> replicate original.")
                replicate_cub_metadata_for_training(args)

    elif args.dataset == 'CelebA':
        if args.shift:
            # shift=True => distribution shift
            print("[INFO] (CelebA) shift=True -> apply distribution shift.")
            apply_distribution_shift(args)
        else:
            # shift=False => ensure & replicate
            print("[INFO] (CelebA) shift=False -> ensure & replicate celeb_df_for_training.")
            ensure_celeb_df_csv_exists(args)
            replicate_celeb_df_for_training(args)

    elif args.dataset == 'CMNIST':
        pass
    else:
        print(f"[INFO] No special logic for dataset '{args.dataset}'.")

    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_confounder_data(args, train, return_full_dataset)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        train_data, val_data = prepare_label_shift_data(args, train)
        test_data = None

    if args.dataset == 'CMNIST' and args.shift:
        print("[INFO] CMNIST shift=True => rotating group #3 by 90 degrees in test_data.")
        if test_data is not None:
            test_data.rotate_images(angle=90, target_group=3)
        else:
            print("[WARNING] test_data is None. Cannot rotate.")

    return train_data, val_data, test_data

def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')


def create_mislabel_edited_metadata(args):
    """
    Reads the original 'metadata.csv' for CUB (Waterbirds),
    mislabels certain rows (sets y=0 for some matching bird names),
    and writes out 'mislabel_edited_metadata.csv'.
    """
    print("[INFO] Creating mislabel_edited_metadata.csv for CUB.")
    # 1) Load original metadata
    path = os.path.join(args.root_dir, 'data', 'waterbird_complete95_forest2water2')
    original_csv = os.path.join(path, 'metadata.csv')
         
    if not os.path.exists(original_csv):
        print(f"[WARNING] Cannot find {original_csv}. Are you sure your root_dir is correct?")
        return

    df = pd.read_csv(original_csv)

    # 2) Define keywords to identify certain birds that you want to mislabel
    keywords = ["Western_Wood_Pewee", "Eastern_Towhee", "Western_Meadowlark"]
    mask = df['img_filename'].str.contains('|'.join(keywords), na=False)
    df.loc[mask, 'y'] = 0

    out_csv = os.path.join(path, 'mislabel_edited_metadata.csv')
    df.to_csv(out_csv, index=False)
    print(f"[INFO] mislabel_edited_metadata.csv created at {out_csv}.")
    out_csv = os.path.join(path, 'metadata_for_training.csv')
    df.to_csv(out_csv, index=False)
    print(f"[INFO] mislabel_edited_metadata.csv created at {out_csv}.")


def replicate_cub_metadata_for_training(args):
    """
    If shift=False and we're using standard metadata.csv,
    we also want to ensure there's a metadata_for_training.csv.
    If not, replicate from metadata.csv.
    """
    path = os.path.join(args.root_dir, 'data', 'waterbird_complete95_forest2water2')
    meta_csv = os.path.join(path, 'metadata.csv')
    training_csv = os.path.join(path, 'metadata_for_training.csv')

    if not os.path.exists(meta_csv):
        print(f"[WARNING] {meta_csv} not found. Cannot replicate for training.")
        return

    print("[INFO] Replicating metadata.csv => metadata_for_training.csv for CUB (no shift).")
    df = pd.read_csv(meta_csv)
    df.to_csv(training_csv, index=False)




def ensure_celeb_df_csv_exists(args):
    """
    If celeb_df.csv does NOT exist at {root_dir}/data/celeb_df.csv,
    create it by merging 'list_attr_celeba.csv' and 'list_eval_partition.csv'.
    Then save as 'celeb_df.csv'.

    list_attr_celeba.csv => has columns like [image_id, 40 attribute columns]
    list_eval_partition.csv => has columns like [image_id, partition(0-2)]
    """
    path = os.path.join(args.root_dir, 'data')
    celeb_df_path = os.path.join(path, 'celeb_df.csv')
    if os.path.exists(celeb_df_path):
        print("[INFO] celeb_df.csv already exists. Not creating a new one.")
        return

    list_attr_path = os.path.join(path, 'list_attr_celeba.csv')
    list_eval_path = os.path.join(path, 'list_eval_partition.csv')

    if not (os.path.exists(list_attr_path) and os.path.exists(list_eval_path)):
        print(f"[WARNING] Could not find {list_attr_path} or {list_eval_path}.")
        return

    print("[INFO] Creating celeb_df.csv by merging list_attr_celeba.csv and list_eval_partition.csv.")

    # 1) Load
    list_attr_celeba = pd.read_csv(list_attr_path)
    list_eval_partition = pd.read_csv(list_eval_path)

    # 2) Replace -1 with 0 in attributes
    list_attr_celeba = list_attr_celeba.replace(-1, 0)

    # 3) Merge on 'image_id'
    celeb_df = pd.merge(list_attr_celeba, list_eval_partition, on='image_id')

    # 4) Save
    celeb_df.to_csv(celeb_df_path, index=False)
    print(f"[INFO] celeb_df.csv created at {celeb_df_path}.")


def replicate_celeb_df_for_training(args):
    """
    For the no-shift scenario with CelebA, simply copy celeb_df.csv
    into celeb_df_for_training.csv (so that the code can consistently
    read from celeb_df_for_training.csv, if needed).
    """
    path = os.path.join(args.root_dir, 'data')
    original_csv = os.path.join(path, 'celeb_df.csv')
    target_csv = os.path.join(path, 'celeb_df_for_training.csv')

    if not os.path.exists(original_csv):
        print(f"[WARNING] {original_csv} not found. Cannot replicate for training.")
        return

    df = pd.read_csv(original_csv)
    df.to_csv(target_csv, index=False)


# waterbird lists for CUB "waterbird" dataset
sea_birds_list = [
    'Albatross', 'Auklet', 'Cormorant', 'Frigatebird', 'Fulmar',
    'Gull', 'Jaeger', 'Kittiwake', 'Pelican', 'Puffin', 'Tern', 'Guillemot'
]
waterfowl_birds_list = [
    'Gadwall', 'Grebe', 'Mallard', 'Merganser', 'Pacific_Loon'
]

def metadata_add_col(args, metadata_name):
    """
    Loads a CSV and adds columns like 'bird_name' and 'bird_type'.
    The path is derived from args.root_dir for CUB.
    """
    path = os.path.join(args.root_dir, 'data', 'waterbird_complete95_forest2water2')
    csv_path = os.path.join(path, metadata_name)
    df = pd.read_csv(csv_path)

    # bird_name extraction
    df['bird_name'] = (
        df['img_filename'].str.split('/')
           .str.get(0)
           .str.split('_')
           .str[-1:].str.join('_')
    )
    df['bird_name'] = df['bird_name'].str.split('.').str.get(-1)
    df['bird_name'] = df['bird_name'].replace('Loon', 'Pacific_Loon')

    # species and bird_type columns
    df['species'] = df['img_filename'].apply(lambda x: x.split('/')[0])

    def classify_bird(name):
        if name in sea_birds_list:
            return 'sea bird'
        elif name in waterfowl_birds_list:
            return 'waterfowl bird'
        else:
            return 'land bird'

    df['bird_type'] = df['bird_name'].apply(classify_bird)
    return df

def save_dist_shift_waterbirds(args, metadata, new_filename):
    """
    Moves certain waterfowl birds from test->train/val,
    Moves certain sea birds from train/val->test.
    """
    np.random.seed(0)
    df = metadata.copy()

    # (1) Waterfowl birds in test -> some train(0), some val(1)
    change_settings = [
        ('Gadwall', 10, 9),
        ('Grebe', 4, 58),
        ('Mallard', 10, 5),
        ('Merganser', 8, 20),
        ('Pacific_Loon', 10, 6)
    ]
    for bird_name, zero_count, one_count in change_settings:
        subset = df[
            (df['bird_name'] == bird_name) &
            (df['split'] == 2) &  # test set
            (df['y'] == 1) &
            (df['place'] == 0)
        ]
        if len(subset) >= zero_count + one_count:
            idx_zero = np.random.choice(subset.index, size=zero_count, replace=False)
            df.loc[idx_zero, 'split'] = 0
            remaining = subset.index.difference(idx_zero)
            if one_count <= len(remaining):
                idx_one = np.random.choice(remaining, size=one_count, replace=False)
                df.loc[idx_one, 'split'] = 1
            else:
                print(f"[Warning] Not enough leftover rows for bird '{bird_name}' to set {one_count} to split=1.")
        else:
            print(f"[Warning] For '{bird_name}', needed {zero_count + one_count} rows but found {len(subset)}.")

    # (2) Sea birds in train/val -> test
    sea_cond = (
        (df['bird_type'] == 'sea bird') &
        (df['place'] == 0) &
        (df['y'] == 1) &
        (df['split'] != 2)
    )
    df.loc[sea_cond, 'split'] = 2

    # Save to CSV
    path = os.path.join(args.root_dir, 'data', 'waterbird_complete95_forest2water2')
    out_csv = os.path.join(path, new_filename)
    df.to_csv(out_csv, index=False)
    return df


def save_dist_shift_celebdata(args, celeb_df, shift_attr, new_filename, only_test=0):
    """
    For CelebA, moves minority group from test->train/val or train/val->test
    based on shift_attr (e.g., 'Eyeglasses').
    """
    np.random.seed(0)
    df = celeb_df.copy()

    # Example minority group
    minority_cond = (df['Blond_Hair'] == 1) & (df['Male'] == 1)

    # Move from test->train(0)/val(1)
    subset = df[
        (df[shift_attr] == 1 - only_test) &
        (df['partition'] == 2) &
        minority_cond
    ]
    zero_count = len(subset)
    one_count = len(subset) - zero_count

    if len(subset) >= zero_count + one_count:
        idx_zero = np.random.choice(subset.index, size=zero_count, replace=False)
        df.loc[idx_zero, 'partition'] = 0

        remain_idx = subset.index.difference(idx_zero)
        if one_count <= len(remain_idx):
            idx_one = np.random.choice(remain_idx, size=one_count, replace=False)
            df.loc[idx_one, 'partition'] = 1
        else:
            print(f"[Warning] Not enough rows left for setting partition=1 with '{shift_attr}'.")
    else:
        print(f"[Warning] For '{shift_attr}', needed {zero_count + one_count} rows but found {len(subset)}.")

    # Move from train/val->test
    cond = (df[shift_attr] == only_test) & (df['partition'] != 2) & minority_cond
    df.loc[cond, 'partition'] = 2

    out_path = os.path.join(args.root_dir, 'data')
    out_csv = os.path.join(out_path, new_filename)
    df.to_csv(out_csv, index=False)
    return df

def apply_distribution_shift(args):
    """
    If args.shift == True, modifies the CSV for CUB or CelebA so
    that subsequent data loading reflects the new distribution.
    """
    if args.dataset == 'CUB':
        path = os.path.join(args.root_dir, 'data', 'waterbird_complete95_forest2water2')
        mislabel_csv = os.path.join(path, 'mislabel_edited_metadata.csv')
        if not os.path.exists(mislabel_csv):
            # If it doesn't exist, create it from original metadata
            create_mislabel_edited_metadata(args)

        print("[INFO] Applying distribution shift for CUB. Using 'mislabel_edited_metadata.csv'.")
        out_csv_name = 'metadata_for_training.csv'
        df = metadata_add_col(args, 'mislabel_edited_metadata.csv')
        shifted_df = save_dist_shift_waterbirds(args, df, out_csv_name)
        shifted_df.to_csv(os.path.join(path, out_csv_name), index=False)

    elif args.dataset == 'CelebA':
        print("[INFO] Applying distribution shift for CelebA.")
        path = os.path.join(args.root_dir, 'data')
        ensure_celeb_df_csv_exists(args)

        csv_file_name = 'celeb_df_for_training.csv'
        main_csv = os.path.join(path, 'celeb_df.csv')
        if not os.path.exists(main_csv):
            print(f"[Warning] {main_csv} not found. Cannot apply shift.")
            return
        celeb_df = pd.read_csv(main_csv)
        shifted_celeb = save_dist_shift_celebdata(
            args, celeb_df,
            shift_attr='Eyeglasses',
            new_filename=csv_file_name,
            only_test=1
        )
        shifted_celeb.to_csv(os.path.join(path, csv_file_name), index=False)
