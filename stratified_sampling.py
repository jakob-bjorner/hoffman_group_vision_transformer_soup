import torch
from collections import Counter
import tensorflow as tf
import tensorflow_datasets as tfds

def data_profile(dataset):
    """ 
    Count the number of elements in each class so we know how many we need to split up.
    
    Parameters
    ----------
    dataset : tensorflow dataset | torch.utils.data.DataLoader
        The dataset to count classes from.
    
    Returns
    -------
    dict
        The count of every class with class names as the keys.
    """
    # class_names = list(map(ds_info.features['label'].int2str, range(37)))
    classes = Counter()
    for sample in dataset:
        # print(sample)
        class_name = None
        if isinstance(sample["file_name"], list):
            # print(sample["file_name"])
            class_name = sample["file_name"][0].decode()
        elif isinstance(sample["file_name"], bytes):
            class_name = sample["file_name"].decode()
        else:    
            class_name = sample["file_name"].numpy().decode()
        index = class_name[::-1].index("_")
        class_name = class_name[:-index -1]
        classes[class_name] += 1
    return classes
def stratify_tfds(dataset, pct_split, seed=24):
    """
    Divide a dataset into two components stratifyied on their classes
    
    This function takes in a dataset and splits it into two smaller datasets
    the size of the splits depends on the pct_split parameter, and the
    elements in each dataset depends on the random seed given.

    Parameters
    ----------
    dataset: tensorflow data object
        The dataset to split.
    pct_split: float
        Float between 0 and 1, dictating the size of the two datasets produced.
    seed: int
        Random seed determining the element distribution.
    
    Returns
    ----------
    Tuple(torch DataLoader, torch DataLoader)
        The two splits with the frist dataset containing pct_split portion of the original dataset.
    """ 
    classes = data_profile(dataset)
    # generate the split of the classes for each split of the test set. right now 75 25.
    gen = torch.Generator()
    gen = gen.manual_seed(seed)
    splits = {}
    for name, sz in classes.items():
        val_splt = int(sz * pct_split)
        perm = torch.randperm(sz, generator=gen)
        splits[name] = {
            "ds_split_0": perm[:val_splt],
            "ds_split_1": perm[val_splt:]
        }
    # plan: send ds_test to numpy array, for each class split into a 0 or 1 depending on split defined above.
    ds_split_0 = []
    ds_split_1 = []
    class_counter = Counter()
    all_examples = tfds.as_numpy(dataset)
    for sample in all_examples:
        class_name = sample["file_name"].decode("utf-8")
        index = class_name[::-1].index("_")
        class_name = class_name[:-index -1]
        sample_number = class_counter[class_name]
        if sample_number in splits[class_name]["ds_split_0"]:
            ds_split_0.append(sample)
        else:
            ds_split_1.append(sample)
        class_counter[class_name] += 1
    # ds_split_0 = torch.utils.data.DataLoader(ds_split_0)
    # ds_split_1 = torch.utils.data.DataLoader(ds_split_1)
    return ds_split_0, ds_split_1

if __name__ == "__main__":
    tfds_name = "oxford_iiit_pet"

    ds, ds_info = tfds.load(tfds_name, with_info=True)
    ds = ds["test"]
    pct_split = 0.3
    ds_split_0, ds_split_1 = stratify_tfds(ds, pct_split)
    profile_ds = data_profile(ds)
    profile_split_0 = data_profile(ds_split_0)
    profile_split_1 = data_profile(ds_split_1)
    
    correctly_split = True
    for key, total in profile_ds.items():
        if abs(profile_split_0[key]- total * pct_split) > 1:
            correctly_split = False
            break
        if abs(profile_split_1[key] - total * (1 - pct_split) > 1):
            correctly_split = False
            break
    if len(ds_split_0) + len(ds_split_1) != len(ds):
        print("*** FAILED to stratify data correctly")
        print("length of test split 0", len(ds_split_0))
        print("length of test split 1", len(ds_split_1))
        print("total length", len(ds_split_0) + len(ds_split_1), "from an original ds of size", len(ds))
        print()
    elif not correctly_split:
        print("*** FAILED to split correctly")
        print("prfile of split 0")
        print(profile_split_0)
        print()
        print("profile of split 1")
        print(profile_split_1)
    else:
        print("success")
    
    