"""
Copyright 2023 Rahil Mehrizi, Cuong Nguyen, GSK plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import image2omics.utils as utils


def _is_number(s: str):
    """Returns True if string is a number"""
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


class SingleIndexLabels:
    """Create an instance of LabelsToIndex
    and check if the new labels contain labels that don't exist in the original labels
    Parameters
    ----------
    unique_labels : Sequence
        a sequence-like object containing all unique labels
        e.g. ["pos1", "pos2", "neg"]
    Returns
    -------
    LabelsToIndex
        An LabelsToIndex object
    """
    def __init__(self, unique_labels) -> None:
        self._labels2idx_mapping = {k: i for i, k in enumerate(sorted(unique_labels))}

    def __getitem__(self, label):
        return self._labels2idx_mapping[label]

    def __iter__(self):
        for label in self._labels2idx_mapping:
            yield label

    def __len__(self):
        return len(self._labels2idx_mapping)

    def save(self, path_to_save):
        with open(path_to_save, "w") as f:
            f.write(json.dumps(self._labels2idx_mapping))

    def update(self, new_labels: Sequence) -> None:
        """Update the labels
        The new labels may or may not contain unknown labels

        Parameters
        ----------
        new_labels : Sequence
            The new labels to be added to the original ones
        """
        unknown_labels = set(new_labels) - self.labels
        original_num_labels = len(self._labels2idx_mapping)
        for i, label in enumerate(sorted(unknown_labels)):
            self._labels2idx_mapping[label] = original_num_labels + i

    @property
    def mapping(self):
        return self._labels2idx_mapping

    @property
    def labels(self):
        return set(self._labels2idx_mapping.keys())

class MultitaskIndexLabels:
    """Create an instance of LabelsToIndex
    Independent indexing: In this strategy, each label column has its own counter and is
    treated separately. This is the strategy used for adversarial
    training.

    Labels are separated by a semi colon, for example: train_MCCC1;E1.
    Note that only the first label can have a prefix.

    Parameters
    ----------
    labels : Sequence
        a sequence-like object containing all unique labels

    Returns
    -------
    LabelsToIndex
        An LabelsToIndex object
    """
    def __init__(self, labels):        

        if isinstance(labels,list):
            labels = self._to_dataframe(labels)

        self.num_label_cols = len(labels.columns)
        self.num_rows = len(labels)
        self.indices_counters = np.zeros(self.num_label_cols, dtype=int)
        self.indices_dicts = [{} for i in range(self.num_label_cols)]
        self.unique_label_arrays = set()
        
        self.update(labels)
        pass


    def _to_dataframe(self,labels):
        self.num_label_cols = len(labels)
        lcols = [f"LABEL_{i}" for i in range(self.num_label_cols)]
        labels = pd.DataFrame(data=labels,columns=lcols)
        return labels

    def __getitem__(self, labels):
        if not isinstance(labels, list):
            labels = labels.to_list()            
        indices_list = []  
        for label, label_dict in zip(labels,self.indices_dicts):
            indices_list.append(label_dict[label])        
        return indices_list

    def __len__(self):
        return len(self.num_rows)

    def __iter__(self):
        for unique_str in self.unique_label_arrays:     
            labels_list = unique_str.split(";") 
            for i in range(len(labels_list)):
                try:
                    labels_list[i] = float(labels_list[i])
                except:
                    pass
            yield labels_list


    def save(self, path_to_save):
        with open(path_to_save, "w") as f:
            f.write(json.dumps(self.indices_dicts))

    def update(self, labels) -> None:    
        if isinstance(labels,list):
            labels = self._to_dataframe(labels)

        for col, col_name in enumerate(labels.columns):
            unique_labels = labels[col_name].unique()            
            for ulabel in unique_labels:    
                if ulabel not in self.indices_dicts[col]:            
                    if _is_number(ulabel):
                        self.indices_dicts[col][ulabel] = float(ulabel)
                    else:
                        self.indices_dicts[col][ulabel] = self.indices_counters[col]
                        self.indices_counters[col] += 1

        for i,row in labels.iterrows():
            unique_label_str = ";".join([str(c) for c in row.to_list()])
            self.unique_label_arrays.add(unique_label_str)        
        

    @property
    def mapping(self):
        return self.indices_dicts

    @property
    def labels(self):
        all_label_lists = []
        for unique_str in self.unique_label_arrays:     
            labels_list = unique_str.split(";") 
            for i in range(len(labels_list)):
                try:
                    labels_list[i] = float(labels_list[i])
                except:
                    pass
            all_label_lists.append(labels_list)
        return all_label_lists
    

class MulticlassIndexLabels:
    """Create an instance of LabelsToIndex

    Continuous indexing: This is the strategy where all categorical labels are associated
    with a single counter. This can be used for multilabel classification
    where labels are arrays and the model predicts the probability of a
    sample belonging to each class instead of predicting the class itself,
    as exemplified in this post:
    https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45

    Labels are separated by a semi colon, for example: train_MCCC1;E1.
    Note that only the first label can have a prefix.

    Parameters
    ----------
    labels : Sequence
        a sequence-like object containing all unique labels
        
    Returns
    -------
    LabelsToIndex
        An LabelsToIndex object
    """
    def __init__(self, labels):    
        self._categorical = {}
        self._continuous = {}
        self._labels2idx_mapping = {}
        self.category_count = 0
        self.all_labels = set()
        if isinstance(labels, list):
            self.all_labels.update(labels)
        else:
            for col_name in labels.columns:
                self.all_labels.update(labels[col_name].to_list())
        for label in self.all_labels:
            self._add_label(label)

    def _add_label(self,label):
        if _is_number(label):
            self._continuous[label] = float(label)
            self._labels2idx_mapping[label] = self._continuous
        else:
            if label not in self._categorical:
                self._categorical[label] = self.category_count
                self._labels2idx_mapping[label] = self._categorical
                self.category_count += 1

    def __getitem__(self, label):
        if isinstance(label,list):
            index_list = []
            for l in label:
                index_list.append(self._labels2idx_mapping[l][l])
            return index_list
        else:
            return self._labels2idx_mapping[label][label]

    def __len__(self):
        return len(self._labels2idx_mapping)

    def save(self, path_to_save):
        with open(path_to_save, "w") as f:
            f.write(json.dumps(self._labels2idx_mapping))

    def update(self, new_labels: Sequence) -> None:
        """Update the labels
        The new labels may or may not contain unknown labels

        Parameters
        ----------
        new_labels : Sequence
            The new labels to be added to the original ones
        """      
        if isinstance(new_labels,list):
            unknown_labels = set(new_labels) - self.all_labels
        else:
            unknown_labels = set()
            for col_name in new_labels.columns:
                unknown_labels.update(new_labels[col_name].to_list())
            unknown_labels = unknown_labels - self.all_labels
        for label in sorted(unknown_labels):
            self._add_label(label)

    @property
    def mapping(self):
        return self._labels2idx_mapping

    @property
    def labels(self):
        return set(self._labels2idx_mapping.keys())


# define some constant values to avoid hard-coding it around
SPLIT_CHAR = ";"
SINGLE_LABEL = "singlelabel"
CONTINUOUS_LABEL = "multiclass"
INDEPENDENT_LABEL = "multitask"

LABEL_TYPES_DICT = {
    SINGLE_LABEL: SingleIndexLabels,
    CONTINUOUS_LABEL: MulticlassIndexLabels,
    INDEPENDENT_LABEL: MultitaskIndexLabels
}

class LabelsToIndex(metaclass=utils.Singleton):
    """Create an instance of LabelsToIndex
    Class responsible for mapping the label to index
    mapping. This class is implemented as a Singleton,
    meaning that only one instance will exist at a given time
    and will be used for all dataset insanteces.

    Parameters
    ----------
    labels : Sequence
        a sequence-like object containing all unique labels
    labeltype : str
        type of label index strategy. Available options:
        singlelabel
        multiclass
        multitask
        
    Returns
    -------
    LabelsToIndex
        An LabelsToIndex object
    """
    def __init__(self, labels, label_type = SINGLE_LABEL) -> None:
        self.labels_handler = LABEL_TYPES_DICT[label_type](labels)

    def __getitem__(self, label):
        return self.labels_handler[label]

    def __iter__(self):
        self.labels_handler.__iter__()

    def __len__(self):
        return len(self.labels_handler)

    def save(self, path_to_save):
        self.labels_handler.save(path_to_save)

    def update(self, new_labels: Sequence) -> None:    
        self.labels_handler.update(new_labels)

    @property
    def mapping(self):
        return self.labels_handler.mapping

    @property
    def labels(self):
        return self.labels_handler.labels


def create_labels2idx(all_labels, label_type) -> LabelsToIndex:
    """Create an instance of LabelsToIndex
    and check if the new labels contain labels that don't exist in the original labels
    Parameters
    ----------
    all_unique_labels : Sequence
        a sequence-like object containing all unique labels
        e.g. ["pos1", "pos2", "neg"]
    Returns
    -------
    LabelsToIndex
        An LabelsToIndex object
    """
    labels2idx = LabelsToIndex(all_labels, label_type)
    if all_labels is not None:
        labels2idx.update(all_labels)

    return labels2idx


def split_train_val(
    labels_df: pd.DataFrame,
    random_split_train_size: float,
    low_count_max: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Split wells into training and validation sets, and return one of them
    Labels with very low number of occurrences (<= low_count_max) will only be used in
    training, otherwise at least one occurrence of the label will be used in val set
    In the end, a fraction of labels_df will be returned
    Parameters
    ----------
    labels_df : pd.DataFrame
        Dataframe of the wells, with each row corresponding to 1 well in a plate.
        It should have at least one column named "LABEL".
        One example:
        ROW  COLUMN  LABEL  PLATE_BARCODE
        1     2      pos       AAAA
        2     2      pos       AAAA
        1     3      neg       BBBB
        2     3      neg       BBBB
    random_split_train_size: float
        The percentage of training data in the whole set
    is_training
        If the training or validation part of labels_df will be returned
    low_count_max: int, optional
        If the number of occurrences of a label <= low_count_max, all of them will be
        used as training data, otherwise at least one will be used as validation data
    seed: int, optional
        Seed for reproducibility of random operations
    Returns
    -------
    pd.DataFrame
        The training or validation part of labels_df
    """
    if random_split_train_size == 1:
        if is_training:
            return labels_df
        else:
            return pd.DataFrame(columns=labels_df.columns)  # empty dataframe
    count_col = "count"
    split_col = "split"
    labels_df.loc[:, split_col] = "train"

    # calculate the number of occurrences of each label
    unique_label_counts = labels_df["LABEL"].value_counts()
    label2count = dict(unique_label_counts)
    labels_df[count_col] = labels_df["LABEL"].map(label2count)

    middle_count_max = int(1 / (1 - random_split_train_size))
    # if the number of occurrences of a label is between low_count_max and
    # middle_count_max, 1 occurrence will be chosen as val data
    middle_labels = labels_df[
        (labels_df[count_col] > low_count_max)
        & (labels_df[count_col] <= middle_count_max)
    ]["LABEL"].unique()
    np.random.seed(seed)
    for label in middle_labels:
        indices = labels_df[labels_df["LABEL"] == label].index
        chosen_idx = np.random.choice(indices)
        labels_df.loc[chosen_idx, split_col] = "val"

    # when the number of occurrences of a label is larger than middle_count_max,
    # at least one val data will be chosen from it by sklearn train_test_split
    high_count_df = labels_df[labels_df[count_col] > middle_count_max]
    train_idx, val_idx = train_test_split(
        high_count_df.index,
        train_size=random_split_train_size,
        stratify=high_count_df.LABEL,
        random_state=seed,
    )
    labels_df.loc[val_idx, split_col] = "val"

    train_idx = list(labels_df[labels_df["split"] == "train"].index)
    val_idx = list(labels_df[labels_df["split"] == "val"].index)
    return train_idx, val_idx