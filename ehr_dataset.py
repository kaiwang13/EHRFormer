from collections import defaultdict
from functools import partial
from pathlib import Path
import random
from typing import Iterator, List, Optional
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, default_collate, Sampler, BatchSampler
import numpy as np
import torch.nn.functional as F
import pyarrow.parquet as pq
import diskcache
from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices

def save_parquet(df, path):
    tmp_df = df.copy()
    for col in tmp_df.columns:
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and len(tmp_df.iloc[0][col].shape) == 2:
            tmp_df[col] = tmp_df[col].map(lambda x: list(x))
    tmp_df.to_parquet(path)

def load_parquet(path):
    parquet_file = pq.ParquetFile(path)
    tmp_df = []
    for sub_df in parquet_file.iter_batches(batch_size=100000):
        tmp_df.append(sub_df.to_pandas())
    tmp_df = pd.concat(tmp_df, axis=0)
    for col in tmp_df.columns:
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and isinstance(tmp_df.iloc[0][col][0], np.ndarray):
            tmp_df[col] = tmp_df[col].map(lambda x: np.array(list(x)))
    return tmp_df

def sample_subset_2d(features, mask, prob):
    M, N = features.shape
    random_matrix = np.random.random(features.shape)
    random_matrix[~mask] = float('inf')
    valid_counts = np.sum(mask, axis=1, keepdims=True)
    sample_counts = np.round(valid_counts * prob).astype(int)
    sorted_indices = np.argsort(random_matrix, axis=1)
    row_indices = np.arange(M)[:, np.newaxis]
    col_positions = np.arange(N)[np.newaxis, :]
    output_mask = np.zeros_like(mask, dtype=bool)
    selection_matrix = col_positions < sample_counts
    output_mask[row_indices, sorted_indices] = selection_matrix
    output_mask = output_mask & mask
    input_mask = mask & ~output_mask
    
    
    return input_mask, output_mask

def process_nested_dict(d, func):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = process_nested_dict(value, func)
        elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            result[key] = func(value)
        else:
            result[key] = value
    return result


def truncate(v, max_len):
    return v[..., :max_len]

def collate_fn(data):
    batch = default_collate(data)
    max_len = min(16, int(batch['length'].max()))
    batch = process_nested_dict(batch, partial(truncate, max_len=max_len))
    return batch

class LengthBasedBatchSampler(Sampler):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.length_groups = defaultdict(list)
        for idx, length in enumerate(lengths):
            self.length_groups[length].append(idx)
        
        self.sorted_lengths = sorted(self.length_groups.keys())
        
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            for length in self.length_groups:
                np.random.shuffle(self.length_groups[length])

        length_iterators = {
            length: iter(self.length_groups[length])
            for length in self.sorted_lengths
        }
    
        remaining_samples = {
            length: len(indices)
            for length, indices in self.length_groups.items()
        }
        
        batches = []
        
        for current_length in self.sorted_lengths:
            while remaining_samples[current_length] > 0:
                batch = []
                while (
                    remaining_samples[current_length] > 0 
                    and len(batch) < self.batch_size
                ):
                    try:
                        batch.append(next(length_iterators[current_length]))
                        remaining_samples[current_length] -= 1
                    except StopIteration:
                        break
                
                if len(batch) < self.batch_size:
                    next_length_idx = self.sorted_lengths.index(current_length) + 1
                    while (
                        len(batch) < self.batch_size 
                        and next_length_idx < len(self.sorted_lengths)
                    ):
                        next_length = self.sorted_lengths[next_length_idx]
                        while (
                            remaining_samples[next_length] > 0 
                            and len(batch) < self.batch_size
                        ):
                            try:
                                batch.append(next(length_iterators[next_length]))
                                remaining_samples[next_length] -= 1
                            except StopIteration:
                                break
                        next_length_idx += 1
                
                if len(batch) == self.batch_size or (not self.drop_last and batch):
                    batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)
            
        return iter(batches)

    def __len__(self) -> int:
        total_samples = sum(len(indices) for indices in self.length_groups.values())
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size

class DynamicLengthBasedBatchSampler(Sampler):
    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int,  
        shuffle: bool = True,
        drop_last: bool = False,
        min_batch_size: int = 1     
    ):
        self.lengths = lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        
        
        self.length_groups = defaultdict(list)
        for idx, length in enumerate(lengths):
            self.length_groups[length].append(idx)
        
        self.sorted_lengths = sorted(self.length_groups.keys())
        
    def get_batch_size_for_length(self, length: int) -> int:
        batch_size = self.max_tokens_per_batch // length
        return max(min(batch_size, len(self.length_groups[length])), self.min_batch_size)
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            for length in self.length_groups:
                np.random.shuffle(self.length_groups[length])
        
        length_iterators = {
            length: iter(self.length_groups[length])
            for length in self.sorted_lengths
        }
        
        remaining_samples = {
            length: len(indices)
            for length, indices in self.length_groups.items()
        }
        
        batches = []
        
        for current_length in self.sorted_lengths:
            
            current_batch_size = self.get_batch_size_for_length(current_length)
            
            while remaining_samples[current_length] > 0:
                batch = []
                
                while (
                    remaining_samples[current_length] > 0 
                    and len(batch) < current_batch_size
                ):
                    try:
                        batch.append(next(length_iterators[current_length]))
                        remaining_samples[current_length] -= 1
                    except StopIteration:
                        break
                
                
                if len(batch) < current_batch_size:
                    next_length_idx = self.sorted_lengths.index(current_length) + 1
                    while (
                        len(batch) < current_batch_size 
                        and next_length_idx < len(self.sorted_lengths)
                    ):
                        next_length = self.sorted_lengths[next_length_idx]
                        
                        max_additional = min(
                            current_batch_size - len(batch),
                            self.max_tokens_per_batch // next_length
                        )
                        while (
                            remaining_samples[next_length] > 0 
                            and len(batch) < current_batch_size
                            and len(batch) < max_additional
                        ):
                            try:
                                batch.append(next(length_iterators[next_length]))
                                remaining_samples[next_length] -= 1
                            except StopIteration:
                                break
                        next_length_idx += 1
                
                if len(batch) >= self.min_batch_size or (not self.drop_last and batch):
                    batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        return iter(batches)

    def __len__(self) -> int:
        
        total_tokens = sum(length * len(indices) for length, indices in self.length_groups.items())
        if self.drop_last:
            return total_tokens // self.max_tokens_per_batch
        return (total_tokens + self.max_tokens_per_batch - 1) // self.max_tokens_per_batch
    

@dataclass
class EHRDataset(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict

    def __post_init__(self):
        self.lengths = self.data['n_visit'].values
        self.train_length = self.config['train_length']
        self.mask_ratio = self.config['mask_ratio']
        self.float_name_col = 'float_feat_cols'
        self.input_float_col = 'tokenized_float_feats'
        self.output_float_col = 'float_feats'
        self.mean_std = np.array([(self.config['feat_info']['float_cols'][x]['mean'], self.config['feat_info']['float_cols'][x]['std']) for x in self.data.iloc[0][self.float_name_col]], dtype=np.float32)
        self.cat_name_col = 'category_feat_cols'
        self.input_cat_col = 'tokenized_category_feats'

        self.df_path = self.config['df_path']
        self.cache = diskcache.Cache(self.df_path, eviction_policy='none')
        
        self.config['reg_label_names'] = self.data.iloc[0][self.float_name_col]
        self.config['cls_label_names'] = ['GENDER']

    def read_sample_label(self, row):
        feats = self.cache[row['pid']]
        tokenized_category_feats = feats['tokenized_category_feats']
        tokenized_float_feats = feats['tokenized_float_feats']
        float_feats = feats['float_feats']

        input_float_feats = np.nan_to_num(tokenized_float_feats, nan=-1).astype(np.int64)
        float_mask = input_float_feats != -1
        output_float_feats = np.nan_to_num(float_feats, nan=-1.0).astype(np.float32)

        input_cat_feats = np.nan_to_num(tokenized_category_feats, nan=-1).astype(np.int64)
        output_cat_feats = np.nan_to_num(tokenized_category_feats, nan=0).astype(np.int64)
        output_cat_feats[output_cat_feats == -1] = 0
        cat_mask = input_cat_feats != -1

        time_index = row['time_index'].astype(np.int64)
        valid_mask = row['valid_mask'].astype(bool)

        if self.mask_ratio != 0:
            input_mask, output_mask = sample_subset_2d(input_float_feats, float_mask, self.mask_ratio)
            input_sample = np.where(input_mask, input_float_feats, -1)
            output_sample = np.where(output_mask, output_float_feats, -1.0)
            output_sample = (output_sample - self.mean_std[:, 0][:, np.newaxis]) / self.mean_std[:, 1][:, np.newaxis]
        
        sample = {
            'time_index': time_index,
            'valid_mask': valid_mask,
            'cat_feats': input_cat_feats,
            'float_feats': input_sample,
            'float_valid_mask': input_mask
        }
        label = {
            'cat_feats': output_cat_feats,
            'cat_valid_mask': cat_mask,
            'float_feats': output_sample,
            'float_valid_mask': output_mask
        }
        return sample, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample, label = self.read_sample_label(row)
        result = {
            'pid': row['pid'],
            'length': min(16, row['n_visit']),
            'data': sample,
            'label': label,
        }
        result = process_nested_dict(result, partial(truncate, max_len=self.train_length))
        return result
    
class EHRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_length = self.config['train_length']
        self.df_path = config.get('df_path', None)
        self.use_cache = config.get('use_cache', False)

        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        
    def setup(self, stage=None):
        if isinstance(self.df_path, pd.DataFrame):
            df = self.df_path
        else:
            
            df = pd.read_parquet(Path(self.df_path) / 'metadata.parquet')
        df = df[df['n_visit'] != 1]
        df_train = df[df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
        df_valid = df[df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
        self.ds_train = EHRDataset(df_train, 'train', self.config)
        self.ds_valid = EHRDataset(df_valid, 'valid', self.config)

        if self.config.get('test_df', '') != '':
            df_test = pd.read_csv(self.config['test_df'])
        else:
            df_test = df[df[self.dataset_col].isin(self.test_folds)]
        self.ds_test = EHRDataset(df_test, 'test', self.config)
        
    def train_dataloader(self):
        sampler = LengthBasedBatchSampler(batch_size=self.batch_size, lengths=self.ds_train.lengths)
        
        return DataLoader(self.ds_train, num_workers=8,
                          batch_sampler=sampler,
                          collate_fn=collate_fn,
                          pin_memory=False)

    def val_dataloader(self):
        sampler = LengthBasedBatchSampler(batch_size=self.batch_size, lengths=self.ds_valid.lengths)
        
        return DataLoader(self.ds_valid, num_workers=8,
                          batch_sampler=sampler,
                          collate_fn=collate_fn,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8,
                        
                          pin_memory=False, shuffle=False)

    def teardown(self, stage=None):
        pass
