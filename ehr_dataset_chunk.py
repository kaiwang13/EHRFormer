from pathlib import Path
import random
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
import pyarrow.parquet as pq
import pyarrow as pa
import diskcache
import gc
from Utils_ehr import *

def optimize_dtypes(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_min, col_max = df[col].min(), df[col].max()
            
            if col_min >= 0:
                if col_max < 2**8:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 2**16:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 2**32:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df[col] = df[col].astype(np.int32)
        
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
            
        elif pd.api.types.is_string_dtype(df[col]) and df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
            
    return df

@dataclass
class ChunkedEHRDataset(Dataset):
    data_chunks: list  
    mode: str
    config: dict

    def __post_init__(self):
        self.df_paths = self.config['df_paths']
        self.use_cache = self.config['use_cache']
        if self.use_cache:
            
            self.caches = [
                diskcache.Cache(path, eviction_policy='none')
                for path in self.df_paths
            ]
            
        self.chunk_lengths = [len(chunk) for chunk in self.data_chunks]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.total_length = self.cumulative_lengths[-1] if self.chunk_lengths else 0
        
        
        self._setup_config()

    def _setup_config(self):
        self.dataframe_cols = set().union(*[set(chunk.columns) for chunk in self.data_chunks])
        self.config['cls_label_names'] = []
        self.config['reg_label_names'] = []
        self.config['n_cls'] = []
        
        
        first_row = self.data_chunks[0].iloc[0]
        
        for label_cols, cols, n_cls in zip(
            self.config['cls_label_cols'],
            self.config['cls_label_name_cols'],
            self.config['cls_label_n_cls']
        ):
            for col in first_row[cols]:
                self.config['cls_label_names'].append(f'{label_cols}_{col}')
                self.config['n_cls'].append(n_cls)
                
        for label_cols, cols in zip(
            self.config['reg_label_cols'],
            self.config['reg_label_name_cols']
        ):
            for col in first_row[cols]:
                self.config['reg_label_names'].append(f'{label_cols}_{col}')
                self.config['n_cls'].append(1)
        
        self.seq_max_len = self.config['seq_max_len']

    def get_from_cache(self, pid):
        if not self.use_cache:
            return None
            
        for cache in self.caches:
            try:
                data = cache.get(pid, default=None)
                if data is not None:
                    return data
            except:
                continue
        return None

    def read_col(self, chunk_idx: int, within_chunk_idx: int, data: dict, col: str):
        if col in self.dataframe_cols:
            return self.data_chunks[chunk_idx].iloc[within_chunk_idx][col]
        return data[col]

    def read_sample(self, chunk_idx: int, within_chunk_idx: int, data: dict):
        cat_feats = self.read_col(chunk_idx, within_chunk_idx, data, 'tokenized_category_feats')
        float_feats = self.read_col(chunk_idx, within_chunk_idx, data, 'tokenized_float_feats')
        valid_mask = self.read_col(chunk_idx, within_chunk_idx, data, 'valid_mask')
        time_index = self.read_col(chunk_idx, within_chunk_idx, data, 'time_index')

        
        tensors = {
            'cat_feats': torch.from_numpy(cat_feats.astype(np.int64)),
            'float_feats': torch.from_numpy(float_feats.astype(np.int64)),
            'valid_mask': torch.from_numpy(valid_mask.astype(bool)),
            'time_index': torch.from_numpy(time_index.astype(np.int64))
        }
        
        return tensors

    def read_label(self, chunk_idx: int, within_chunk_idx: int, data: dict):
        labels = {
            'cls': {'values': [], 'masks': []},
            'reg': {'values': [], 'masks': []},
            'time_index': self.read_col(chunk_idx, within_chunk_idx, data, 'time_index').astype(np.int64)
        }
        
        
        for value_col in self.config['cls_label_cols']:
            values = self.read_col(chunk_idx, within_chunk_idx, data, value_col)
            if len(values.shape) == 1:
                values = values.reshape(1, -1)
            for value in values:
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels['cls']['values'].append(value)
                labels['cls']['masks'].append(mask)
        
        
        for value_col in self.config['reg_label_cols']:
            values = self.read_col(chunk_idx, within_chunk_idx, data, value_col)
            if len(values.shape) == 1:
                values = values.reshape(1, -1)
            for value, (mean, std) in zip(values, self.config['reg_label_info']):
                value = (value - mean) / std
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels['reg']['values'].append(value)
                labels['reg']['masks'].append(mask)

        
        if labels['cls']['values']:
            values = torch.tensor(np.stack(labels['cls']['values'], axis=0), dtype=torch.long)
            values[values == -1] = 0
            labels['cls']['values'] = values
            labels['cls']['masks'] = torch.tensor(np.stack(labels['cls']['masks'], axis=0), dtype=torch.bool)

        if labels['reg']['values']:
            labels['reg']['values'] = torch.tensor(np.stack(labels['reg']['values'], axis=0), dtype=torch.float32)
            labels['reg']['masks'] = torch.tensor(np.stack(labels['reg']['masks'], axis=0), dtype=torch.bool)

        labels['time_index'] = torch.tensor(np.stack(labels['time_index'], axis=0), dtype=torch.long)
        return labels

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if chunk_idx > 0:
            within_chunk_idx = idx - self.cumulative_lengths[chunk_idx - 1]
        else:
            within_chunk_idx = idx
            
        
        if self.use_cache:
            pid = self.data_chunks[chunk_idx].iloc[within_chunk_idx]['pid']
            data = self.get_from_cache(pid)
        else:
            data = None
            
        return {
            'pid': self.read_col(chunk_idx, within_chunk_idx, data, 'pid'),
            'data': self.read_sample(chunk_idx, within_chunk_idx, data),
            'label': self.read_label(chunk_idx, within_chunk_idx, data)
        }

class EHRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_paths = config.get('df_paths', [])  
        self.use_cache = config.get('use_cache', False)
        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        self.chunk_size = config.get('chunk_size', 1000000)

    def setup(self, stage=None):
        if not isinstance(self.df_paths, list):
            self.df_paths = [self.df_paths]
        if isinstance(self.df_paths[0], pd.DataFrame):
            self._setup_from_dataframe()
        else:
            self._setup_from_parquet()

    def _setup_from_dataframe(self):
        df = pd.concat(sorted(self.df_paths), axis=0, ignore_index=True)
        df = optimize_dtypes(df)
        
        
        df_train = df[df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
        df_valid = df[df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
        df_test = df[df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
        
        
        self.config['df_paths'] = self.df_paths  
        
        
        self.ds_train = ChunkedEHRDataset([df_train], 'train', self.config)
        self.ds_valid = ChunkedEHRDataset([df_valid], 'valid', self.config)
        self.ds_test = ChunkedEHRDataset([df_test], 'test', self.config)

    def _setup_from_parquet(self):
        dfs = []
        for path in sorted(self.df_paths):
            metadata_path = Path(path) / 'metadata.parquet'
            df = load_parquet(metadata_path)
            dfs.append(df)
        
        
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df = optimize_dtypes(merged_df)
        
        
        df_train = merged_df[merged_df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
        df_valid = merged_df[merged_df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
        df_test = merged_df[merged_df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
        
        
        self.config['df_paths'] = self.df_paths
        
        
        self.ds_train = ChunkedEHRDataset([df_train], 'train', self.config)
        self.ds_valid = ChunkedEHRDataset([df_valid], 'valid', self.config)
        
        if self.config.get('test_df', '') != '':
            df_test = pd.read_csv(self.config['test_df'])
            df_test = optimize_dtypes(df_test)
            self.ds_test = ChunkedEHRDataset([df_test], 'test', self.config)
        else:
            self.ds_test = ChunkedEHRDataset([df_test], 'test', self.config)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=8,
            multiprocessing_context='fork'
        )

    def teardown(self, stage=None):
        gc.collect()