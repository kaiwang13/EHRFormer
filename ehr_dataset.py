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
import gc
from Utils import json_load

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
def sample_subset(Mask, Prob):
    num_true_in_Mask = np.sum(Mask)
    num_samples = int(round(num_true_in_Mask * Prob))
    valid_indices = np.where(Mask)[0]
    
    if num_samples > 0 and len(valid_indices) > 0:
        output_indices = np.random.choice(valid_indices, size=num_samples, replace=False)
    else:
        output_indices = valid_indices
    input_indices = np.setdiff1d(valid_indices, output_indices)
    return input_indices, output_indices

@dataclass
class ChunkedEHRDataset(Dataset):
    data_chunks: list  
    mode: str
    config: dict
    def __post_init__(self):
        self.mask_ratio = self.config['mask_ratio']
        self.float_cols = sorted(json_load(self.config['float_feats']))
        self.input_float_cols = [f'tokenized.{x}' for x in self.float_cols]
        self.output_float_cols = self.float_cols
        self.mean_std = np.array([(self.config['feat_info']['float_cols'][x]['mean'], self.config['feat_info']['float_cols'][x]['std']) for x in self.output_float_cols], dtype=np.float32)
        self.cat_cols = [f'tokenized.{x}' for x in sorted(self.config['feat_info']['category_cols'])]
        self.config['mean_std'] = self.mean_std
        self.config['reg_label_names'] = self.float_cols
        self.config['cls_label_names'] = sorted(self.config['feat_info']['category_cols'])
        self.chunk_lengths = [len(chunk) for chunk in self.data_chunks]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.total_length = self.cumulative_lengths[-1] if self.chunk_lengths else 0
    def read_sample(self, row):
        sample = {
            'cat_feats': torch.tensor(row[self.cat_cols].values.astype(np.int64), dtype=torch.long),
            'float_feats': torch.tensor(row[self.input_float_cols].values.astype(np.int64), dtype=torch.long),
        }
        return sample
    
    def read_label(self, row):
        float_feats = row[self.output_float_cols].values.astype(np.float32)
        float_mask = (float_feats != -1) & ~np.isnan(float_feats)
        float_feats = np.nan_to_num(float_feats, nan=-1.0)
        float_feats = (float_feats - self.mean_std[:, 0]) / self.mean_std[:, 1]
        cat_feats = np.nan_to_num(row[self.cat_cols].values, nan=-1)
        cat_feats = cat_feats.astype(np.int64)
        cat_mask = (cat_feats != -1)
        cat_feats[~cat_mask] = 0
        label = {
            'cat_feats': torch.tensor(cat_feats, dtype=torch.long),
            'cat_valid_mask': torch.tensor(cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(float_feats, dtype=torch.float32),
            'float_valid_mask': torch.tensor(float_mask, dtype=torch.bool)
        }
        return label
    def read_sample_label(self, row):
        input_float_feats = row[self.input_float_cols].values.astype(np.int64)
        float_mask = input_float_feats != -1
        cat_feats = np.nan_to_num(row[self.cat_cols].values.astype(np.float32), nan=-1).astype(np.int64)
        cat_mask = cat_feats != -1
        if self.mask_ratio != 0 and self.mode != 'test':
            
            float_input_indices, float_output_indices = sample_subset(float_mask, self.mask_ratio)
            
            input_bool_array = np.zeros_like(float_mask, dtype=bool)
            input_bool_array[float_input_indices] = True
            input_values_array = np.zeros_like(input_float_feats) - 1
            input_values_array[float_input_indices] = input_float_feats[float_input_indices]
            
            output_float_feats = row[self.output_float_cols].values.astype(np.float32)
            output_float_feats = np.nan_to_num(output_float_feats, nan=-1.0)
            output_float_feats = (output_float_feats - self.mean_std[:, 0]) / (self.mean_std[:, 1] + 1e-10)
            
            output_bool_array = np.zeros_like(float_mask, dtype=bool)
            output_bool_array[float_output_indices] = True
            output_values_array = np.zeros_like(output_float_feats)
            output_values_array[float_output_indices] = output_float_feats[float_output_indices]
            
            valid_cat_indices = np.where(cat_mask)[0]
            n_valid_cats = len(valid_cat_indices)
            
            if n_valid_cats > 0:
                random_values = np.random.random(n_valid_cats)
                output_selector = random_values < self.mask_ratio
                cat_output_indices = valid_cat_indices[output_selector]
                cat_input_indices = valid_cat_indices[~output_selector]
            else:
                cat_input_indices = np.array([], dtype=int)
                cat_output_indices = np.array([], dtype=int)

            input_cat_mask = np.zeros_like(cat_mask, dtype=bool)
            input_cat_mask[cat_input_indices] = True
            
            output_cat_mask = np.zeros_like(cat_mask, dtype=bool)
            output_cat_mask[cat_output_indices] = True
            
            
            input_cat_values = np.zeros_like(cat_feats) - 1
            input_cat_values[cat_input_indices] = cat_feats[cat_input_indices]
            
            output_cat_values = np.zeros_like(cat_feats)
            output_cat_values[cat_output_indices] = cat_feats[cat_output_indices]
        else:
            
            input_values_array = input_float_feats
            output_values_array = row[self.output_float_cols].values.astype(np.float32)
            output_values_array = np.nan_to_num(output_values_array, nan=-1.0)
            output_values_array = (output_values_array - self.mean_std[:, 0]) / (self.mean_std[:, 1] + 1e-10)
            
            input_bool_array = float_mask
            output_bool_array = float_mask
            input_cat_values = cat_feats
            input_cat_mask = cat_mask
            output_cat_values = cat_feats
            output_cat_mask = cat_mask
        output_cat_values[output_cat_values == -1] = 0
        sample = {
            'cat_feats': torch.tensor(input_cat_values, dtype=torch.long),
            'cat_valid_mask': torch.tensor(input_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(input_values_array, dtype=torch.long),
            'float_valid_mask': torch.tensor(input_bool_array, dtype=torch.bool)
        }
        label = {
            'cat_feats': torch.tensor(output_cat_values, dtype=torch.long),
            'cat_valid_mask': torch.tensor(output_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(output_values_array, dtype=torch.float32),
            'float_valid_mask': torch.tensor(output_bool_array, dtype=torch.bool)
        }
        return sample, label
    
    def read_sample_label_old(self, row):
        input_float_feats = row[self.input_float_cols].values.astype(np.int64)
        float_mask = input_float_feats != -1
        cat_feats = np.nan_to_num(row[self.cat_cols].values.astype(np.float32), nan=-1).astype(np.int64)
        cat_mask = cat_feats != -1
        if self.mask_ratio != 0:
            input_indices, output_indices = sample_subset(float_mask, self.mask_ratio)
            input_bool_array = np.zeros_like(float_mask, dtype=bool)
            input_bool_array[input_indices] = True
            input_values_array = np.zeros_like(input_float_feats)
            input_values_array[input_indices] = input_float_feats[input_indices]
            output_float_feats = row[self.output_float_cols].values.astype(np.float32)
            output_float_feats = np.nan_to_num(output_float_feats, nan=-1.0)
            output_float_feats = (output_float_feats - self.mean_std[:, 0]) / self.mean_std[:, 1]
            output_bool_array = np.zeros_like(float_mask, dtype=bool)
            output_bool_array[output_indices] = True
            output_values_array = np.zeros_like(output_float_feats)
            output_values_array[output_indices] = output_float_feats[output_indices]
            
            input_cat_mask = cat_mask & (np.random.randn(1) < self.mask_ratio)
            input_cat = cat_feats if input_cat_mask[0] else np.ones_like(cat_feats) * -1
            
            output_cat_mask = cat_mask & ~input_cat_mask
            output_cat = cat_feats if output_cat_mask[0] else np.zeros_like(cat_feats)
        else:
            input_values_array = input_float_feats
            output_values_array = row[self.output_float_cols].values.astype(np.float32)
            input_bool_array = float_mask
            output_bool_array = float_mask
            input_cat = cat_feats
            input_cat_mask = cat_mask
            output_cat = cat_feats
            output_cat_mask = cat_mask
        
        sample = {
            'cat_feats': torch.tensor(input_cat, dtype=torch.long),
            'cat_valid_mask': torch.tensor(input_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(input_values_array, dtype=torch.long),
            'float_valid_mask': torch.tensor(input_bool_array, dtype=torch.bool)
        }
        label = {
            'cat_feats': torch.tensor(output_cat, dtype=torch.long),
            'cat_valid_mask': torch.tensor(output_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(output_values_array, dtype=torch.float32),
            'float_valid_mask': torch.tensor(output_bool_array, dtype=torch.bool)
        }
        return sample, label
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if chunk_idx > 0:
            
            within_chunk_idx = idx - self.cumulative_lengths[chunk_idx - 1]
        else:
            within_chunk_idx = idx
        row = self.data_chunks[chunk_idx].iloc[within_chunk_idx]
        sample, label = self.read_sample_label(row)
        result = {
            'pid': str(row['pid']),
            'vid': str(row['vid']),
            
            'data': sample,
            'label': label,
        }
        return result

@dataclass
class EHRDataset(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict
    def __post_init__(self):
        self.mask_ratio = self.config['mask_ratio']
        self.float_cols = json_load(self.config['float_feats'])
        self.input_float_cols = [f'tokenized.{x}' for x in self.float_cols]
        self.output_float_cols = self.float_cols
        self.mean_std = np.array([(self.config['feat_info']['float_cols'][x]['mean'], self.config['feat_info']['float_cols'][x]['std']) for x in self.output_float_cols], dtype=np.float32)
        self.cat_cols = ['GENDER']
        self.config['mean_std'] = self.mean_std
        self.config['reg_label_names'] = self.float_cols
        self.config['cls_label_names'] = ['GENDER']
    def read_sample(self, row):
        sample = {
            'cat_feats': torch.tensor(row[self.cat_cols].values.astype(np.int64), dtype=torch.long),
            'float_feats': torch.tensor(row[self.input_float_cols].values.astype(np.int64), dtype=torch.long),
        }
        return sample
    
    def read_label(self, row):
        float_feats = row[self.output_float_cols].values.astype(np.float32)
        float_mask = (float_feats != -1) & ~np.isnan(float_feats)
        float_feats = np.nan_to_num(float_feats, nan=-1.0)
        float_feats = (float_feats - self.mean_std[:, 0]) / (self.mean_std[:, 1] + 1e-10)
        cat_feats = np.nan_to_num(row[self.cat_cols].values, nan=-1)
        cat_feats = cat_feats.astype(np.int64)
        cat_mask = (cat_feats != -1)
        cat_feats[~cat_mask] = 0
        label = {
            'cat_feats': torch.tensor(cat_feats, dtype=torch.long),
            'cat_valid_mask': torch.tensor(cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(float_feats, dtype=torch.float32),
            'float_valid_mask': torch.tensor(float_mask, dtype=torch.bool)
        }
        return label
    
    def read_sample_label(self, row):
        input_float_feats = row[self.input_float_cols].values.astype(np.int64)
        float_mask = input_float_feats != -1
        cat_feats = np.nan_to_num(row[self.cat_cols].values.astype(np.float32), nan=-1).astype(np.int64)
        cat_mask = cat_feats != -1
        if self.mask_ratio != 0:
            input_indices, output_indices = sample_subset(float_mask, self.mask_ratio)
            input_bool_array = np.zeros_like(float_mask, dtype=bool)
            input_bool_array[input_indices] = True
            input_values_array = np.zeros_like(input_float_feats)
            input_values_array[input_indices] = input_float_feats[input_indices]
            output_float_feats = row[self.output_float_cols].values.astype(np.float32)
            output_float_feats = np.nan_to_num(output_float_feats, nan=-1.0)
            output_float_feats = (output_float_feats - self.mean_std[:, 0]) / self.mean_std[:, 1]
            output_bool_array = np.zeros_like(float_mask, dtype=bool)
            output_bool_array[output_indices] = True
            output_values_array = np.zeros_like(output_float_feats)
            output_values_array[output_indices] = output_float_feats[output_indices]
            
            input_cat_mask = cat_mask & (np.random.randn(1) < self.mask_ratio)
            input_cat = cat_feats if input_cat_mask[0] else np.ones_like(cat_feats) * -1
            
            output_cat_mask = cat_mask & ~input_cat_mask
            output_cat = cat_feats if output_cat_mask[0] else np.zeros_like(cat_feats)
        else:
            input_values_array = input_float_feats
            output_values_array = row[self.output_float_cols].values.astype(np.float32)
            input_bool_array = float_mask
            output_bool_array = float_mask
            input_cat = cat_feats
            input_cat_mask = cat_mask
            output_cat = cat_feats
            output_cat_mask = cat_mask
        
        sample = {
            'cat_feats': torch.tensor(input_cat, dtype=torch.long),
            'cat_valid_mask': torch.tensor(input_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(input_values_array, dtype=torch.long),
            'float_valid_mask': torch.tensor(input_bool_array, dtype=torch.bool)
        }
        label = {
            'cat_feats': torch.tensor(output_cat, dtype=torch.long),
            'cat_valid_mask': torch.tensor(output_cat_mask, dtype=torch.bool),
            'float_feats': torch.tensor(output_values_array, dtype=torch.float32),
            'float_valid_mask': torch.tensor(output_bool_array, dtype=torch.bool)
        }
        return sample, label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample, label = self.read_sample_label(row)
        result = {
            'pid': str(row['PATIENT_SN']),
            'vid': str(row['VISIT_SN']),
            'data': sample,
            'label': label,
        }
        return result
    
class EHRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_path = config.get('df_path', None)
        self.use_cache = config.get('use_cache', False)
        self.dataset_col = config.get('dataset_col', None)
        self.batch_size = config.get('batch_size', None)
        self.train_folds = config.get('train_folds', None)
        self.valid_folds = config.get('valid_folds', None)
        self.test_folds = config.get('test_folds', None)
        self.float_cols = json_load(config['float_feats'])
        self.input_float_cols = [f'tokenized.{x}' for x in self.float_cols]
        self.cat_cols = [f'tokenized.{x}' for x in sorted(self.config['feat_info']['category_cols'])]
        
        self.needed_columns = ['pid', 'vid', self.dataset_col] + self.cat_cols + self.float_cols + self.input_float_cols
        self.chunk_size = config.get('chunk_size', 1000000)  
        
    def setup(self, stage=None):
        if isinstance(self.df_path, pd.DataFrame):
            df = self.df_path
            df = optimize_dtypes(df)
            df_train = df[df[self.dataset_col].isin(self.train_folds)].reset_index(drop=True)
            df_valid = df[df[self.dataset_col].isin(self.valid_folds)].reset_index(drop=True)
            df_test = df[df[self.dataset_col].isin(self.test_folds)].reset_index(drop=True)
            
            self.ds_train = EHRDataset(df_train, 'train', self.config)
            self.ds_valid = EHRDataset(df_valid, 'valid', self.config)
            self.ds_test = EHRDataset(df_test, 'test', self.config)
        else:
            parquet_file = pq.ParquetFile(Path(self.df_path))
            df_train_chunks = []
            df_valid_chunks = []
            df_test_chunks = []
            dataset_col_table = pq.read_table(self.df_path, columns=[self.dataset_col])
            dataset_values = dataset_col_table.to_pandas()[self.dataset_col]
            train_mask = dataset_values.isin(self.train_folds)
            valid_mask = dataset_values.isin(self.valid_folds)
            test_mask = dataset_values.isin(self.test_folds)
            train_indices = train_mask[train_mask].index.tolist()
            valid_indices = valid_mask[valid_mask].index.tolist()
            test_indices = test_mask[test_mask].index.tolist()
            
            del dataset_col_table, dataset_values, train_mask, valid_mask, test_mask
            gc.collect()
            
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=self.needed_columns):
                chunk_df = optimize_dtypes(batch.to_pandas())
                start_idx = len(df_train_chunks) * self.chunk_size
                end_idx = start_idx + len(chunk_df)
                chunk_train_indices = [i for i in train_indices if start_idx <= i < end_idx]
                chunk_valid_indices = [i for i in valid_indices if start_idx <= i < end_idx]
                chunk_test_indices = [i for i in test_indices if start_idx <= i < end_idx]
                chunk_train_indices = [i - start_idx for i in chunk_train_indices]
                chunk_valid_indices = [i - start_idx for i in chunk_valid_indices]
                chunk_test_indices = [i - start_idx for i in chunk_test_indices]
                if chunk_train_indices:
                    df_train_chunks.append(chunk_df.iloc[chunk_train_indices].reset_index(drop=True))
                if chunk_valid_indices:
                    df_valid_chunks.append(chunk_df.iloc[chunk_valid_indices].reset_index(drop=True))
                if chunk_test_indices:
                    df_test_chunks.append(chunk_df.iloc[chunk_test_indices].reset_index(drop=True))
                del chunk_df
                gc.collect()
            self.ds_train = ChunkedEHRDataset(df_train_chunks, 'train', self.config)
            self.ds_valid = ChunkedEHRDataset(df_valid_chunks, 'valid', self.config)
            
            if self.config.get('test_df', '') != '':
                df_test = pd.read_csv(self.config['test_df'])
                df_test = optimize_dtypes(df_test)
                self.ds_test = EHRDataset(df_test, 'test', self.config)
            else:
                self.ds_test = ChunkedEHRDataset(df_test_chunks, 'test', self.config)
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8, 
                          pin_memory=True, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size, num_workers=8, 
                          pin_memory=True, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8,
                          pin_memory=True, shuffle=False)
    def teardown(self, stage=None):
        gc.collect()