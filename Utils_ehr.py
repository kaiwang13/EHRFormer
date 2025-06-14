from Utils import *
import sys
import statsmodels.api as sm
import pandas as pd
import re
import pyarrow.parquet as pq

def save_parquet(df, path):
    tmp_df = df.copy()
    for col in tqdm(tmp_df.columns):
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and len(tmp_df.iloc[0][col].shape) == 2:
            tmp_df[col] = [x.tolist() for x in tmp_df[col]]
    tmp_df.to_parquet(path)

def load_parquet(path, reset_index=True):
    parquet_file = pq.ParquetFile(path)
    tmp_df = []
    for sub_df in parquet_file.iter_batches(batch_size=100000):
        tmp_df.append(sub_df.to_pandas())
    tmp_df = pd.concat(tmp_df, axis=0)
    for col in tqdm(tmp_df.columns):
        if isinstance(tmp_df.iloc[0][col], np.ndarray) and isinstance(tmp_df.iloc[0][col][0], np.ndarray):
            tmp_df[col] = [np.array(list(x)) for x in tmp_df[col]]
    if reset_index:
        tmp_df = tmp_df.reset_index(drop=True)
    return tmp_df

def get_feat_cols(df, prefix):
    if isinstance(prefix, list):
        result = []
        for p in prefix:
            result += list(filter(lambda x: x.startswith(p), df.columns))
        return result
    return list(filter(lambda x: x.startswith(prefix), df.columns))
