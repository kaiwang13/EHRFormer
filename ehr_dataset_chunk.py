"""EHRFormer data loading.

Each patient is stored as one row of a single ``data.parquet`` holding *raw*
per-visit features (NaN = missing) plus per-visit labels. There is no external
diskcache: continuous features are discretized **on the fly** in ``__getitem__``
via a :class:`FeatureSchema`, following the paper's Stage-1 tokenization
(``floor((x - min)/(max - min) * n_bins)``, missing -> -1). Records are stored
raw and share a single feature schema.

The batch structure produced here is unchanged from the previous loader, so the
pretraining / finetuning Lightning modules consume it without modification.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def load_parquet(path):
    return pd.read_parquet(path)


def _stack(cell, dtype):
    """Reconstruct a 2-D numpy array from a parquet list<list<...>> cell."""
    return np.stack(list(cell)).astype(dtype)


class FeatureSchema:
    """Per-feature discretization statistics for on-the-fly tokenization.

    Continuous features are binned into ``n_float_bins`` levels using the min/max
    fitted on the training cohort; ``mean_std`` is used to normalize continuous
    reconstruction targets during pretraining. The float-feature order is the
    sorted ``feat_info['float_cols']`` order, which must match the row order of
    the ``float_raw`` arrays written by ``prepare_data.py``.
    """

    def __init__(self, feat_info: dict, n_float_bins: int):
        self.category_cols = sorted(feat_info["category_cols"])
        self.float_cols = sorted(feat_info["float_cols"].keys())
        self.n_float_bins = n_float_bins
        fc = feat_info["float_cols"]
        self.float_min = np.array([fc[c]["min"] for c in self.float_cols], dtype=np.float32)[:, None]
        self.float_max = np.array([fc[c]["max"] for c in self.float_cols], dtype=np.float32)[:, None]
        self.mean_std = np.array([(fc[c]["mean"], fc[c]["std"]) for c in self.float_cols], dtype=np.float32)
        self.float_mean = self.mean_std[:, 0][:, None]
        self.float_std = self.mean_std[:, 1][:, None] + 1e-8
        self.age_mean = float(feat_info.get("age_mean", 0.0))
        self.age_std = float(feat_info.get("age_std", 1.0)) + 1e-8
        self.n_cohorts = int(feat_info.get("n_cohorts", 1))

    def discretize(self, float_raw: np.ndarray) -> np.ndarray:
        """float_raw: [n_float, L] with NaN for missing -> int tokens [n_float, L], -1 missing."""
        rng = np.maximum(self.float_max - self.float_min, 1e-8)
        b = np.floor((float_raw - self.float_min) / rng * self.n_float_bins)
        b = np.clip(b, 0, self.n_float_bins - 1)
        b = np.where(np.isnan(float_raw), -1, b)
        return b.astype(np.int64)

    @classmethod
    def from_config(cls, config: dict) -> "FeatureSchema":
        with open(config["feat_info_path"]) as f:
            feat_info = json.load(f)
        return cls(feat_info, config["n_float_values"])


class EHRDataset(Dataset):
    """Reads raw per-patient records from a dataframe and tokenizes on the fly."""

    def __init__(self, df: pd.DataFrame, schema: FeatureSchema, config: dict, mode: str):
        self.df = df.reset_index(drop=True)
        self.schema = schema
        self.config = config
        self.mode = mode
        self.mask_ratio = config.get("mask_ratio", 0.5)
        self.seq_max_len = config["seq_max_len"]
        self._setup_config()

    def _setup_config(self):
        """Populate cls/reg label names + head sizes on the shared config."""
        if self.mode == "pretrain":
            # reconstruction targets: categorical (CE) and continuous (MSE) features
            self.config["cls_label_names"] = list(self.schema.category_cols)
            self.config["reg_label_names"] = list(self.schema.float_cols)
            self.config["mean_std"] = self.schema.mean_std
            self.config["mask_ratio"] = self.mask_ratio
            self.config["n_cls"] = []
            return
        first = self.df.iloc[0]
        cls_names, n_cls = [], []
        for label_col, name_col, n in zip(
            self.config["cls_label_cols"],
            self.config["cls_label_name_cols"],
            self.config["cls_label_n_cls"],
        ):
            for col in first[name_col]:
                cls_names.append(f"{label_col}_{col}")
                n_cls.append(n)
        reg_names = []
        for label_col, name_col in zip(self.config["reg_label_cols"], self.config["reg_label_name_cols"]):
            for col in first[name_col]:
                reg_names.append(f"{label_col}_{col}")
                n_cls.append(1)
        self.config["cls_label_names"] = cls_names
        self.config["reg_label_names"] = reg_names
        self.config["n_cls"] = n_cls

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        float_raw = _stack(row["float_raw"], np.float32)      # [n_float, L] (NaN=missing)
        cat = _stack(row["cat_feats"], np.int64)              # [n_cat, L] (raw tokens, -1=missing)
        valid_mask = np.asarray(row["valid_mask"], dtype=bool)
        time_index = np.asarray(row["time_index"], dtype=np.int64)

        tok_float = self.schema.discretize(float_raw)         # on-the-fly discretization
        tok_cat = np.where(cat < 0, -1, cat).astype(np.int64)

        if self.mode == "pretrain":
            sample, label = self._mask_pretrain(tok_cat, tok_float, cat, float_raw, valid_mask, time_index)
            n_cat, L = tok_cat.shape
            n_float = tok_float.shape[0]

            # (a) missingness target: original absence per feature per visit [n_cat+n_float, L]
            missing = np.concatenate([cat < 0, np.isnan(float_raw)], axis=0).astype(np.float32)

            # (b) next-examination targets: visit t predicts visit t+1
            pair_valid = np.zeros(L, dtype=bool)
            pair_valid[: L - 1] = valid_mask[: L - 1] & valid_mask[1:]
            cat_next = np.concatenate([cat[:, 1:], np.full((n_cat, 1), -1)], axis=1)
            float_next = np.concatenate([float_raw[:, 1:], np.full((n_float, 1), np.nan)], axis=1)
            next_cat_mask = (cat_next >= 0) & pair_valid[None, :]
            next_cat = np.where(cat_next >= 0, cat_next, 0).astype(np.int64)
            next_float_mask = (~np.isnan(float_next)) & pair_valid[None, :]
            next_float = np.where(np.isnan(float_next), 0.0,
                                  (float_next - self.schema.float_mean) / self.schema.float_std).astype(np.float32)

            # (c) age clock (normalized), restricted to healthy visits
            age = _stack(row["c_reg_labels"], np.float32)[0]      # [L] raw age
            age_norm = np.nan_to_num((age - self.schema.age_mean) / self.schema.age_std, nan=0.0).astype(np.float32)
            healthy = np.asarray(row["healthy"], dtype=bool)
            age_mask = healthy & valid_mask

            sample["missing_mask"] = torch.tensor(missing, dtype=torch.float32)
            sample["cohort"] = torch.tensor(int(row["cohort"]), dtype=torch.long)
            label["next_cat"] = torch.tensor(next_cat, dtype=torch.long)
            label["next_cat_mask"] = torch.tensor(next_cat_mask, dtype=torch.bool)
            label["next_float"] = torch.tensor(next_float, dtype=torch.float32)
            label["next_float_mask"] = torch.tensor(next_float_mask, dtype=torch.bool)
            label["age"] = torch.tensor(age_norm, dtype=torch.float32)
            label["age_mask"] = torch.tensor(age_mask, dtype=torch.bool)
            return {"pid": row["pid"], "data": sample, "label": label}

        data = {
            "cat_feats": torch.from_numpy(tok_cat),
            "float_feats": torch.from_numpy(tok_float),
            "valid_mask": torch.from_numpy(valid_mask),
            "time_index": torch.from_numpy(time_index),
        }
        return {"pid": row["pid"], "data": data, "label": self._read_label(row)}

    def _mask_pretrain(self, tok_cat, tok_float, orig_cat, orig_float, valid_mask, time_index):
        """Dual stochastic masking (paper Stage 2): mask a fraction of the valid
        features at each visit; reconstruct the masked categorical (CE) and
        continuous (MSE, normalized) values."""
        n_cat, L = tok_cat.shape
        n_float = tok_float.shape[0]
        input_cat, input_float = tok_cat.copy(), tok_float.copy()
        input_cat_masks = np.zeros((n_cat, L), dtype=bool)
        input_float_masks = np.zeros((n_float, L), dtype=bool)
        out_cat = np.zeros((n_cat, L), dtype=np.int64)
        out_cat_masks = np.zeros((n_cat, L), dtype=bool)
        out_float = np.zeros((n_float, L), dtype=np.float32)
        out_float_masks = np.zeros((n_float, L), dtype=bool)

        for t in range(L):
            if not valid_mask[t]:
                continue
            valid = [(i, "cat") for i in range(n_cat) if tok_cat[i, t] != -1]
            valid += [(i, "float") for i in range(n_float) if tok_float[i, t] != -1]
            if not valid:
                continue
            k = max(1, int(len(valid) * self.mask_ratio))
            np.random.shuffle(valid)
            to_mask, to_keep = valid[:k], valid[k:]
            for i, ty in to_keep:
                (input_cat_masks if ty == "cat" else input_float_masks)[i, t] = True
            for i, ty in to_mask:
                if ty == "cat":
                    input_cat[i, t] = -1
                    out_cat_masks[i, t] = True
                    out_cat[i, t] = orig_cat[i, t] if orig_cat[i, t] >= 0 else 0
                else:
                    input_float[i, t] = -1
                    out_float_masks[i, t] = True
                    v = orig_float[i, t]
                    if not np.isnan(v):
                        mean, std = self.schema.mean_std[i]
                        out_float[i, t] = (v - mean) / (std + 1e-10)

        sample = {
            "cat_feats": torch.tensor(input_cat, dtype=torch.long),
            "cat_valid_mask": torch.tensor(input_cat_masks, dtype=torch.bool),
            "float_feats": torch.tensor(input_float, dtype=torch.long),
            "float_valid_mask": torch.tensor(input_float_masks, dtype=torch.bool),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "time_index": torch.tensor(time_index, dtype=torch.long),
        }
        label = {
            "cat_feats": torch.tensor(out_cat, dtype=torch.long),
            "cat_valid_mask": torch.tensor(out_cat_masks, dtype=torch.bool),
            "float_feats": torch.tensor(out_float, dtype=torch.float32),
            "float_valid_mask": torch.tensor(out_float_masks, dtype=torch.bool),
        }
        return sample, label

    def _read_label(self, row):
        labels = {
            "cls": {"values": [], "masks": []},
            "reg": {"values": [], "masks": []},
            "time_index": np.asarray(row["time_index"], dtype=np.int64),
        }
        for value_col in self.config["cls_label_cols"]:
            values = _stack(row[value_col], np.float64)   # [n_category, L]
            for value in values:
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels["cls"]["values"].append(value)
                labels["cls"]["masks"].append(mask)
        for value_col, (mean, std) in zip(self.config["reg_label_cols"], self.config["reg_label_info"]):
            values = _stack(row[value_col], np.float64)   # [n_reg, L]
            for value in values:
                value = (value - mean) / std
                value = np.nan_to_num(value, nan=-1)
                mask = (value != -1) & ~np.isnan(value)
                labels["reg"]["values"].append(value)
                labels["reg"]["masks"].append(mask)

        if labels["cls"]["values"]:
            v = torch.tensor(np.stack(labels["cls"]["values"]), dtype=torch.long)
            v[v == -1] = 0
            labels["cls"]["values"] = v
            labels["cls"]["masks"] = torch.tensor(np.stack(labels["cls"]["masks"]), dtype=torch.bool)
        if labels["reg"]["values"]:
            labels["reg"]["values"] = torch.tensor(np.stack(labels["reg"]["values"]), dtype=torch.float32)
            labels["reg"]["masks"] = torch.tensor(np.stack(labels["reg"]["masks"]), dtype=torch.bool)
        labels["time_index"] = torch.tensor(labels["time_index"], dtype=torch.long)
        return labels


class EHRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.df_paths = config.get("df_paths", [])
        self.dataset_col = config.get("dataset_col", "dataset_fold10")
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.train_folds = config.get("train_folds")
        self.valid_folds = config.get("valid_folds")
        self.test_folds = config.get("test_folds")
        self.mode = "pretrain" if config.get("mode") == "pretrain" else "finetune"
        # expose cohort count + age stats on the config before the model is built
        try:
            with open(config["feat_info_path"]) as f:
                fi = json.load(f)
            config.setdefault("n_cohorts", int(fi.get("n_cohorts", 1)))
            config.setdefault("age_mean", float(fi.get("age_mean", 0.0)))
            config.setdefault("age_std", float(fi.get("age_std", 1.0)))
        except (KeyError, FileNotFoundError):
            pass

    def setup(self, stage=None):
        paths = self.df_paths if isinstance(self.df_paths, list) else [self.df_paths]
        df = pd.concat([load_parquet(Path(p) / "data.parquet") for p in sorted(paths)], ignore_index=True)
        schema = FeatureSchema.from_config(self.config)
        tr = df[df[self.dataset_col].isin(self.train_folds)]
        va = df[df[self.dataset_col].isin(self.valid_folds)]
        te = df[df[self.dataset_col].isin(self.test_folds)]
        self.ds_train = EHRDataset(tr, schema, self.config, self.mode)
        self.ds_valid = EHRDataset(va, schema, self.config, self.mode)
        self.ds_test = EHRDataset(te, schema, self.config, self.mode)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=shuffle)

    def train_dataloader(self):
        return self._loader(self.ds_train, True)

    def val_dataloader(self):
        return self._loader(self.ds_valid, False)

    def test_dataloader(self):
        return self._loader(self.ds_test, False)

    def teardown(self, stage=None):
        pass
