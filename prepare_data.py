"""Prepare a small EHRFormer finetuning dataset for major disease-category
prediction from laboratory tests.

Samples N patients from a wide per-visit CHAI parquet and writes the exact
on-disk contract the loader (``ehr_dataset_chunk.EHRDataModule``) expects:

  <out>/sample/            diskcache keyed by pid  (+ metadata.parquet)
  <out>/feat_info.json     per-feature normalization statistics
  <out>/float_feats.json   ordered list of continuous feature columns

Each cached patient record holds the tokenized/original per-visit feature
matrices, the valid-visit mask, the time index, and the current/future
disease-category labels and age target used by the finetune module.

Features  : continuous  = lab_float_* + sign_*  (discretized to n_float_values bins)
            categorical = gender                (tokenized to {0,1})
Labels    : the top-N most prevalent diag_* disease categories, as
            c_cls (present at the visit) and f_cls (appears at a later visit).

Usage:
  python prepare_data.py --src /path/to/per_visit_ehr.parquet \
                         --out sample_data/chai500 --n-patients 500 --top-n 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import diskcache
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def fold_of(pid: str, n_fold: int = 10) -> int:
    """Deterministic patient-level fold assignment in [0, n_fold)."""
    h = hashlib.md5(str(pid).encode()).hexdigest()
    return int(h, 16) % n_fold


def discretize(x: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    """floor((x-lo)/(hi-lo) * n_bins) clipped to [0, n_bins-1]; NaN -> -1."""
    rng = max(hi - lo, 1e-8)
    b = np.floor((x - lo) / rng * n_bins)
    b = np.clip(b, 0, n_bins - 1)
    b = np.where(np.isnan(x), -1, b)
    return b.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="wide per-visit EHR parquet (pid, vid, visit_date, age, gender, lab_float_*, sign_*, diag_*)")
    ap.add_argument("--out", default="sample_data/chai500")
    ap.add_argument("--n-patients", type=int, default=500)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--seq-max-len", type=int, default=16)
    ap.add_argument("--n-float-values", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pf = pq.ParquetFile(args.src)
    all_cols = pf.schema_arrow.names

    float_cols = sorted([c for c in all_cols if c.startswith("lab_float_") or c.startswith("sign_")])
    diag_cols_all = sorted([c for c in all_cols if c.startswith("diag_")])
    id_cols = ["pid", "vid", "visit_date", "age", "gender"]
    print(f"[prep] {len(float_cols)} continuous feats, {len(diag_cols_all)} diag categories available")

    # 1) sample N patients by unique pid ------------------------------------
    pid_tbl = pf.read(columns=["pid"]).column("pid").to_pandas()
    uniq = pid_tbl.dropna().unique()
    n = min(args.n_patients, len(uniq))
    sampled = set(rng.choice(uniq, size=n, replace=False).tolist())
    print(f"[prep] sampled {len(sampled)} / {len(uniq)} patients")

    # 2) stream the file, keep only sampled patients' visits ----------------
    needed = id_cols + float_cols + diag_cols_all
    parts = []
    for batch in pf.iter_batches(columns=needed, batch_size=100_000):
        df = batch.to_pandas()
        df = df[df["pid"].isin(sampled)]
        if len(df):
            parts.append(df)
    data = pd.concat(parts, ignore_index=True)
    print(f"[prep] collected {len(data)} visits for {data['pid'].nunique()} patients")

    # 3) per-feature stats + top-N diagnoses --------------------------------
    fstats = {}
    for c in float_cols:
        v = data[c].to_numpy(dtype=np.float64)
        v = v[~np.isnan(v)]
        if v.size == 0:
            fstats[c] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
        else:
            fstats[c] = {"mean": float(v.mean()), "std": float(v.std() + 1e-8),
                         "min": float(np.percentile(v, 0.5)), "max": float(np.percentile(v, 99.5))}

    # prevalence = fraction of patients with >=1 positive visit for the category
    prev = {}
    for c in diag_cols_all:
        pos_pat = data.loc[data[c] == 1, "pid"].nunique()
        prev[c] = pos_pat
    diag_cols = [c for c, _ in sorted(prev.items(), key=lambda kv: kv[1], reverse=True)[: args.top_n]]
    print(f"[prep] top-{len(diag_cols)} disease categories (by #patients):")
    for c in diag_cols:
        print(f"        {prev[c]:4d}  {c}")

    age_all = data["age"].to_numpy(dtype=np.float64)
    age_all = age_all[~np.isnan(age_all)]
    age_mean, age_std = float(age_all.mean()), float(age_all.std() + 1e-8)

    # 4) build per-patient records ------------------------------------------
    out = Path(args.out)
    sample_dir = out / "sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cache = diskcache.Cache(str(sample_dir), eviction_policy="none")
    meta_rows = []
    T = args.seq_max_len
    nF, nC, nD = len(float_cols), 1, len(diag_cols)

    for pid, g in data.groupby("pid", sort=False):
        g = g.sort_values("visit_date").head(T)
        L = len(g)

        float_raw = g[float_cols].to_numpy(dtype=np.float64).T          # (nF, L)
        gender = g["gender"].to_numpy(dtype=np.float64)                 # (L,)
        diag = g[diag_cols].to_numpy(dtype=np.float64)                  # (L, nD)
        ages = g["age"].to_numpy(dtype=np.float64)                      # (L,)

        # tokenize
        tok_float = np.full((nF, T), -1, dtype=np.int64)
        orig_float = np.zeros((nF, T), dtype=np.float32)
        for j, c in enumerate(float_cols):
            tok_float[j, :L] = discretize(float_raw[j], fstats[c]["min"], fstats[c]["max"], args.n_float_values)
            orig_float[j, :L] = np.nan_to_num(float_raw[j], nan=0.0)
        tok_cat = np.full((nC, T), -1, dtype=np.int64)
        gcodes = np.where(np.isnan(gender), -1, np.clip(gender, 0, 1)).astype(np.int64)
        tok_cat[0, :L] = gcodes
        orig_cat = np.zeros((nC, T), dtype=np.int64)
        orig_cat[0, :L] = np.where(gcodes < 0, 0, gcodes)

        valid_mask = np.zeros(T, dtype=bool)
        valid_mask[:L] = True
        dt = pd.to_datetime(g["visit_date"], errors="coerce")
        days_since = np.nan_to_num((dt - dt.min()).dt.days.to_numpy(dtype="float64"), nan=0.0)
        time_index = np.zeros(T, dtype=np.int64)
        time_index[:L] = days_since.astype(np.int64)

        # labels: current (present at visit t) and future (appears at t' > t)
        c_cls = np.full((nD, T), -1, dtype=np.int64)
        f_cls = np.full((nD, T), -1, dtype=np.int64)
        for d in range(nD):
            col = np.where(diag[:, d] == 1, 1, 0)                       # (L,) present per visit
            c_cls[d, :L] = col
            fut = np.zeros(L, dtype=np.int64)
            for t in range(L):
                fut[t] = 1 if (t + 1 < L and col[t + 1:].any()) else 0
            f_cls[d, :L] = fut

        c_reg = np.full((1, T), -1.0, dtype=np.float32)
        c_reg[0, :L] = np.nan_to_num(ages, nan=-1.0)

        cache[pid] = {
            "pid": pid,
            "tokenized_category_feats": tok_cat,
            "tokenized_float_feats": tok_float,
            "category_feats": orig_cat,
            "float_feats": orig_float,
            "valid_mask": valid_mask,
            "time_index": time_index,
            "c_cls_labels": c_cls,
            "f_cls_labels": f_cls,
            "c_reg_labels": c_reg,
        }
        meta_rows.append({
            "pid": pid,
            "diag_cols": diag_cols,
            "reg_cols": ["age"],
            "dataset_fold10": fold_of(pid),
        })
    cache.close()

    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(sample_dir / "metadata.parquet", index=False)

    # 5) feature-info side files --------------------------------------------
    feat_info = {"category_cols": ["gender"],
                 "float_cols": {c: {k: fstats[c][k] for k in ("mean", "std", "min", "max")} for c in float_cols}}
    (out / "feat_info.json").write_text(json.dumps(feat_info, indent=2))
    (out / "float_feats.json").write_text(json.dumps(float_cols, indent=2))

    print(f"[prep] wrote {len(meta_rows)} patients -> {sample_dir}")
    print(f"[prep] n_float_feats={nF}  n_category_feats={nC}  n_diseases={nD}")
    print(f"[prep] age mean/std = {age_mean:.3f} / {age_std:.3f}  (for reg_label_info)")
    print(f"[prep] fold counts: {meta['dataset_fold10'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
