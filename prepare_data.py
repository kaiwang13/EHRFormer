"""Prepare an EHRFormer dataset for major disease-category prediction from
laboratory tests.

Samples N patients from a wide per-visit EHR parquet and writes a single
self-contained parquet of *raw* per-visit features (no diskcache). Continuous
features are discretized on the fly by the dataset (``ehr_dataset_chunk``) via a
shared ``FeatureSchema``, following the paper's Stage-1 tokenization.

  <out>/sample/data.parquet   one row per patient: raw features + labels + fold
  <out>/feat_info.json        per-feature min/max/mean/std for on-the-fly binning
  <out>/float_feats.json      ordered continuous feature list

Features  : continuous  = lab_float_* + sign_*  (raw values, NaN = missing)
            categorical = gender                ({0,1}, -1 = missing)
Labels    : top-N most prevalent diag_* categories as current (c_cls) and future
            (f_cls) per-visit targets, age (c_reg), a per-visit "healthy" flag and a
            cohort id (for pretraining's age clock and domain-adversarial losses).

Usage:
  python prepare_data.py --src /path/to/per_visit_ehr.parquet \
                         --out sample_data --n-patients 500 --top-n 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def fold_of(pid: str, n_fold: int = 10) -> int:
    return int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % n_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="wide per-visit EHR parquet (pid, vid, visit_date, age, gender, lab_float_*, sign_*, diag_*)")
    ap.add_argument("--out", default="sample_data")
    ap.add_argument("--n-patients", type=int, default=500)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--seq-max-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pf = pq.ParquetFile(args.src)
    all_cols = pf.schema_arrow.names

    float_cols = sorted([c for c in all_cols if c.startswith("lab_float_") or c.startswith("sign_")])
    diag_cols_all = sorted([c for c in all_cols if c.startswith("diag_")])
    id_cols = ["pid", "vid", "source", "visit_date", "age", "gender"]
    print(f"[prep] {len(float_cols)} continuous feats, {len(diag_cols_all)} diag categories available")

    # 1) sample N patients by unique pid ------------------------------------
    uniq = pf.read(columns=["pid"]).column("pid").to_pandas().dropna().unique()
    n = min(args.n_patients, len(uniq))
    sampled = set(rng.choice(uniq, size=n, replace=False).tolist())
    print(f"[prep] sampled {len(sampled)} / {len(uniq)} patients")

    # 2) stream file, keep sampled patients' visits -------------------------
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

    prev = {c: data.loc[data[c] == 1, "pid"].nunique() for c in diag_cols_all}
    diag_cols = [c for c, _ in sorted(prev.items(), key=lambda kv: kv[1], reverse=True)[: args.top_n]]
    print(f"[prep] top-{len(diag_cols)} disease categories (by #patients):")
    for c in diag_cols:
        print(f"        {prev[c]:4d}  {c}")

    age_all = data["age"].to_numpy(dtype=np.float64)
    age_all = age_all[~np.isnan(age_all)]
    age_mean, age_std = float(age_all.mean()), float(age_all.std() + 1e-8)

    # cohort map (from the data 'source') for domain-adversarial de-biasing
    sources = sorted(data["source"].fillna("NA").astype(str).unique().tolist())
    cohort_index = {s: i for i, s in enumerate(sources)}
    n_cohorts = len(sources)
    print(f"[prep] {n_cohorts} cohorts (from 'source')")

    # 4) build one raw record per patient -----------------------------------
    T, nF, nD = args.seq_max_len, len(float_cols), len(diag_cols)
    rows = []
    for k, (pid, g) in enumerate(data.groupby("pid", sort=True)):
        opid = f"P{k + 1:05d}"   # opaque study id; the source patient id is NOT stored
        g = g.sort_values("visit_date").head(T)
        L = len(g)

        float_raw = np.full((nF, T), np.nan, dtype=np.float32)
        float_raw[:, :L] = g[float_cols].to_numpy(dtype=np.float32).T       # raw values, NaN = missing

        gender = g["gender"].to_numpy(dtype=np.float64)
        cat = np.full((1, T), -1, dtype=np.int64)
        cat[0, :L] = np.where(np.isnan(gender), -1, np.clip(gender, 0, 1)).astype(np.int64)

        valid_mask = np.zeros(T, dtype=bool)
        valid_mask[:L] = True
        dt = pd.to_datetime(g["visit_date"], errors="coerce")
        days = np.nan_to_num((dt - dt.min()).dt.days.to_numpy(dtype="float64"), nan=0.0)
        time_index = np.zeros(T, dtype=np.int64)
        time_index[:L] = days.astype(np.int64)

        diag = g[diag_cols].to_numpy(dtype=np.float64)                      # (L, nD)
        c_cls = np.full((nD, T), -1, dtype=np.int64)
        f_cls = np.full((nD, T), -1, dtype=np.int64)
        for d in range(nD):
            col = np.where(diag[:, d] == 1, 1, 0)
            c_cls[d, :L] = col
            fut = np.array([1 if (t + 1 < L and col[t + 1:].any()) else 0 for t in range(L)], dtype=np.int64)
            f_cls[d, :L] = fut

        c_reg = np.full((1, T), np.nan, dtype=np.float32)
        c_reg[0, :L] = g["age"].to_numpy(dtype=np.float32)                  # raw age; normalized by the loader

        # per-visit "healthy" flag (no diagnosis of ANY category) for the age clock
        diag_all = np.nan_to_num(g[diag_cols_all].to_numpy(dtype=np.float64), nan=0.0)
        healthy = np.zeros(T, dtype=bool)
        healthy[:L] = (diag_all != 1).all(axis=1)
        src = g["source"].iloc[0]
        cohort = cohort_index["NA" if pd.isna(src) else str(src)]

        rows.append({
            "pid": opid,
            "float_raw": float_raw.tolist(),
            "cat_feats": cat.tolist(),
            "valid_mask": valid_mask.tolist(),
            "time_index": time_index.tolist(),
            "c_cls_labels": c_cls.tolist(),
            "f_cls_labels": f_cls.tolist(),
            "c_reg_labels": c_reg.tolist(),
            "healthy": healthy.tolist(),
            "cohort": int(cohort),
            "diag_cols": diag_cols,
            "reg_cols": ["age"],
            "dataset_fold10": fold_of(opid),
        })

    # 5) write outputs -------------------------------------------------------
    out = Path(args.out)
    sample_dir = out / "sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_parquet(sample_dir / "data.parquet", index=False)

    feat_info = {"category_cols": ["gender"],
                 "float_cols": {c: {k: fstats[c][k] for k in ("mean", "std", "min", "max")} for c in float_cols},
                 "age_mean": age_mean, "age_std": age_std,
                 "n_cohorts": n_cohorts}
    (out / "feat_info.json").write_text(json.dumps(feat_info, indent=2))
    (out / "float_feats.json").write_text(json.dumps(float_cols, indent=2))

    folds = pd.Series([r["dataset_fold10"] for r in rows]).value_counts().sort_index().to_dict()
    print(f"[prep] wrote {len(rows)} patients -> {sample_dir / 'data.parquet'}")
    print(f"[prep] n_float_feats={nF}  n_category_feats=1  n_diseases={nD}")
    print(f"[prep] age mean/std = {age_mean:.3f} / {age_std:.3f}  (for reg_label_info)")
    print(f"[prep] fold counts: {folds}")


if __name__ == "__main__":
    main()
