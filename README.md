# EHRFormer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-ee4c2c.svg)](https://pytorch-lightning.readthedocs.io/)

A unified transformer-based foundation model for Electronic Health Records (EHR) that supports both self-supervised pretraining and supervised finetuning on sequential clinical data with advanced feature-level masking.

## 🔥 Features

- **Unified Architecture**: Single `EHRFormer` model for both pretraining and finetuning
- **Sequential Modeling**: Handles temporal EHR data with timestamp-aware attention
- **Feature-Level Masking**: Advanced pretraining with timestamp-wise feature masking (50% of valid features per timestamp)
- **Multi-Task Learning**: Supports both classification and regression tasks
- **Distributed Training**: PyTorch Lightning with multi-GPU support
- **Flexible Data Loading**: Efficient chunked data loading with caching support

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on H100)
- **Memory**: 16GB+ GPU memory recommended for default settings
- **Storage**: SSD recommended for large datasets

### Software
- Python 3.8+
- CUDA 11.7+ compatible with PyTorch
- PyTorch Lightning 2.0+

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EHRFormer
   ```

2. **Create conda environment**:
   ```bash
   conda create -n ehrformer python=3.10
   conda activate ehrformer
   ```

3. **Install dependencies**:
   ```bash
   # install the torch build matching your CUDA (see requirements.txt), e.g.:
   pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```

### Run with Demo Data

The repository includes demo data ready for immediate testing:

```bash
# Pretraining with demo data
python pretrain.py

# Finetuning with demo data  
python finetune.py
```

The demo data (`sample_data/sample/data.parquet`) is a raw per-visit dataset of
500 de-identified patients with 150 continuous lab/vital
features and 20 major disease categories; continuous features are discretized
on the fly by the loader. Regenerate it with `prepare_data.py` (see below).

## 📊 Data Format Requirements

### For Custom Data Training

To train EHRFormer on your custom data, you need to process your data into the following format:

#### Required DataFrame Columns

```python
# One row per patient in a single data.parquet (arrays stored as nested lists).
# Continuous features are RAW (NaN = missing); the loader discretizes on the fly.
{
    'pid': str,                # Patient ID
    'float_raw': ndarray,      # (n_float, seq_len)   raw continuous values, NaN = missing
    'cat_feats': ndarray,      # (n_cat, seq_len)     categorical token ids, -1 = missing
    'valid_mask': ndarray,     # (seq_len,)           real vs padded visits
    'time_index': ndarray,     # (seq_len,)           days since first visit
    'c_cls_labels': ndarray,   # (n_disease, seq_len) current diagnosis (finetune)
    'f_cls_labels': ndarray,   # (n_disease, seq_len) future diagnosis (finetune)
    'c_reg_labels': ndarray,   # (n_reg, seq_len)     regression targets, e.g. age
    'healthy': ndarray,        # (seq_len,)           1 = no diagnosis at visit (age clock)
    'cohort': int,             # data-source id (domain-adversarial de-biasing)
    'diag_cols': list,         # disease-category names
    'reg_cols': list,          # regression target names
    'dataset_fold10': int      # patient-level fold for train/val/test
}
```

#### Data Processing Steps

1. **On-the-fly tokenization**: store RAW continuous values (NaN = missing); the
   loader discretizes them to `n_float_values` bins per feature at read time via a
   shared `FeatureSchema` (from `feat_info.json`). Categorical features are stored
   as integer token ids (-1 = missing). No pre-tokenization / diskcache is needed.

2. **Sequence Formatting**: Organize data by timestamps
   - Each feature becomes a sequence across time
   - Pad or truncate sequences to `seq_max_len`
   - Create `valid_mask` to indicate actual vs padded timestamps

3. **Feature Statistics**: Create `feat_info.json` with normalization statistics
   ```json
   {
       "category_cols": ["GENDER", "AGE_GROUP", ...],
       "float_cols": {
           "LAB_VALUE_1": {"mean": 50.0, "std": 15.0},
           "LAB_VALUE_2": {"mean": 100.0, "std": 25.0},
           ...
       }
   }
   ```

### Demo Data

The repository includes demo data in `sample_data/`:
- **500 de-identified patients** in `sample_data/sample/data.parquet` (a single
  single raw parquet; no diskcache)
- **150 continuous lab/vital features** (`lab_float_*`, `sign_*`) + gender
- **20 major disease categories** as current/future per-visit labels
- **Feature statistics** in `sample_data/feat_info.json`, feature list in
  `sample_data/float_feats.json`

### Major disease-category finetuning from lab tests

`prepare_data.py` builds this dataset for **major disease-category prediction**
directly from a wide per-visit EHR parquet, writing a single self-contained
`sample_data/sample/data.parquet` plus `feat_info.json` / `float_feats.json`.
Continuous inputs are the `lab_float_*` and `sign_*` columns (stored raw,
discretized on the fly); labels are the top-N most prevalent `diag_*` categories
as current (`c_cls_labels`) and future (`f_cls_labels`) per-visit targets.

```bash
python prepare_data.py --src /path/to/per_visit_ehr.parquet \
                       --out sample_data --n-patients 500 --top-n 20
python finetune.py   # configs/finetune.json points at sample_data/sample
```

## 🎯 Usage

### Pretraining

Train the model with self-supervised feature-level masking:

```bash
python pretrain.py
```

The pretraining process jointly optimizes the paper's self-supervised objectives:
1. **Dual stochastic masking**: masks 50% of valid features per visit and reconstructs
   them in the **current and next** examination (categorical CE + continuous MSE)
2. **Variational framework (ELBO)**: VAE latent with a weighted KL term
3. **Age clock**: per-visit age regression on *healthy* visits (biological age)
4. **Domain-adversarial de-biasing**: cohort and missingness discriminators behind a
   gradient-reversal layer → cohort-invariant and missing-invariant representations


### Finetuning

Finetune the pretrained model for downstream tasks:

```bash
python finetune.py
```

The finetuning process:
1. Loads pretrained EHRFormer weights
2. Adds task-specific prediction heads
3. Trains on labeled data for specific clinical tasks

### Configuration

Modify `configs/pretrain.json` or `configs/finetune.json`:

```json
{
    "mode": "pretrain",                    // "pretrain" or "finetune"
    "n_gpus": [0],                        // GPU devices to use
    "batch_size": 32,                     // Batch size
    "lr": 1e-4,                          // Learning rate
    "n_epoch": 100,                      // Number of epochs
    "seq_max_len": 16,                   // Maximum sequence length (visits)
    "n_encoder_layers": 2,               // examination-encoder depth (paper: 24)
    "encoder_hidden": 768,               // examination-encoder width (paper: 1024)
    "mask_ratio": 0.5,                   // fraction of features masked per visit
    "predict_next_visit": true,          // next-examination prediction
    "w_next": 1.0, "w_age": 1.0,         // next-visit / age-clock loss weights
    "w_domain": 0.1, "w_missing": 0.1,   // cohort / missingness adversarial weights
    "w_kl": 0.001,                       // KL weight (ELBO)
    "df_paths": "sample_data/sample",    // Data directory (demo data)
    "feat_info_path": "sample_data/feat_info.json", // Feature statistics
    "float_feats": "sample_data/float_feats.json",  // Float feature list
    "train_folds": [1,2,3,4,5,6,7,8,9], // Training folds
    "valid_folds": [0]                   // Validation folds
}
```

### Notice

The demo data is a small de-identified 500-patient sample; the original large-scale EHR pre-training data cannot be published due to privacy and ethical considerations, so the model may not converge to clinically meaningful performance on this sample. The total time for pretraining and finetuning steps is highly correlated with the data scale, GPU type, and the number of GPUs. Currently, the test results show that on a 1xH100 80G, each step of the pre-training and fine-tuning steps takes less than 10s.

## 📁 Project Structure

```
EHRFormer/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── pretrain.py                  # Pretraining script
├── finetune.py                  # Finetuning script
├── ehrformer.py                 # Core model architecture
├── ehr_model_module_pretrain.py # Pretraining Lightning module
├── ehr_model_module_finetune.py # Finetuning Lightning module
├── ehr_dataset_chunk.py         # Data loading and processing
├── Utils.py                     # Utility functions
├── configs/                     # Configuration files
│   ├── pretrain.json           
│   └── finetune.json           
├── sample_data/                 # Demo data
│   ├── sample/                  # 1,000 synthetic patient samples
│   ├── feat_info.json          # Feature statistics
│   └── float_feats.json        # Float feature list
└── output/                      # Training outputs and checkpoints
```

## ⚠️ Important Notes

### Data Requirements
- **Custom data must be processed** into the specified format before training
- **Feature tokenization** is required for both categorical and continuous features
- **Normalization statistics** must be provided in `feat_info.json`
- **Demo data** (1,000 samples, 5 features) is provided in `sample_data/` for testing only

### Computational Requirements
- Default configuration requires significant GPU memory
- Adjust `batch_size` and model dimensions based on available hardware
- Consider using gradient checkpointing for memory-constrained environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use EHRFormer in your research, please cite:

```bibtex
@article{ehrformer,
  title={A full lifecycle biological clock based on routine clinical data and its impact in health and diseases},
  author={...},
  journal={...},
  year={...}
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Documentation**: See code comments and configuration files

---

**Note**: This implementation demonstrates advanced feature-level masking for EHR pretraining. The included demo data (1,000 samples, 5 features) in `sample_data/` serves as a reference implementation. For production use with custom data, ensure proper data preprocessing following the specified format requirements. 