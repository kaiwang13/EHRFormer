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
   conda create -n ehrformer python=3.8
   conda activate ehrformer
   ```

3. **Install dependencies**:
   ```bash
   pip install torch pytorch-lightning transformers
   pip install pandas numpy scikit-learn tqdm
   pip install diskcache pyarrow wandb statsmodels
   ```

### Run with Demo Data

The repository includes demo data ready for immediate testing:

```bash
# Pretraining with demo data
python pretrain.py

# Finetuning with demo data  
python finetune.py
```

The demo data includes:
- 1,000 synthetic patient samples
- 5 features (1 categorical, 4 continuous)
- Pre-configured paths in `configs/*.json`

## 📊 Data Format Requirements

### For Custom Data Training

To train EHRFormer on your custom data, you need to process your data into the following format:

#### Required DataFrame Columns

```python
# Each row represents one patient sequence
{
    'pid': str,                           # Patient ID (unique identifier)
    'tokenized_category_feats': ndarray,  # (n_cat, seq_len) - Discretized categorical features
    'tokenized_float_feats': ndarray,     # (n_float, seq_len) - Discretized continuous features  
    'category_feats': ndarray,            # (n_cat, seq_len) - Original categorical values
    'float_feats': ndarray,               # (n_float, seq_len) - Original continuous values
    'valid_mask': ndarray,                # (seq_len,) - Valid timestamp mask
    'time_index': ndarray,                # (seq_len,) - Time indices
    'dataset_fold10': int                 # Dataset fold for train/val/test split
}
```

#### Data Processing Steps

1. **Tokenization**: Convert categorical and continuous features to token indices
   - Categorical: Map categories to integer tokens (0, 1, 2, ...)
   - Continuous: Quantize values to discrete tokens (e.g., 0-255 for 256 bins)
   - Missing values: Use -1 as missing token

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

The repository includes demo data in `sample_data/` with:
- **1,000 synthetic patient samples** in `sample_data/sample/`
- **5 features** (1 categorical: GENDER, 4 continuous: lab_float_CBC_RBC, lab_float_CBC_WBC, sign_SBP, sign_DBP)
- **Pre-processed format** for immediate testing
- **Feature statistics** in `sample_data/feat_info.json`
- **Feature list** in `sample_data/float_feats.json`

This serves as a reference for data format and processing pipeline.

### Major disease-category finetuning from lab tests

`prepare_data.py` builds a finetuning dataset for **major disease-category
prediction** directly from a wide per-visit CHAI parquet, writing the exact
loader contract (diskcache + `metadata.parquet` + `feat_info.json` +
`float_feats.json`). Continuous inputs are the `lab_float_*` and `sign_*`
columns; labels are the top-N most prevalent `diag_*` categories, provided as
both current (`c_cls_labels`) and future (`f_cls_labels`) targets.

```bash
python prepare_data.py --src /path/to/per_visit_ehr.parquet \
                       --out sample_data/chai500 --n-patients 500 --top-n 20
python finetune.py   # configs/finetune.json points at sample_data/chai500
```

The generated dataset contains patient-derived data and is git-ignored; run
`prepare_data.py` locally to reproduce it.

## 🎯 Usage

### Pretraining

Train the model with self-supervised feature-level masking:

```bash
python pretrain.py
```

The pretraining process:
1. **Feature-level masking**: Randomly masks 50% of valid features at each timestamp
2. **Multi-task prediction**: Predicts original feature values at masked positions
3. **Balanced losses**: Uses both categorical (cross-entropy) and continuous (MSE) reconstruction losses


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
    "seq_max_len": 64,                   // Maximum sequence length
    "df_paths": "sample_data/sample",    // Data directory (demo data)
    "feat_info_path": "sample_data/feat_info.json", // Feature statistics
    "float_feats": "sample_data/float_feats.json",  // Float feature list
    "use_cache": true,                   // Enable data caching
    "train_folds": [1,2,3,4,5,6,7,8,9], // Training folds
    "valid_folds": [0]                   // Validation folds
}
```

### Notice

The current data is a small amount of synthetic data because the original large-scale EHR pre-training data cannot be published due to privacy and ethical considerations, so the model may not converge normally. The total time for pretraining and finetuning steps are highly correlated with the data scale, GPU type, and the number of GPUs. Currently, the test results show that on a 1xH100 80G, each step of the pre-training and fine-tuning steps takes less than 10s.

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