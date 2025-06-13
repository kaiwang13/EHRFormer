# EHRFormer

EHRFormer is a foundation model for learning robust, universal patient representations from large-scale, longitudinal, and multi-cohort electronic health records (EHRs). It is designed to address the inherent heterogeneity and biases in real-world EHR data, enabling downstream clinical analyses such as biological age (BA) prediction, missing value imputation, and cross-cohort batch effect correction.

## Overview

EHRFormer leverages a stochastic masking strategy for representation encoding and reconstruction, capturing complex feature interactions and imputing missing clinical indicators. An adversarial module is incorporated to remove subjective biases, and a cohort-agnostic adversarial training approach is used to eliminate batch effects across different hospital cohorts. The model is pretrained in an autoregressive manner to capture both current and temporal dynamics of patient health trajectories.

Key features:
- **Stochastic masking** for robust representation learning and imputation.
- **Adversarial training** to remove cohort-specific and subjective biases.
- **Autoregressive pretraining** to model temporal EHR dynamics.
- **Supports multiple clinical indicators** (lab tests, vital signs, etc.).
- **PyTorch Lightning** based training and evaluation pipeline.

## System requirements

- **Hardware**:
  - **GPU**: 8x NVIDIA H100 (80GB) for default training configuration
  - **CPU**: 64 cores recommended
  - **RAM**: 1TB memory
  
- **Software**:
  - Python 3.8+
  - CUDA 11.7+ and cuDNN compatible with your PyTorch version
  

**Note**: The model training parameters can be adjusted based on your available hardware. For systems with less GPU memory or fewer GPUs, consider:
- Reducing batch size in config files
- Decreasing model dimensions
- Training with fewer data folds in parallel

## Installation

1. Clone the repository:
   ```bash
   git clone <your_repo_url>
   cd EHRFormer
   ```

2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install torch pytorch-lightning transformers scikit-learn pandas numpy tqdm matplotlib opencv-python pillow wandb
   ```

   Additional dependencies may be required depending on your environment and GPU setup.

## Data Preparation

- Prepare your EHR data as a DataFrame and feature information JSON files.
- Update the configuration files in `configs/` (e.g., `pretrain.json`, `finetune.json`) with the correct paths:
  - `"df_path"`: Path to your EHR DataFrame.
  - `"feat_info_path"`: Path to the feature info JSON.
  - `"float_feats"`: Path to the float features JSON.
  - `"ckpt_path"`: Path to a checkpoint for inference (if testing).

## Pretraining

To pretrain the EHRFormer model on your EHR data:

```bash
python pretrain.py
```

- The script uses the configuration in `configs/train.json` by default.
- Adjust GPU settings, epochs, and data paths in the config as needed.

## Finetuning

To finetune the pretrained model for downstream tasks (e.g., BA prediction):

```bash
python finetune.py
```

- The script uses the configuration in `configs/finetune.json` by default.
- Update the config for your specific task and data.

## Configuration

Configuration files are in JSON format and control all aspects of training and evaluation. Example (`configs/train.json`):

```json
{
    "debug": false,
    "project": "ehrformer-pretrain",
    "stage": "pretrain",
    "name": "ehrformer",
    "model": "ehrformer",
    "output_dir": "output/pretrain",
    "n_gpus": [0,1,2,3,4,5,6,7],
    "n_epoch": 300,
    "train": true,
    "test": false,
    "lr": 0.0001,
    "wd": 0.05,
    "output_dim": 768,
    "proj_dim": 768,
    "n_category_feats": 28,
    "n_float_feats": 150,
    "n_category_values": 2,
    "n_float_values": 256,
    "mask_ratio": 0.5,
    "pool_emb": false,
    "df_path": "path/to/dataframe",
    "train_length": 1,
    "use_cache": false,
    "dataset_col": "dataset_fold10",
    "batch_size": 768,
    "train_folds": [1,2,3,4,5,6,7,8,9],
    "valid_folds": [0],
    "test_folds": [0,1,2,3,4,5,6,7,8,9],
    "feat_info_path": "path/to/feat_info.json",
    "float_feats": "path/to/float_feats.json",
    "ckpt_path": "path/to/ckpt for inference",
    "pred_folder": "pred"
}
```

## Logging & Checkpoints

- Training logs and checkpoints are saved in the `output/` directory.
- Supports both CSV and [Weights & Biases (wandb)](https://wandb.ai/) logging.

## Applications

- **Biological Age (BA) prediction** across the full human lifecycle.
- **Imputation** of missing clinical indicators.
- **Batch effect correction** for multi-cohort EHR integration.
- **Generalizable patient representation** for downstream clinical tasks.

## Citation

If you use EHRFormer in your research, please cite the corresponding paper:

```
@article{ehrformer,
  title={A full lifecycle biological clock and its impact in health and diseases},
  author={...},
  journal={...},
  year={...}
}
```

---

**Note:** For more details on data formatting, feature selection, and advanced usage, please refer to the comments in the configuration files and the source code. 