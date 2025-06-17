"""EHRFormer pretraining module with feature-level masking."""
from collections import defaultdict
from functools import reduce
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import functional as F
from torchmetrics import *
from torchmetrics.utilities.data import dim_zero_cat
import torch
import torchmetrics.functional as MF
from ddp_Utils import *
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from ehrformer import EHRFormer
from Utils import str_hash
import gc
from tqdm import tqdm


def group_by_M_dimension(A_list, B_list, C_list):
    """Group tensors by sequence length for efficient processing."""
    device = A_list[0].device
    M_values = torch.tensor([A.size(1) for A in A_list], device=device)
    unique_M = torch.unique(M_values)
    
    groups = {}
    for M in unique_M.cpu().numpy():
        mask = M_values == M
        indices = torch.where(mask)[0]
        groups[M] = (
            torch.cat([A_list[i] for i in indices], dim=0),
            torch.cat([B_list[i] for i in indices], dim=0),
            torch.cat([C_list[i] for i in indices], dim=0)
        )
    
    return groups


class EHRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.config['mode'] = 'pretrain'
        
        self.data_type = config.get('data_type', '')
        self.momentum = config.get('momentum', 0.9)
        self.wd = config.get('wd', 1e-6)
        self.lr = config.get('lr', 1e-3)
        self.n_nodes = config.get('n_nodes', 1)
        self.n_gpus = config.get('n_gpus', 1)
        if isinstance(self.n_gpus, list):
            self.n_gpus = len(self.n_gpus)
        self.n_epoch = config.get('n_epoch', None)
        
        if 'feat_info' in config:
            self.cls_label_names = sorted(config['feat_info']['category_cols'])
            self.reg_label_names = sorted([x for x in config['feat_info']['float_cols'].keys()])
        else:
            self.cls_label_names = config.get('cls_label_names', [])
            self.reg_label_names = config.get('reg_label_names', [])
        
        self.config['cls_label_names'] = self.cls_label_names
        self.config['reg_label_names'] = self.reg_label_names
        
        self.model = EHRFormer(config)

        self.output_dir = config.get('output_dir', None)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pred_folder = config.get('pred_folder', 'pred')

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                  first_cycle_steps=self.n_epoch,
                                                  max_lr=self.lr, 
                                                  min_lr=1e-8, 
                                                  warmup_steps=int(self.n_epoch * 0.1))
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step()

    def on_train_start(self):
        self.loggers[0].log_hyperparams(self.config)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        
        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        cat_valid_mask = data['cat_valid_mask']
        float_valid_mask = data['float_valid_mask']
        
        reg_label = label['float_feats']
        reg_mask = label['float_valid_mask']
        cls_label = label['cat_feats']
        cls_mask = label['cat_valid_mask']

        valid_mask = (cat_valid_mask.any(dim=1) | float_valid_mask.any(dim=1))
        time_index = torch.arange(cat_feats.shape[-1], device=cat_feats.device).unsqueeze(0).repeat(cat_feats.shape[0], 1)

        cls_preds, reg_preds = self.model(cat_feats, float_feats, valid_mask, time_index)
        
        # Regression loss
        reg_loss = 0
        n_reg_tasks = 0
        if len(reg_preds) > 0 and reg_label.shape[1] > 0:
            for i in range(min(len(reg_preds), reg_label.shape[1])):
                y_reg_loss = reg_preds[i].reshape(-1)
                reg_label_loss = reg_label[:, i, :].reshape(-1)
                reg_mask_loss = reg_mask[:, i, :].reshape(-1)
                reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
                reg_loss += (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))
                n_reg_tasks += 1
        
        if n_reg_tasks > 0:
            reg_loss = reg_loss / n_reg_tasks

        # Classification loss
        cls_loss = 0
        n_cls_tasks = 0
        if len(cls_preds) > 0 and cls_label.shape[1] > 0:
            for i in range(min(len(cls_preds), cls_label.shape[1])):
                y_cls_loss = cls_preds[i].reshape(-1, 2)
                cls_label_loss = cls_label[:, i, :].reshape(-1)
                cls_mask_loss = cls_mask[:, i, :].reshape(-1)
                cls_losses = F.cross_entropy(y_cls_loss, cls_label_loss, reduction='none')
                cls_loss += (cls_losses * cls_mask_loss).sum() / (cls_mask_loss.sum().clip(1))
                n_cls_tasks += 1
        
        if n_cls_tasks > 0:
            cls_loss = cls_loss / n_cls_tasks

        loss = cls_loss + reg_loss
        self.log("train_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("train_reg_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        
        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        cat_valid_mask = data['cat_valid_mask']
        float_valid_mask = data['float_valid_mask']
        
        reg_label = label['float_feats']
        reg_mask = label['float_valid_mask']
        cls_label = label['cat_feats']
        cls_mask = label['cat_valid_mask']

        valid_mask = (cat_valid_mask.any(dim=1) | float_valid_mask.any(dim=1))
        time_index = torch.arange(cat_feats.shape[-1], device=cat_feats.device).unsqueeze(0).repeat(cat_feats.shape[0], 1)

        cls_preds, reg_preds = self.model(cat_feats, float_feats, valid_mask, time_index)
        
        reg_loss = 0
        n_reg_tasks = 0
        if len(reg_preds) > 0 and reg_label.shape[1] > 0:
            for i in range(min(len(reg_preds), reg_label.shape[1])):
                y_reg_loss = reg_preds[i].reshape(-1)
                reg_label_loss = reg_label[:, i, :].reshape(-1)
                reg_mask_loss = reg_mask[:, i, :].reshape(-1)
                reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
                reg_loss += (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))
                n_reg_tasks += 1
        
        # Average regression loss over number of regression tasks
        if n_reg_tasks > 0:
            reg_loss = reg_loss / n_reg_tasks

        # Compute classification loss
        cls_loss = 0
        n_cls_tasks = 0
        if len(cls_preds) > 0 and cls_label.shape[1] > 0:
            for i in range(min(len(cls_preds), cls_label.shape[1])):
                y_cls_loss = cls_preds[i].reshape(-1, 2)
                cls_label_loss = cls_label[:, i, :].reshape(-1)
                cls_mask_loss = cls_mask[:, i, :].reshape(-1)
                cls_losses = F.cross_entropy(y_cls_loss, cls_label_loss, reduction='none')
                cls_loss += (cls_losses * cls_mask_loss).sum() / (cls_mask_loss.sum().clip(1))
                n_cls_tasks += 1
        
        # Average classification loss over number of classification tasks
        if n_cls_tasks > 0:
            cls_loss = cls_loss / n_cls_tasks

        loss = cls_loss + reg_loss

        # Gather tensors for metric computation (following original logic)
        tensor_to_gather = [
            cls_label.contiguous(), cls_mask.contiguous(),
            reg_label.contiguous(), reg_mask.contiguous(),
        ] + [pred.contiguous() for pred in cls_preds] + [pred.contiguous() for pred in reg_preds]
        
        tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
        
        cls_label = tensor_gathered[0]
        cls_mask = tensor_gathered[1]
        reg_label = tensor_gathered[2]
        reg_mask = tensor_gathered[3]
        
        gathered_cls_preds = tensor_gathered[4:4+len(cls_preds)]
        gathered_reg_preds = tensor_gathered[4+len(cls_preds):4+len(cls_preds)+len(reg_preds)]

        # Accumulate predictions following original pretrain logic
        for i in range(len(self.cls_label_names)):
            if i < len(gathered_cls_preds) and i < cls_label.shape[1]:
                mask = cls_mask[:, i, :].reshape(-1)  # (B*L,)
                pp = gathered_cls_preds[i].reshape(-1, 2)  # (B*L, 2)
                yy = cls_label[:, i, :].reshape(-1)  # (B*L,)
                
                # Only use valid masked positions
                valid_indices = mask.bool()
                if valid_indices.sum() > 2:
                    self.preds[f'cls_{i}'].append(pp[valid_indices, 1])  # Positive class prob
                    self.targets[f'cls_{i}'].append(yy[valid_indices])

        for i in range(len(self.reg_label_names)):
            if i < len(gathered_reg_preds) and i < reg_label.shape[1]:
                mask = reg_mask[:, i, :].reshape(-1)  # (B*L,)
                pp = gathered_reg_preds[i].reshape(-1)  # (B*L,)
                yy = reg_label[:, i, :].reshape(-1)  # (B*L,)
                
                # Only use valid masked positions
                valid_indices = mask.bool()
                if valid_indices.sum() > 2:
                    self.preds[f'reg_{i}'].append(pp[valid_indices])
                    self.targets[f'reg_{i}'].append(yy[valid_indices])

        self.log("val_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_cls_loss", cls_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_reg_loss", reg_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        mauc = []
        mf1 = []
        for i, k in enumerate(self.cls_label_names):
            if len(self.preds[f'cls_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'cls_{i}']).double()
                target = dim_zero_cat(self.targets[f'cls_{i}'])
                
                precision, recall, _ = precision_recall_curve(target.numpy(), pred.numpy())
                precision += 1e-10
                recall += 1e-10
                f1 = 2*recall*precision/(recall+precision)
                best_precision = precision[np.argmax(f1)]
                best_recall = recall[np.argmax(f1)]
                best_f1 = np.max(2*recall*precision/(recall+precision))
                mf1.append(best_f1)
                auc_score = MF.auroc(pred, target, task='binary')
                self.log(f"val_{k}_auc", auc_score, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_f1", best_f1, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_precision", best_precision, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_recall", best_recall, prog_bar=False, rank_zero_only=True)
                mauc.append(auc_score.item())
            else:
                self.log(f"val_{k}_auc", torch.nan, prog_bar=False, rank_zero_only=True)
        
        if len(mauc) > 0:
            mauc_mean = sum(mauc) / len(mauc)
            self.log(f"val_mauc", mauc_mean, prog_bar=True, rank_zero_only=True)

        if len(mf1) > 0:
            mf1_mean = sum(mf1) / len(mf1)
            self.log(f"val_mf1", mf1_mean, prog_bar=True, rank_zero_only=True)

        mpcc = []
        mr2 = []
        for i, k in enumerate(self.reg_label_names):
            if len(self.preds[f'reg_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double()
                target = dim_zero_cat(self.targets[f'reg_{i}']).double()
                self.log(f"val_{k}_mse", MF.mean_squared_error(pred, target), prog_bar=False, rank_zero_only=True)
                pcc_score = MF.pearson_corrcoef(pred, target)
                r2_score = MF.r2_score(pred, target)
                self.log(f"val_{k}_pcc", pcc_score, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", r2_score, prog_bar=False, rank_zero_only=True)
                mpcc.append(pcc_score.item())
                mr2.append(r2_score.item())
        
        if len(mpcc) > 0:
            mpcc_mean = sum(mpcc) / len(mpcc)
            self.log(f"val_mpcc", mpcc_mean, prog_bar=True, rank_zero_only=True)
        
        if len(mr2) > 0:
            mr2_mean = sum(mr2) / len(mr2)
            self.log(f"val_mr2", mr2_mean, prog_bar=True, rank_zero_only=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        data = batch['data']
        pid = batch['pid']
        vid = batch['vid']
        
        label = batch['label']
        
        cat_feats = data['cat_feats']
        float_feats = data['float_feats']
        valid_mask = data['valid_mask']
        time_index = data['time_index']
        
        reg_label = label['float_feats']
        cls_label = label['cat_feats']
        reg_mask = label['float_valid_mask']

        cls_preds, reg_preds = self.model(cat_feats, float_feats, valid_mask, time_index)
        
        # Convert to the format expected by original test logic
        # Stack predictions along feature dimension
        if len(cls_preds) > 0:
            y_cls = torch.stack([torch.softmax(pred, dim=-1) for pred in cls_preds], dim=1)  # (B, n_cls, L, 2)
        else:
            y_cls = torch.empty(cat_feats.shape[0], 0, cat_feats.shape[2], 2)
            
        if len(reg_preds) > 0:
            y_reg = torch.stack(reg_preds, dim=1)  # (B, n_reg, L)
        else:
            y_reg = torch.empty(cat_feats.shape[0], 0, cat_feats.shape[2])
        
        y_cls = y_cls.float()
        y_reg = y_reg.float()
        
        self.test_outputs.append({
            'pid': pid, 'vid': vid,
            'cls_preds': y_cls.cpu().numpy(), 
            'cls_label': cls_label.cpu().numpy(), 
            'reg_preds': y_reg.cpu().numpy(), 
            'reg_label': reg_label.cpu().numpy(), 
            'reg_mask': reg_mask.cpu().numpy()
        })
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        output_dir = Path(self.output_dir) / self.pred_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Following original test output logic
        total_samples = sum(len(output['pid']) for output in self.test_outputs)
        
        all_pids = []
        all_vids = []
        
        n_cls = len(self.cls_label_names)
        n_reg = len(self.reg_label_names)
        
        # Initialize storage for predictions
        cls_probs = [[] for _ in range(n_cls)]
        cls_labels = [[] for _ in range(n_cls)]
        
        reg_preds = [[] for _ in range(n_reg)]
        reg_masks = [[] for _ in range(n_reg)]
        reg_labels = [[] for _ in range(n_reg)]
        
        for output in tqdm(self.test_outputs):
            all_pids.extend(output['pid'])
            all_vids.extend(output['vid'])

            # Process each sample in the batch
            for b_idx in range(len(output['pid'])):
                for i in range(n_cls):
                    if i < output['cls_preds'].shape[1]:
                        # Take mean over sequence length for each sample
                        cls_probs[i].append(output['cls_preds'][b_idx, i, :, 1].mean())
                        cls_labels[i].append(output['cls_label'][b_idx, i, :].mean())

                for i in range(n_reg):
                    if i < output['reg_preds'].shape[1]:
                        # Take mean over sequence length for each sample  
                        reg_preds[i].append(output['reg_preds'][b_idx, i, :].mean())
                        reg_masks[i].append(output['reg_mask'][b_idx, i, :].any())
                        reg_labels[i].append(output['reg_label'][b_idx, i, :].mean())
        
        # Create DataFrame following original format
        data_dict = {
            'pid': all_pids,
            'vid': all_vids,
        }

        for i, col in enumerate(self.cls_label_names):
            if i < len(cls_probs):
                data_dict[f"{col}_prob_1"] = cls_probs[i]
                data_dict[col] = cls_labels[i]
        
        for i, col in enumerate(self.reg_label_names):
            if i < len(reg_preds):
                data_dict[f"{col}_prob_0"] = reg_preds[i]
                data_dict[f"{col}_mask"] = reg_masks[i]
                data_dict[col] = reg_labels[i]
        
        df = pd.DataFrame(data_dict)
        output_path = output_dir / f'test_pred.{self.global_rank}.parquet'
        df.to_parquet(output_path)
        self.test_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()
