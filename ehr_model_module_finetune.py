"""EHRFormer finetuning module for downstream tasks."""
from collections import defaultdict
from functools import reduce
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, roc_auc_score
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import functional as F
from torchmetrics import *
from torchmetrics.utilities.data import dim_zero_cat
import torch
import torchmetrics.functional as MF
from tqdm import tqdm
from Utils import save_parquet
from ddp_Utils import *
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from ehrformer import EHRFormer
from sklearn.preprocessing import label_binarize

def group_by_M_dimension(A_list, B_list, C_list):
    groups = {}
    for A, B, C in zip(A_list, B_list, C_list):
        M = A.shape[1]

        if M not in groups:
            groups[M] = ([], [], [])

        groups[M][0].append(A)
        groups[M][1].append(B)
        groups[M][2].append(C)

    return groups

def group_by_M_dimension(A_list, B_list, C_list):
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

def encode(nums):
    result = 0
    for num in nums:
        result = result * 41 + num
    return result

def batch_top5_encode(probs: np.ndarray) -> np.ndarray:
    top5_indices = np.argsort(probs, axis=1)[:, -5:]  
    
    
    encoded = np.zeros(len(probs), dtype=np.int32)
    for i, indices in enumerate(top5_indices):
        
        val = 0
        for idx in indices[::-1]:  
            val = val * 41 + idx
        encoded[i] = val
    
    return encoded


def decode(encoded):
    nums = []
    for _ in range(5):
        nums.append(encoded % 41)
        encoded //= 41
    return nums[::-1]


class EHRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.data_type = config.get('data_type', '')
        self.momentum = config.get('momentum', 0.9)
        self.wd = config.get('wd', 1e-6)
        self.lr = config.get('lr', 1e-3)
        self.n_nodes = config.get('n_nodes', 1)
        self.n_gpus = config.get('n_gpus', 1)
        if isinstance(self.n_gpus, list):
            self.n_gpus = len(self.n_gpus)
        self.n_epoch = config.get('n_epoch', None)
        self.n_cls = config.get('n_cls', [])
        self.cls_label_names = config.get('cls_label_names', [])
        self.reg_label_names = config.get('reg_label_names', [])
        self.model = EHRFormer(config)
        

        self.criterion = FocalLoss(alpha=0.25, gamma=2, reduction='none')
        self.output_dir = config.get('output_dir', None)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_folder = 'pred'

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

    def kl_divergence_loss(self, mu, logvar):
        """Calculate KL divergence loss for VAE."""
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_loss.mean()

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        valid_mask = data['valid_mask']
        cls_label = label['cls']['values']
        cls_mask = valid_mask.unsqueeze(1).expand_as(label['cls']['masks']) & label['cls']['masks']
        reg_label = label['reg']['values']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']

        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  
        valid_mask = data['valid_mask']  
        time_index = data['time_index']  

        # Get model outputs including VAE components
        y_cls, mu_z, std_z = self.model(cat_feats, float_feats, valid_mask, time_index)
        n_cls = cls_label.shape[1]
        n_reg = reg_label.shape[1]
        
        
        cls_preds = [y.reshape(-1, y.shape[-1]) for y in y_cls[:n_cls]]
        cls_labels = cls_label.reshape(cls_label.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_labels = [l.reshape(-1) for l in cls_labels]
        cls_masks = cls_mask.reshape(cls_mask.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_masks = [m.reshape(-1) for m in cls_masks]

        
        cls_groups = group_by_M_dimension(cls_preds, cls_labels, cls_masks)
        cls_loss = sum((self.criterion(pred, label) * mask).sum() / mask.sum().clip(1)
                    for pred, label, mask in cls_groups.values())
        
        reg_preds = torch.stack([y_cls[i + n_cls].reshape(-1) for i in range(n_reg)], dim=0)
        reg_labels = reg_label.reshape(reg_label.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_masks = reg_mask.reshape(reg_mask.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)

        reg_losses = F.mse_loss(reg_preds, reg_labels, reduction='none')
        reg_loss = (reg_losses * reg_masks).sum() / reg_masks.sum().clip(1)

        # Calculate KL divergence loss for VAE
        kl_loss = self.kl_divergence_loss(mu_z, std_z)
        
        # ELBO loss = Reconstruction Loss + KL Divergence Loss
        reconstruction_loss = cls_loss + reg_loss
        elbo_loss = reconstruction_loss + kl_loss
        
        self.log("train_loss", elbo_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("train_recon_loss", reconstruction_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("train_kl_loss", kl_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return elbo_loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)
    
    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        valid_mask = data['valid_mask']
        cls_label = label['cls']['values']
        cls_mask = valid_mask.unsqueeze(1).expand_as(label['cls']['masks']) & label['cls']['masks']
        reg_label = label['reg']['values']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']

        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  
        valid_mask = data['valid_mask']  
        time_index = data['time_index']  

        # Get model outputs including VAE components
        y_cls, mu_z, std_z = self.model(cat_feats, float_feats, valid_mask, time_index)
        n_cls = cls_label.shape[1]
        n_reg = reg_label.shape[1]
        
        
        cls_preds = [y.reshape(-1, y.shape[-1]) for y in y_cls[:n_cls]]
        cls_labels = cls_label.reshape(cls_label.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_labels = [l.reshape(-1) for l in cls_labels]
        cls_masks = cls_mask.reshape(cls_mask.shape[0], n_cls, -1).permute(1, 0, 2)
        cls_masks = [m.reshape(-1) for m in cls_masks]

        
        cls_groups = group_by_M_dimension(cls_preds, cls_labels, cls_masks)
        cls_loss = sum((self.criterion(pred, label) * mask).sum() / mask.sum().clip(1)
                    for pred, label, mask in cls_groups.values())
        
        reg_preds = torch.stack([y_cls[i + n_cls].reshape(-1) for i in range(n_reg)], dim=0)
        reg_labels = reg_label.reshape(reg_label.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)
        reg_masks = reg_mask.reshape(reg_mask.shape[0], n_reg, -1).permute(1, 0, 2).reshape(n_reg, -1)

        reg_losses = F.mse_loss(reg_preds, reg_labels, reduction='none')
        reg_loss = (reg_losses * reg_masks).sum() / reg_masks.sum().clip(1)

        # Calculate KL divergence loss for VAE
        kl_loss = self.kl_divergence_loss(mu_z, std_z)
        
        # ELBO loss = Reconstruction Loss + KL Divergence Loss
        reconstruction_loss = cls_loss + reg_loss
        elbo_loss = reconstruction_loss + kl_loss

        loss = elbo_loss
        tensor_to_gather = [
            cls_label.contiguous(), cls_mask.contiguous(), 
            reg_label.contiguous(), reg_mask.contiguous()
        ] + y_cls
        tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
        cls_label = tensor_gathered[0]
        cls_mask = tensor_gathered[1]
        reg_label = tensor_gathered[2]
        reg_mask = tensor_gathered[3]
        y_cls = tensor_gathered[4:]

        for i in range(len(self.cls_label_names)):
            mask = cls_mask[:, i, :]
            
            pp = y_cls[i]
            pp = pp[mask]
            pp = torch.softmax(pp, dim=-1)

            yy = cls_label[:, i, :]
            yy = yy[mask]
            if len(pp) > 0:
                self.preds[f'cls_{i}'].append(pp)
                self.targets[f'cls_{i}'].append(yy)

        for i in range(len(self.reg_label_names)):
            mask = reg_mask[:, i, :]

            pp = y_cls[i + n_cls].squeeze(-1)
            pp = pp[mask]

            yy = reg_label[:, i, :]
            yy = yy[mask]
            if len(pp) > 0:
                self.preds[f'reg_{i}'].append(pp)
                self.targets[f'reg_{i}'].append(yy)
        
        self.log("val_loss", elbo_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_recon_loss", reconstruction_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return elbo_loss

    def on_validation_epoch_end(self) -> None:
        mauc = []
        mf1 = []
        for i, k in enumerate(self.cls_label_names):
            if len(self.preds[f'cls_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'cls_{i}']).float()
                target = dim_zero_cat(self.targets[f'cls_{i}'])
                if pred.shape[-1] == 2:
                    pred = pred[..., 1]
                    precision, recall, _ = precision_recall_curve(target.numpy(), pred.numpy())
                    precision += 1e-10
                    recall += 1e-10
                    f1 = 2*recall*precision/(recall+precision)
                    best_precision = precision[np.argmax(f1)]
                    best_recall = recall[np.argmax(f1)]
                    best_f1 = np.max(2*recall*precision/(recall+precision))
                    mf1.append(best_f1)
                    self.log(f"val_{k}_auc", MF.auroc(pred, target, task='binary'), prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_f1", best_f1, prog_bar=False, rank_zero_only=True)
                    
                    
                    mauc.append(MF.auroc(pred, target, task='binary'))
                else:
                    num_classes = pred.shape[-1]
                    
                    
                    top1_acc = MF.accuracy(pred, target, task='multiclass', num_classes=num_classes, top_k=1)
                    top5_acc = MF.accuracy(pred, target, task='multiclass', num_classes=num_classes, top_k=5)
                    
                    
                    target_onehot = label_binarize(target.numpy(), classes=range(num_classes))
                    pred_probs = pred.numpy()
                    
                    micro_auc = roc_auc_score(target_onehot.ravel(), pred_probs.ravel())
                    self.log(f"val_{k}_top1_acc", top1_acc, prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_top5_acc", top5_acc, prog_bar=False, rank_zero_only=True)
                    self.log(f"val_{k}_micro_auc", micro_auc, prog_bar=False, rank_zero_only=True)
            else:
                self.log(f"val_{k}_auc", torch.nan, prog_bar=False, rank_zero_only=True)
        mauc = sum(mauc) / len(mauc)
        self.log(f"val_mauc", mauc, prog_bar=False, rank_zero_only=True)

        mf1 = sum(mf1) / len(mf1)
        self.log(f"val_mf1", mf1, prog_bar=False, rank_zero_only=True)

        mpcc = []
        for i, k in enumerate(self.reg_label_names):
            if len(self.preds[f'reg_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double()
                target = dim_zero_cat(self.targets[f'reg_{i}']).double()
                self.log(f"val_{k}_mse", MF.mean_squared_error(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_pcc", MF.pearson_corrcoef(pred, target), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", MF.r2_score(pred, target), prog_bar=False, rank_zero_only=True)
                mpcc.append(MF.pearson_corrcoef(pred, target))
        mpcc = sum(mpcc) / len(mpcc) if len(mpcc) != 0 else 0
        self.log(f"val_mpcc", mpcc, prog_bar=False, rank_zero_only=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        data = batch['data']
        pid = batch['pid']
        label = batch['label']
        valid_mask = data['valid_mask']
        cls_label = label['cls']['values']
        reg_label = label['reg']['values']

        cls_mask = valid_mask.unsqueeze(1).expand_as(label['cls']['masks']) & label['cls']['masks']
        reg_mask = valid_mask.unsqueeze(1).expand_as(label['reg']['masks']) & label['reg']['masks']

        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  
        valid_mask = data['valid_mask']  
        time_index = data['time_index']  

        # Get model outputs (ignore VAE components for testing)
        y_cls, mu_z, std_z = self.model(cat_feats, float_feats, valid_mask, time_index)
        self.test_outputs.append({
            'pid': pid.cpu(),
            'preds': y_cls, 
            'cls_label': cls_label, 'cls_mask': cls_mask, 
            'reg_label': reg_label, 'reg_mask': reg_mask
        })

    def on_test_epoch_end(self):
        output_dir = Path(self.output_dir) / self.pred_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        n_cls = len(self.cls_label_names)
        n_reg = len(self.reg_label_names)

        pred_cls = [[] for _ in range(n_cls)]
        cls_mask = [[] for _ in range(n_cls)]
        cls_label = [[] for _ in range(n_cls)]
        for i in range(len(self.test_outputs)):
            for j in range(n_cls):
                pred_cls[j].append(self.test_outputs[i]['preds'][j])
                cls_mask[j].append(self.test_outputs[i]['cls_mask'][:, j, :])
                cls_label[j].append(self.test_outputs[i]['cls_label'][:, j, :])
        pred_cls = [torch.softmax(torch.cat(x, dim=0), dim=-1).to('cpu') for x in pred_cls]
        cls_mask = [torch.cat(x, dim=0).to('cpu') for x in cls_mask]
        cls_label = [torch.cat(x, dim=0).to('cpu') for x in cls_label]

        pred_reg = [[] for _ in range(n_reg)]
        reg_mask = [[] for _ in range(n_reg)]
        reg_label = [[] for _ in range(n_reg)]
        for i in range(len(self.test_outputs)):
            for j in range(n_reg):
                pred_reg[j].append(self.test_outputs[i]['preds'][j+n_cls])
                reg_mask[j].append(self.test_outputs[i]['reg_mask'][:, j, :])
                reg_label[j].append(self.test_outputs[i]['reg_label'][:, j, :])
        pred_reg = [torch.cat(x, dim=0).to('cpu') for x in pred_reg]
        reg_mask = [torch.cat(x, dim=0).to('cpu') for x in reg_mask]
        reg_label = [torch.cat(x, dim=0).to('cpu') for x in reg_label]

        pid = torch.cat([x['pid'] for x in self.test_outputs]).numpy()

        n_sample = len(pid)
        df = pd.DataFrame(pid, columns=['pid'])
        for i, col in enumerate(tqdm(self.cls_label_names)):
            if pred_cls[i].shape[-1] == 2:
                tmp1 = pd.DataFrame([{f"{col}_prob_1": pred_cls[i][j, :, 1].numpy()} for j in range(n_sample)])
                tmp2 = pd.DataFrame([{f"{col}_mask": cls_mask[i][j, :].numpy()} for j in range(n_sample)])
                tmp3 = pd.DataFrame([{col: cls_label[i][j, :].numpy()} for j in range(n_sample)])
                df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
            else:
                tmp1 = pd.DataFrame([{f"{col}_probs": batch_top5_encode(pred_cls[i][j, :, :].numpy().astype(np.int32))} for j in range(n_sample)])
                tmp2 = pd.DataFrame([{f"{col}_mask": cls_mask[i][j, :].numpy()} for j in range(n_sample)])
                tmp3 = pd.DataFrame([{col: cls_label[i][j, :].numpy()} for j in range(n_sample)])
                df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
        for i, col in enumerate(self.reg_label_names):
            tmp1 = pd.DataFrame([{f"{col}_prob_0": pred_reg[i][j, :, 0].numpy()} for j in range(n_sample)])
            tmp2 = pd.DataFrame([{f"{col}_mask": reg_mask[i][j, :].numpy()} for j in range(n_sample)])
            tmp3 = pd.DataFrame([{col: reg_label[i][j, :].numpy()} for j in range(n_sample)])
            df = pd.concat([df, tmp1, tmp2, tmp3], axis=1)
        output_path = output_dir / f'test_pred.{self.global_rank}.parquet'
        df.to_parquet(output_path)
        
        self.test_outputs.clear()