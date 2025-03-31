from collections import defaultdict
from functools import reduce
from pathlib import Path
from einops import rearrange
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
from ehrformer import EHRVAE2D
from Utils import str_hash


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
        
        self.model = EHRVAE2D(config)

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

    def training_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        
        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  
        time_index = data['time_index']  
        valid_mask = data['valid_mask']  
        reg_label = label['float_feats']  
        cls_label = label['cat_feats']  
        reg_mask = label['float_valid_mask']  

        y_cls, y_reg, mu_z, std_z = self.model(cat_feats, float_feats, time_index, valid_mask)
        y_reg_loss = y_reg.reshape(-1)
        reg_label_loss = reg_label.reshape(-1)
        reg_mask_loss = reg_mask.reshape(-1)
        reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
        reg_loss = (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))

        y_cls_loss = y_cls.reshape(-1, 2)
        cls_label_loss = cls_label.reshape(-1)
        cls_loss = F.cross_entropy(y_cls_loss, cls_label_loss)

        temp = 1 + std_z - mu_z.pow(2) - std_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

        loss = cls_loss + reg_loss + loss_kld
        self.log("train_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds = defaultdict(list)
        self.targets = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        
        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  
        float_feats = data['float_feats']  
        time_index = data['time_index']  
        valid_mask = data['valid_mask']  
        reg_label = label['float_feats']  
        cls_label = label['cat_feats']  
        cls_mask = label['cat_valid_mask']  
        reg_mask = label['float_valid_mask']  

        y_cls, y_reg, mu_z, std_z = self.model(cat_feats, float_feats, time_index, valid_mask)
        y_reg_loss = y_reg.reshape(-1)
        reg_label_loss = reg_label.reshape(-1)
        reg_mask_loss = reg_mask.reshape(-1)
        reg_losses = F.mse_loss(y_reg_loss, reg_label_loss, reduction='none')
        reg_loss = (reg_losses * reg_mask_loss).sum() / (reg_mask_loss.sum().clip(1))

        cls_label_flat = cls_label.reshape(-1)
        cls_mask_flat = cls_mask.reshape(-1)
        y_cls_flat = y_cls.reshape(-1, 2)
        cls_losses = F.cross_entropy(y_cls_flat, cls_label_flat, reduction='none')
        cls_loss = (cls_losses * cls_mask_flat).sum() / (cls_mask_flat.sum().clip(1))

        
        
        

        temp = 1 + std_z - mu_z.pow(2) - std_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

        loss = cls_loss + reg_loss + loss_kld
        B = cat_feats.shape[0]
        cls_label_perf = rearrange(cls_label, 'b n l -> (b l) n', b=B)
        cls_mask_perf = rearrange(valid_mask, 'b l -> (b l) 1', b=B)
        reg_label_perf = rearrange(reg_label, 'b n l -> (b l) n', b=B)
        reg_mask_perf = rearrange(reg_mask, 'b n l -> (b l) n', b=B)
        tensor_to_gather = [
            cls_label_perf.contiguous(), cls_mask_perf.contiguous(),
            reg_label_perf.contiguous(), reg_mask_perf.contiguous(),
            y_cls.contiguous(), y_reg.contiguous()
        ]
        tensor_gathered = [x.cpu() for x in all_gather(tensor_to_gather)]
        cls_label_perf = tensor_gathered[0]
        cls_mask_perf = tensor_gathered[1]
        reg_label_perf = tensor_gathered[2]
        reg_mask_perf = tensor_gathered[3]
        
        y_cls = tensor_gathered[4]
        y_reg = tensor_gathered[5]

        for i in range(len(self.cls_label_names)):
            mask = cls_mask_perf[:, i]

            pp = y_cls[:, i, 1]
            pp = pp[mask]

            yy = cls_label_perf[:, i]
            yy = yy[mask]

            self.preds[f'cls_{i}'].append(pp)
            self.targets[f'cls_{i}'].append(yy)

        for i in range(len(self.reg_label_names)):
            mask = reg_mask_perf[:, i]

            pp = y_reg[:, i]
            pp = pp[mask]

            yy = reg_label_perf[:, i]
            yy = yy[mask]
            if len(pp) > 2:
                self.preds[f'reg_{i}'].append(pp)
                self.targets[f'reg_{i}'].append(yy)

        self.log("val_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
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
                self.log(f"val_{k}_auc", MF.auroc(pred, target, task='binary'), prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_f1", best_f1, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_precision", best_precision, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_recall", best_recall, prog_bar=False, rank_zero_only=True)
                mauc.append(MF.auroc(pred, target, task='binary'))
            else:
                self.log(f"val_{k}_auc", torch.nan, prog_bar=False, rank_zero_only=True)
        mauc = sum(mauc) / len(mauc)
        self.log(f"val_mauc", mauc, prog_bar=False, rank_zero_only=True)

        mf1 = sum(mf1) / len(mf1)
        self.log(f"val_mf1", mf1, prog_bar=False, rank_zero_only=True)

        mpcc = []
        mr2 = []
        for i, k in enumerate(self.reg_label_names):
            if len(self.preds[f'reg_{i}']) != 0:
                pred = dim_zero_cat(self.preds[f'reg_{i}']).double()
                target = dim_zero_cat(self.targets[f'reg_{i}']).double()
                mse = MF.mean_squared_error(pred, target)
                pcc = MF.pearson_corrcoef(pred, target)
                r2 = MF.r2_score(pred, target)
                self.log(f"val_{k}_mse", mse, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_pcc", pcc, prog_bar=False, rank_zero_only=True)
                self.log(f"val_{k}_r2", r2, prog_bar=False, rank_zero_only=True)
                if not torch.isnan(pcc).any() and pcc > 0:
                    mpcc.append(pcc)
                if not torch.isnan(r2).any() and r2 > 0:
                    mr2.append(r2)
        mpcc = sum(mpcc) / len(mpcc) if len(mpcc) != 0 else 0
        mr2 = sum(mr2) / len(mr2) if len(mr2) != 0 else 0
        self.log(f"val_mpcc", mpcc, prog_bar=False, rank_zero_only=True)
        self.log(f"val_mr2", mr2, prog_bar=False, rank_zero_only=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        data = batch['data']
        
        cat_feats = data['cat_feats']  
        float_feats = data['float_feats']  

        mu_z = self.model.forward_mu(cat_feats, float_feats)
        for i in range(len(batch['pid'])):
            output_name = str_hash(batch['pid'][i]) % (10**32)
            output_name = f'{output_name:032d}.npz'
            mu_z_i = mu_z[i, :, :].detach().cpu().numpy()
            np.savez_compressed(str(self.output_dir / output_name), mu_z_i)
