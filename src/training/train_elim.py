"""Example of config mim.yaml
data:
  batch_size: 8
  dataset_root: data/
  dataset_name: mvtec_ad
  category_name: bottle
  multi_category: false
  num_workers: 1
  prefetch_factor: 1
  pin_mem: true
logging:
  folder: logs/mim/train/
  write_tag: debug
mask:
  mask_strategy: random
  mask_ratio: 0.5
  aspect_ratio:  # for block mask
  - 0.75
  - 1.5  
  mask_scale:  # for block mask
  - 0.1
  - 0.4
  patch_size: 2
model:
  mim_model_type: mim_base
  backbone_name: vgg19
  backbone_indices:
  - 3
  - 8
  - 17
  - 26
  feature_res: 64
  patch_size: 2
  in_resolution: 224
  use_bfloat16: true
  resume_path: null
optimization:
  epochs: 5
  final_lr: 1.0e-06
  final_weight_decay: 0.4
  warmup_lr: 0.001
  start_lr: 0.0002
  warmup_epochs: 1
  start_weight_decay: 0.04
  grad_clip_norm: 0.0
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import yaml
import random
import argparse

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from torch.cuda import amp
import tensorboardX

from models.build_model import build_mim_model, build_predictor_model
from datasets import build_dataset, build_transforms
from util import AverageMeter, CosineAnealingLRSchedulerWithWarmup, CosineAnealingWDScheduler
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator
from models.vision_transformer import indices_to_mask

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='MIM [Training]')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def train(args):
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data settings
    batch_size = config['data']['batch_size']
    dataset_root = config['data']['dataset_root']
    dataset_name = config['data']['dataset_name']
    category_name = config['data']['category_name']
    multi_category = config['data']['multi_category']
    transform_type = config['data']['transform_type']
    num_workers = config['data']['num_workers']
    prefetch_factor = config['data']['prefetch_factor']
    pin_mem = config['data']['pin_mem']
    
    # logging
    log_folder = config['logging']['folder']
    write_tag = config['logging']['write_tag']
    ckpt_interval = config['logging']['ckpt_interval']
    
    # mask
    mask_strategy = config['mask']['mask_strategy']
    mask_ratio = config['mask']['mask_ratio']
    aspect_min, aspect_max = config['mask']['aspect_ratio']
    scale_min, scale_max = config['mask']['mask_scale']
    min_divisor, max_divisor = config['mask']['divisor']
    
    # model
    mim_model_type = config['model']['mim_model_type']
    eliminator_model_type = config['model']['elim_model_type']
    out_channels = config['model']['out_channels']
    backbone_name = config['model']['backbone_name']
    backbone_indices = config['model']['backbone_indices']
    feature_res = config['model']['feature_res']
    in_resolution = config['model']['in_resolution']
    use_bfloat16 = config['model']['use_bfloat16']
    resume_path = config['model']['resume_path']
    patch_size = config['model']['patch_size']
    
    # optimization
    epochs = config['optimization']['epochs']
    final_lr = config['optimization']['final_lr']
    final_weight_decay = config['optimization']['final_weight_decay']
    warmup_lr = config['optimization']['warmup_lr']
    start_lr = config['optimization']['start_lr']
    warmup_epochs = config['optimization']['warmup_epochs']
    start_weight_decay = config['optimization']['start_weight_decay']
    grad_clip_norm = config['optimization']['grad_clip_norm']
    
    # build model
    model = build_mim_model(mim_model_type)(**config['model'])
    model.to(device)
    assert resume_path is not None, "Resume path must be provided for Eliminator training"
    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path, weights_only=True))
        logger.info(f"Loaded model from {resume_path}")
    model.eval()
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    # build eliminator
    out_channels = patch_size * patch_size * out_channels
    eliminator = build_predictor_model(eliminator_model_type)(num_patches=model.num_patches, in_channels=model.enc_emb_size, out_channels=out_channels)
    eliminator.to(device)
    
    # build dataset
    dataset_path = os.path.join(dataset_root, dataset_name)
    transform = build_transforms(img_size=in_resolution, transform_type=transform_type)
    dataset = build_dataset(
        in_resolution, 'train', -1, dataset_path, category_name, transform, multi_category=multi_category
    )
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(dataset)}")
    
    log_folder = os.path.join(log_folder, write_tag)
    tb_logger = tensorboardX.SummaryWriter(log_folder)
    # save config to log
    tb_logger.add_text("config", yaml.dump(config), 0)
    
    # build data loader
    if mask_strategy == "random":
        mask_collator = RandomMaskCollator(
            ratio=mask_ratio, input_size=feature_res, patch_size=patch_size
        )
    elif mask_strategy == "block":
        mask_collator = BlockRandomMaskCollator(
            input_size=feature_res, patch_size=patch_size, aspect_min=aspect_min, aspect_max=aspect_max,
            scale_min=scale_min, scale_max=scale_max, mask_ratio=mask_ratio
        )
    elif mask_strategy == "checkerboard":
        mask_collator = CheckerBoardMaskCollator(
            input_size=feature_res, patch_size=patch_size, min_divisor=min_divisor, max_divisor=max_divisor,
        )
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_mem,
        collate_fn=mask_collator
    )
    total_iter = len(dataloader) * epochs
    logging.info(f"Total iterations: {total_iter/1000}K steps")
    
    # optimizer
    optimizer = optim.AdamW(
        eliminator.parameters(), lr=start_lr, weight_decay=start_weight_decay
    )
    lr_scheduler = CosineAnealingLRSchedulerWithWarmup(
        optimizer, warmup_epochs, epochs, warmup_lr, final_lr, start_lr
    )
    wd_scheduler = CosineAnealingWDScheduler(
        optimizer, epochs, start_weight_decay, final_weight_decay
    )
    
    log_interval = 10
    eliminator.train()
    logger.info(f"Start training for {epochs} epochs")
    for i in range(epochs):
        mim_loss_meter = AverageMeter()
        elim_loss_meter = AverageMeter()
        
        for j, (batch, mask_indices) in enumerate(dataloader):
            x = batch["samples"].to(device)
            mask, _ = indices_to_mask(mask_indices.to(device), model.num_patches)  
            # mask: 
            
            # model forward pass 
            with torch.no_grad():
                if use_bfloat16:
                    with torch.amp.autocast("cuda"):
                        outputs = model(x, mask)
                else:
                    outputs = model(x, mask)
                loss = outputs["loss"]
                target_features = outputs["target_features"]  # (B, c, h, w)
                pred_features = outputs["pred_features"]  # (B, c, h, w)
                enc_features = outputs["enc_features"].detach()  # (B, V, d)
                err_map = torch.mean((target_features - pred_features)**2, dim=1)  # (B, h, w)
                err_map = err_map.detach()
            mim_loss_meter.update(loss.item(), x.size(0))
            
            # Predict error map using eliminator
            elim_outputs, _ = eliminator(enc_features, mask, return_all_patches=True)  # (B, V+M, 1*p*p)
            # Compute prediction loss
            elim_outputs = rearrange(elim_outputs, 'b (h w) (p1 p2) -> b (h p1) (w p2)', p1=patch_size, p2=patch_size, \
                h=feature_res//patch_size, w=feature_res//patch_size)  # (B, h, w)
            elim_loss = nn.functional.mse_loss(elim_outputs, err_map, reduction='none')  # (B, h, w)
            
            reshaped_mask = rearrange(mask, 'b (h w) -> b h w', h=feature_res//patch_size, w=feature_res//patch_size)   # (B, h//patch_size, w//patch_size)
            # (B, h//patch_size, w//patch_size) -> (B, h, w)
            reshaped_mask = torch.repeat_interleave(reshaped_mask, patch_size, dim=1)
            reshaped_mask = torch.repeat_interleave(reshaped_mask, patch_size, dim=2)
            reshaped_mask = reshaped_mask.half() if use_bfloat16 else reshaped_mask.float()
            
            elim_loss = torch.sum(elim_loss * reshaped_mask) / torch.sum(reshaped_mask) 
            elim_loss_meter.update(elim_loss.item(), x.size(0))  
            
            # backward pass
            elim_loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(eliminator.parameters(), grad_clip_norm)
            optimizer.step()
            
            assert not torch.isnan(elim_loss).any(), "Loss is NaN"
            
            if j % log_interval == 0:
                logger.info(f"Epoch: {i+1}/{epochs}, Iter: {j}/{len(dataloader)}, MIM Loss: {mim_loss_meter.avg:.4f}, Elim Loss: {elim_loss_meter.avg:.4f}")
                tb_logger.add_scalar("train/mim_loss", mim_loss_meter.avg, i*len(dataloader)+j)
                tb_logger.add_scalar("train/elim_loss", elim_loss_meter.avg, i*len(dataloader)+j)
                
        # update lr and wd
        lr_scheduler.step()
        wd_scheduler.step()
        tb_logger.add_scalar("train/lr", lr_scheduler.get_lr(), i)
        tb_logger.add_scalar("train/wd", wd_scheduler.get_wd(), i)
        
        logger.info(f"Epoch: {i+1}/{epochs}, MIM Loss: {mim_loss_meter.avg:.4f}, Elim Loss: {elim_loss_meter.avg:.4f}")
        
        if (i+1) % ckpt_interval == 0:
            save_path = os.path.join(log_folder, f"{write_tag}_epoch_{i+1}.pth")
            torch.save(eliminator.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
    
    # save model
    save_path = os.path.join(log_folder, f"{write_tag}.pth")
    torch.save(eliminator.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    
    logger.info(f"To visualize training logs, run: tensorboard --logdir {log_folder}")
    logger.info(f"Training finished")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    
            
            
        
            
    
    
    