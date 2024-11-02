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
from sklearn.metrics import roc_auc_score
import tensorboardX

from models.build_model import build_mim_model
from datasets import build_dataset, build_transforms, EvalDataLoader
from util import AverageMeter, CosineAnealingLRSchedulerWithWarmup, CosineAnealingWDScheduler, \
    calculate_mask_coverage, gaussian_filter
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
    eval_interval = config['logging']['eval_interval']
    
    # mask
    mask_strategy = config['mask']['mask_strategy']
    mask_ratio = config['mask']['mask_ratio']
    aspect_min, aspect_max = config['mask']['aspect_ratio']
    scale_min, scale_max = config['mask']['mask_scale']
    min_divisor, max_divisor = config['mask']['divisor']
    
    # model
    mim_model_type = config['model']['mim_model_type']
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
    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path, weights_only=True))
        logger.info(f"Loaded model from {resume_path}")
    
    # build dataset
    dataset_path = os.path.join(dataset_root, dataset_name)
    transform = build_transforms(img_size=in_resolution, transform_type=transform_type)
    train_dataset = build_dataset(
        in_resolution, 'train', -1, dataset_path, category_name, transform, multi_category=multi_category
    )
    test_dataset = build_dataset(
        in_resolution, 'test', -1, dataset_path, category_name, transform, multi_category=multi_category
    )
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(train_dataset)}")
    
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
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_mem,
        collate_fn=mask_collator
    )
    total_iter = len(train_dataloader) * epochs
    
    # optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=start_lr, weight_decay=start_weight_decay
    )
    lr_scheduler = CosineAnealingLRSchedulerWithWarmup(
        optimizer, warmup_epochs, epochs, warmup_lr, final_lr, start_lr
    )
    wd_scheduler = CosineAnealingWDScheduler(
        optimizer, epochs, start_weight_decay, final_weight_decay
    )
    
    log_interval = 10
    model.train()
    logger.info(f"Start training for {epochs} epochs")
    for i in range(epochs):
        loss_meter = AverageMeter()
        for j, (batch, mask_indices) in enumerate(train_dataloader):
            x = batch["samples"].to(device)
            mask, _ = indices_to_mask(mask_indices.to(device), model.num_patches)
            
            # forward pass
            if use_bfloat16:
                with torch.cuda.amp.autocast():
                    outputs = model(x, mask)
                    loss = outputs["loss"]
            else:
                outputs = model(x, mask)
                loss = outputs["loss"]
            loss_meter.update(loss.item(), x.size(0))
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            
            assert not torch.isnan(loss).any(), "Loss is NaN"
            
            if j % log_interval == 0:
                logger.info(f"Epoch: {i+1}/{epochs}, Iter: {j}/{len(train_dataloader)}, Loss: {loss_meter.avg:.4f}")
                tb_logger.add_scalar("train/loss", loss_meter.avg, i*len(train_dataloader)+j)
                
        # update lr and wd
        lr_scheduler.step()
        wd_scheduler.step()
        tb_logger.add_scalar("train/lr", lr_scheduler.get_lr(), i)
        tb_logger.add_scalar("train/wd", wd_scheduler.get_wd(), i)
        
        logger.info(f"Epoch: {i+1}/{epochs}, Loss: {loss_meter.avg:.4f}")
        
        if (i+1) % ckpt_interval == 0:
            save_path = os.path.join(log_folder, f"{write_tag}_epoch_{i+1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
            
        if (i+1) % eval_interval == 0:
            evaluate(
                test_dataset, mask_collator, model, device, logger, tb_logger, i+1
            )
    
    # save model
    save_path = os.path.join(log_folder, f"{write_tag}.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    
    logger.info(f"To visualize training logs, run: tensorboard --logdir {log_folder}")
    logger.info(f"Training finished")

def evaluate(
    test_dataset,
    mask_collator,
    model, 
    device, 
    logger, 
    tb_logger, 
    epoch, 
    num_masks=10,
    apply_gaussian_filter=True,
):
    logger.info(f"Evaluating model")
    test_dataloader = EvalDataLoader(
        test_dataset, num_repeat=num_masks, collate_fn=mask_collator
    )
    loss_meter = AverageMeter()
    mask_coverage_meter = AverageMeter()
    
    results = {
        "err_maps": [],
        "filenames": [],
        "cls_names": [],
        "labels": [],
        "anom_types": []
    } 
    model.eval()
    for i, (batch, mask_indices) in enumerate(test_dataloader):
        images = batch["samples"].to(device)
        results["labels"].append(batch["labels"][0].item())
        results["anom_types"].append(batch["anom_type"][0])
        results["filenames"].append(batch["filenames"][0])
        results["cls_names"].append(batch["clsnames"][0])

        feature_res = model.feature_res
        patch_size = model.patch_size
        
        mask_coverage = calculate_mask_coverage(mask_indices, feature_res//patch_size, feature_res//patch_size)
        mask_coverage_meter.update(mask_coverage, 1)
        
        mask, _ = indices_to_mask(mask_indices, model.num_patches)
        mask = mask.to(device)

        with torch.no_grad():
            outputs = model(images, mask)
            loss = outputs["loss"]
            target_features = outputs["target_features"]  # (B, c, h, w)
            pred_features = outputs["pred_features"]  # (B, c, h, w)
            err_map = torch.mean((target_features - pred_features)**2, dim=1)  # (B, h, w)
            reshaped_mask = rearrange(mask, 'b (h w) -> b h w', h=feature_res//patch_size, w=feature_res//patch_size)   # (B, h//patch_size, w//patch_size)
            # (B, h//patch_size, w//patch_size) -> (B, h, w)
            reshaped_mask = torch.repeat_interleave(reshaped_mask, patch_size, dim=1)
            reshaped_mask = torch.repeat_interleave(reshaped_mask, patch_size, dim=2)
            reshaped_mask = reshaped_mask.float()  

            # Apply mask
            err_map = err_map * reshaped_mask  # (B, h, w)
            if apply_gaussian_filter:
                err_map = gaussian_filter(err_map)
            
            err_map = torch.max(err_map, dim=0)[0]  # (h, w)
            results["err_maps"].append(err_map)
            loss_meter.update(loss.item(), images.size(0))

            if i % 10 == 0:
                logger.info(f"Iter: {i}/{len(test_dataloader)}")
    logger.info(f"Loss: {loss_meter.avg:.4f}")
    tb_logger.add_scalar("eval/loss", loss_meter.avg, epoch)
    logger.info(f"Mask coverage: {mask_coverage_meter.avg:.4f}")

    # Calculate AUC score
    img_level_scores = np.array([torch.max(err_map).cpu().item() for err_map in results["err_maps"]])
    labels = results["labels"]  # 0: good, 1: anomaly
    auc_score = roc_auc_score(labels, img_level_scores)
    logger.info(f"AUC score: {auc_score:.4f}")
    tb_logger.add_scalar("eval/auc_all", auc_score, epoch)

    # Calculate the auROC score for each class
    unique_anom_types = list(sorted(set(results["anom_types"])))
    normal_indices = [i for i, x in enumerate(results["anom_types"]) if x == "good"]
    for anom_type in unique_anom_types:
        if anom_type == "good":
            continue
        anom_indices = [i for i, x in enumerate(results["anom_types"]) if x == anom_type]
        normal_scores = img_level_scores[normal_indices]
        anom_scores = img_level_scores[anom_indices]
        scores = np.concatenate([normal_scores, anom_scores])
        labels = [0] * len(normal_scores) + [1] * len(anom_scores)
        auc = roc_auc_score(labels, scores)
        logger.info(f'auROC: {auc:.4f} on {anom_type}')
        tb_logger.add_scalar(f"eval/auroc_{anom_type}", auc, epoch)
    
    logger.info(f"Eval finished")

if __name__ == "__main__":
    args = parse_args()
    train(args)
    
            
            
        
            
    
    
    