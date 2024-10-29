# data:
#   batch_size: 1
#   dataset_root: data/
#   dataset_name: mvtec_ad
#   category_name: bottle
#   multi_category: false
#   transform_type: default
#   num_workers: 1
#   prefetch_factor: 1
#   pin_mem: true
# logging:
#   folder: logs/mim/eval/
#   write_tag: debug
# mask:
#   mask_strategy: random
#   mask_ratio: 0.5
#   num_masks: 10
#   aspect_ratio:  # for block mask
#   - 0.75
#   - 1.5  
#   mask_scale:  # for block mask
#   - 0.1
#   - 0.4
# model:
#   mim_model_type: mim_base
#   backbone_name: vgg19
#   backbone_indices:
#   - 3
#   - 8
#   - 17
#   - 26
#   feature_res: 64
#   in_resolution: 224
#   use_bfloat16: false
#   resume_path: null
#   patch_size: 2

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import yaml
import random
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from einops import rearrange

from models.build_model import build_mim_model
from datasets import build_dataset, build_transforms, EvalDataLoader
from mask import RandomMaskCollator
from models.vision_transformer import indices_to_mask
from util import AverageMeter, pidx_to_pmask


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

def gaussian_kernel(size: int, sigma: float):
    """Generate a 2D Gaussian kernel
    Args:
        size (int): size of the kernel
        sigma (float): standard deviation of the Gaussian distribution
    Returns:
        kernel (tensor): 2D Gaussian kernel
    """
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = torch.exp(-x**2 / (2 * sigma**2))
    x = x / x.sum()  # normalize the kernel
    kernel_2d = x[:, None] * x[None, :]
    return kernel_2d

def calculate_mask_coverage(mask_batch, h, w):
    """Calculate mask coverage. 

    Args:
        mask_batch (tensor): Indices of masked patches. Shape: (B, L)
        h (int): Height of the feature map
        w (int): Width of the feature map
    Returns:
        mask_coverage (float): Mask coverage
    """
    mask = pidx_to_pmask(mask_batch, h, w)  # (B, h, w)
    mask_or = torch.any(mask, dim=0).float()  # (h, w)
    mask_coverage = torch.mean(mask_or)  # scalar
    return mask_coverage  

def gaussian_filter(err_map, sigma=1.4, ksize=7):
    """Apply Gaussian filter to the error map

    Args:
        err_map (tensor): Error map. Shape: (B, H, W)
        sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 1.4.
        ksize (int, optional): Kernel size of the Gaussian filter. Defaults to 7.
    Returns:
        err_map (tensor): Error map after applying Gaussian filter, Shape: (B, H, W)
    """
    err_map = err_map.detach().cpu()
    kernel = gaussian_kernel(ksize, sigma) 
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(err_map.device)  # (1, 1, ksize, ksize)
    padding = ksize // 2
    err_map = F.pad(err_map, (padding, padding, padding, padding), mode='reflect')
    err_map = F.conv2d(err_map.unsqueeze(1), kernel, padding=0).squeeze(1)
    return err_map

def evaluate(args):
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    
    with open(args.config, 'r') as f:   
        config = yaml.safe_load(f)
    
    # Set random seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
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
    
    # logging settings
    log_folder = config['logging']['folder']
    write_tag = config['logging']['write_tag']
    save_heatmap = config['logging']['save_heatmap']
    
    # mask settings
    mask_strategy = config['mask']['mask_strategy']
    mask_ratio = config['mask']['mask_ratio']
    num_masks = config['mask']['num_masks']
    aspect_ratio = config['mask']['aspect_ratio']
    mask_scale = config['mask']['mask_scale']
    
    # model settings
    mim_model_type = config['model']['mim_model_type']
    backbone_name = config['model']['backbone_name']
    backbone_indices = config['model']['backbone_indices']
    feature_res = config['model']['feature_res']
    in_resolution = config['model']['in_resolution']
    use_bfloat16 = config['model']['use_bfloat16']
    resume_path = config['model']['resume_path']
    patch_size = config['model']['patch_size']
    
    if resume_path is None:
        raise ValueError("Please provide the path to the weights for evaluation")
    
    # Build model
    model = build_mim_model(mim_model_type)(**config['model'])
    model.load_state_dict(torch.load(resume_path, weights_only=True))
    model.eval()    
    model.to(device)
    logger.info(f"Model loaded from {resume_path}")
    
    # Build dataset
    dataset_path = os.path.join(dataset_root, dataset_name)
    transform = build_transforms(img_size=in_resolution, transform_type=transform_type)
    test_dataset = build_dataset(
        in_resolution, 'test', -1, dataset_path, category_name, transform, multi_category=multi_category
    )
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Category: {category_name}")
    logger.info(f"Number of samples: {len(test_dataset)}")
    
    # Create mask collator
    if mask_strategy == 'random':
        mask_collator = RandomMaskCollator(ratio=mask_ratio, input_size=feature_res, patch_size=patch_size)
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    # Create dataloader
    test_dataloader = EvalDataLoader(
        test_dataset, num_masks, collate_fn=mask_collator
    )
    
    # Evaluation
    logger.info(f"Evaluation started")
    loss_meter = AverageMeter()
    mask_coverage_meter = AverageMeter()
    
    results = {
        "err_maps": [],
        "filenames": [],
        "cls_names": [],
        "labels": [],
        "anom_types": []
    } 
    for i, (batch, mask_indices) in enumerate(test_dataloader):
        images = batch["samples"].to(device)
        results["labels"].append(batch["labels"][0].item())
        results["anom_types"].append(batch["anom_type"][0])
        results["filenames"].append(batch["filenames"][0])
        results["cls_names"].append(batch["clsnames"][0])

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
            
            if config['mask'].get('gaussian_filter', False):
                err_map = gaussian_filter(err_map)
            
            err_map = torch.max(err_map, dim=0)[0]  # (h, w)
            results["err_maps"].append(err_map)
            
            loss_meter.update(loss.item(), images.size(0))
        
        if i % 10 == 0:
            logger.info(f"Iter: {i}/{len(test_dataloader)}")
    logger.info(f"Loss: {loss_meter.avg:.4f}")
    logger.info(f"Mask coverage: {mask_coverage_meter.avg:.4f}")
    
    # Calculate AUC score
    img_level_scores = np.array([torch.max(err_map).cpu().item() for err_map in results["err_maps"]])
    labels = results["labels"]  # 0: good, 1: anomaly
    auc_score = roc_auc_score(labels, img_level_scores)
    logger.info(f"AUC score: {auc_score:.4f}")
    
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
    
    # Save heatmaps
    if save_heatmap:
        save_dir = os.path.join(log_folder, write_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        global_min = np.min([torch.min(err_map).cpu().item() for err_map in results["err_maps"]])
        global_max = np.max([torch.max(err_map).cpu().item() for err_map in results["err_maps"]])
        for i, (err_map, filename) in tqdm(enumerate(zip(results["err_maps"], results["filenames"])), total=len(results["err_maps"])):
            save_path = os.path.join(save_dir, f"{filename}")
            err_map = err_map.cpu().numpy()  # (h, w)
            err_map = (err_map - global_min) / (global_max - global_min + 1e-6) 
            
            parent_dir = os.path.dirname(save_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            plt.imsave(save_path, err_map, cmap='hot')
    
    logger.info(f"Evaluation completed")

if __name__ == "__main__":            
    args = parse_args()
    evaluate(args)
    
