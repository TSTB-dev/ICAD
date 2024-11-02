import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

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

def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def pidx_to_ppos(patch_idx, h, w):
    """
    Convert patch index to patch position
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.zeros((patch_idx.shape[0], patch_idx.shape[1], 2), dtype=torch.int)
    for i, idx in enumerate(patch_idx):
        patch_pos[i] = torch.stack([idx % w, idx // w], dim=1)
    return patch_pos

def ppos_to_pidx(patch_pos, h, w):
    """
    Convert patch position to patch index
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_idx = patch_pos[:, :, 1] * w + patch_pos[:, :, 0]
    return patch_idx

def ppos_to_pmask(patch_pos, h, w):
    """
    Convert patch position to binary mask of patches
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    mask = torch.zeros((patch_pos.shape[0], 1, h, w), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y, x] = 1
    return mask

def pmask_to_ppos(mask):
    """
    Convert binary mask of patches to patch position
    Args:
        mask: (N, H, W), binary mask
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    return patch_pos

def pidx_to_pmask(patch_idx, h, w):
    """
    Convert patch index to binary mask of patches
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_pmask(patch_pos, h, w)
    return mask

def pmask_to_pidx(mask, h, w):
    """
    Convert binary mask of patches to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = pmask_to_ppos(mask)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx

def ppos_to_imask(patch_pos, h, w, patch_size):
    """Convert patch position to binary mask of image
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask (float32)
    """
    H, W = h * patch_size, w * patch_size
    mask = torch.ones((patch_pos.shape[0], 1, H, W), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = 0
    return mask

def imask_to_ppos(mask, patch_size):
    """Convert binary mask of image to patch position
    Args:
        mask: (N, H, W), binary mask
        patch_size: size of patch
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    patch_pos = patch_pos[:, :, [1, 2]]
    patch_pos = patch_pos - patch_pos % patch_size
    return patch_pos

def pidx_to_imask(patch_idx, h, w, patch_size):
    """Convert patch index to binary mask of image
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_imask(patch_pos, h, w, patch_size)
    return mask

def imask_to_pidx(mask, h, w, patch_size):
    """Convert binary mask of image to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = imask_to_ppos(mask, patch_size)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx
