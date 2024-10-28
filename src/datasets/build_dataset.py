from .mvtec_ad import MVTecAD, AD_CLASSES
from .mvtec_loco import MVTecLOCO, LOCO_CLASSES

import torch
from torchvision import transforms
from torch.utils.data import Subset

from PIL import Image

import random
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from torchvision import transforms

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

class ImageNetTransforms():
    def __init__(self, input_res: int):
        
        self.mean = torch.Tensor(IMNET_MEAN).view(1, 3, 1, 1)
        self.std = torch.Tensor(IMNET_STD).view(1, 3, 1, 1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((input_res, input_res)),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD)
        ])
    
    def __call__(self, img: Image) -> torch.Tensor:
        return self.img_transform(img)
    
    def inverse_affine(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.std.device)
        return img * self.std + self.mean

def build_dataset(img_size, split, num_normal_samples, data_root, category_name, transform, eval=False, multi_category=False, **kwargs):    
    if eval:
        num_normal_samples = -1
    if 'mvtec_ad' in data_root:
        if category_name not in AD_CLASSES:
            raise ValueError(f"Invalid class_name: {category_name}")
        dataset = MVTecAD(data_root, category_name, img_size, split, custom_transforms=transform)
        if num_normal_samples > 0:
            dataset = Subset(dataset, random.sample(range(len(dataset)), num_normal_samples))
    elif 'mvtec_loco' in data_root:  
        if category_name not in LOCO_CLASSES:
            raise ValueError(f"Invalid class_name: {category_name}")
        dataset = MVTecLOCO(data_root, category_name, img_size, split, custom_transforms=transform)
        if num_normal_samples > 0:
            dataset = Subset(dataset, random.sample(range(len(dataset)), num_normal_samples))
    else:
        raise ValueError(f"Invalid dataset: {data_root}")
    return dataset

def build_transforms(img_size, transform_type='default', **kwargs):
    default_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if transform_type == 'default':
        return default_transform
    else:
        raise ValueError(f"Invalid transform: {transform_type}")


    
    