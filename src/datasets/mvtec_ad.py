import os
from pathlib import Path
from typing import *

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

AD_CLASSES = [
    "metal_nut",
    "tile",
    "screw",
    "zipper",
    "grid",
    "pill",
    "capsule",
    "transistor",
    "toothbrush",
    "cable",
    "carpet",
    "wood",
    "bottle",
    "leather",
    "hazelnut"
]

class MVTecAD(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        custom_transforms: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False
    ):
        """Dataset for MVTec AD.
        Args:
            data_root: Root directory of MVTecAD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'hazelnut'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = custom_transforms
        self.is_mask = is_mask
        self.cls_label = cls_label
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'
        
        # # load files from the dataset
        self.img_files = self.get_files()
        if self.split == 'test':
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res),
                    transforms.ToTensor(),
                ]
            )

            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
    
    def __getitem__(self, index):
        inputs = {}
        
        img_file = self.img_files[index]
        cls_label = str(img_file).split("/")[-4]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsnames"] = cls_label
        inputs["filenames"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["samples"] = sample
            return inputs
        else:
            if not self.is_mask:
                inputs["samples"] = sample
                inputs["labels"] = self.labels[index]
                if "good" in str(img_file):
                    inputs["anom_type"] = "good"
                else:
                    inputs["anom_type"] = str(img_file).split("/")[-2]
                return inputs
            else:
                raise NotImplementedError
    
    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            anomalous_img_files = []
            anomalous_dirs = sorted(Path(os.path.join(self.data_root, self.category, 'test')).glob('*'))
            for anomalous_dir in anomalous_dirs:
                if "good" in str(anomalous_dir):
                    continue
                anomalous_img_files += sorted(anomalous_dir.glob('*.png'))
            anomalous_img_files = sorted(anomalous_img_files)
            
            files = normal_img_files + anomalous_img_files
        return files
    