from logging import getLogger
from multiprocessing import Value

import torch
logger = getLogger()

class RandomMaskCollator(object):
    def __init__(
        self,
        ratio=0.75, # ratio of masked patches
        input_size=(224, 224),
        patch_size=16,
        mask_seed = None,
    ):
        super(RandomMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
        self.mask_seed = mask_seed
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes (for distributed training)
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def __call__(self, batch):
        '''
        Create random masks for each sample in the batch
        
        Ouptut:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        '''
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)  # Collates original batch
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()  # use the shared counter to generate seed
        g = torch.Generator()
        g.manual_seed(seed)
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - self.ratio))
        
        collated_masks = []
        for _ in range(B):
            m = torch.randperm(num_patches)
            collated_masks.append(m[num_keep:])
        collated_masks = torch.stack(collated_masks, dim=0)  # (B, M), M: num of masked patches
        return collated_batch_org, collated_masks

class BlockRandomMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        mask_ratio: float = 0.75,
        aspect_min: float = 0.75,
        aspect_max: float = 1.5,
        scale_min: float = 0.1,
        scale_max: float = 0.4,
        mask_seed = None,
    ):
        super(BlockRandomMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mask_seed = mask_seed
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self._itr_counter = Value('i', -1)

    def step(self): 
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def generate_block_mask(self, aspect_ratio, scale):
        """Generate block mask with random aspect ratio and scale
        Args:
            aspect_ratio: aspect ratio of the mask
            scale: scale of the mask
        Returns:
            mask: block mask, (M)
        """
        
        mask = torch.zeros(self.height, self.width)
        h = int(self.height * scale)
        w = int(self.width * scale / aspect_ratio)
        y = torch.randint(0, self.height - h + 1, (1,)).item()
        x = torch.randint(0, self.width - w + 1, (1,)).item()

        mask[y:y+h, x:x+w] = 1
        
        # (1, H, W) -> (M)
        mask = torch.nonzero(mask.view(-1)).squeeze(1)  # (M)
        return mask
    
    def restrict_mask_ratio(self, mask, num_remove, num_total):
        """Restrict the number of masked patches to num_keep
        Args:
            mask: block mask, (M)
            num_remove: number of patches to keep
            num_total: total number of patches
        Returns:
            mask: restricted mask, (M')
        """
        num_masked_patches = mask.size(0)
        
        if num_remove < num_masked_patches:
            mask = mask[:num_remove]
        elif num_remove > num_masked_patches:
            # fill mask indices with random patches
            num_fill = num_remove - num_masked_patches
            m = ~torch.isin(torch.arange(num_total), mask)
            unmasked_indices = torch.arange(num_total)[m]
            new_mask = unmasked_indices[torch.randperm(unmasked_indices.size(0))][:num_fill]
            mask = torch.cat([mask, new_mask], dim=0)
            
        return mask
    
    def __call__(self, batch):
        """Create random block mask for each sample in the batch
        Args:
            original batch
        Returns:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        """
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - self.mask_ratio))
        num_remove = num_patches - num_keep
        
        collated_masks = []
        for _ in range(B):
            aspect_ratio = torch.empty(1).uniform_(self.aspect_min, self.aspect_max, generator=g)
            scale = torch.empty(1).uniform_(self.scale_min, self.scale_max, generator=g)
            mask = self.generate_block_mask(aspect_ratio, scale)
            mask = self.restrict_mask_ratio(mask, num_remove, num_patches)
            collated_masks.append(mask)
        collated_masks = torch.stack(collated_masks, dim=0)
        return collated_batch_org, collated_masks

class CheckerBoardMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        min_divisor=1,
        max_divisor=4,
        mode = "random", # random or fixed
        mask_seed = None,
    ):
        super(CheckerBoardMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.min_divisor = min_divisor
        self.max_divisor = max_divisor
        self.mode = mode
        self.mask_seed = mask_seed
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self._itr_counter = Value('i', -1)
        
        self.num_patterns = 2 * (max_divisor - min_divisor + 1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def collate_all_masks(self):
        """Create all possible checkerboard masks for evaluation"""
        masks = []
        for div in range(self.min_divisor, self.max_divisor+1):
            div = 2 ** div
            mask = self.checkerboard_mask(div)  
            inv_mask = 1 - mask
            mask = torch.nonzero(mask.view(-1)).squeeze(1)
            inv_mask = torch.nonzero(inv_mask.view(-1)).squeeze(1)
            masks.append(mask)
            masks.append(inv_mask)
        masks = torch.stack(masks, dim=0)  # (2 * (max_divisor - min_divisor + 1), M)
        return masks

    def checkerboard_mask(self, divisor):
        tile_h = self.height // divisor
        tile_w = self.width // divisor
        y_indices = torch.arange(divisor).view(-1, 1).expand(divisor, divisor)
        x_indices = torch.arange(divisor).view(1, -1).expand(divisor, divisor)
        checkerboard_pattern = (x_indices + y_indices) % 2
        mask = checkerboard_pattern.repeat_interleave(tile_h, dim=0).repeat_interleave(tile_w, dim=1)
        return mask
    
    def generate_checkerboard_mask(self, divisor):
        """Generate checkerboard mask with random divisor
        Args:
            divisor: divisor of the mask
        Returns:
            mask: checkerboard mask, (M)
        """
        mask = self.checkerboard_mask(divisor)
        inv_mask = 1 - mask
        mask = torch.nonzero(mask.view(-1)).squeeze(1)
        inv_mask = torch.nonzero(inv_mask.view(-1)).squeeze(1)
        return mask, inv_mask
    
    def __call__(self, batch):
        """Create checkerboard mask for each sample in the batch
        Args:
            original batch
        Returns:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        """
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)
        
        if self.mode == "fixed":
            collated_masks = self.collate_all_masks()
            assert collated_masks.size(0) == B, "Number of masks should be equal to batch size"
            return collated_batch_org, collated_masks
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)
        
        collated_masks = []
        for _ in range(B):
            divisor = 2 ** torch.randint(self.min_divisor, self.max_divisor, (1,)).item() 
            mask, inv_mask = self.generate_checkerboard_mask(divisor)
            if torch.rand(1) > 0.5:
                collated_masks.append(mask)
            else:
                collated_masks.append(inv_mask)
        collated_masks = torch.stack(collated_masks, dim=0)
        return collated_batch_org, collated_masks

if __name__ == '__main__':
    # collator = RandomMaskCollator(ratio=0.75, input_size=(224, 224), patch_size=16)
    # collator = BlockRandomMaskCollator(input_size=(224, 224), patch_size=16, mask_ratio=0.75, aspect_min=0.75, aspect_max=1.5, scale_min=0.1, scale_max=0.4)
    collator = CheckerBoardMaskCollator(input_size=(64, 64), patch_size=4, min_divisor=1, max_divisor=4)
    batch = [torch.randn(3, 64, 64) for _ in range(64)]
    collated_batch_org, collated_masks = collator(batch)
    print(collated_batch_org.shape, collated_masks.shape)
    print(collated_masks)
    print(collated_masks[0].shape)
    for i in range(64):
        print(collated_masks[i].shape)