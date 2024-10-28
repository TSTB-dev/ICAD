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
    ):
        super(RandomMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
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
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - self.ratio))
        
        collated_masks = []
        for _ in range(B):
            m = torch.randperm(num_patches)
            collated_masks.append(m[num_keep:])
        collated_masks = torch.stack(collated_masks, dim=0)  # (B, M), M: num of masked patches
        return collated_batch_org, collated_masks
    
if __name__ == '__main__':
    collator = RandomMaskCollator(ratio=0.75, input_size=(224, 224), patch_size=16)
    batch = [torch.randn(3, 224, 224) for _ in range(4)]
    collated_batch_org, collated_masks = collator(batch)
    print(collated_batch_org.shape, collated_masks.shape)
    print(collated_masks)
    print(collated_masks[0].shape)