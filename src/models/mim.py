from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from .vision_transformer import VisionTransformerEncoder, VisionTransformerPredictor
# from .backbone import BackboneModel

import sys
sys.path.append("src/models")

from vision_transformer import VisionTransformerEncoder, VisionTransformerPredictor
from backbone import BackboneModel

SUPPORTEED_MODELS = [
    "mim_tiny",
    "mim_small",
    "mim_base",
    "mim_large",
    "mim_huge",
    "mim_gigant"
]

class MaskedImageModelingModel(nn.Module):
    def __init__(
        self, 
        backbone_model="vgg19",
        backbone_indices=[3, 8, 17, 26],
        feature_res=64,
        in_resolution=224, 
        in_channels=3, 
        patch_size=2, 
        enc_emb_size=256,
        pred_emb_size=256,
        num_enc_blocks=4, 
        num_pred_blocks=3,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    ):
        super(MaskedImageModelingModel, self).__init__()
        self.backbone_model = backbone_model
        self.backbone_indices = backbone_indices
        self.feature_res = feature_res
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.enc_emb_size = enc_emb_size
        self.pred_emb_size = pred_emb_size
        self.num_enc_blocks = num_enc_blocks
        self.num_pred_blocks = num_pred_blocks
        self.num_enc_heads = num_enc_heads
        self.num_pred_heads = num_pred_heads
        self.num_enc_mlp_ratio = num_enc_mlp_ratio
        self.num_pred_mlp_ratio = num_pred_mlp_ratio
        self.layer_norm = layer_norm
        
        self.backbone = BackboneModel(backbone_model, backbone_indices, feature_res)
        
        self.feature_dim = self.backbone.feature_dim
        self.num_patches = (feature_res // patch_size) ** 2
        
        self.encoder = VisionTransformerEncoder(
            feature_res, self.feature_dim, patch_size, enc_emb_size, num_enc_blocks, num_enc_heads, num_enc_mlp_ratio, layer_norm
        )
        
        self.predictor = VisionTransformerPredictor(
            self.num_patches, enc_emb_size, self.feature_dim*patch_size*patch_size, pred_emb_size, num_pred_blocks, num_pred_heads, num_pred_mlp_ratio, layer_norm
        )
    
    def calculate_loss(self, pred_features, target_features, mask, loss_on_all_patches=False):
        """Calculate the loss between the predicted features and the encoded features. 
        Args:
            pred_features (torch.Tensor): Predicted features, shape (B, L, c*p*p)
            target_features (torch.Tensor): Target features, shape (B, L, c*p*p)
            mask (torch.Tensor): Mask tensor, shape (B, L)
            loss_on_all_patches (bool): If True, calculate the loss on all patches. Otherwise, calculate the loss on the masked patches only.
        Returns:
            torch.Tensor: Loss value
        """
        if loss_on_all_patches:
            loss = F.mse_loss(pred_features, target_features)
        else:
            mask = mask.to(torch.bool)
            loss = F.mse_loss(pred_features[mask], target_features[mask])
        return loss
            
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        loss_on_all_patches: bool = False,
    ):
        """Forward pass of MaskedImageModelingModel. 
        Args: 
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            mask (torch.Tensor): Mask tensor, shape (B, L)
            loss_on_all_patches (bool): If True, calculate the loss on all patches. Otherwise, calculate the loss on the masked patches only. 
        Returns:
            {
                "loss": torch.Tensor,
                "enc_features": torch.Tensor, shape (B, V, d),
                "enc_attention_maps": torch.Tensor, shape (B, h, V, V),
                "pred_features": torch.Tensor, shape (B, L, d),
                "pred_attention_maps": torch.Tensor, shape (B, h, L, L),
            }
        """
        # Extract features from the backbone model
        features = self.backbone(x)  # (B, c, h, w)
        
        target_features = features.clone().detach()
        target_features = rearrange(target_features, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size, \
            h=self.feature_res//self.patch_size, w=self.feature_res//self.patch_size)
        
        # Feed the features to the encoder
        enc_features, enc_attention_maps = self.encoder(features, mask)  # (B, V, d), (B, h, V, V)
        M = mask.shape[1] - enc_features.shape[1]
        
        # Masked patch prediction
        pred_features, pred_attention_maps = self.predictor(enc_features, mask, return_all_patches=True) # (B, L, d), (B, h, L, L)
        
        # loss calculation
        loss = self.calculate_loss(pred_features, target_features, mask, loss_on_all_patches)
        
        return {
            "loss": loss,
            "enc_features": enc_features,
            "enc_attention_maps": enc_attention_maps,
            "pred_features": pred_features,
            "pred_attention_maps": pred_attention_maps,
        }

def mim_tiny(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=256,
        pred_emb_size=256,
        num_enc_blocks=4, 
        num_pred_blocks=3,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )

def mim_small(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=512,
        pred_emb_size=512,
        num_enc_blocks=6, 
        num_pred_blocks=4,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )

def mim_base(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=768,
        pred_emb_size=768,
        num_enc_blocks=8, 
        num_pred_blocks=6,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )

def mim_large(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=1024,
        pred_emb_size=1024,
        num_enc_blocks=12, 
        num_pred_blocks=8,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )

def mim_huge(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=2048,
        pred_emb_size=2048,
        num_enc_blocks=16, 
        num_pred_blocks=12,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )

def mim_gigant(backbone_name="vgg19", backbone_indices=[3, 8, 17, 26], feature_res=64, patch_size=2, in_resolution=224, **kwargs):
    return MaskedImageModelingModel(
        backbone_model=backbone_name,
        backbone_indices=backbone_indices,
        feature_res=feature_res,
        in_resolution=in_resolution, 
        in_channels=3, 
        patch_size=patch_size, 
        enc_emb_size=4096,
        pred_emb_size=4096,
        num_enc_blocks=20, 
        num_pred_blocks=16,
        num_enc_heads=8,
        num_pred_heads=8,
        num_enc_mlp_ratio=4,
        num_pred_mlp_ratio=4,
        layer_norm=nn.LayerNorm,
    )
        
if __name__ == "__main__":
    model = MaskedImageModelingModel()
    x = torch.randn(2, 3, 224, 224)
    mask = torch.zeros(2, 32*32)
    mask[:, :32] = 1
    output = model(x, mask)
    print(output["loss"])
        