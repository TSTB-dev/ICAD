import torch
from torch import nn, einsum
from torchvision import models
from torchvision.models import VGG19_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
from einops import rearrange

SUPPOTED_BACKBONES = [
    "vgg19", "efficientnet-s", "efficientnet-m", "efficientnet-l"
]

def get_backbone_model(model_name):
    assert model_name in SUPPOTED_BACKBONES, f"Model {model_name} not supported"
    if model_name == "vgg19":
        return models.vgg19(weights=VGG19_Weights.DEFAULT)
    elif model_name == "efficientnet-s":
        return models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    elif model_name == "efficientnet-m":
        return models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    elif model_name == "efficientnet-l":
        return models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

def get_intermediate_output_hook(layer, input, output):
    BackboneModel.intermediate_cache.append(output)
    
class BackboneModel(nn.Module):
    intermediate_cache = []
    
    def __init__(self, model_name: str, extract_indices: list, feature_res: int = 64):
        super(BackboneModel, self).__init__()
        self.model = get_backbone_model(model_name)
        self.model.eval()
        self.extract_indices = extract_indices
        self.feature_res = feature_res
        
        self._register_hook()
    
    def _register_hook(self):
        self.layer_hooks = []
        feature_dim = 0
        for i, layer_idx in enumerate(self.extract_indices):
            module = self.model.features[layer_idx-1]
            if isinstance(module, nn.Conv2d):
                feature_dim += module.out_channels
            elif isinstance(module, nn.Sequential):
                if isinstance(module[-1], nn.SiLU):
                    feature_dim += module[-3].out_channels
                else:
                    feature_dim += module[-1].out_channels
            layer_to_hook = self.model.features[layer_idx]
            hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
            self.layer_hooks.append(hook)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor):
        """Extract features from the backbone model. 
        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            extract_indices (list): List of indices to extract features from the backbone model.
        Returns:
            torch.Tensor: Extracted features, shape (B, C, H', W')
        Examples:
            >>> backbone = get_backbone_model("vgg19", [3, 8, 17, 26])
            >>> features = backbone.extract_features(x)  # x shape (B, 960, 64, 64)
        """
        
        with torch.no_grad():
            _ = self.model(x)
        self.intermediate_outputs = BackboneModel.intermediate_cache
        self._reset_cache()
        
        for i, intermediate_output in enumerate(self.intermediate_outputs):
            self.intermediate_outputs[i] = nn.functional.interpolate(intermediate_output, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
        features = torch.cat(self.intermediate_outputs, dim=1)

        return features

    def _reset_cache(self):
        BackboneModel.intermediate_cache = []

if __name__ == "__main__":
    model = BackboneModel("efficientnet-s", [1, 2, 3, 4])
    x = torch.randn(1, 3, 224, 224)
    features = model(x)
    print(features.shape)