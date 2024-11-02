import torch
from torch import nn, einsum
from torchvision import models
from torchvision.models import VGG19_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
from einops import rearrange

SUPPOTED_BACKBONES = [
    "vgg19", "efficientnet-s", "efficientnet-m", "efficientnet-l", "pdn_small", "pdn_medium"
]

PDN_SMALL_PATH = "./weights/backbone/pdn/teacher_small.pth"
PDN_MEDIUM_PATH = "./weights/backbone/pdn/teacher_medium.pth"

def get_pdn_small(ckpt_path, out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    pdn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )
    pdn.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return pdn

def get_pdn_medium(ckpt_path, out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    pdn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )
    pdn.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return pdn

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
    elif model_name == "pdn_small":
        return get_pdn_small(ckpt_path=PDN_SMALL_PATH)
    elif model_name == "pdn_medium":
        return get_pdn_medium(ckpt_path=PDN_MEDIUM_PATH)

def get_intermediate_output_hook(layer, input, output):
    BackboneModel.intermediate_cache.append(output)
    
class BackboneModel(nn.Module):
    intermediate_cache = []
    
    def __init__(self, model_name: str, extract_indices: list, feature_res: int = 64):
        super(BackboneModel, self).__init__()
        self.model_name = model_name
        self.model = get_backbone_model(model_name)
        self.model.eval()
        self.extract_indices = extract_indices
        self.feature_res = feature_res
        
        if model_name in ["pdn_small", "pdn_medium"]:
            self.feature_dim = 384
        else:
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
        
        if self.model_name in ["pdn_small", "pdn_medium"]:
            with torch.no_grad():
                features = self.model(x)
                features = nn.functional.interpolate(features, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
            return features
            
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
    model = BackboneModel("pdn_small", [], ckpt_path="/home/haselab/projects/ICAD/weights/backbone/pdn/teacher_small.pth")
    x = torch.randn(1, 3, 224, 224)
    features = model(x)
    print(features.shape)
    
    