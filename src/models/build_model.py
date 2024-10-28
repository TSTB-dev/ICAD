
from .mim import mim_tiny, mim_small, mim_base, mim_large, mim_huge, mim_gigant, SUPPORTEED_MODELS
from . import mim

def build_mim_model(model_name: str):
    assert model_name in SUPPORTEED_MODELS, f"Model {model_name} not supported"
    return getattr(mim, model_name)