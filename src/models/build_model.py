
from .mim import mim_tiny, mim_small, mim_base, mim_large, mim_huge, mim_gigant, predictor_tiny, \
    predictor_small, predictor_base, predictor_large, predictor_huge, predictor_gigant, PREDICTOR_SUPPORTEED_MODELS, MIM_SUPPORTEED_MODELS
from . import mim

def build_mim_model(model_name: str):
    assert model_name in MIM_SUPPORTEED_MODELS, f"Model {model_name} not supported"
    return getattr(mim, model_name)

def build_predictor_model(model_name: str):
    assert model_name in PREDICTOR_SUPPORTEED_MODELS, f"Model {model_name} not supported"
    return getattr(mim, model_name)