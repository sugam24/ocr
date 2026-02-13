from typing import Optional
from .base import OCRModel
from .hf_impl import HuggingFaceLightOnOCRModel
from .vllm_impl import VLLMLightOnOCRModel
from ..core.config import settings

_model_instance: Optional[OCRModel] = None

def get_model() -> OCRModel:
    """
    Get the global model instance. Initializes it if not already loaded.
    This ensures the model is loaded exactly once (Singleton pattern).
    """
    global _model_instance
    if _model_instance is None:
        if settings.MODEL_SOURCE == "huggingface":
            _model_instance = HuggingFaceLightOnOCRModel()
            _model_instance.load()
        elif settings.MODEL_SOURCE == "vllm":
            _model_instance = VLLMLightOnOCRModel()
            _model_instance.load()
        else:
            raise ValueError(f"Unknown model source: {settings.MODEL_SOURCE}")
    return _model_instance
