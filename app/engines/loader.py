from typing import Optional
from .base import OCRModel
from .base import OCRModel
from .hf_impl import HuggingFaceLexiSightModel
from .vllm_impl import VLLMLexiSightModel
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
            _model_instance = HuggingFaceLexiSightModel()
            _model_instance.load()
        elif settings.MODEL_SOURCE == "vllm":
            _model_instance = VLLMLexiSightModel()
            _model_instance.load()
        # Future extension: Add other sources here
        else:
            raise ValueError(f"Unknown model source: {settings.MODEL_SOURCE}")
    return _model_instance
