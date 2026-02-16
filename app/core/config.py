import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LexiSight"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    MODEL_SOURCE: str = "vllm" # huggingface, local, vllm
    MODEL_NAME: str = "rednote-hilab/dots.ocr" # or path to local weights
    MODEL_CACHE_DIR: str = "Model"
    DEVICE: str = "cuda" # Default to cuda as requested
    
    # Service Constraints
    MAX_FILE_SIZE_MB: int = 10
    
    # vLLM Specific Configuration
    # 0.4 (40%) is recommended for shared T4 GPU (16GB) to allow other services.
    # Default is 0.9 (90%) if running dedicated.
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.9 
    VLLM_MAX_MODEL_LEN: int = 8192

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
