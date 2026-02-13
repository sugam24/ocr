import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LightOnOCR-Service"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    MODEL_SOURCE: str = "huggingface" # huggingface or vllm
    MODEL_NAME: str = "lightonai/LightOnOCR-2-1B"
    MODEL_CACHE_DIR: str = "Model"
    DEVICE: str = "cuda" # cuda, cpu, or mps
    
    # Service Constraints
    MAX_FILE_SIZE_MB: int = 10
    
    # vLLM Specific Configuration
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.9 
    VLLM_MAX_MODEL_LEN: int = 8192

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
