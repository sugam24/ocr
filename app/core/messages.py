class LogMessages:
    # Startup/Shutdown
    SERVICE_STARTUP = "LightOnOCR Service starting up..."
    SERVICE_SHUTDOWN = "LightOnOCR Service shutting down..."
    LOGGING_INITIALIZED = "LightOnOCR Logging initialized"
    
    # Model Loading
    MODEL_PREPARING = "Preparing LightOnOCR model '{}' with {}..."
    MODEL_OFFLINE_FOUND = "Found verified model in {}. Using offline mode."
    MODEL_DOWNLOADING = "Model invalid or missing in {}. Downloading..."
    MODEL_DOWNLOAD_SUCCESS = "Model downloaded successfully."
    MODEL_DOWNLOAD_FAIL = "Failed to download model: {}"
    MODEL_LOAD_SUCCESS = "LightOnOCR model loaded successfully."
    MODEL_LOAD_FAIL = "Failed to load model: {}"
    MODEL_INIT_FAIL = "Model init failed: {}"
    
    # vLLM Specific
    VLLM_ENGINE_INIT_SUCCESS = "vLLM Engine initialized successfully."
    VLLM_INIT_FAIL = "Failed to initialize vLLM: {}"
    VLLM_INFERENCE_FAIL = "vLLM Inference failed: {}"
    
    # HF Specific
    HF_INFERENCE_FAIL = "Inference failed: {}"
    CUDA_NOT_AVAILABLE = "CUDA requested but not available. Falling back to CPU."
    
    # Inference
    INFERENCE_OUTPUT = "Final Output Text: {}"
