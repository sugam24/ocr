class LogMessages:
    # Startup/Shutdown
    SERVICE_STARTUP = "LexiSight Service starting up..."
    SERVICE_SHUTDOWN = "LexiSight Service shutting down..."
    LOGGING_INITIALIZED = "LexiSight Logging initialized"
    
    # Model Loading
    MODEL_PREPARING = "Preparing LexiSight model '{}' with {}..." # model_name, source
    MODEL_OFFLINE_FOUND = "Found verified model in {}. Using offline mode."
    MODEL_DOWNLOADING = "Model invalid or missing in {}. Downloading..."
    MODEL_DOWNLOAD_SUCCESS = "Model downloaded successfully."
    MODEL_DOWNLOAD_FAIL = "Failed to download model: {}"
    MODEL_LOAD_SUCCESS = "LexiSight model loaded successfully."
    MODEL_LOAD_FAIL = "Failed to load model: {}"
    MODEL_INIT_FAIL = "Model init failed: {}"
    
    # vLLM Specific
    VLLM_ENGINE_INIT_SUCCESS = "vLLM Engine initialized successfully."
    VLLM_INIT_FAIL = "Failed to initialize vLLM: {}"
    VLLM_INFERENCE_FAIL = "vLLM Inference failed: {}"
    
    # HF Specific
    HF_PROCESSOR_FAIL = "Standard AutoProcessor not found or incompatible ({}). Swapping to manual tokenizer/image_processor loading."
    HF_INFERENCE_FAIL = "Inference failed: {}"
    
    # Inferences
    INFERENCE_INPUT_SHAPE = "Input IDs shape: {}"
    INFERENCE_PIXEL_SHAPE = "Pixel Values shape: {}"
    INFERENCE_GRID_THW = "Image Grid THW: {}"
    INFERENCE_GENERATED = "Generated IDs: {}"
    INFERENCE_OUTPUT = "Final Output Text: {}"
    
    # Parsing
    JSON_PARSE_FAIL = "Failed to parse JSON directly: {}. Attempting regex fallback."
    JSON_FALLBACK_FAIL = "Fallback JSON parsing failed: {}"
