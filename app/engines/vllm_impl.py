import logging
import os
import asyncio
import json
import re
from typing import Dict, Any, List
# We import AsyncLLMEngine and SamplingParams. 
# Note: For offline inference matching standard "predict" call patterns in a service,
# wrapping the AsyncLLMEngine is best for concurrency.
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from huggingface_hub import snapshot_download

from .base import OCRModel
from ..core.config import settings
from ..utils.image import load_image_from_bytes

# Import the correct prompt from hf_impl (or duplicate it if internal import issues)
# For safety, let's duplicate the constant or import it if we refactor. 
# Simplest: duplicate for isolated file stability.
from .prompts import OCR_PROMPT

from ..core.messages import LogMessages

logger = logging.getLogger("lexisight")

class VLLMLexiSightModel(OCRModel):
    def __init__(self):
        self.engine: AsyncLLMEngine = None
        self.device = settings.DEVICE
        self.model_name = settings.MODEL_NAME
        self.cache_dir = settings.MODEL_CACHE_DIR

    def load(self) -> None:
        logger.info(LogMessages.MODEL_PREPARING.format(self.model_name, "vLLM"))

        # 1. Strict Manual Verification
        model_path = self.cache_dir
        
        has_config = os.path.exists(os.path.join(model_path, "config.json"))
        # Check for at least one safetensors/bin file that is not empty
        model_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") or f.endswith(".bin")] if os.path.exists(model_path) else []
        has_weights = any(os.path.getsize(os.path.join(model_path, f)) > 1000 for f in model_files) # > 1KB check

        if has_config and has_weights:
            logger.info(LogMessages.MODEL_OFFLINE_FOUND.format(model_path))
        else:
            logger.info(LogMessages.MODEL_DOWNLOADING.format(model_path))
            try:
                # Force online download
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                logger.info(LogMessages.MODEL_DOWNLOAD_SUCCESS)
            except Exception as download_error:
                logger.error(LogMessages.MODEL_DOWNLOAD_FAIL.format(download_error))
                raise RuntimeError(LogMessages.MODEL_DOWNLOAD_FAIL.format(download_error))
        
        # 2. Initialize vLLM Engine
        # trust_remote_code=True is essential for dots.ocr
        try:
           engine_args = AsyncEngineArgs(
                model=model_path,
                trust_remote_code=True,
                # Use bfloat16 to match Qwen2-VL requirements better
                dtype="bfloat16", 
                max_model_len=settings.VLLM_MAX_MODEL_LEN, # Configurable
                enforce_eager=True, # Often helps with stability for custom models
                limit_mm_per_prompt={"image": 1}, # Multi-modal optimization
                gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION, # Configurable
            )
           self.engine = AsyncLLMEngine.from_engine_args(engine_args)
           logger.info(LogMessages.VLLM_ENGINE_INIT_SUCCESS)
        except Exception as e:
            logger.error(LogMessages.VLLM_INIT_FAIL.format(e))
            raise RuntimeError(LogMessages.VLLM_INIT_FAIL.format(e))

    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Async prediction using vLLM engine.
        """
        if not self.engine:
             raise RuntimeError("Model is not loaded. Call load() first.")

        image = load_image_from_bytes(image_data)
        
        # Dots OCR Prompt format
        # Qwen2-VL chat template usually
        # We construct a conversation-like prompt structure similar to hf_impl
        # But vLLM handles templates if we use the chat API. 
        # Here we are using low-level generate.
        
        # Standard Qwen2-VL raw prompt usually expects:
        # <|im_start|>user<|vision_start|><|image_pad|><|vision_end|>PROMPT<|im_end|><|im_start|>assistant
        
        # Note: vLLM might handle the image tags automatically if we pass the image, but explicit doesn't hurt.
        prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{OCR_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
        
        request_id = os.urandom(8).hex()
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)
        
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image 
            }
        }
        
        try:
            # Generate
            results_generator = self.engine.generate(
                inputs,
                sampling_params,
                request_id=request_id
            )
            
            # Iterate to get final result
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            output_text = final_output.outputs[0].text
            
            # Parse JSON logic (Duplicated from hf_impl for consistency)
            # Parse JSON
            from ..utils.parsing import parse_json_output
            blocks = parse_json_output(output_text)

            return {
                "text": output_text,
                "blocks": blocks,
                "model_version": "lexisight-vllm"
            }
            
        except Exception as e:
             logger.error(LogMessages.VLLM_INFERENCE_FAIL.format(e))
             raise e
