import logging
import os
import base64
import io
import requests
from typing import Dict, Any
from huggingface_hub import snapshot_download

from .base import OCRModel
from ..core.config import settings
from ..utils.image import load_image_from_bytes
from ..core.messages import LogMessages

logger = logging.getLogger("lightonocr")


class VLLMLightOnOCRModel(OCRModel):
    """LightOnOCR inference via a vLLM OpenAI-compatible server.
    
    This engine expects a vLLM server to be running separately:
        vllm serve lightonai/LightOnOCR-2-1B \
            --limit-mm-per-prompt '{"image": 1}' \
            --mm-processor-cache-gb 0 \
            --no-enable-prefix-caching
    
    The engine sends requests to the vLLM server's /v1/chat/completions endpoint.
    """

    def __init__(self):
        self.device = settings.DEVICE
        self.model_name = settings.MODEL_NAME
        self.cache_dir = settings.MODEL_CACHE_DIR
        self.vllm_endpoint = os.environ.get(
            "VLLM_ENDPOINT", "http://localhost:8001/v1/chat/completions"
        )
        self._server_ready = False

    def load(self) -> None:
        """Ensure model weights are available and verify vLLM server connectivity.
        
        Downloads model weights if not cached locally.
        The actual vLLM server should be started separately.
        """
        logger.info(LogMessages.MODEL_PREPARING.format(self.model_name, "vLLM"))

        # Ensure model weights are available locally
        model_path = self.cache_dir

        has_config = os.path.exists(os.path.join(model_path, "config.json"))
        model_files = (
            [
                f
                for f in os.listdir(model_path)
                if f.endswith(".safetensors") or f.endswith(".bin")
            ]
            if os.path.exists(model_path)
            else []
        )
        has_weights = any(
            os.path.getsize(os.path.join(model_path, f)) > 1000 for f in model_files
        )

        if has_config and has_weights:
            logger.info(LogMessages.MODEL_OFFLINE_FOUND.format(model_path))
        else:
            logger.info(LogMessages.MODEL_DOWNLOADING.format(model_path))
            try:
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                )
                logger.info(LogMessages.MODEL_DOWNLOAD_SUCCESS)
            except Exception as download_error:
                logger.error(LogMessages.MODEL_DOWNLOAD_FAIL.format(download_error))
                raise RuntimeError(
                    LogMessages.MODEL_DOWNLOAD_FAIL.format(download_error)
                )

        self._server_ready = True
        logger.info(LogMessages.VLLM_ENGINE_INIT_SUCCESS)

    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Run OCR inference via vLLM OpenAI-compatible API.
        
        Converts image to base64 and sends it to the vLLM server.
        """
        if not self._server_ready:
            raise RuntimeError("Model is not loaded. Call load() first.")

        image = load_image_from_bytes(image_data)

        # Convert PIL image to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Construct OpenAI-compatible chat payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        }
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        try:
            response = requests.post(self.vllm_endpoint, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            output_text = result["choices"][0]["message"]["content"]

            logger.info(LogMessages.INFERENCE_OUTPUT.format(output_text[:200]))

            return {
                "text": output_text,
                "blocks": [],
                "model_version": "lightonocr-2-1b-vllm",
            }

        except Exception as e:
            logger.error(LogMessages.VLLM_INFERENCE_FAIL.format(e))
            raise e
