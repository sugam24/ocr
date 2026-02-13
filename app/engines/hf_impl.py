import torch
import logging
import os
from typing import Dict, Any
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from .base import OCRModel
from ..core.config import settings
from ..utils.image import load_image_from_bytes
from ..core.messages import LogMessages

logger = logging.getLogger("lightonocr")


class HuggingFaceLightOnOCRModel(OCRModel):
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()
        self.model_name = settings.MODEL_NAME
        self.cache_dir = settings.MODEL_CACHE_DIR

    def _resolve_device(self) -> str:
        """Resolve the best available device."""
        device = settings.DEVICE.lower()
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the appropriate dtype for the device."""
        device = self._resolve_device()
        if device == "mps":
            return torch.float32  # MPS doesn't support bfloat16
        elif device == "cuda":
            return torch.bfloat16
        else:
            return torch.float32

    def load(self) -> None:
        """Load the LightOnOCR model and processor.
        
        Handles:
        - Model downloading if not cached locally
        - Device-specific configuration (CPU/CUDA/MPS)
        - Appropriate dtype selection
        """
        logger.info(LogMessages.MODEL_PREPARING.format(self.model_name, self.device))
        
        model_path = self.cache_dir
        if not os.path.exists(os.path.join(model_path, "config.json")):
            logger.info(LogMessages.MODEL_DOWNLOADING.format(model_path))
            try:
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                logger.info(LogMessages.MODEL_DOWNLOAD_SUCCESS)
            except Exception as e:
                logger.error(LogMessages.MODEL_DOWNLOAD_FAIL.format(e))
                raise RuntimeError(LogMessages.MODEL_DOWNLOAD_FAIL.format(e))
        else:
            logger.info(LogMessages.MODEL_OFFLINE_FOUND.format(model_path))

        try:
            # Load model with appropriate dtype and device
            self.model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

            # Load processor
            self.processor = LightOnOcrProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )

            logger.info(LogMessages.MODEL_LOAD_SUCCESS)
            
        except Exception as e:
            logger.error(LogMessages.MODEL_LOAD_FAIL.format(e))
            raise RuntimeError(LogMessages.MODEL_LOAD_FAIL.format(e))

    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Run OCR inference on an image using LightOnOCR-2-1B.
        
        Pipeline:
        1. Decode image from bytes
        2. Construct conversation with image using chat template
        3. Run model generation
        4. Decode output text
        
        Args:
            image_data: Raw image bytes (JPEG, PNG, or PDF)
            
        Returns:
            Dictionary with 'text' (OCR output) and 'model_version'
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            # Step 1: Load image from bytes
            image = load_image_from_bytes(image_data)

            # Step 2: Construct conversation in LightOnOCR chat format
            # LightOnOCR takes an image directly — no explicit text prompt needed
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                    ],
                }
            ]

            # Step 3: Process inputs using the chat template
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move inputs to the correct device and dtype
            inputs = {
                k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }

            # Step 4: Generate OCR output
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                )

            # Step 5: Decode — strip the input tokens to get only the generated text
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            logger.info(LogMessages.INFERENCE_OUTPUT.format(output_text[:200]))

            return {
                "text": output_text,
                "blocks": [],
                "model_version": "lightonocr-2-1b",
            }

        except Exception as e:
            logger.error(LogMessages.HF_INFERENCE_FAIL.format(e))
            raise e