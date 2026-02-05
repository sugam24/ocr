import torch
import logging
import os
import math
from typing import Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from .base import OCRModel
from ..core.config import settings
from ..utils.image import load_image_from_bytes
from qwen_vl_utils import process_vision_info

from ..core.messages import LogMessages

logger = logging.getLogger("lexisight")

# Constants from Reference
MIN_PIXELS = 3136
MAX_PIXELS = 11289600
IMAGE_FACTOR = 28

# Reference Prompt
from .prompts import OCR_PROMPT

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 11289600):
    if max(height, width) / min(height, width) > 200:
        # Fallback or clamp if aspect ratio is insane, but typically just error
        # We will warn and clamp effectively by logic below
        pass
        
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = round_by_factor(height / beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = round_by_factor(height * beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    return h_bar, w_bar

class HuggingFaceLexiSightModel(OCRModel):
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        self.device = torch.device(settings.DEVICE)
        self.model_name = settings.MODEL_NAME
        self.cache_dir = settings.MODEL_CACHE_DIR

    def load(self) -> None:
        logger.info(LogMessages.MODEL_PREPARING.format(self.model_name, self.device))
        
        model_path = self.cache_dir
        if not os.path.exists(os.path.join(model_path, "config.json")):
            logger.info(LogMessages.MODEL_DOWNLOADING.format(model_path))
            try:
                snapshot_download(repo_id=self.model_name, local_dir=model_path, local_dir_use_symlinks=False)
                logger.info(LogMessages.MODEL_DOWNLOAD_SUCCESS)
            except Exception as e:
                logger.error(LogMessages.MODEL_DOWNLOAD_FAIL.format(e))
                raise RuntimeError(LogMessages.MODEL_DOWNLOAD_FAIL.format(e))
        else:
            logger.info(LogMessages.MODEL_OFFLINE_FOUND.format(model_path))

        try:
            # Determine device_map based on DEVICE setting
            if settings.DEVICE == "cpu":
                device_map = "cpu"
            else:
                device_map = "auto"
            
            # Check if this is a pre-quantized 4-bit model
            # The helizac/dots.ocr-4bit model has quantization config embedded
            # so we don't need to specify torch_dtype - it will use the quantized weights
            is_4bit_model = "4bit" in self.model_name.lower() or "4-bit" in self.model_name.lower()
            
            if is_4bit_model:
                logger.info("Loading 4-bit quantized model...")
                # For pre-quantized models, explicitly override the compute dtype to float16
                # to prevent dtype mismatches (BFloat16 vs Float16)
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    device_map=device_map,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                ).eval()
            else:
                # For non-quantized models, use original logic
                model_dtype = torch.float32 if settings.DEVICE == "cpu" else torch.bfloat16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    torch_dtype=model_dtype, 
                    device_map=device_map,
                    local_files_only=True
                ).eval()
            
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.tokenizer = self.processor.tokenizer
            except Exception as ve:
                logger.info(LogMessages.HF_PROCESSOR_FAIL.format(ve))
                from transformers import Qwen2VLImageProcessor, AutoTokenizer
                self.image_processor = Qwen2VLImageProcessor.from_pretrained(model_path, local_files_only=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.processor = None
            
            logger.info(LogMessages.MODEL_LOAD_SUCCESS)
            
        except Exception as e:
            logger.error(LogMessages.MODEL_LOAD_FAIL.format(e))
            raise RuntimeError(LogMessages.MODEL_LOAD_FAIL.format(e))

    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            # 1. Load and Resize Image
            image = load_image_from_bytes(image_data)
            h, w = smart_resize(image.height, image.width)
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            
            # 2. Prepare Messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": OCR_PROMPT}, 
                    ],
                }
            ]
            
            # 3. Process Vision Info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 4. Process Image Inputs FIRST to get patch count
            if self.processor:
                 image_tensor_dict = self.processor.image_processor(
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
            else:
                 image_tensor_dict = self.image_processor(
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
            
            # Calculate number of visual tokens
            # pixel_values shape is (num_patches, channels * patch*patch)
            # Qwen2-VL merges 2x2 patches into 1 visual token.
            # Thus, the number of image tokens should be num_patches // 4
            num_patches = image_tensor_dict['pixel_values'].shape[0] // 4
            
            # 5. Process Text Inputs Manually
            # Ensure we get the correct image_token_id
            image_token_id = getattr(self.model.config, "image_token_id", 151665)
            
            prompt_prefix = "<|im_start|>user\n<|vision_start|>"
            prompt_suffix = f"<|vision_end|>{OCR_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
            
            ids_prefix = self.tokenizer.encode(prompt_prefix, add_special_tokens=False)
            ids_suffix = self.tokenizer.encode(prompt_suffix, add_special_tokens=False)
            
            # Construct input_ids: prefix + [tokens * num_patches] + suffix
            input_ids = ids_prefix + ([image_token_id] * num_patches) + ids_suffix
            
            # Construct final input dictionary
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long).to(self.device),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(self.device),
            }
            
            # Update with image tensors, ensuring correct dtype
            # For 4-bit models, need to use float16; for regular models, use model.dtype
            is_4bit = "4bit" in self.model_name.lower() or "4-bit" in self.model_name.lower()
            for k, v in image_tensor_dict.items():
                if k == 'pixel_values':
                    # visual features must match model dtype
                    # 4-bit quantized models use float16
                    if is_4bit:
                        model_dtype = torch.float16
                    else:
                        model_dtype = self.model.dtype
                    inputs[k] = v.to(self.device, dtype=model_dtype)
                else:
                    # other params like grid_thw are usually int/long
                    inputs[k] = v.to(self.device)

            # Debug Logging
            logger.info(LogMessages.INFERENCE_INPUT_SHAPE.format(inputs['input_ids'].shape))
            logger.info(LogMessages.INFERENCE_PIXEL_SHAPE.format(inputs['pixel_values'].shape))
            if 'image_grid_thw' in inputs:
                 logger.info(LogMessages.INFERENCE_GRID_THW.format(inputs['image_grid_thw']))

            # 5. Inference
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048, # Full generation
                    do_sample=False, 
                    temperature=0.1 # explicit temp from reference
                )
                
            logger.info(LogMessages.INFERENCE_GENERATED.format(generated_ids.tolist()))

            # 6. Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(LogMessages.INFERENCE_OUTPUT.format(output_text))
            
            # Parse JSON
            from ..utils.parsing import parse_json_output
            blocks = parse_json_output(output_text)

            return {
                "text": output_text, # Raw text still useful
                "blocks": blocks, # Strucured JSON
                "model_version": "lexisight-v1"
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e