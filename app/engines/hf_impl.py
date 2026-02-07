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
# These constants define the pixel constraints for the Qwen2-VL vision model
MIN_PIXELS = 3136  # Minimum total pixels required for image processing
MAX_PIXELS = 11289600  # Maximum total pixels allowed (optimal range: 3136-11289600)
IMAGE_FACTOR = 28  # Factor for rounding image dimensions (Qwen2-VL requirement)

# Reference Prompt
from .prompts import OCR_PROMPT

def round_by_factor(number: int, factor: int) -> int:
    """Round a number to the nearest multiple of factor.
    Used to ensure image dimensions are compatible with vision model's patch structure.
    """
    return round(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 11289600):
    """Intelligently resize image dimensions to meet vision model constraints.
    
    Strategy:
    1. Round dimensions to factor multiples (28 for Qwen2-VL patches)
    2. Ensure total pixels stay within [min_pixels, max_pixels] range
    3. Maintain aspect ratio while scaling
    
    Args:
        height, width: Original image dimensions
        factor: Dimension rounding factor (default 28 for Qwen2-VL)
        min_pixels, max_pixels: Valid pixel range for the model
    
    Returns:
        Tuple of (resized_height, resized_width)
    """
    # Check for extremely skewed aspect ratios (>200:1)
    if max(height, width) / min(height, width) > 200:
        # Fallback or clamp if aspect ratio is insane, but typically just error
        # We will warn and clamp effectively by logic below
        pass
        
    # First, round both dimensions to nearest factor multiple
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    # Second, ensure total pixel count is within valid range
    if h_bar * w_bar > max_pixels:
        # Scale down proportionally to meet max_pixels constraint
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = round_by_factor(height / beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        # Scale up proportionally to meet min_pixels constraint
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
        """Load the vision-language model and processor.
        
        Handles:
        - Model downloading if not cached locally
        - Device-specific configuration (CPU/GPU)
        - 4-bit quantization detection and special dtype handling
        - Fallback processor loading if AutoProcessor fails
        """
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
                device_map = "cpu"  # Force CPU execution
            else:
                device_map = "auto"  # Let transformers choose GPU device(s)
            
            # Detect if model is pre-quantized to 4-bit
            # Pre-quantized models (e.g., helizac/dots.ocr-4bit) have quantization config embedded
            # They require special dtype handling to prevent bfloat16/float16 mismatches
            is_4bit_model = "4bit" in self.model_name.lower() or "4-bit" in self.model_name.lower()
            
            if is_4bit_model:
                logger.info("Loading 4-bit quantized model...")
                # 4-bit quantized models require special handling:
                # - Force float16 compute dtype to prevent bfloat16/float16 conflicts
                # - Use eager attention (not SDPA) for quantized inference stability
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Enable 4-bit quantization loading
                    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype must match model expectation
                    bnb_4bit_quant_type="nf4",  # Use normal float 4-bit quantization
                    bnb_4bit_use_double_quant=True,  # Apply double quantization for better compression
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Use eager attention for quantized models
                    device_map=device_map,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,  # Match the quantization compute dtype
                ).eval()
            else:
                # Full-precision models: use float32 on CPU, bfloat16 on GPU
                # bfloat16 provides better performance/precision tradeoff on modern GPUs
                model_dtype = torch.float32 if settings.DEVICE == "cpu" else torch.bfloat16
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="sdpa",  # Use efficient scaled-dot-product attention
                    torch_dtype=model_dtype, 
                    device_map=device_map,
                    local_files_only=True
                ).eval()
            
            try:
                # Try to load unified processor (handles both image and text processing)
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.tokenizer = self.processor.tokenizer
            except Exception as ve:
                # Fallback: Load separate image processor and tokenizer if unified processor fails
                logger.info(LogMessages.HF_PROCESSOR_FAIL.format(ve))
                from transformers import Qwen2VLImageProcessor, AutoTokenizer
                self.image_processor = Qwen2VLImageProcessor.from_pretrained(model_path, local_files_only=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.processor = None  # Mark processor as unavailable for later checks
            
            logger.info(LogMessages.MODEL_LOAD_SUCCESS)
            
        except Exception as e:
            logger.error(LogMessages.MODEL_LOAD_FAIL.format(e))
            raise RuntimeError(LogMessages.MODEL_LOAD_FAIL.format(e))

    async def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Run OCR inference on an image.
        
        Pipeline:
        1. Decode and resize image to model constraints
        2. Process vision info and extract visual features
        3. Construct input_ids with image tokens manually
        4. Run model generation with constrained decoding
        5. Decode output and parse JSON blocks
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with 'text' (raw output), 'blocks' (parsed JSON), and 'model_version'
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            # STEP 1: Load and intelligently resize image to meet model constraints
            image = load_image_from_bytes(image_data)
            h, w = smart_resize(image.height, image.width)  # Ensures pixel count in [3136, 11289600]
            image = image.resize((w, h), Image.Resampling.LANCZOS)  # Use high-quality resampling
            
            # STEP 2: Construct message with image and OCR prompt for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},  # Vision content
                        {"type": "text", "text": OCR_PROMPT},  # OCR instruction prompt
                    ],
                }
            ]
            
            # STEP 3: Extract vision data from messages (separates images/videos from text)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # STEP 4: Process image into visual patches via image processor
            # This converts the image to tensor format suitable for vision encoder
            if self.processor:
                 image_tensor_dict = self.processor.image_processor(
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",  # Return PyTorch tensors
                )
            else:
                 image_tensor_dict = self.image_processor(
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
            
            # Calculate the number of visual tokens needed in input_ids
            # Qwen2-VL's image_processor outputs patches; the model then:
            # - Merges 2x2 adjacent patches into 1 visual token
            # - Thus num_visual_tokens = num_patches // 4
            # Note: pixel_values shape is (num_patches, channels, height, width)
            num_patches = image_tensor_dict['pixel_values'].shape[0] // 4
            
            # STEP 5: Manually construct input_ids with image tokens
            # We insert actual image_token_ids into the sequence to match visual patches
            
            # Get the special token ID used to represent images in the sequence
            # Default to 151665 if not found in config (Qwen2-VL standard)
            image_token_id = getattr(self.model.config, "image_token_id", 151665)
            
            # Define text prompts with Qwen2-VL special formatting
            prompt_prefix = "<|im_start|>user\n<|vision_start|>"  # Begin user message with vision
            prompt_suffix = f"<|vision_end|>{OCR_PROMPT}<|im_end|>\n<|im_start|>assistant\n"  # End vision, add prompt
            
            # Tokenize prefix and suffix, without special tokens (we manage them manually)
            ids_prefix = self.tokenizer.encode(prompt_prefix, add_special_tokens=False)
            ids_suffix = self.tokenizer.encode(prompt_suffix, add_special_tokens=False)
            
            # Build final input_ids sequence: prefix -> [visual tokens] -> suffix
            # The repeated image_token_ids will be replaced with actual visual embeddings
            input_ids = ids_prefix + ([image_token_id] * num_patches) + ids_suffix
            
            # Prepare model inputs dictionary with text and visual components
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long).to(self.device),  # Text token IDs
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(self.device),  # All tokens attend
            }
            
            # Add visual tensors with appropriate dtype handling
            # Different model variants require different dtypes:
            # - 4-bit quantized: must use float16
            # - Regular: use model's native dtype (usually bfloat16 on GPU, float32 on CPU)
            is_4bit = "4bit" in self.model_name.lower() or "4-bit" in self.model_name.lower()
            for k, v in image_tensor_dict.items():
                if k == 'pixel_values':
                    # Visual feature tensor must match what the model expects
                    if is_4bit:
                        model_dtype = torch.float16  # 4-bit models use float16 for visual features
                    else:
                        model_dtype = self.model.dtype  # Use model's native dtype
                    inputs[k] = v.to(self.device, dtype=model_dtype)
                else:
                    # Metadata like grid_thw (height/width grid) are typically int/long
                    inputs[k] = v.to(self.device)

            # Log input shapes for debugging
            logger.info(LogMessages.INFERENCE_INPUT_SHAPE.format(inputs['input_ids'].shape))
            logger.info(LogMessages.INFERENCE_PIXEL_SHAPE.format(inputs['pixel_values'].shape))
            if 'image_grid_thw' in inputs:
                 logger.info(LogMessages.INFERENCE_GRID_THW.format(inputs['image_grid_thw']))

            # STEP 6: Run model generation
            # Generate OCR text output greedily (deterministic, no sampling)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Allow up to 2048 tokens for full OCR output
                    do_sample=False,  # Greedy decoding for reproducibility
                    temperature=0.1  # Temperature parameter (ignored with do_sample=False)
                )
                
            logger.info(LogMessages.INFERENCE_GENERATED.format(generated_ids.tolist()))

            # STEP 7: Decode generated tokens back to text
            # Remove input token IDs to keep only generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            # Convert token IDs to text, removing special tokens
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,  # Remove <|im_start|>, <|im_end|>, etc.
                clean_up_tokenization_spaces=False  # Preserve original spacing
            )[0]
            
            logger.info(LogMessages.INFERENCE_OUTPUT.format(output_text))
            
            # STEP 8: Parse model output into structured JSON blocks
            # The model outputs JSON with recognized text blocks, bounding boxes, etc.
            from ..utils.parsing import parse_json_output
            blocks = parse_json_output(output_text)

            return {
                "text": output_text,  # Raw model output (includes JSON)
                "blocks": blocks,  # Parsed and structured JSON blocks with OCR data
                "model_version": "lexisight-v1"  # Version identifier
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e