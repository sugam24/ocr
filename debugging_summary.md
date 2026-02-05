# Debugging Summary: running `helizac/dots.ocr-4bit` on CPU/Low-VRAM

## Objective
Run the dots.ocr model on a system with 4GB VRAM and limited RAM.
 Chosen approach: Use the 4-bit quantized version `helizac/dots.ocr-4bit`.

## System State
- **Project Path**: `/home/sugam/Desktop/helios-ocr`
- **Model Path**: `/home/sugam/Desktop/helios-ocr/Model-4bit` (Local download of `helizac/dots.ocr-4bit`)
- **Current Error**: `RuntimeError: Input type (c10::BFloat16) and bias type (c10::Half) should be the same` during inference.

## Changes Applied

### 1. Configuration
- Updated `.env` to use `MODEL_NAME="helizac/dots.ocr-4bit"`.
- Installed `bitsandbytes`.

### 2. Code Modifications (`app/engines/hf_impl.py`)
- Modified model loading to detect "4bit" in the name.
- Forces `attn_implementation="eager"` to avoid `flash_attn` requirement.
- **Attempted Fix for Dtype Mismatch**: Added logic to cast `pixel_values` to `torch.float16` if it detects a 4-bit model.
  ```python
  is_4bit = "4bit" in self.model_name.lower() ...
  if is_4bit: model_dtype = torch.float16
  inputs[k] = v.to(self.device, dtype=model_dtype)
  ```

### 3. Model Patches (Inside `Model-4bit/`)
The pre-quantized model code had hardcoded dependencies causing crashes. We patched them locally:
- **`Model-4bit/modeling_dots_vision.py`**:
  - Wrapped `from flash_attn import ...` in a try-except block to prevent `ModuleNotFoundError`.
- **`Model-4bit/config.json`**:
  - However, `bnb_4bit_compute_dtype` is still set to `"bfloat16"`, which might be the root cause of the current error.
  - Changed `vision_config.attn_implementation` from `"flash_attention_2"` to `"eager"`.

## Current Issue & Next Steps
The application runs and loads the model, but fails at inference with:
`Inference failed: Input type (c10::BFloat16) and bias type (c10::Half) should be the same`

**Hypothesis:**
The model's `config.json` specifies `"bnb_4bit_compute_dtype": "bfloat16"`, causing internal computations to attempt BFloat16, while the quantized weights usually effectively act as Float16 (Half) or the system expects them to match.

**Task for Next Model:**
Investigate why BFloat16 tensors are still being generated or expected.
1. Check `Model-4bit/config.json` and change `"bnb_4bit_compute_dtype": "bfloat16"` to `"float16"`.
2. Verify if `torch.autocast` is active or if other inputs (like hidden states inside the model) are being cast to BFloat16.
