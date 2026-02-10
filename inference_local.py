#!/usr/bin/env python3
"""
Local Inference Script for dots.ocr AWQ 4-bit Model
Supports uploading images/documents for OCR.

Usage:
    python inference_local.py --image path/to/image.png
    python inference_local.py --image path/to/document.pdf
    python inference_local.py  # Opens file dialog
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer


# ============ CONFIGURATION ============
# Change this to your HuggingFace model ID or local path
MODEL_ID = "sugam24/dots-ocr-awq-4bit"  # or local path like "./dots_ocr_awq_4bit"
DEFAULT_IMAGE = "data/doc1.jpeg"  # Default image for testing
# =======================================


def load_model(model_id: str, device: str = "cuda"):
    """Load the quantized model and processor."""
    print(f"Loading model from {model_id}...")
    
    # Patch the cached model code to handle CompressedLinear layers
    import importlib
    import sys
    
    # Pre-load the model config to get the module cached
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Find and patch the modeling module in the cache
    for module_name in list(sys.modules.keys()):
        if "modeling_dots_vision" in module_name:
            module = sys.modules[module_name]
            if hasattr(module, "DotsVisionModel"):
                original_init_weights = module.DotsVisionModel._init_weights
                def patched_init_weights(self, module):
                    # Skip CompressedLinear layers - they don't have .weight
                    if type(module).__name__ == "CompressedLinear":
                        return
                    return original_init_weights(self, module)
                module.DotsVisionModel._init_weights = patched_init_weights
                print("✓ Patched _init_weights for CompressedLinear compatibility")
                break

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    # NOTE: `device_map="cuda"` is NOT a valid value in Transformers.
    # Use `device_map="auto"` for GPU, or load then `.to("cpu")` for CPU.
    max_memory = None
    if use_cuda:
        # Keep some headroom to reduce OOMs; allow CPU offload if needed.
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_budget_gib = max(1.0, total_gib * 0.80)
        # Accelerate expects GPU keys as integers (device indices).
        max_memory = {0: f"{cuda_budget_gib:.1f}GiB", "cpu": "64GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if not use_cuda:
        model = model.to("cpu", dtype=torch.float32)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Some chatty VLMs don't define a pad token; generation may warn/error otherwise.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    return model, tokenizer, image_processor


def load_image(image_path: str, max_side: int | None = 1024) -> Image.Image:
    """Load an image from path. Supports common formats.

    `max_side` downsizes large inputs to reduce VRAM usage.
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Handle PDF (first page only)
    if path.suffix.lower() == ".pdf":
        try:
            import pdf2image
            images = pdf2image.convert_from_path(str(path), first_page=1, last_page=1)
            return images[0]
        except ImportError:
            print("For PDF support, install: pip install pdf2image")
            print("Also requires poppler: apt-get install poppler-utils")
            sys.exit(1)
    
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        if max(w, h) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def run_ocr(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str = None,
    max_new_tokens: int = 1024,
) -> str:
    """Run OCR on an image and return extracted text."""
    
    if prompt is None:
        prompt = "Extract all the text from this image."
    
    # Prepare the conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize prompt + encode image. We avoid `AutoProcessor` here because some
    # Transformers builds error when a `video_processor=None` is present.
    text_inputs = tokenizer([text], padding=True, return_tensors="pt")
    image_inputs = image_processor(images=image, return_tensors="pt")
    inputs = {**text_inputs, **image_inputs}
    
    # Move to device
    device = next(model.parameters()).device
    # Some modules/params may be bf16 on some installs; keep a stable dtype.
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(device)
            # Ensure vision tensors match model dtype (prevents bf16/fp16 mismatch)
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                v = v.to(model_dtype)
        moved[k] = v
    inputs = moved
    
    # Generate
    with torch.no_grad():
        # Autocast to model dtype for safety on GPU
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    # Decode output
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return result.strip()


def format_markdown_result(model_id: str, image_path: str, image: Image.Image, prompt: str, extracted_text: str) -> str:
    return (
        "# OCR Results\n\n"
        "## Inference Details\n"
        f"- **Model**: {model_id}\n"
        f"- **Input**: {image_path}\n"
        f"- **Image Size**: {image.size}\n\n"
        "## Prompt\n"
        "```text\n"
        f"{prompt}\n"
        "```\n\n"
        "## Extracted Text\n"
        "```text\n"
        f"{extracted_text}\n"
        "```\n"
    )


def select_file_dialog() -> str:
    """Open a file dialog to select an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select Image or Document",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ]
        )
        
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Could not open file dialog: {e}")
        return input("Enter image path: ").strip()


def main():
    parser = argparse.ArgumentParser(description="OCR with dots.ocr model")
    parser.add_argument("--image", "-i", type=str, default=DEFAULT_IMAGE, help="Path to image file")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt")
    parser.add_argument("--model", "-m", type=str, default=MODEL_ID, help="Model ID or path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--out", "-o", type=str, default=None, help="Optional path to save Markdown output")
    parser.add_argument("--max-side", type=int, default=1024, help="Downscale input so max(width,height)<=max-side (reduces VRAM)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate")
    args = parser.parse_args()
    
    # Get image path
    image_path = args.image
    print(f"Using image: {image_path}")
    
    # Load model
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda"
        if torch.cuda.is_available():
            total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_gib < 8.0:
                print(f"⚠️ GPU VRAM is {total_gib:.1f} GiB; this model often OOMs on <8 GiB.")
                print("   Falling back to CPU. (Use a larger GPU to run on CUDA.)")
                device = "cpu"
        else:
            device = "cpu"
    model, tokenizer, image_processor = load_model(args.model, device)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path, max_side=args.max_side)
    print(f"Image size: {image.size}")
    
    # Run OCR
    print("\nRunning OCR...")
    used_prompt = args.prompt or "Extract all the text from this image."
    result = run_ocr(
        model,
        tokenizer,
        image_processor,
        image,
        used_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    
    print("\n" + "="*50)
    print("EXTRACTED TEXT:")
    print("="*50)
    print(result)
    print("="*50)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = format_markdown_result(args.model, image_path, image, used_prompt, result)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n✓ Saved Markdown to: {out_path}")
    
    return result


if __name__ == "__main__":
    main()

