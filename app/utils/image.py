import io
from PIL import Image
from typing import List, Union
from pdf2image import convert_from_bytes

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load an image from bytes and convert to RGB.
    If data is PDF, converts first page to image.
    """
    try:
        # Check if PDF signature
        if data.startswith(b"%PDF"):
            # Convert first page only for now as per requirement "multimedia file" 
            # implying single unit processing or we could return list.
            # The model accepts one image.
            images = convert_from_bytes(data)
            if not images:
                raise ValueError("Empty PDF")
            image = images[0]
        else:
            image = Image.open(io.BytesIO(data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid image/PDF data: {e}")

def get_image_format(data: bytes) -> str:
    """
    Detect image format from header bytes.
    """
    if data.startswith(b"%PDF"):
        return "PDF"
    return "IMAGE"
