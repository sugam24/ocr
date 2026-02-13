import io
from PIL import Image
from typing import List

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load an image from bytes and convert to RGB.
    If data is PDF, converts first page to image at 200 DPI.
    """
    try:
        # Check if PDF signature
        if data.startswith(b"%PDF"):
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(data)
            if len(pdf) == 0:
                raise ValueError("Empty PDF")
            page = pdf[0]
            # Render at 200 DPI (scale factor = 200/72 ≈ 2.77)
            # as recommended by LightOnOCR preprocessing tips
            image = page.render(scale=2.77).to_pil()
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
