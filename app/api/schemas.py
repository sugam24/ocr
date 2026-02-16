from pydantic import BaseModel
from typing import List, Optional, Any

class OCRBlock(BaseModel):
    # Depending on model output, we might have bbox, text, confidence here.
    # For now, generic placeholder as model output is raw text currently.
    # Updated to match Dots OCR output
    bbox: Optional[List[int]] = None # [x1, y1, x2, y2]
    category: Optional[str] = None
    text: Optional[str] = None

class OCRResponse(BaseModel):
    text: str
    blocks: List[OCRBlock] = []
    model_version: str
