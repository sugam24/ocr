from pydantic import BaseModel
from typing import List, Optional, Any

class OCRBlock(BaseModel):
    """Reserved for future structured output (e.g., bounding boxes, categories)."""
    bbox: Optional[List[int]] = None
    category: Optional[str] = None
    text: Optional[str] = None

class OCRResponse(BaseModel):
    text: str
    blocks: List[OCRBlock] = []
    model_version: str
