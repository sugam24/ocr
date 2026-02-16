from fastapi import APIRouter, UploadFile, File, HTTPException
from ..core.config import settings
from ..engines.loader import get_model
from .schemas import OCRResponse

router = APIRouter()

@router.post("/api/inference", response_model=OCRResponse, tags=["ocr"])
async def predict_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on an uploaded image or PDF file.
    """
    # 1. Validate file size (Reader check)
    # Note: real content-length check happens on read, but we can check header first
    filesize = file.size
    if filesize and filesize > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
         raise HTTPException(status_code=413, detail="File too large")

    content = await file.read()
    
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # 2. Get Model
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Model not ready")

    # 3. Predict
    try:
        result = await model.predict(content)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log generic error in middleware/handler
        raise HTTPException(status_code=500, detail="Inference failed")

@router.get("/info", tags=["info"])
def get_service_info():
    """
    Returns information about the running OCR service.
    """
    return {
        "engine": settings.MODEL_SOURCE,
        "device": settings.DEVICE,
        "model": settings.MODEL_NAME,
        "api_version": "v1"
    }
