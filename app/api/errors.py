from fastapi import Request, status
from fastapi.responses import JSONResponse

async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to prevent stack traces from leaking.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal server error occurred.", "detail": str(exc)},
    )

async def validation_exception_handler(request: Request, exc: Exception):
    """
    Handle validation errors explicitly.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid request", "detail": str(exc)},
    )
