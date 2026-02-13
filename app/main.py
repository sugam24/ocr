from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.config import settings
from .core.logging import setup_logging
from .api.routes import router
from .api.errors import generic_exception_handler, validation_exception_handler
from .engines.loader import get_model
from .core.messages import LogMessages
import logging

setup_logging()
logger = logging.getLogger("lightonocr")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(LogMessages.SERVICE_STARTUP)
    
    try:
        get_model()
    except Exception as e:
        logger.error(LogMessages.MODEL_INIT_FAIL.format(e))
    
    yield
    
    # Shutdown
    logger.info(LogMessages.SERVICE_SHUTDOWN)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.include_router(router, prefix="")

app.add_exception_handler(Exception, generic_exception_handler)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "Welcome to LightOnOCR Service",
        "docs": "/docs",
        "health": "/health"
    }