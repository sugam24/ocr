import logging
import sys

def setup_logging():
    """
    Configure the logging system to utilize standard output.
    """
    import os
    from logging.handlers import RotatingFileHandler
    
    # Ensure logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Add Rotating File Handler
    # 10MB per file, keep last 5 backups
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "lightonocr.log"), 
        maxBytes=10*1024*1024, 
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    from .messages import LogMessages
    logger = logging.getLogger("lightonocr")
    logger.info(LogMessages.LOGGING_INITIALIZED)
