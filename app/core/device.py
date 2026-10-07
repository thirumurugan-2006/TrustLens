import torch
from app.core.config import settings, logger

def get_device() -> str:
    """
    Returns the appropriate device (cuda or cpu) based on configuration and availability.
    """
    if settings.device.lower() == "cpu":
        return "cpu"
    
    if settings.device.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif settings.device.lower() == "cuda":
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    
    if settings.device.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
        
    logger.warning(f"Unknown device setting '{settings.device}'. Defaulting to auto detection.")
    return "cuda" if torch.cuda.is_available() else "cpu"
