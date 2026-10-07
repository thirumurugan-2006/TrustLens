import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from typing import Optional

# Load environment variables from .env
load_dotenv()

class ModelConfig(BaseModel):
    name: Optional[str] = None
    enabled: bool = False

class Settings(BaseModel):
    app_name: str = "TrustLens"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    model_cache_dir: str = Field(default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "./models"))
    
    language_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name=os.getenv("LANGUAGE_MODEL"), 
            enabled=bool(os.getenv("LANGUAGE_MODEL"))
        )
    )
    embedding_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name=os.getenv("EMBEDDING_MODEL"), 
            enabled=bool(os.getenv("EMBEDDING_MODEL"))
        )
    )
    reranker_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name=os.getenv("RERANKER_MODEL"), 
            enabled=bool(os.getenv("RERANKER_MODEL"))
        )
    )
    graph_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name=os.getenv("GRAPH_MODEL"), 
            enabled=bool(os.getenv("GRAPH_MODEL"))
        )
    )
    ocr_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name=os.getenv("OCR_MODEL", "easyocr"), 
            enabled=True
        )
    )

    vector_index_path: str = Field(default_factory=lambda: os.getenv("VECTOR_INDEX_PATH", "./data/index"))
    evidence_data_path: str = Field(default_factory=lambda: os.getenv("EVIDENCE_DATA_PATH", "./data/evidence"))
    retrieval_top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10")))

    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    risk_threshold: Optional[float] = Field(default_factory=lambda: float(os.getenv("RISK_THRESHOLD")) if os.getenv("RISK_THRESHOLD") else None)
    abstention_threshold: Optional[float] = Field(default_factory=lambda: float(os.getenv("ABSTENTION_THRESHOLD")) if os.getenv("ABSTENTION_THRESHOLD") else None)

settings = Settings()

def setup_logging():
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(settings.app_name)

logger = setup_logging()

def validate_settings():
    if not os.path.exists(settings.model_cache_dir):
        os.makedirs(settings.model_cache_dir, exist_ok=True)
    
    if settings.device not in ["auto", "cpu", "cuda"]:
        logger.warning(f"Invalid DEVICE {settings.device}. Falling back to 'auto'.")
        settings.device = "auto"
    
    logger.info("Configuration validated.")
