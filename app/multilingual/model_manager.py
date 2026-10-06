from typing import Optional, List, Dict
import logging

from app.multilingual.xlm_roberta import XLMRobertaEncoder
from app.multilingual.muril import MurilEncoder

logger = logging.getLogger(__name__)

class MultilingualModelManager:
    """
    Manages loading and caching of multilingual representation models.
    """
    def __init__(self):
        self._xlm_roberta = None
        self._muril = None
        self._available_models = ["xlm-roberta"]
        
        # We assume XLM-R is always available if transformers is installed.
        # We can try to test MuRIL availability.
        self._check_availability()

    def _check_availability(self):
        # We won't eagerly download MuRIL here, just mark it as available 
        # to try if requested.
        self._available_models.append("muril")

    def get_xlm_roberta(self) -> XLMRobertaEncoder:
        if self._xlm_roberta is None:
            logger.info("Initializing XLM-RoBERTa encoder...")
            self._xlm_roberta = XLMRobertaEncoder()
        return self._xlm_roberta

    def get_muril(self) -> Optional[MurilEncoder]:
        if self._muril is None:
            try:
                logger.info("Initializing MuRIL encoder...")
                self._muril = MurilEncoder()
            except Exception as e:
                logger.error(f"Failed to load MuRIL model: {e}")
                if "muril" in self._available_models:
                    self._available_models.remove("muril")
                return None
        return self._muril

    def get_available_models(self) -> List[str]:
        return self._available_models.copy()

    def get_model(self, model_name: str):
        if model_name == "muril":
            return self.get_muril()
        elif model_name == "xlm-roberta":
            return self.get_xlm_roberta()
        else:
            raise ValueError(f"Unknown model: {model_name}")

# Global instance
model_manager = MultilingualModelManager()
