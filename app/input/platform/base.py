from abc import ABC, abstractmethod
from app.input.schemas import UniversalSocialPost
from typing import Dict, Any

class BasePlatformAdapter(ABC):
    @abstractmethod
    def retrieve(self, identifier: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        pass
        
    @abstractmethod
    def normalize(self, raw_data: Dict[str, Any]) -> UniversalSocialPost:
        pass
