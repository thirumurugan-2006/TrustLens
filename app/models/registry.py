from typing import Dict, Any, Callable
from app.core.config import logger

class ModelRegistry:
    def __init__(self):
        self._builders: Dict[str, Callable[..., Any]] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, model_type: str, name: str, builder: Callable[..., Any]):
        key = f"{model_type}:{name}"
        self._builders[key] = builder
        logger.debug(f"Registered model builder for {key}")

    def get(self, model_type: str, name: str, **kwargs) -> Any:
        key = f"{model_type}:{name}"
        if key not in self._instances:
            if key not in self._builders:
                raise ValueError(f"No model registered for {key}")
            logger.info(f"Loading model {key}")
            self._instances[key] = self._builders[key](**kwargs)
        else:
            logger.debug(f"Reusing cached model {key}")
        return self._instances[key]

# Global registry instance
registry = ModelRegistry()
