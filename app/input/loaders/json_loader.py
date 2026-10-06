import json
from typing import List
from app.input.schemas import UniversalSocialPost
from app.input.platform.adapters.generic_adapter import GenericAdapter

class JSONLoader:
    def load(self, file_path: str) -> List[UniversalSocialPost]:
        adapter = GenericAdapter()
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for record in data:
                try:
                    results.append(adapter.normalize(record))
                except Exception:
                    pass
        return results
