import csv
from typing import List, Dict
from app.schemas import NormalizedPost as UniversalSocialPost
from app.input.platform.adapters.generic_adapter import GenericAdapter

class CSVLoader:
    def load(self, file_path: str, mapping: Dict[str, str]) -> List[UniversalSocialPost]:
        adapter = GenericAdapter()
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapped_row = {"text": row.get(mapping.get("text", "text"), "")}
                try:
                    results.append(adapter.normalize(mapped_row))
                except Exception:
                    pass
        return results
