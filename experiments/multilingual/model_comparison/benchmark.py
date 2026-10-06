import time
import json
import torch
import sys
import os

# Add to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.multilingual.xlm_roberta import XLMRobertaEncoder
from app.multilingual.muril import MurilEncoder

test_cases = {
    "English": "ABC Wealth guarantees 20% monthly returns.",
    "Tamil": "ABC Wealth நிறுவனம் மாதம் 20% வருமானம் வழங்கும் என்று உறுதி செய்கிறது.",
    "Hindi": "ABC Wealth हर महीने 20% रिटर्न की गारंटी देता है।",
    "Tamil-English": "ABC Wealth monthly 20% return guarantee panranga.",
    "Tanglish": "ABC Wealth monthly 20% return guarantee panranga nu solranga.",
    "Hinglish": "ABC Wealth har month 20% return guarantee karta hai."
}

def run_benchmark():
    results = {}
    
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    for model_class, name in [(XLMRobertaEncoder, "XLM-R"), (MurilEncoder, "MuRIL")]:
        print(f"\nEvaluating {name}...")
        try:
            model = model_class()
            model._load_model()
            
            # warmup
            _ = model.encode(["warmup"])
            
            res = {}
            for lang, text in test_cases.items():
                start = time.time()
                emb = model.encode([text])
                end = time.time()
                
                res[lang] = {
                    "success": True,
                    "embedding_dimension": len(emb[0]),
                    "latency_ms": round((end - start) * 1000, 2)
                }
            
            results[name] = res
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}
            
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("Benchmark complete. Results saved to results.json.")

if __name__ == "__main__":
    run_benchmark()
