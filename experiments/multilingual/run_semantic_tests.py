import json
import time
import math
from app.multilingual.embeddings import embedding_service
from app.multilingual.model_manager import model_manager

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def run_tests():
    texts = {
        "english": "This investment guarantees 20% monthly returns.",
        "tamil": "இந்த முதலீடு மாதத்திற்கு 20% லாபத்தை உறுதி செய்கிறது.",
        "tamil_english": "இந்த investment மிகவும் safe என்று சொல்கிறார்கள்.",
        "tanglish": "Indha investment romba safe nu solranga.",
        "short": "Hi",
        "empty": "",
        "long": "Hello everyone, I recently found out about this amazing opportunity. You can easily make money from home by just referring friends. It's 100% genuine and risk-free. I have already earned a lot of money and thought I should share it with you all. Don't miss this golden chance!"
    }

    results = {}
    
    models_to_test = model_manager.get_available_models()
    
    for model in models_to_test:
        print(f"Running tests for model: {model}")
        model_results = {}
        embeddings = {}
        
        for name, text in texts.items():
            if name == "empty":
                try:
                    embedding_service.encode(text, model_name=model)
                    model_results[name] = "Failed (did not raise error on empty input)"
                except ValueError:
                    model_results[name] = "Passed (raised ValueError correctly)"
                continue
                
            res = embedding_service.encode(text, model_name=model)
            embeddings[name] = res["embedding"]
            model_results[name] = {
                "quality": res["embedding_quality"],
                "latency": res["latency_seconds"],
                "dimension": res["embedding_dimension"]
            }
            
        # Cross language similarity
        sim_en_ta = cosine_similarity(embeddings["english"], embeddings["tamil"])
        sim_te_tan = cosine_similarity(embeddings["tamil_english"], embeddings["tanglish"])
        
        model_results["cross_language_similarities"] = {
            "english_vs_tamil": sim_en_ta,
            "tamil_english_vs_tanglish": sim_te_tan
        }
        
        # Batch test
        batch_res = embedding_service.encode_batch([texts["english"], texts["tamil"]], model_name=model)
        model_results["batch_test"] = {
            "num_results": len(batch_res),
            "latency": batch_res[0]["batch_latency_seconds"] if batch_res else 0
        }
        
        results[model] = model_results
        
    with open("experiments/multilingual/semantic_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to experiments/multilingual/semantic_results.json")

if __name__ == "__main__":
    run_tests()
