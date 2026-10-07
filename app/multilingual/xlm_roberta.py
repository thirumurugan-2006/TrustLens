import os
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel

class XLMRobertaEncoder:
    """
    XLM-RoBERTa representation encoder.
    Uses 'xlm-roberta-base' with masked mean pooling.
    """
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lazy loading
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            except Exception:
                try:
                    import huggingface_hub.constants
                    huggingface_hub.constants.HF_HUB_OFFLINE = True
                except Exception:
                    pass
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            self.model.eval()

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts into embeddings using masked mean pooling.
        """
        if not texts:
            return []

        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract last hidden states
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_dim]
        
        # Masked mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_states.size()).float()
        
        sum_embeddings = torch.sum(last_hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings.cpu().tolist()

    @property
    def embedding_dimension(self) -> int:
        return 768
