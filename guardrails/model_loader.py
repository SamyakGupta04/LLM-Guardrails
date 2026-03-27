"""
Central Model Loader for Prompt Injection Detection
Loads the ProtectAI/deberta-v3-base-prompt-injection model.
Uses ONNX Runtime via Optimum for low-latency CPU inference.
"""

from transformers import AutoTokenizer
import torch
import os
from typing import List, Dict, Any

# Configure logger
import logging
logger = logging.getLogger("guardrails")

# Global variables for model instance
_model_instance = None
_tokenizer_instance = None
MODEL_NAME = "ProtectAI/deberta-v3-base-prompt-injection"
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")

def get_model():
    """
    Get the shared ProtectAI model and tokenizer.
    Uses ONNX Runtime for high performance and works OFFFLINE.
    """
    global _model_instance, _tokenizer_instance
    if _model_instance is None or _tokenizer_instance is None:
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            
            # Essential files for a complete offline cache
            tokenizer_file = os.path.join(ONNX_PATH, "tokenizer.json")
            model_file = os.path.join(ONNX_PATH, "model.onnx")
            
            if not os.path.exists(ONNX_PATH) or not os.path.exists(model_file):
                logger.info(f"Exporting model to ONNX for caching. INTERNET REQUIRED for this step.")
                _tokenizer_instance = AutoTokenizer.from_pretrained(MODEL_NAME, fix_mistral_regex=True)
                _model_instance = ORTModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, 
                    export=True
                )
                _model_instance.save_pretrained(ONNX_PATH)
                _tokenizer_instance.save_pretrained(ONNX_PATH)
                logger.info(f"Model and tokenizer exported and saved to {ONNX_PATH}")
            elif not os.path.exists(tokenizer_file):
                logger.info(f"Model exists but tokenizer is missing. Repairing cache. INTERNET REQUIRED for this step.")
                _tokenizer_instance = AutoTokenizer.from_pretrained(MODEL_NAME, fix_mistral_regex=True)
                _tokenizer_instance.save_pretrained(ONNX_PATH)
                _model_instance = ORTModelForSequenceClassification.from_pretrained(ONNX_PATH, local_files_only=True)
                logger.info(f"Tokenizer repaired and saved to {ONNX_PATH}")
            else:
                logger.info(f"Loading cached ONNX model and tokenizer from {ONNX_PATH} (Offline mode)...")
                _tokenizer_instance = AutoTokenizer.from_pretrained(ONNX_PATH, local_files_only=True, fix_mistral_regex=True)
                _model_instance = ORTModelForSequenceClassification.from_pretrained(ONNX_PATH, local_files_only=True)
                
            logger.info("ProtectAI model loaded successfully (ONNX/Local).")
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}. Falling back to PyTorch.")
            from transformers import AutoModelForSequenceClassification
            try:
                # Try loading from local PyTorch cache or local path
                _tokenizer_instance = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
                _model_instance = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
            except Exception:
                # Last resort: remote load
                _tokenizer_instance = AutoTokenizer.from_pretrained(MODEL_NAME)
                _model_instance = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _model_instance.eval()
            
    return _model_instance, _tokenizer_instance


def predict_injection(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Single query prediction."""
    results = predict_batch([text], threshold)
    return results[0]


def predict_batch(texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Predict if multiple segments contain prompt injection (Batch Inference).
    """
    if not texts:
        return []
        
    model, tokenizer = get_model()
    
    # Batch tokenization
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        
    results = []
    # Deberta v3 injection labels: 0=Safe, 1=Injection
    for i in range(len(texts)):
        score = float(probabilities[i][1].item())
        is_injection = score >= threshold
        results.append({
            "is_injection": is_injection,
            "score": score,
            "label": "INJECTION" if is_injection else "SAFE"
        })
    return results
