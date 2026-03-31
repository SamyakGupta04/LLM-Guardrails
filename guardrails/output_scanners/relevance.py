"""
Relevance Scanner
Verifies that LLM output is semantically relevant to the input prompt.

Uses embedding-based cosine similarity to detect off-topic or irrelevant responses.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("guardrails")

# Model configuration
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.65

# Get the directory where this file is located, then go up to llmguard
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_LLMGUARD_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
MODEL_CACHE_DIR = os.path.join(_LLMGUARD_DIR, "models")


# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None


def _ensure_model_loaded():
    """Lazy-load the embedding model and tokenizer."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        raise ImportError("Please install transformers and torch: pip install transformers torch")
    
    # Ensure cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    logger.info(f"Loading embedding model to {MODEL_CACHE_DIR}...")
    
    _tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    _model = AutoModel.from_pretrained(
        DEFAULT_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    # Set to evaluation mode
    _model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    
    logger.info(f"Embedding model loaded successfully on {device}")


def _encode(text: str, max_length: int = 512) -> np.ndarray:
    """
    Encode text into a normalized embedding vector.
    Uses CLS token pooling and L2 normalization.
    """
    import torch
    
    _ensure_model_loaded()
    
    device = next(_model.parameters()).device
    
    # Tokenize
    inputs = _tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = _model(**inputs, return_dict=True)
        # CLS pooling (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        embeddings = embeddings.cpu().numpy()
    
    return embeddings[0]


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts.
    
    Returns:
        Similarity score between -1 and 1 (higher = more similar).
    """
    emb1 = _encode(text1)
    emb2 = _encode(text2)
    
    # Cosine similarity (embeddings are already normalized)
    similarity = float(np.dot(emb1, emb2))
    return similarity


def detect_irrelevance(
    prompt: str,
    output: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Detect if LLM output is irrelevant to the input prompt.
    
    Args:
        prompt: The user's input prompt.
        output: The LLM's generated output.
        threshold: Minimum similarity score (0-1). Default 0.5.
    
    Returns:
        List of issues if output is irrelevant, empty list if relevant.
    """
    if not prompt or not prompt.strip():
        # No prompt provided, skip relevance check
        return []
    
    if not output or not output.strip():
        # Empty output, skip (handled elsewhere)
        return []
    
    try:
        similarity = compute_similarity(prompt, output)
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return []  # Fail open on errors
    
    if similarity < threshold:
        logger.warning(
            f"Output relevance below threshold: {similarity:.3f} < {threshold}"
        )
        return [
            {
                "guard": "relevance_scanner",
                "description": f"Output may not be relevant to the prompt (similarity: {similarity:.2f})",
                "score": 1.0 - similarity,  # Risk score (higher = worse)
            }
        ]
    
    logger.debug(f"Output is relevant to prompt (similarity: {similarity:.3f})")
    return []


# Pre-load model on import (optional, can be disabled for faster startup)
# Uncomment the line below to enable eager loading:
# _ensure_model_loaded()
