"""
Guardrails Service
Exposes /api/validate endpoint to validate prompts before LLM processing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from input_scanners.pii_detector import detect_pii
from input_scanners.injection_detector import detect_injection
from input_scanners.code_detector import detect_code
from input_scanners.competitor_detector import detect_competitors
from input_scanners.token_limit_detector import detect_token_limit
from input_scanners.invisible_text_detector import detect_invisible_text

from output_scanners.deanonymization import detect_deanonymization
from output_scanners.malicious_url_detector import detect_malicious_urls
from output_scanners.factual_consistency import detect_factual_inconsistency
from output_scanners.relevance import detect_irrelevance
from output_scanners.json_scanner import detect_json_syntax
from output_scanners.ban_topics import detect_banned_topics


app = FastAPI(title="Guardrails Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValidateRequest(BaseModel):
    prompt: str


class ValidateOutputRequest(BaseModel):
    output: str
    prompt: Optional[str] = None
    check_relevance: bool = True  # Enable/disable relevance check
    check_json: bool = False      # Enable/disable JSON syntax check
    check_banned_topics: bool = True # Enable/disable banned topics check


class ValidateRAGOutputRequest(BaseModel):
    """Request for RAG output validation with context."""
    output: str
    context: str
    strict: bool = False  # If True, require all claims to be supported


class ValidationIssue(BaseModel):
    guard: str
    description: str
    matched: Optional[str] = None
    pii_type: Optional[str] = None
    score: Optional[float] = None


class ValidateResponse(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue]


@app.post("/api/validate", response_model=ValidateResponse)
def validate_prompt(request: ValidateRequest):
    """Validate a prompt against all (input) guardrails."""
    issues = []
    
    issues.extend(detect_pii(request.prompt))
    issues.extend(detect_injection(request.prompt))
    issues.extend(detect_code(request.prompt))
    issues.extend(detect_competitors(request.prompt))
    issues.extend(detect_token_limit(request.prompt))
    issues.extend(detect_invisible_text(request.prompt))
    
    return ValidateResponse(
        is_valid=len(issues) == 0,
        issues=[ValidationIssue(**issue) for issue in issues]
    )


@app.post("/api/validate_output", response_model=ValidateResponse)
def validate_output(request: ValidateOutputRequest):
    """Validate an output against all output guardrails."""
    issues = []
    
    # Check relevance if prompt is provided and check_relevance is enabled
    if request.prompt and request.check_relevance:
        issues.extend(detect_irrelevance(request.prompt, request.output))
    
    # Check JSON syntax if enabled
    if request.check_json:
        issues.extend(detect_json_syntax(request.output, repair=True))
    
    # Check banned topics if enabled
    if request.check_banned_topics:
        issues.extend(detect_banned_topics(request.output))
    
    issues.extend(detect_deanonymization(request.output))
    issues.extend(detect_malicious_urls(request.output))
    
    return ValidateResponse(
        is_valid=len(issues) == 0,
        issues=[ValidationIssue(**issue) for issue in issues]
    )


@app.post("/api/validate_rag_output", response_model=ValidateResponse)
def validate_rag_output(request: ValidateRAGOutputRequest):
    """
    Validate RAG output for factual consistency with context.
    
    This endpoint checks if the LLM output is grounded in the provided context.
    Use this for RAG pipelines where output should be faithful to retrieved documents.
    """
    issues = []
    
    # Run factual consistency check
    issues.extend(detect_factual_inconsistency(
        output=request.output,
        context=request.context,
        strict_all_claims=request.strict,
    ))
    
    # Also run other output scanners
    issues.extend(detect_deanonymization(request.output))
    issues.extend(detect_malicious_urls(request.output))
    
    return ValidateResponse(
        is_valid=len(issues) == 0,
        issues=[ValidationIssue(**issue) for issue in issues]
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "guardrails"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

