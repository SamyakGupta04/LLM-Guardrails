from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import httpx

# Load environment variables
load_dotenv()

app = FastAPI()

# Guardrails service URL
GUARDRAILS_URL = os.getenv("GUARDRAILS_URL", "http://localhost:8001")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # First, validate with guardrails
    async with httpx.AsyncClient() as client:
        try:
            guard_response = await client.post(
                f"{GUARDRAILS_URL}/api/validate",
                json={"prompt": request.message},
                timeout=30.0
            )
            guard_data = guard_response.json()
            
            if not guard_data.get("is_valid", True):
                print(f"Refused by guardrails: {guard_data.get('issues')}")
                # Return blocked response with issues
                return JSONResponse(
                    status_code=400,
                    content={
                        "blocked": True,
                        "issues": guard_data.get("issues", []),
                        "message": "Your message was blocked by safety guardrails."
                    }
                )
        except httpx.RequestError as e:
            # If guardrails service is down, log and continue (fail-open)
            print(f"Guardrails service unavailable or timed out: {e}")
    
    # Proceed with LLM call
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        response = model.generate_content(request.message)
        return {"response": response.text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

