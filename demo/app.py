"""
SAHAYAK-TB — FastAPI Backend + Static Frontend.

Serves:
  - Static HTML/CSS/JS frontend at /
  - POST /api/assess — inference endpoint

Deployment:
  - Local: python app.py
  - HF Spaces: Docker (see Dockerfile)
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Add parent directory to path for imports (works locally and in Docker)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from inference.postprocess import correct_output
from inference.schema import SYSTEM_PROMPT, RISK_ASSESSMENT_SCHEMA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_REPO = os.environ.get("SAHAYAK_MODEL_REPO", "Roopalgn/sahayak-tb-gemma4-e2b-gguf")
MODEL_FILENAME = os.environ.get("SAHAYAK_MODEL_FILENAME", "gemma-4-e2b-it.Q4_K_M.gguf")
MODEL_PATH = os.environ.get("SAHAYAK_MODEL_PATH", "")

if not MODEL_PATH:
    print(f"Downloading model from {MODEL_REPO}/{MODEL_FILENAME}...")
    MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    print(f"Model downloaded to: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------
llm = None
_model_loading = False

def get_model():
    global llm
    if llm is None:
        from llama_cpp import Llama
        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=4096,
            verbose=False,
        )
        print("Model loaded.")
    return llm

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="SAHAYAK-TB", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Load model in background thread so server starts serving immediately."""
    import threading
    def _load():
        global _model_loading
        _model_loading = True
        get_model()
        _model_loading = False
    threading.Thread(target=_load, daemon=True).start()

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AssessRequest(BaseModel):
    visit_notes: str


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/assess")
async def assess_risk(request: AssessRequest):
    if not request.visit_notes.strip():
        raise HTTPException(status_code=400, detail="Visit notes cannot be empty")

    if llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait 1-2 minutes and try again.")

    model = llm

    try:
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.visit_notes},
            ],
            response_format={
                "type": "json_object",
                "schema": RISK_ASSESSMENT_SCHEMA,
            },
            temperature=0.0,
            max_tokens=1024,
        )
        raw = response["choices"][0]["message"]["content"]
        result = json.loads(raw)
        result = correct_output(result, user_text=request.visit_notes)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    if llm is not None:
        return {"status": "ok", "model_loaded": True}
    elif _model_loading:
        return {"status": "loading", "model_loaded": False}
    else:
        return {"status": "idle", "model_loaded": False}


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

