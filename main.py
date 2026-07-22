import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.embeddings import get_model
from app.search import router as search_router

logger = logging.getLogger("main")

app = FastAPI(title="Zatch Search API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.mount("/demo", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="demo")


@app.on_event("startup")
def warm_up_embedding_model():
    """Loads the local embedding model at startup instead of on the first
    search request -- otherwise the first user's query could time out
    against get_query_vector()'s 2s budget while the model is still
    loading, and silently fall back to keyword-only for no good reason."""
    try:
        get_model()
        logger.info("Embedding model loaded and ready.")
    except Exception as e:
        logger.warning("Embedding model failed to preload (semantic search "
                        "will retry lazily on first query): %s", e)


@app.get("/")
def home():
    return RedirectResponse(url="/demo/")


@app.get("/health")
def health():
    return {"status": "ok"}