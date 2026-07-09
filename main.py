from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.search import router as search_router

app = FastAPI(title="Zatch Search API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)


@app.get("/")
def home():
    return {"message": "Zatch Search API v3 running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}