"""
FastAPI application entry point.
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load repo-root .env (TYPESAFE_API_KEY / JEV_API_KEY) if present.
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")
load_dotenv()  # also allow cwd override

from .routes import router

app = FastAPI(
    title="Soccer Simulation API",
    description="Turn-based 5v5 physics soccer game",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "game": "soccer-sim", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
