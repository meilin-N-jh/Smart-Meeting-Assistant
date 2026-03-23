"""Smart Meeting Assistant - Main FastAPI Application."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from loguru import logger

from backend.core.config import settings
from backend.api.routes import router


# Configure logging
def setup_logging():
    """Configure application logging."""
    logger.remove()
    logger.add(
        "meeting_assistant.log",
        rotation="10 MB",
        retention="7 days",
        level=settings.log_level,
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=settings.log_level,
    )


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Smart Meeting Assistant starting up...")
    logger.info(f"vLLM: {settings.vllm_base_url}")
    logger.info(f"Model: {settings.vllm_model}")

    # Check if vLLM is available
    try:
        from backend.services import get_llm_client
        llm = get_llm_client()
        logger.info("LLM client initialized")
    except Exception as e:
        logger.warning(f"LLM not available: {e}")

    yield

    # Shutdown
    logger.info("Smart Meeting Assistant shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Smart Meeting Assistant",
    description="AI-powered meeting transcription, summarization, and analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    """Avoid stale inline frontend assets during local development."""
    response = await call_next(request)
    content_type = str(response.headers.get("content-type", "")).lower()
    if request.url.path == "/" or content_type.startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Include API routes
app.include_router(router)

# Mount frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


# Root endpoint - serve frontend
@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(frontend_path.read_text())
    return {"message": "Smart Meeting Assistant API", "docs": "/docs"}


def main():
    """Run the application."""
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
