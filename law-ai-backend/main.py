"""
Law AI Backend - Main Entry Point
Organized project structure with separated concerns.
"""
import os

# Disable telemetry before any LlamaIndex imports
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("LLAMA_INDEX_DISABLE_TELEMETRY", "1")
os.environ.setdefault("LLAMA_INDEX_TELEMETRY_ENABLED", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Config
from app.core.logger import logger
from app.core.models import model_manager
from app.db import init_db
from app.api.routes import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Law AI Assistant",
        description="Australian Family Law AI Assistant with RAG",
        version="2.0.0"
    )

    # CORS configuration
    origins = Config.CORS_ORIGINS_LIST if Config.ENV == "prd" else ["http://localhost:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    @app.on_event("startup")
    async def startup_event():
        """Initialize models and database on startup."""
        logger.info("Starting up Law AI Backend...")
        logger.info("Initializing models and vector indices...")
        model_manager.init_models()
        model_manager.create_or_load_index()
        init_db()
        logger.info("Startup complete!")

    @app.get("/")
    async def root():
        return {
            "message": "Law AI Backend API",
            "version": "2.0.0",
            "status": "running"
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
