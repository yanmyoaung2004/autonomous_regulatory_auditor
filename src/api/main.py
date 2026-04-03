import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from src.core.config import settings
from src.core.graph.workflow import run_audit, resume_audit, get_audit_state, list_active_audits
from src.services.vector.qdrant_service import qdrant_service
from src.api.routes import router as audit_router


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ARA Auditor API v2.0")
    
    # Initialize vector database
    try:
        await qdrant_service.initialize_collection()
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize vector database", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ARA Auditor API v2.0")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Autonomous Regulatory Auditor (ARA) v2.0 - Multi-Agent LangGraph System for Automated Legal Compliance in SE Asia",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check API and service health."""
    try:
        # Check vector database health
        vector_health = await qdrant_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.VERSION,
            "services": {
                "vector_db": vector_health,
                "api": "healthy"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ARA Auditor API v2.0",
        "version": settings.VERSION,
        "description": "Autonomous Regulatory Auditor for Southeast Asian Compliance",
        "docs": "/docs" if settings.DEBUG else "Documentation not available in production",
        "health": "/health"
    }


# Include audit routes
app.include_router(audit_router, prefix="/api/v1", tags=["Audit"])


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured logging."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing information."""
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time=process_time,
        client_ip=request.client.host if request.client else None
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
