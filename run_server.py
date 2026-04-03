#!/usr/bin/env python3
"""
ARA Auditor API Server Launcher
"""
import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Run the ARA Auditor API server."""
    print("🚀 Starting ARA Auditor v2.0 API Server...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚠️  Make sure Qdrant is running on localhost:6333")
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent / "src"))
    
    # Run the server
    try:
        import uvicorn
        
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install fastapi uvicorn langchain langchain-groq langgraph pydantic pydantic-settings python-dotenv qdrant-client sqlalchemy pyyaml lxml beautifulsoup4 sentence-transformers rank-bm25 python-multipart httpx structlog rich")
        return 1
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
