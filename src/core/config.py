import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with comprehensive security and performance settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # === LLM Configuration ===
    GROQ_API_KEY: str = Field(..., description="Groq API key for LLM inference")
    DEFAULT_MODEL: str = Field(default="gpt-oss-120b", description="Default model name")
    MAX_TOKENS: int = Field(default=4000, description="Maximum tokens per LLM call")
    TEMPERATURE: float = Field(default=0.1, description="LLM temperature for deterministic outputs")
    
    # === Vector Database Configuration ===
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant vector database URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key if required")
    VECTOR_COLLECTION: str = Field(default="ara_regulations", description="Qdrant collection name")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    VECTOR_SIZE: int = Field(default=384, description="Vector embedding dimension")
    
    # === Database Configuration ===
    DATABASE_URL: str = Field(default="sqlite:///./data/audit_checkpoints.db", description="Database connection URL")
    DATABASE_POOL_SIZE: int = Field(default=5, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="Database max overflow connections")
    
    # === Application Configuration ===
    PROJECT_NAME: str = Field(default="ARA_Auditor_v2", description="Application name")
    VERSION: str = Field(default="2.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # === Security Configuration ===
    SECRET_KEY: str = Field(..., description="Secret key for JWT signing")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration time")
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Maximum file upload size in MB")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".html"], 
        description="Allowed file types for upload"
    )
    
    # === Performance Configuration ===
    MAX_CONCURRENT_AUDITS: int = Field(default=10, description="Maximum concurrent audit processes")
    RETRIEVAL_TOP_K: int = Field(default=20, description="Number of documents to retrieve")
    RERANK_TOP_K: int = Field(default=5, description="Number of documents after reranking")
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Cache TTL in seconds")
    
    # === LangGraph Configuration ===
    CHECKPOINTER_DB_PATH: str = Field(default="./data/audit_checkpoints.db", description="LangGraph checkpointer database")
    WORKFLOW_TIMEOUT_SECONDS: int = Field(default=1800, description="Workflow timeout in seconds")
    
    @validator("GROQ_API_KEY", "SECRET_KEY")
    def validate_required_keys(cls, v):
        if not v or len(v) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v
    
    @validator("TEMPERATURE")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("MAX_TOKENS")
    def validate_max_tokens(cls, v):
        if v <= 0 or v > 32000:
            raise ValueError("MAX_TOKENS must be between 1 and 32000")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG and self.LOG_LEVEL == "INFO"
    
    @property
    def llm_config(self) -> dict:
        """Get LLM configuration dictionary."""
        return {
            "api_key": self.GROQ_API_KEY,
            "model": self.DEFAULT_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
        }


# Global settings instance
settings = Settings()