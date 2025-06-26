"""
Configuration settings for FinDoc AI Platform
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "FinDoc AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    database_url: str = "postgresql://user:password@localhost/findoc_ai"
    redis_url: str = "redis://localhost:6379"
    
    # NVIDIA HPC
    nvidia_gpu_enabled: bool = True
    nvidia_gpu_memory_fraction: float = 0.8
    nvidia_mixed_precision: bool = True
    
    # Document Processing
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list = ["pdf", "jpg", "jpeg", "png", "tiff"]
    ocr_languages: list = ["spa", "eng", "por"]
    
    # AI Models
    model_cache_dir: str = "./models"
    use_gpu: bool = True
    batch_size: int = 32
    
    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Vector Database
    weaviate_url: str = "http://localhost:8080"
    vector_dimension: int = 768
    
    # Stream Processing
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_documents: str = "documents"
    kafka_topic_analytics: str = "analytics"
    
    # Regional Settings
    default_language: str = "es"
    supported_languages: list = ["es", "en", "pt"]
    timezone: str = "America/Mexico_City"
    
    # File Storage
    upload_dir: str = "./uploads"
    processed_dir: str = "./processed"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "WARNING"
    nvidia_gpu_memory_fraction: float = 0.9


class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    database_url: str = "postgresql://test:test@localhost/findoc_ai_test"
    use_gpu: bool = False


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export settings
settings = get_settings() 