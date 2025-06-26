#!/usr/bin/env python3
"""
Startup script for FinDoc AI Platform
"""

import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting FinDoc AI Platform on port 8001...")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.info("Trying port 8002...")
        try:
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8002,
                reload=True,
                log_level="info"
            )
        except Exception as e2:
            logger.error(f"Failed to start server on port 8002: {e2}")
            logger.info("Trying port 8003...")
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8003,
                reload=True,
                log_level="info"
            ) 