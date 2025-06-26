#!/usr/bin/env python3
"""
FinDoc AI - Latin American Bank Document Processing Platform
Main application entry point with FastAPI backend
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.core.config import settings
from backend.services.document_processor import DocumentProcessor
from backend.services.predictive_analytics import PredictiveAnalytics
from backend.services.digital_assistant import DigitalAssistant
from backend.services.nvidia_hpc import NVIDIAHPCManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
document_processor: Optional[DocumentProcessor] = None
predictive_analytics: Optional[PredictiveAnalytics] = None
digital_assistant: Optional[DigitalAssistant] = None
nvidia_hpc: Optional[NVIDIAHPCManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    global document_processor, predictive_analytics, digital_assistant, nvidia_hpc
    
    # Startup
    logger.info("🚀 Starting FinDoc AI Platform...")
    
    try:
        # Initialize NVIDIA HPC manager
        nvidia_hpc = NVIDIAHPCManager()
        await nvidia_hpc.initialize()
        logger.info("✅ NVIDIA HPC initialized successfully")
        
        # Initialize document processor
        document_processor = DocumentProcessor(nvidia_hpc)
        await document_processor.initialize()
        logger.info("✅ Document processor initialized successfully")
        
        # Initialize predictive analytics
        predictive_analytics = PredictiveAnalytics(nvidia_hpc)
        await predictive_analytics.initialize()
        logger.info("✅ Predictive analytics initialized successfully")
        
        # Initialize digital assistant
        digital_assistant = DigitalAssistant(nvidia_hpc)
        await digital_assistant.initialize()
        logger.info("✅ Digital assistant initialized successfully")
        
        logger.info("🎉 FinDoc AI Platform started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize FinDoc AI Platform: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FinDoc AI Platform...")
    
    if digital_assistant:
        await digital_assistant.cleanup()
    if predictive_analytics:
        await predictive_analytics.cleanup()
    if document_processor:
        await document_processor.cleanup()
    if nvidia_hpc:
        await nvidia_hpc.cleanup()
    
    logger.info("✅ FinDoc AI Platform shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="FinDoc AI - Latin American Bank Document Processing Platform",
    description="Advanced document processing and predictive analytics for Latin American banks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentResponse(BaseModel):
    document_id: str
    status: str
    extracted_data: dict
    confidence_score: float
    processing_time: float

class RiskScoreRequest(BaseModel):
    document_id: str
    customer_data: dict
    loan_amount: float
    loan_type: str

class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_category: str
    confidence: float
    factors: List[str]
    recommendations: List[str]

class AssistantQuery(BaseModel):
    query: str
    document_context: Optional[str] = None
    language: str = "es"

class AssistantResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    suggested_actions: List[str]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "FinDoc AI Platform is running! 🏦",
        "version": "1.0.0",
        "status": "healthy",
        "features": [
            "Document Processing",
            "Predictive Analytics", 
            "Digital Assistant",
            "NVIDIA HPC Integration"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "document_processor": document_processor is not None,
            "predictive_analytics": predictive_analytics is not None,
            "digital_assistant": digital_assistant is not None,
            "nvidia_hpc": nvidia_hpc is not None
        },
        "gpu_info": await nvidia_hpc.get_gpu_info() if nvidia_hpc else None
    }


@app.post("/api/v1/documents/process", response_model=DocumentResponse)
async def process_document(file: UploadFile = File(...)):
    """Process uploaded financial document"""
    try:
        if not document_processor:
            raise HTTPException(status_code=503, detail="Document processor not available")
        
        # Validate file type
        allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_types}"
            )
        
        # Process document
        result = await document_processor.process_document(file)
        
        return DocumentResponse(
            document_id=result["document_id"],
            status="completed",
            extracted_data=result["extracted_data"],
            confidence_score=result["confidence_score"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/risk-score", response_model=RiskScoreResponse)
async def calculate_risk_score(request: RiskScoreRequest):
    """Calculate risk score for loan application"""
    try:
        if not predictive_analytics:
            raise HTTPException(status_code=503, detail="Predictive analytics not available")
        
        result = await predictive_analytics.calculate_risk_score(
            document_id=request.document_id,
            customer_data=request.customer_data,
            loan_amount=request.loan_amount,
            loan_type=request.loan_type
        )
        
        return RiskScoreResponse(
            risk_score=result["risk_score"],
            risk_category=result["risk_category"],
            confidence=result["confidence"],
            factors=result["factors"],
            recommendations=result["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/assistant/query", response_model=AssistantResponse)
async def query_assistant(query: AssistantQuery):
    """Query the digital assistant"""
    try:
        if not digital_assistant:
            raise HTTPException(status_code=503, detail="Digital assistant not available")
        
        result = await digital_assistant.process_query(
            query=query.query,
            document_context=query.document_context,
            language=query.language
        )
        
        return AssistantResponse(
            response=result["response"],
            confidence=result["confidence"],
            sources=result["sources"],
            suggested_actions=result["suggested_actions"]
        )
        
    except Exception as e:
        logger.error(f"Error processing assistant query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        if not predictive_analytics:
            raise HTTPException(status_code=503, detail="Predictive analytics not available")
        
        dashboard_data = await predictive_analytics.get_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/nvidia/status")
async def get_nvidia_status():
    """Get NVIDIA HPC status and GPU information"""
    try:
        if not nvidia_hpc:
            raise HTTPException(status_code=503, detail="NVIDIA HPC not available")
        
        status_info = await nvidia_hpc.get_status()
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting NVIDIA status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
