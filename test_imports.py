#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all service imports"""
    try:
        logger.info("Testing service imports...")
        
        # Test core imports
        from backend.core.config import settings
        logger.info("✅ Core config imported successfully")
        
        # Test service imports
        from backend.services.nvidia_hpc import NVIDIAHPCManager
        logger.info("✅ NVIDIA HPC Manager imported successfully")
        
        from backend.services.document_processor import DocumentProcessor
        logger.info("✅ Document Processor imported successfully")
        
        from backend.services.predictive_analytics import PredictiveAnalytics
        logger.info("✅ Predictive Analytics imported successfully")
        
        from backend.services.digital_assistant import DigitalAssistant
        logger.info("✅ Digital Assistant imported successfully")
        
        # Test main app import
        import main
        logger.info("✅ Main application imported successfully")
        
        logger.info("🎉 All imports successful! The application is ready to run.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 