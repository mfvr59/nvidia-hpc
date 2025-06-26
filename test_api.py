#!/usr/bin/env python3
"""
Test script for FinDoc AI Platform API
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_nvidia_status():
    """Test NVIDIA status endpoint"""
    print("\n🖥️  Testing NVIDIA status...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/nvidia/status")
        if response.status_code == 200:
            print("✅ NVIDIA status endpoint working")
            data = response.json()
            print(f"   Initialized: {data.get('initialized', False)}")
            print(f"   NVIDIA Available: {data.get('nvidia_available', False)}")
            return True
        else:
            print(f"❌ NVIDIA status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ NVIDIA status error: {e}")
        return False

def test_analytics_dashboard():
    """Test analytics dashboard endpoint"""
    print("\n📊 Testing analytics dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/analytics/dashboard")
        if response.status_code == 200:
            print("✅ Analytics dashboard working")
            data = response.json()
            print(f"   Total Applications: {data.get('total_applications', 0)}")
            print(f"   Average Risk Score: {data.get('average_risk_score', 0)}")
            return True
        else:
            print(f"❌ Analytics dashboard failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Analytics dashboard error: {e}")
        return False

def test_assistant_query():
    """Test digital assistant query"""
    print("\n🤖 Testing digital assistant...")
    try:
        query_data = {
            "query": "¿Cómo funciona la evaluación de riesgo?",
            "language": "es"
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/assistant/query",
            json=query_data
        )
        if response.status_code == 200:
            print("✅ Digital assistant working")
            data = response.json()
            print(f"   Response: {data.get('response', '')[:100]}...")
            print(f"   Confidence: {data.get('confidence', 0)}")
            return True
        else:
            print(f"❌ Digital assistant failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Digital assistant error: {e}")
        return False

def test_risk_score_calculation():
    """Test risk score calculation"""
    print("\n📈 Testing risk score calculation...")
    try:
        risk_data = {
            "document_id": "test-doc-123",
            "customer_data": {
                "age": 35,
                "income": 50000,
                "credit_score": 650,
                "debt_to_income": 0.3,
                "employment_years": 5
            },
            "loan_amount": 100000,
            "loan_type": "personal"
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/analytics/risk-score",
            json=risk_data
        )
        if response.status_code == 200:
            print("✅ Risk score calculation working")
            data = response.json()
            print(f"   Risk Score: {data.get('risk_score', 0)}")
            print(f"   Risk Category: {data.get('risk_category', 'unknown')}")
            print(f"   Confidence: {data.get('confidence', 0)}")
            return True
        else:
            print(f"❌ Risk score calculation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Risk score calculation error: {e}")
        return False

def create_sample_document():
    """Create a sample document for testing"""
    print("\n📄 Creating sample document...")
    try:
        # Create a simple text file as a sample document
        sample_content = """
        SOLICITUD DE PRÉSTAMO PERSONAL
        
        Datos del solicitante:
        Nombre: Juan Pérez
        Edad: 35 años
        Ingresos mensuales: $50,000 MXN
        Puntaje de crédito: 650
        Tiempo en empleo: 5 años
        
        Detalles del préstamo:
        Monto solicitado: $100,000 MXN
        Plazo: 24 meses
        Propósito: Consolidación de deudas
        
        Documentos adjuntos:
        - Identificación oficial
        - Comprobante de ingresos
        - Estados de cuenta bancarios
        """
        
        sample_file = Path("sample_loan_application.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        print(f"✅ Sample document created: {sample_file}")
        return sample_file
    except Exception as e:
        print(f"❌ Error creating sample document: {e}")
        return None

def test_document_processing():
    """Test document processing endpoint"""
    print("\n📋 Testing document processing...")
    try:
        sample_file = create_sample_document()
        if not sample_file:
            print("❌ Could not create sample document")
            return False
        
        with open(sample_file, "rb") as f:
            files = {"file": ("sample_loan_application.txt", f, "text/plain")}
            response = requests.post(
                f"{BASE_URL}/api/v1/documents/process",
                files=files
            )
        
        if response.status_code == 200:
            print("✅ Document processing working")
            data = response.json()
            print(f"   Document ID: {data.get('document_id', 'unknown')}")
            print(f"   Document Type: {data.get('document_type', 'unknown')}")
            print(f"   Confidence Score: {data.get('confidence_score', 0)}")
            print(f"   Processing Time: {data.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Document processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 FinDoc AI Platform API Tests")
    print("=" * 50)
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(10)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_nvidia_status,
        test_analytics_dashboard,
        test_assistant_query,
        test_risk_score_calculation,
        test_document_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FinDoc AI Platform is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for more details.")
    
    print("\n📚 Next Steps:")
    print("   1. Visit http://localhost:8000/docs for API documentation")
    print("   2. Try uploading real documents")
    print("   3. Explore the analytics dashboard")
    print("   4. Chat with the digital assistant")

if __name__ == "__main__":
    main() 