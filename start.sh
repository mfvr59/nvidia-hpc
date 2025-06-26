#!/bin/bash

# FinDoc AI Platform Startup Script

echo "🚀 Starting FinDoc AI - Latin American Bank Document Processing Platform"
echo "=================================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available (optional)
if docker info | grep -q "nvidia"; then
    echo "✅ NVIDIA Docker runtime detected"
else
    echo "⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not be available."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads processed models data

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env 2>/dev/null || echo "No .env.example found, creating basic .env"
    echo "ENVIRONMENT=development" > .env
    echo "DATABASE_URL=postgresql://findoc_user:findoc_password@localhost:5432/findoc_ai" >> .env
    echo "REDIS_URL=redis://localhost:6379" >> .env
    echo "WEAVIATE_URL=http://localhost:8080" >> .env
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ FinDoc AI Backend is healthy"
else
    echo "❌ FinDoc AI Backend is not responding"
fi

if curl -f http://localhost:5432 &> /dev/null; then
    echo "✅ PostgreSQL is running"
else
    echo "❌ PostgreSQL is not responding"
fi

if curl -f http://localhost:6379 &> /dev/null; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
fi

if curl -f http://localhost:8080/v1/meta &> /dev/null; then
    echo "✅ Weaviate is running"
else
    echo "❌ Weaviate is not responding"
fi

echo ""
echo "🎉 FinDoc AI Platform is starting up!"
echo ""
echo "📊 Service URLs:"
echo "   - FinDoc AI API: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - PostgreSQL: localhost:5432"
echo "   - Redis: localhost:6379"
echo "   - Weaviate: http://localhost:8080"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "📚 Quick Start:"
echo "   1. Visit http://localhost:8000/docs to see the API documentation"
echo "   2. Upload a document using the /api/v1/documents/process endpoint"
echo "   3. Calculate risk scores using /api/v1/analytics/risk-score"
echo "   4. Chat with the assistant using /api/v1/assistant/query"
echo ""
echo "🛑 To stop the platform: docker-compose down"
echo "📋 To view logs: docker-compose logs -f" 