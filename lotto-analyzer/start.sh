#!/bin/bash
# Lotto Analyzer - Startup Script for Linux/Mac

echo "🎲 Starting Lotto Analyzer..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi

echo "✓ Docker is running"

# Start services
echo ""
echo "📦 Starting services with Docker Compose..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start services"
    exit 1
fi

echo "✓ Services started"

# Wait for database to be ready
echo ""
echo "⏳ Waiting for database to be ready..."
sleep 5

# Run migrations
echo ""
echo "🔄 Running database migrations..."
docker-compose exec -T backend alembic upgrade head

if [ $? -ne 0 ]; then
    echo "❌ Failed to run migrations"
    exit 1
fi

echo "✓ Migrations completed"

# Display status
echo ""
echo "✅ Lotto Analyzer is ready!"
echo ""
echo "Access the application at:"
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
