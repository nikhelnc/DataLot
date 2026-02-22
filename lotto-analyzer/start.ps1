# Lotto Analyzer - Startup Script for Windows
Write-Host "🎲 Starting Lotto Analyzer..." -ForegroundColor Cyan

# Check if Docker is running
$dockerRunning = docker info 2>&1 | Select-String "Server Version"
if (-not $dockerRunning) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker is running" -ForegroundColor Green

# Start services
Write-Host "`n📦 Starting services with Docker Compose..." -ForegroundColor Cyan
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Services started" -ForegroundColor Green

# Wait for database to be ready
Write-Host "`n⏳ Waiting for database to be ready..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Run migrations
Write-Host "`n🔄 Running database migrations..." -ForegroundColor Cyan
docker-compose exec -T backend alembic upgrade head

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to run migrations" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Migrations completed" -ForegroundColor Green

# Display status
Write-Host "`n✅ Lotto Analyzer is ready!" -ForegroundColor Green
Write-Host "`nAccess the application at:" -ForegroundColor Cyan
Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor White

Write-Host "`nTo view logs:" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f" -ForegroundColor White

Write-Host "`nTo stop services:" -ForegroundColor Yellow
Write-Host "  docker-compose down" -ForegroundColor White
