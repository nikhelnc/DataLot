from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import health, games, draws, imports, analyses, alerts, forensics, fraud, jackpot

app = FastAPI(
    title="Lotto Analyzer API",
    version=settings.code_version,
    description="Statistical analysis and probability modeling for lottery draws",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(games.router)
app.include_router(draws.router)
app.include_router(imports.router)
app.include_router(analyses.router)
app.include_router(alerts.router)
app.include_router(forensics.router)
app.include_router(fraud.router)
app.include_router(jackpot.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
