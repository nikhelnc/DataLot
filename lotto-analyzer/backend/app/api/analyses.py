from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from sqlalchemy.orm import Session
import io

from app.db.database import get_db
from app.db.models import Analysis, Game, Draw
from app.schemas.analysis import AnalysisRunRequest, AnalysisResponse
from app.services.analysis_service import AnalysisService
from app.analysis.backtest import WalkForwardBacktest

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("/run", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRunRequest, db: Session = Depends(get_db)):
    service = AnalysisService(db)
    result = await service.run_analysis(
        request.game_id, request.analysis_name, request.params or {}
    )
    return result


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: UUID, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisResponse(
        analysis_id=analysis.id,
        game_id=analysis.game_id,
        name=analysis.name,
        dataset_hash=analysis.dataset_hash,
        code_version=analysis.code_version,
        params=analysis.params_json,
        results=analysis.results_json,
        created_at=analysis.created_at,
    )


@router.get("/{analysis_id}/export.csv")
async def export_analysis_csv(analysis_id: UUID, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    service = AnalysisService(db)
    csv_content = service.export_to_csv(analysis)
    
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"},
    )


@router.get("/{analysis_id}/report.html", response_class=HTMLResponse)
async def get_analysis_report(analysis_id: UUID, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    service = AnalysisService(db)
    html_content = service.generate_html_report(analysis)
    
    return HTMLResponse(content=html_content)


@router.get("/predict/{game_id}")
async def get_next_prediction(
    game_id: UUID,
    n_combinations: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Generate Anti-Consensus prediction for the next draw.
    Uses all historical data to predict numbers NOT predicted by other models.
    """
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    draws = db.query(Draw).filter(Draw.game_id == game_id).order_by(Draw.draw_date.asc()).all()
    if len(draws) < 10:
        raise HTTPException(status_code=400, detail="Insufficient draws for prediction (minimum 10)")
    
    backtester = WalkForwardBacktest(
        draws=draws,
        game=game,
        n_test_draws=1,
        top_n=10,
        n_combinations=n_combinations
    )
    
    prediction = backtester.generate_next_prediction()
    
    return prediction
