"""
API endpoints for jackpot analysis.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID
from datetime import date
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.db.models import Game, Draw, JackpotAnalysis
from app.analysis.jackpot import (
    JackpotIndependenceTest, PlayerBiasAnalyzer, 
    RDDAnalyzer, MustBeWonAnalyzer
)

router = APIRouter(prefix="/games/{game_id}/jackpot", tags=["jackpot"])


class JackpotAnalysisRequest(BaseModel):
    """Request body for jackpot analysis"""
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    alpha: float = Field(default=0.05, ge=0.001, le=0.1)
    rdd_thresholds: Optional[List[float]] = None


def _get_game_or_404(game_id: UUID, db: Session) -> Game:
    """Get game by ID or raise 404"""
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    return game


def _parse_game_rules(game: Game) -> dict:
    """Parse game rules"""
    rules = game.rules_json or {}
    numbers = rules.get("numbers", {})
    main = rules.get("main", {})
    return {
        "n_max": main.get("max", numbers.get("max", 45)),
        "k": main.get("pick", numbers.get("count", 7))
    }


@router.post("/analyze")
async def run_jackpot_analysis(
    game_id: UUID,
    request: JackpotAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Run comprehensive jackpot analysis.
    
    Includes:
    - Independence tests (jackpot vs draw characteristics)
    - Player bias analysis
    - RDD analysis at jackpot thresholds
    - Must-be-won analysis
    """
    game = _get_game_or_404(game_id, db)
    rules = _parse_game_rules(game)
    
    # Get draws with jackpot data
    query = db.query(Draw).filter(Draw.game_id == game_id)
    
    if request.period_start:
        query = query.filter(Draw.draw_date >= request.period_start)
    if request.period_end:
        query = query.filter(Draw.draw_date <= request.period_end)
    
    query = query.order_by(Draw.draw_date)
    draws_db = query.all()
    
    # Extract data
    draws = [draw.numbers for draw in draws_db]
    jackpots = [float(draw.jackpot_amount) if draw.jackpot_amount else None for draw in draws_db]
    must_be_won = [draw.must_be_won or False for draw in draws_db]
    n_winners = [draw.n_winners_div1 for draw in draws_db]
    
    n_max = rules["n_max"]
    
    # Check if we have jackpot data
    valid_jackpots = [j for j in jackpots if j and j > 0]
    if len(valid_jackpots) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient jackpot data. Need >= 30 draws with jackpot, got {len(valid_jackpots)}"
        )
    
    # Run analyses
    independence = JackpotIndependenceTest(alpha=request.alpha)
    player_bias = PlayerBiasAnalyzer(alpha=request.alpha)
    rdd = RDDAnalyzer(alpha=request.alpha)
    mbw = MustBeWonAnalyzer(alpha=request.alpha)
    
    independence_results = independence.run_all_tests(draws, jackpots, n_max)
    player_bias_results = player_bias.run_all_analyses(draws, jackpots, n_winners, n_max)
    rdd_results = rdd.run_full_analysis(draws, jackpots, request.rdd_thresholds)
    mbw_results = mbw.run_full_analysis(draws, must_be_won, n_max, jackpots, n_winners)
    
    # Save to database
    import hashlib
    import json
    dataset_hash = hashlib.sha256(json.dumps([d for d in draws[:100]]).encode()).hexdigest()[:16]
    
    db_analysis = JackpotAnalysis(
        game_id=game_id,
        generator_independence_test=independence_results,
        player_bias_analysis=player_bias_results,
        threshold_effect=rdd_results,
        must_be_won_analysis=mbw_results,
        n_draws_analyzed=len(draws),
        jackpot_range_min=min(valid_jackpots),
        jackpot_range_max=max(valid_jackpots),
        dataset_hash=dataset_hash
    )
    
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    
    return {
        "analysis_id": str(db_analysis.id),
        "game_id": str(game_id),
        "n_draws": len(draws),
        "n_draws_with_jackpot": len(valid_jackpots),
        "jackpot_range": {
            "min": min(valid_jackpots),
            "max": max(valid_jackpots),
            "mean": sum(valid_jackpots) / len(valid_jackpots)
        },
        "independence_tests": independence_results,
        "player_bias": player_bias_results,
        "rdd_analysis": rdd_results,
        "must_be_won": mbw_results
    }


@router.get("/stats")
async def get_jackpot_stats(
    game_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get jackpot statistics for a game.
    """
    game = _get_game_or_404(game_id, db)
    
    draws = db.query(Draw).filter(
        Draw.game_id == game_id,
        Draw.jackpot_amount.isnot(None)
    ).order_by(Draw.draw_date).all()
    
    if not draws:
        return {
            "game_id": str(game_id),
            "has_jackpot_data": False,
            "message": "No jackpot data available"
        }
    
    jackpots = [float(d.jackpot_amount) for d in draws if d.jackpot_amount]
    rollovers = [d for d in draws if d.jackpot_rollover]
    mbw_draws = [d for d in draws if d.must_be_won]
    
    import numpy as np
    
    return {
        "game_id": str(game_id),
        "has_jackpot_data": True,
        "n_draws_with_jackpot": len(jackpots),
        "jackpot_stats": {
            "min": float(min(jackpots)),
            "max": float(max(jackpots)),
            "mean": float(np.mean(jackpots)),
            "median": float(np.median(jackpots)),
            "std": float(np.std(jackpots))
        },
        "rollover_stats": {
            "n_rollovers": len(rollovers),
            "rollover_rate": len(rollovers) / len(draws) if draws else 0
        },
        "must_be_won_stats": {
            "n_mbw": len(mbw_draws),
            "mbw_rate": len(mbw_draws) / len(draws) if draws else 0
        },
        "date_range": {
            "first": draws[0].draw_date.isoformat() if draws else None,
            "last": draws[-1].draw_date.isoformat() if draws else None
        }
    }


@router.get("/history")
async def get_jackpot_history(
    game_id: UUID,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get jackpot history for a game.
    """
    game = _get_game_or_404(game_id, db)
    
    draws = db.query(Draw).filter(
        Draw.game_id == game_id,
        Draw.jackpot_amount.isnot(None)
    ).order_by(Draw.draw_date.desc()).limit(limit).all()
    
    return {
        "game_id": str(game_id),
        "count": len(draws),
        "history": [
            {
                "draw_date": d.draw_date.isoformat(),
                "draw_number": d.draw_number,
                "jackpot_amount": float(d.jackpot_amount) if d.jackpot_amount else None,
                "rollover": d.jackpot_rollover,
                "consecutive_rollovers": d.jackpot_consecutive_rollovers,
                "must_be_won": d.must_be_won,
                "n_winners": d.n_winners_div1,
                "numbers": d.numbers
            }
            for d in draws
        ]
    }


@router.get("/analyses")
async def get_jackpot_analyses(
    game_id: UUID,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get history of jackpot analyses for a game.
    """
    game = _get_game_or_404(game_id, db)
    
    analyses = db.query(JackpotAnalysis).filter(
        JackpotAnalysis.game_id == game_id
    ).order_by(JackpotAnalysis.computed_at.desc()).limit(limit).all()
    
    return {
        "game_id": str(game_id),
        "count": len(analyses),
        "analyses": [
            {
                "id": str(a.id),
                "computed_at": a.computed_at.isoformat(),
                "n_draws_analyzed": a.n_draws_analyzed,
                "jackpot_range": {
                    "min": float(a.jackpot_range_min) if a.jackpot_range_min else None,
                    "max": float(a.jackpot_range_max) if a.jackpot_range_max else None
                },
                "independence_conclusion": a.generator_independence_test.get("summary", {}).get("conclusion") if a.generator_independence_test else None,
                "mbw_conclusion": a.must_be_won_analysis.get("summary", {}).get("conclusion") if a.must_be_won_analysis else None
            }
            for a in analyses
        ]
    }


@router.get("/analyses/{analysis_id}")
async def get_jackpot_analysis_detail(
    game_id: UUID,
    analysis_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed jackpot analysis results.
    """
    game = _get_game_or_404(game_id, db)
    
    analysis = db.query(JackpotAnalysis).filter(
        JackpotAnalysis.id == analysis_id,
        JackpotAnalysis.game_id == game_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    
    return {
        "id": str(analysis.id),
        "game_id": str(analysis.game_id),
        "computed_at": analysis.computed_at.isoformat(),
        "n_draws_analyzed": analysis.n_draws_analyzed,
        "jackpot_range": {
            "min": float(analysis.jackpot_range_min) if analysis.jackpot_range_min else None,
            "max": float(analysis.jackpot_range_max) if analysis.jackpot_range_max else None
        },
        "independence_tests": analysis.generator_independence_test,
        "player_bias": analysis.player_bias_analysis,
        "rdd_analysis": analysis.threshold_effect,
        "must_be_won": analysis.must_be_won_analysis,
        "dataset_hash": analysis.dataset_hash
    }
