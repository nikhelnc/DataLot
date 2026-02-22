"""
API endpoints for forensic analysis of lottery generators.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, List, AsyncGenerator
from uuid import UUID
from datetime import datetime, date
from pydantic import BaseModel, Field
import asyncio
import json

from app.db.database import get_db
from app.db.models import Game, Draw, GeneratorProfile
from app.analysis.forensics import GeneratorProfiler

router = APIRouter(prefix="/games/{game_id}/forensics", tags=["forensics"])


class ForensicsRunRequest(BaseModel):
    """Request body for running forensic analysis"""
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    n_simulations: int = Field(default=1000, ge=100, le=10000)
    compute_ci: bool = True
    alpha: float = Field(default=0.01, ge=0.001, le=0.1)


class ForensicsResponse(BaseModel):
    """Response for forensic profile"""
    id: Optional[UUID] = None
    game_id: UUID
    computed_at: datetime
    n_draws: int
    conformity_score: float
    conformity_ci_low: float
    conformity_ci_high: float
    conformity_interpretation: str
    generator_type: str
    category_scores: dict
    summary: dict
    
    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    """Response for async task"""
    task_id: str
    status: str
    message: str


def _get_game_or_404(game_id: UUID, db: Session) -> Game:
    """Get game by ID or raise 404"""
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    return game


def _parse_game_rules(game: Game) -> dict:
    """Parse game rules to extract n_max and k"""
    rules = game.rules_json or {}
    numbers = rules.get("numbers", {})
    return {
        "n_max": numbers.get("max", 45),
        "k": numbers.get("count", 7),
        "bonus_enabled": rules.get("bonus", {}).get("enabled", False),
        "bonus_max": rules.get("bonus", {}).get("max", 12)
    }


@router.get("", response_model=Optional[ForensicsResponse])
async def get_latest_forensics(game_id: UUID, db: Session = Depends(get_db)):
    """
    Get the latest forensic profile for a game.
    
    Returns the most recent computed forensic profile, or null if none exists.
    """
    game = _get_game_or_404(game_id, db)
    
    profile = db.query(GeneratorProfile)\
        .filter(GeneratorProfile.game_id == game_id)\
        .order_by(GeneratorProfile.computed_at.desc())\
        .first()
    
    if not profile:
        return None
    
    return ForensicsResponse(
        id=profile.id,
        game_id=profile.game_id,
        computed_at=profile.computed_at,
        n_draws=profile.n_draws,
        conformity_score=profile.conformity_score or 0.0,
        conformity_ci_low=profile.conformity_ci_low or 0.0,
        conformity_ci_high=profile.conformity_ci_high or 1.0,
        conformity_interpretation=profile.standard_tests.get("interpretation", "") if profile.standard_tests else "",
        generator_type=profile.generator_type or "unknown",
        category_scores=profile.params.get("category_scores", {}) if profile.params else {},
        summary=profile.structural_tests.get("summary", {}) if profile.structural_tests else {}
    )


@router.post("/run", response_model=dict)
async def run_forensics(
    game_id: UUID,
    request: ForensicsRunRequest,
    db: Session = Depends(get_db)
):
    """
    Run forensic analysis on a game's draw data.
    
    This performs a complete forensic profile including:
    - NIST-adapted randomness tests
    - Physical bias tests (if emission order available)
    - RNG vulnerability tests
    - Structural tests
    - Conformity score with Monte Carlo confidence intervals
    """
    game = _get_game_or_404(game_id, db)
    rules = _parse_game_rules(game)
    
    # Get draws
    query = db.query(Draw).filter(Draw.game_id == game_id)
    
    if request.period_start:
        query = query.filter(Draw.draw_date >= request.period_start)
    if request.period_end:
        query = query.filter(Draw.draw_date <= request.period_end)
    
    query = query.order_by(Draw.draw_date)
    draws_db = query.all()
    
    if len(draws_db) < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient draws for forensic analysis. Need at least 50, got {len(draws_db)}"
        )
    
    # Extract draw data
    draws = [draw.numbers for draw in draws_db]
    
    # Extract emission orders if available
    emission_orders = None
    draws_with_emission = [d for d in draws_db if d.emission_order]
    if draws_with_emission:
        emission_orders = [d.emission_order for d in draws_with_emission]
    
    # Run forensic profile
    profiler = GeneratorProfiler(
        alpha=request.alpha,
        n_simulations=request.n_simulations,
        seed=42
    )
    
    profile_data = profiler.run_full_profile(
        draws=draws,
        n_max=rules["n_max"],
        k=rules["k"],
        emission_orders=emission_orders,
        period_start=datetime.combine(request.period_start, datetime.min.time()) if request.period_start else None,
        period_end=datetime.combine(request.period_end, datetime.max.time()) if request.period_end else None,
        compute_ci=request.compute_ci
    )
    
    # Save to database
    db_profile = GeneratorProfile(
        game_id=game_id,
        period_start=request.period_start,
        period_end=request.period_end,
        n_draws=len(draws),
        conformity_score=profile_data["conformity_score"],
        conformity_ci_low=profile_data["conformity_ci_low"],
        conformity_ci_high=profile_data["conformity_ci_high"],
        conformity_n_simulations=profile_data["conformity_n_simulations"],
        generator_type=profile_data["generator_type"],
        standard_tests={"interpretation": profile_data["conformity_interpretation"]},
        nist_tests=profile_data["nist_tests"],
        physical_tests=profile_data["physical_tests"],
        rng_tests=profile_data["rng_tests"],
        structural_tests=profile_data["structural_tests"],
        dataset_hash=profile_data["dataset_hash"],
        app_version=profile_data["app_version"],
        params={
            **profile_data["params"],
            "category_scores": profile_data["category_scores"]
        },
        seed=profile_data["seed"]
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    return {
        "profile_id": str(db_profile.id),
        "status": "completed",
        "conformity_score": profile_data["conformity_score"],
        "conformity_ci": [profile_data["conformity_ci_low"], profile_data["conformity_ci_high"]],
        "interpretation": profile_data["conformity_interpretation"],
        "generator_type": profile_data["generator_type"],
        "summary": profile_data["summary"],
        "computation_time_seconds": profile_data["computation_time_seconds"]
    }


@router.get("/run/stream")
async def run_forensics_stream(
    game_id: UUID,
    n_simulations: int = 1000,
    compute_ci: bool = True,
    alpha: float = 0.01,
    db: Session = Depends(get_db)
):
    """
    Run forensic analysis with Server-Sent Events for progress updates.
    
    Returns a stream of SSE events with progress updates and final results.
    """
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    rules = _parse_game_rules(game)
    
    # Get draws
    draws_db = db.query(Draw).filter(Draw.game_id == game_id).order_by(Draw.draw_date).all()
    
    if len(draws_db) < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient draws for forensic analysis. Need at least 50, got {len(draws_db)}"
        )
    
    draws = [draw.numbers for draw in draws_db]
    emission_orders = None
    draws_with_emission = [d for d in draws_db if d.emission_order]
    if draws_with_emission:
        emission_orders = [d.emission_order for d in draws_with_emission]
    
    async def event_generator() -> AsyncGenerator[str, None]:
        progress_queue = asyncio.Queue()
        result_holder = {"data": None, "error": None}
        
        def progress_callback(step: str, progress: int, total: int):
            try:
                asyncio.get_event_loop().call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"step": step, "progress": progress, "total": total}
                )
            except:
                pass
        
        def run_analysis():
            try:
                profiler = GeneratorProfiler(
                    alpha=alpha,
                    n_simulations=n_simulations,
                    seed=42
                )
                result_holder["data"] = profiler.run_full_profile(
                    draws=draws,
                    n_max=rules["n_max"],
                    k=rules["k"],
                    emission_orders=emission_orders,
                    compute_ci=compute_ci,
                    progress_callback=progress_callback
                )
            except Exception as e:
                result_holder["error"] = str(e)
                progress_callback("Erreur", -1, 100)
        
        import threading
        thread = threading.Thread(target=run_analysis)
        thread.start()
        
        while thread.is_alive() or not progress_queue.empty():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(progress)}\n\n"
            except asyncio.TimeoutError:
                continue
        
        thread.join()
        
        if result_holder["error"]:
            yield f"data: {json.dumps({'error': result_holder['error']})}\n\n"
        elif result_holder["data"]:
            # Save to database
            profile_data = result_holder["data"]
            db_profile = GeneratorProfile(
                game_id=game_id,
                n_draws=len(draws),
                conformity_score=profile_data["conformity_score"],
                conformity_ci_low=profile_data["conformity_ci_low"],
                conformity_ci_high=profile_data["conformity_ci_high"],
                conformity_n_simulations=profile_data["conformity_n_simulations"],
                generator_type=profile_data["generator_type"],
                standard_tests={"interpretation": profile_data["conformity_interpretation"]},
                nist_tests=profile_data["nist_tests"],
                physical_tests=profile_data["physical_tests"],
                rng_tests=profile_data["rng_tests"],
                structural_tests=profile_data["structural_tests"],
                dataset_hash=profile_data["dataset_hash"],
                app_version=profile_data["app_version"],
                params={
                    **profile_data["params"],
                    "category_scores": profile_data["category_scores"]
                },
                seed=profile_data["seed"]
            )
            db.add(db_profile)
            db.commit()
            db.refresh(db_profile)
            
            final_result = {
                "completed": True,
                "profile_id": str(db_profile.id),
                "conformity_score": profile_data["conformity_score"],
                "conformity_ci": [profile_data["conformity_ci_low"], profile_data["conformity_ci_high"]],
                "interpretation": profile_data["conformity_interpretation"],
                "generator_type": profile_data["generator_type"],
                "summary": profile_data["summary"],
                "computation_time_seconds": profile_data["computation_time_seconds"]
            }
            yield f"data: {json.dumps(final_result)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/{profile_id}")
async def get_forensics_profile(
    game_id: UUID,
    profile_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific forensic profile by ID.
    
    Returns the complete profile with all test results.
    """
    game = _get_game_or_404(game_id, db)
    
    profile = db.query(GeneratorProfile)\
        .filter(GeneratorProfile.id == profile_id, GeneratorProfile.game_id == game_id)\
        .first()
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
    
    return {
        "id": str(profile.id),
        "game_id": str(profile.game_id),
        "computed_at": profile.computed_at.isoformat(),
        "period_start": profile.period_start.isoformat() if profile.period_start else None,
        "period_end": profile.period_end.isoformat() if profile.period_end else None,
        "n_draws": profile.n_draws,
        "conformity_score": profile.conformity_score,
        "conformity_ci_low": profile.conformity_ci_low,
        "conformity_ci_high": profile.conformity_ci_high,
        "conformity_n_simulations": profile.conformity_n_simulations,
        "generator_type": profile.generator_type,
        "standard_tests": profile.standard_tests,
        "nist_tests": profile.nist_tests,
        "physical_tests": profile.physical_tests,
        "rng_tests": profile.rng_tests,
        "structural_tests": profile.structural_tests,
        "dataset_hash": profile.dataset_hash,
        "app_version": profile.app_version,
        "params": profile.params,
        "seed": profile.seed
    }


@router.get("/{profile_id}/heatmap")
async def get_position_heatmap(
    game_id: UUID,
    profile_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get position × number heatmap data from a forensic profile.
    
    Returns the position bias data for visualization.
    """
    game = _get_game_or_404(game_id, db)
    
    profile = db.query(GeneratorProfile)\
        .filter(GeneratorProfile.id == profile_id, GeneratorProfile.game_id == game_id)\
        .first()
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
    
    physical_tests = profile.physical_tests or {}
    position_test = physical_tests.get("tests", {}).get("Emission Position Bias", {})
    
    if not position_test or not position_test.get("details"):
        return {
            "available": False,
            "message": "No emission order data available for heatmap"
        }
    
    details = position_test["details"]
    
    return {
        "available": True,
        "n_positions": details.get("n_positions", 0),
        "n_draws": details.get("n_draws", 0),
        "position_counts": details.get("position_counts", []),
        "chi2_by_position": details.get("chi2_by_position", []),
        "p_values_by_position": details.get("p_values_by_position", []),
        "most_biased_position": details.get("most_biased_position"),
        "most_biased_p_value": details.get("most_biased_p_value")
    }


@router.get("/history")
async def get_forensics_history(
    game_id: UUID,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get history of forensic profiles for a game.
    
    Returns a list of past forensic analyses with their conformity scores.
    """
    game = _get_game_or_404(game_id, db)
    
    profiles = db.query(GeneratorProfile)\
        .filter(GeneratorProfile.game_id == game_id)\
        .order_by(GeneratorProfile.computed_at.desc())\
        .limit(limit)\
        .all()
    
    return {
        "game_id": str(game_id),
        "count": len(profiles),
        "profiles": [
            {
                "id": str(p.id),
                "computed_at": p.computed_at.isoformat(),
                "n_draws": p.n_draws,
                "conformity_score": p.conformity_score,
                "conformity_ci": [p.conformity_ci_low, p.conformity_ci_high],
                "generator_type": p.generator_type,
                "period_start": p.period_start.isoformat() if p.period_start else None,
                "period_end": p.period_end.isoformat() if p.period_end else None
            }
            for p in profiles
        ]
    }
