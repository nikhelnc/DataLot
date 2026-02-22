"""
API endpoints for fraud detection.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID
from datetime import date
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.db.models import Game, Draw, FraudAlert
from app.analysis.fraud import (
    DispersionTests, BenfordTests, ClusteringTests, 
    JackpotFraudTests, FraudScoreCalculator, AlertManager
)

router = APIRouter(prefix="/games/{game_id}/fraud", tags=["fraud"])


class FraudAnalysisRequest(BaseModel):
    """Request body for fraud analysis"""
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    include_jackpot_tests: bool = True
    alpha: float = Field(default=0.01, ge=0.001, le=0.1)


class AlertUpdateRequest(BaseModel):
    """Request body for updating an alert"""
    status: str = Field(..., pattern="^(OPEN|INVESTIGATING|CLOSED|FALSE_POSITIVE)$")
    resolution_note: Optional[str] = None
    assigned_to: Optional[str] = None


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
    main = rules.get("main", {})
    return {
        "n_max": main.get("max", numbers.get("max", 45)),
        "k": main.get("pick", numbers.get("count", 7))
    }


@router.post("/analyze")
async def run_fraud_analysis(
    game_id: UUID,
    request: FraudAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Run comprehensive fraud detection analysis.
    
    Performs multiple statistical tests to detect potential fraud:
    - Dispersion tests (variance, VMR, sum, gaps)
    - Benford's Law tests
    - Clustering tests (duplicates, near-duplicates, temporal)
    - Jackpot correlation tests (if data available)
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
    
    if len(draws_db) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient draws for fraud analysis. Need at least 30, got {len(draws_db)}"
        )
    
    # Extract data
    draws = [draw.numbers for draw in draws_db]
    jackpots = [float(draw.jackpot_amount) if draw.jackpot_amount else None for draw in draws_db]
    rollovers = [draw.jackpot_rollover for draw in draws_db]
    must_be_won = [draw.must_be_won for draw in draws_db]
    
    n_max = rules["n_max"]
    
    # Run tests
    dispersion = DispersionTests(alpha=request.alpha)
    benford = BenfordTests(alpha=request.alpha)
    clustering = ClusteringTests(alpha=request.alpha)
    jackpot_tests = JackpotFraudTests(alpha=request.alpha)
    
    dispersion_results = dispersion.run_all_tests(draws, n_max)
    benford_results = benford.run_all_tests(draws, n_max, jackpots if request.include_jackpot_tests else None)
    clustering_results = clustering.run_all_tests(draws, n_max)
    
    jackpot_results = None
    if request.include_jackpot_tests and any(j for j in jackpots if j):
        jackpot_results = jackpot_tests.run_all_tests(
            draws, n_max, jackpots, rollovers, must_be_won
        )
    
    # Calculate fraud score
    calculator = FraudScoreCalculator()
    score_result = calculator.compute_score(
        dispersion_results, benford_results, clustering_results, jackpot_results
    )
    
    # Generate alerts
    alerts = calculator.generate_alert_summary(
        dispersion_results, benford_results, clustering_results, jackpot_results
    )
    
    # Create alerts in database
    alert_manager = AlertManager(db)
    for alert_data in alerts:
        if alert_data["severity"] in ["HIGH", "CRITICAL"]:
            alert_manager.create_alert(
                game_id=game_id,
                severity=alert_data["severity"],
                signal_type=alert_data["test_name"],
                title=f"{alert_data['category']}: {alert_data['test_name']}",
                category=alert_data["category"].lower(),
                description=alert_data["description"],
                statistical_evidence={
                    "p_value": alert_data["p_value"],
                    "statistic": alert_data["statistic"],
                    "details": alert_data["details"]
                }
            )
    
    return {
        "game_id": str(game_id),
        "n_draws": len(draws),
        "period": {
            "start": draws_db[0].draw_date.isoformat() if draws_db else None,
            "end": draws_db[-1].draw_date.isoformat() if draws_db else None
        },
        "fraud_score": {
            "score": score_result.score,
            "risk_level": score_result.risk_level,
            "interpretation": score_result.interpretation,
            "category_scores": score_result.category_scores
        },
        "alerts": {
            "total": score_result.n_alerts,
            "by_severity": score_result.alerts_by_severity,
            "details": alerts[:20]  # Limit to 20 most severe
        },
        "test_results": {
            "dispersion": dispersion_results,
            "benford": benford_results,
            "clustering": clustering_results,
            "jackpot": jackpot_results
        }
    }


@router.get("/score")
async def get_fraud_score(
    game_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get quick fraud risk score based on recent analysis.
    
    Returns cached score if available, or runs quick analysis.
    """
    game = _get_game_or_404(game_id, db)
    
    # Get alert summary as proxy for fraud score
    alert_manager = AlertManager(db)
    summary = alert_manager.get_alert_summary(game_id)
    
    # Calculate risk level from open alerts
    critical = summary["open_by_severity"].get("CRITICAL", 0)
    high = summary["open_by_severity"].get("HIGH", 0)
    warning = summary["open_by_severity"].get("WARNING", 0)
    
    if critical > 0:
        risk_level = "CRITICAL"
        score = 75 + min(25, critical * 5)
    elif high > 0:
        risk_level = "HIGH"
        score = 50 + min(25, high * 5)
    elif warning > 0:
        risk_level = "MEDIUM"
        score = 25 + min(25, warning * 5)
    else:
        risk_level = "LOW"
        score = 0
    
    return {
        "game_id": str(game_id),
        "score": score,
        "risk_level": risk_level,
        "open_alerts": summary["total_open"],
        "alerts_by_severity": summary["open_by_severity"]
    }


@router.get("/alerts")
async def get_fraud_alerts(
    game_id: UUID,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get fraud alerts for a game.
    
    Supports filtering by status, severity, and category.
    """
    game = _get_game_or_404(game_id, db)
    
    query = db.query(FraudAlert).filter(FraudAlert.game_id == game_id)
    
    if status:
        query = query.filter(FraudAlert.status == status)
    if severity:
        query = query.filter(FraudAlert.severity == severity)
    if category:
        query = query.filter(FraudAlert.category == category)
    
    alerts = query.order_by(FraudAlert.created_at.desc()).limit(limit).all()
    
    return {
        "game_id": str(game_id),
        "count": len(alerts),
        "alerts": [
            {
                "id": str(alert.id),
                "severity": alert.severity,
                "signal_type": alert.signal_type,
                "category": alert.category,
                "title": alert.title,
                "description": alert.description,
                "status": alert.status,
                "created_at": alert.created_at.isoformat(),
                "statistical_evidence": alert.statistical_evidence,
                "assigned_to": alert.assigned_to,
                "resolution_note": alert.resolution_note,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
    }


@router.get("/alerts/summary")
async def get_alerts_summary(
    game_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get summary of fraud alerts for a game.
    """
    game = _get_game_or_404(game_id, db)
    
    alert_manager = AlertManager(db)
    return alert_manager.get_alert_summary(game_id)


@router.patch("/alerts/{alert_id}")
async def update_alert(
    game_id: UUID,
    alert_id: UUID,
    request: AlertUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update a fraud alert status.
    """
    game = _get_game_or_404(game_id, db)
    
    alert = db.query(FraudAlert)\
        .filter(FraudAlert.id == alert_id, FraudAlert.game_id == game_id)\
        .first()
    
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    alert_manager = AlertManager(db)
    updated = alert_manager.update_alert_status(
        alert_id=alert_id,
        status=request.status,
        resolution_note=request.resolution_note,
        assigned_to=request.assigned_to
    )
    
    return {
        "id": str(updated.id),
        "status": updated.status,
        "resolution_note": updated.resolution_note,
        "assigned_to": updated.assigned_to,
        "resolved_at": updated.resolved_at.isoformat() if updated.resolved_at else None
    }


@router.get("/alerts/{alert_id}")
async def get_alert_detail(
    game_id: UUID,
    alert_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific alert.
    """
    game = _get_game_or_404(game_id, db)
    
    alert = db.query(FraudAlert)\
        .filter(FraudAlert.id == alert_id, FraudAlert.game_id == game_id)\
        .first()
    
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return {
        "id": str(alert.id),
        "game_id": str(alert.game_id),
        "severity": alert.severity,
        "signal_type": alert.signal_type,
        "category": alert.category,
        "title": alert.title,
        "description": alert.description,
        "statistical_evidence": alert.statistical_evidence,
        "draw_ids": [str(d) for d in alert.draw_ids] if alert.draw_ids else [],
        "period_start": alert.period_start.isoformat() if alert.period_start else None,
        "period_end": alert.period_end.isoformat() if alert.period_end else None,
        "status": alert.status,
        "assigned_to": alert.assigned_to,
        "resolution_note": alert.resolution_note,
        "created_at": alert.created_at.isoformat(),
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
    }
