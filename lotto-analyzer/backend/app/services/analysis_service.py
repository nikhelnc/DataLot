import hashlib
from typing import Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session

from app.db.models import Game, Draw, Analysis, Alert
from app.config import settings
from app.analysis.orchestrator import AnalysisOrchestrator
from app.schemas.analysis import AnalysisResponse


class AnalysisService:
    def __init__(self, db: Session):
        self.db = db

    async def run_analysis(
        self, game_id: UUID, analysis_name: str, params: Dict[str, Any]
    ) -> AnalysisResponse:
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")

        draws = (
            self.db.query(Draw)
            .filter(Draw.game_id == game_id)
            .order_by(Draw.draw_date)
            .all()
        )

        if len(draws) < 10:
            raise ValueError("Insufficient data: at least 10 draws required")

        dataset_hash = hashlib.sha256(
            str([d.id for d in draws]).encode()
        ).hexdigest()[:16]

        orchestrator = AnalysisOrchestrator(game, draws)
        results = orchestrator.run(analysis_name, params)

        analysis = Analysis(
            game_id=game_id,
            name=analysis_name,
            params_json=params,
            results_json=results,
            dataset_hash=dataset_hash,
            code_version=settings.code_version,
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)

        if "alerts" in results:
            for alert_data in results["alerts"]:
                alert = Alert(
                    game_id=game_id,
                    analysis_id=analysis.id,
                    severity=alert_data["severity"],
                    score=alert_data["score"],
                    message=alert_data["message"],
                    evidence_json=alert_data.get("evidence_json"),
                )
                self.db.add(alert)
            self.db.commit()

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

    def export_to_csv(self, analysis: Analysis) -> str:
        from app.analysis.reporting.csv_exporter import CSVExporter

        exporter = CSVExporter()
        return exporter.export(analysis)

    def generate_html_report(self, analysis: Analysis) -> str:
        from app.analysis.reporting.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        return reporter.generate(analysis)
