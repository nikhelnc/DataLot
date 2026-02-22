"""
Alert manager for fraud detection.
Handles creation, storage, and management of fraud alerts.
"""

from typing import Dict, List, Any, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models import FraudAlert, Game


class AlertManager:
    """
    Manages fraud detection alerts.
    
    Provides methods to create, update, and query fraud alerts.
    """
    
    def __init__(self, db: Session):
        """
        Initialize alert manager.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def create_alert(self,
                     game_id: UUID,
                     severity: str,
                     signal_type: str,
                     title: str,
                     category: str = None,
                     description: str = None,
                     statistical_evidence: Dict[str, Any] = None,
                     draw_ids: List[UUID] = None,
                     period_start: datetime = None,
                     period_end: datetime = None,
                     analysis_id: UUID = None) -> FraudAlert:
        """
        Create a new fraud alert.
        
        Args:
            game_id: ID of the game
            severity: Alert severity (INFO, WARNING, HIGH, CRITICAL)
            signal_type: Type of signal detected
            title: Short title for the alert
            category: Category (generator, data_quality, behavioral, structural)
            description: Detailed description
            statistical_evidence: Statistical evidence as JSON
            draw_ids: List of affected draw IDs
            period_start: Start of affected period
            period_end: End of affected period
            analysis_id: ID of the analysis that generated this alert
            
        Returns:
            Created FraudAlert object
        """
        alert = FraudAlert(
            game_id=game_id,
            severity=severity,
            signal_type=signal_type,
            title=title,
            category=category,
            description=description,
            statistical_evidence=statistical_evidence,
            draw_ids=draw_ids,
            period_start=period_start.date() if period_start else None,
            period_end=period_end.date() if period_end else None,
            analysis_id=analysis_id,
            status='OPEN'
        )
        
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        
        return alert
    
    def create_alerts_from_results(self,
                                   game_id: UUID,
                                   test_results: Dict[str, Any],
                                   category: str,
                                   analysis_id: UUID = None) -> List[FraudAlert]:
        """
        Create alerts from test results.
        
        Args:
            game_id: ID of the game
            test_results: Results from a test module
            category: Category for the alerts
            analysis_id: ID of the analysis
            
        Returns:
            List of created FraudAlert objects
        """
        alerts = []
        
        if not test_results or "tests" not in test_results:
            return alerts
        
        for test_name, test_data in test_results["tests"].items():
            if not test_data.get("passed", True):
                severity = test_data.get("severity", "WARNING")
                
                alert = self.create_alert(
                    game_id=game_id,
                    severity=severity,
                    signal_type=test_name,
                    title=f"{category}: {test_name} failed",
                    category=category,
                    description=test_data.get("description", ""),
                    statistical_evidence={
                        "p_value": test_data.get("p_value"),
                        "statistic": test_data.get("statistic"),
                        "details": test_data.get("details", {})
                    },
                    analysis_id=analysis_id
                )
                alerts.append(alert)
        
        return alerts
    
    def get_open_alerts(self, game_id: UUID, 
                        severity: str = None,
                        limit: int = 100) -> List[FraudAlert]:
        """
        Get open alerts for a game.
        
        Args:
            game_id: ID of the game
            severity: Filter by severity (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of FraudAlert objects
        """
        query = self.db.query(FraudAlert)\
            .filter(FraudAlert.game_id == game_id)\
            .filter(FraudAlert.status == 'OPEN')
        
        if severity:
            query = query.filter(FraudAlert.severity == severity)
        
        return query.order_by(FraudAlert.created_at.desc()).limit(limit).all()
    
    def get_alerts_by_category(self, game_id: UUID, 
                                category: str,
                                include_closed: bool = False) -> List[FraudAlert]:
        """
        Get alerts for a game by category.
        
        Args:
            game_id: ID of the game
            category: Alert category
            include_closed: Whether to include closed alerts
            
        Returns:
            List of FraudAlert objects
        """
        query = self.db.query(FraudAlert)\
            .filter(FraudAlert.game_id == game_id)\
            .filter(FraudAlert.category == category)
        
        if not include_closed:
            query = query.filter(FraudAlert.status.in_(['OPEN', 'INVESTIGATING']))
        
        return query.order_by(FraudAlert.created_at.desc()).all()
    
    def update_alert_status(self, alert_id: UUID, 
                            status: str,
                            resolution_note: str = None,
                            assigned_to: str = None) -> FraudAlert:
        """
        Update alert status.
        
        Args:
            alert_id: ID of the alert
            status: New status (OPEN, INVESTIGATING, CLOSED, FALSE_POSITIVE)
            resolution_note: Note explaining the resolution
            assigned_to: Person assigned to investigate
            
        Returns:
            Updated FraudAlert object
        """
        alert = self.db.query(FraudAlert).filter(FraudAlert.id == alert_id).first()
        
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert.status = status
        
        if resolution_note:
            alert.resolution_note = resolution_note
        
        if assigned_to:
            alert.assigned_to = assigned_to
        
        if status in ['CLOSED', 'FALSE_POSITIVE']:
            alert.resolved_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(alert)
        
        return alert
    
    def get_alert_summary(self, game_id: UUID) -> Dict[str, Any]:
        """
        Get summary of alerts for a game.
        
        Args:
            game_id: ID of the game
            
        Returns:
            Dictionary with alert counts and statistics
        """
        from sqlalchemy import func
        
        # Count by status
        status_counts = self.db.query(
            FraudAlert.status,
            func.count(FraudAlert.id)
        ).filter(FraudAlert.game_id == game_id)\
         .group_by(FraudAlert.status)\
         .all()
        
        # Count by severity (open only)
        severity_counts = self.db.query(
            FraudAlert.severity,
            func.count(FraudAlert.id)
        ).filter(FraudAlert.game_id == game_id)\
         .filter(FraudAlert.status == 'OPEN')\
         .group_by(FraudAlert.severity)\
         .all()
        
        # Count by category (open only)
        category_counts = self.db.query(
            FraudAlert.category,
            func.count(FraudAlert.id)
        ).filter(FraudAlert.game_id == game_id)\
         .filter(FraudAlert.status == 'OPEN')\
         .group_by(FraudAlert.category)\
         .all()
        
        return {
            "game_id": str(game_id),
            "by_status": {status: count for status, count in status_counts},
            "open_by_severity": {sev: count for sev, count in severity_counts},
            "open_by_category": {cat: count for cat, count in category_counts if cat},
            "total_open": sum(count for sev, count in severity_counts)
        }
    
    def close_duplicate_alerts(self, game_id: UUID, signal_type: str) -> int:
        """
        Close duplicate alerts of the same type.
        
        Keeps only the most recent alert open.
        
        Args:
            game_id: ID of the game
            signal_type: Type of signal
            
        Returns:
            Number of alerts closed
        """
        alerts = self.db.query(FraudAlert)\
            .filter(FraudAlert.game_id == game_id)\
            .filter(FraudAlert.signal_type == signal_type)\
            .filter(FraudAlert.status == 'OPEN')\
            .order_by(FraudAlert.created_at.desc())\
            .all()
        
        if len(alerts) <= 1:
            return 0
        
        # Keep the first (most recent), close the rest
        closed_count = 0
        for alert in alerts[1:]:
            alert.status = 'CLOSED'
            alert.resolution_note = 'Superseded by newer alert'
            alert.resolved_at = datetime.utcnow()
            closed_count += 1
        
        self.db.commit()
        return closed_count
