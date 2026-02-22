from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel


class AlertResponse(BaseModel):
    id: UUID
    game_id: UUID
    analysis_id: Optional[UUID]
    severity: str
    score: int
    message: str
    evidence_json: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
