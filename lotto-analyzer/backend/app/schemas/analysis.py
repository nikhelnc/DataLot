from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel


class AnalysisRunRequest(BaseModel):
    game_id: UUID
    analysis_name: str
    params: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    analysis_id: UUID
    game_id: UUID
    name: str
    dataset_hash: Optional[str]
    code_version: Optional[str]
    params: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
