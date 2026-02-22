from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel


class ImportPreviewRow(BaseModel):
    draw_number: Optional[int] = None
    draw_date: str
    numbers: List[int]
    bonus_numbers: Optional[List[int]] = []
    # V2.0: New columns
    emission_order: Optional[List[int]] = None
    bonus_emission_order: Optional[List[int]] = None
    jackpot_amount: Optional[float] = None
    jackpot_rollover: Optional[bool] = False
    must_be_won: Optional[bool] = False
    n_winners_div1: Optional[int] = None


class ImportError(BaseModel):
    row: int
    field: Optional[str]
    message: str


class ImportResponse(BaseModel):
    import_id: UUID
    mode: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    preview_rows: List[ImportPreviewRow]
    errors: List[ImportError]


class ImportStatusResponse(BaseModel):
    id: UUID
    game_id: UUID
    source: str
    file_hash: Optional[str]
    status: str
    stats_json: Optional[Dict[str, Any]]
    error_log: Optional[List[Dict[str, Any]]]
    created_at: datetime

    class Config:
        from_attributes = True
