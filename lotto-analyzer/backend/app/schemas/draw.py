from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class DrawResponse(BaseModel):
    id: UUID
    game_id: UUID
    draw_number: Optional[int] = None
    draw_date: datetime
    numbers: List[int]
    bonus_numbers: Optional[List[int]] = []
    created_at: datetime

    class Config:
        from_attributes = True


class DrawCreate(BaseModel):
    game_id: UUID
    draw_number: Optional[int] = None
    draw_date: datetime
    numbers: List[int]
    bonus_numbers: Optional[List[int]] = []


class DrawListParams(BaseModel):
    game_id: Optional[UUID] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    page: int = 1
    page_size: int = 100
