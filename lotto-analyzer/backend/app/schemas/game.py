from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class PrizeDivision(BaseModel):
    division: int
    main_numbers: int
    supplementary: int = 0
    description: Optional[str] = None


class GameRules(BaseModel):
    numbers: dict
    bonus: Optional[dict] = None
    calendar: Optional[dict] = None
    prize_divisions: Optional[List[PrizeDivision]] = None


class GameCreate(BaseModel):
    name: str
    description: Optional[str] = None
    rules_json: dict


class GameUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rules_json: Optional[dict] = None


class PrizeDivisionsUpdate(BaseModel):
    prize_divisions: List[PrizeDivision]


class GameResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    rules_json: dict
    version: int
    created_at: datetime

    class Config:
        from_attributes = True
