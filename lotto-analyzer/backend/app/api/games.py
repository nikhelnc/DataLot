from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Game
from app.schemas.game import GameCreate, GameResponse, GameUpdate, PrizeDivisionsUpdate

router = APIRouter(prefix="/games", tags=["games"])


@router.post("", response_model=GameResponse)
async def create_game(game: GameCreate, db: Session = Depends(get_db)):
    existing = db.query(Game).filter(Game.name == game.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Game with this name already exists")
    
    db_game = Game(
        name=game.name,
        description=game.description,
        rules_json=game.rules_json,
    )
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game


@router.get("", response_model=List[GameResponse])
async def list_games(db: Session = Depends(get_db)):
    games = db.query(Game).all()
    return games


@router.get("/{game_id}", response_model=GameResponse)
async def get_game(game_id: UUID, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@router.put("/{game_id}", response_model=GameResponse)
async def update_game(game_id: UUID, game_update: GameUpdate, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if game_update.name is not None:
        game.name = game_update.name
    if game_update.description is not None:
        game.description = game_update.description
    if game_update.rules_json is not None:
        game.rules_json = game_update.rules_json
    
    db.commit()
    db.refresh(game)
    return game


@router.put("/{game_id}/prize-divisions", response_model=GameResponse)
async def update_prize_divisions(game_id: UUID, data: PrizeDivisionsUpdate, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Update rules_json with prize_divisions
    rules = dict(game.rules_json) if game.rules_json else {}
    rules['prize_divisions'] = [div.model_dump() for div in data.prize_divisions]
    game.rules_json = rules
    
    db.commit()
    db.refresh(game)
    return game


@router.delete("/{game_id}")
async def delete_game(game_id: UUID, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    db.delete(game)
    db.commit()
    return {"message": "Game deleted successfully"}
