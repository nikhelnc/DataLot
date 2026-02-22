from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Draw, Game
from app.schemas.draw import DrawResponse, DrawCreate

router = APIRouter(prefix="/draws", tags=["draws"])


@router.post("", response_model=DrawResponse, status_code=201)
async def create_draw(draw_data: DrawCreate, db: Session = Depends(get_db)):
    """Create a new draw manually."""
    # Verify game exists
    game = db.query(Game).filter(Game.id == draw_data.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Validate numbers against game rules
    rules = game.rules_json or {}
    main_count = rules.get("main_numbers", {}).get("count", 0)
    main_min = rules.get("main_numbers", {}).get("min", 1)
    main_max = rules.get("main_numbers", {}).get("max", 99)
    bonus_count = rules.get("bonus_numbers", {}).get("count", 0)
    bonus_min = rules.get("bonus_numbers", {}).get("min", 1)
    bonus_max = rules.get("bonus_numbers", {}).get("max", 99)
    
    # Validate main numbers count
    if main_count > 0 and len(draw_data.numbers) != main_count:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {main_count} main numbers, got {len(draw_data.numbers)}"
        )
    
    # Validate main numbers range
    for num in draw_data.numbers:
        if num < main_min or num > main_max:
            raise HTTPException(
                status_code=400,
                detail=f"Main number {num} is out of range [{main_min}-{main_max}]"
            )
    
    # Validate bonus numbers if applicable
    if bonus_count > 0:
        if draw_data.bonus_numbers and len(draw_data.bonus_numbers) != bonus_count:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {bonus_count} bonus numbers, got {len(draw_data.bonus_numbers)}"
            )
        if draw_data.bonus_numbers:
            for num in draw_data.bonus_numbers:
                if num < bonus_min or num > bonus_max:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bonus number {num} is out of range [{bonus_min}-{bonus_max}]"
                    )
    
    # Check for duplicate draw (same game, same date or same draw_number)
    existing = db.query(Draw).filter(
        Draw.game_id == draw_data.game_id,
        Draw.draw_date == draw_data.draw_date
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="A draw already exists for this game on this date"
        )
    
    if draw_data.draw_number:
        existing_number = db.query(Draw).filter(
            Draw.game_id == draw_data.game_id,
            Draw.draw_number == draw_data.draw_number
        ).first()
        if existing_number:
            raise HTTPException(
                status_code=400,
                detail=f"A draw with number {draw_data.draw_number} already exists for this game"
            )
    
    # Create the draw
    draw = Draw(
        game_id=draw_data.game_id,
        draw_number=draw_data.draw_number,
        draw_date=draw_data.draw_date,
        numbers=draw_data.numbers,
        bonus_numbers=draw_data.bonus_numbers or [],
        raw_payload={"source": "manual_entry"}
    )
    
    db.add(draw)
    db.commit()
    db.refresh(draw)
    
    return draw


@router.get("", response_model=List[DrawResponse])
async def list_draws(
    game_id: Optional[UUID] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    query = db.query(Draw)
    
    if game_id:
        query = query.filter(Draw.game_id == game_id)
    if from_date:
        query = query.filter(Draw.draw_date >= from_date)
    if to_date:
        query = query.filter(Draw.draw_date <= to_date)
    
    query = query.order_by(Draw.draw_date.desc())
    offset = (page - 1) * page_size
    draws = query.offset(offset).limit(page_size).all()
    
    return draws


@router.get("/{draw_id}", response_model=DrawResponse)
async def get_draw(draw_id: UUID, db: Session = Depends(get_db)):
    draw = db.query(Draw).filter(Draw.id == draw_id).first()
    if not draw:
        raise HTTPException(status_code=404, detail="Draw not found")
    return draw


@router.delete("/{draw_id}", status_code=204)
async def delete_draw(draw_id: UUID, db: Session = Depends(get_db)):
    """Delete a draw by ID."""
    draw = db.query(Draw).filter(Draw.id == draw_id).first()
    if not draw:
        raise HTTPException(status_code=404, detail="Draw not found")
    
    db.delete(draw)
    db.commit()
    return None
