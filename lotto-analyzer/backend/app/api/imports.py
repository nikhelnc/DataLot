from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.import_schema import ImportResponse, ImportStatusResponse
from app.services.import_service import ImportService

router = APIRouter(prefix="/draws/import", tags=["imports"])


@router.post("", response_model=ImportResponse)
async def import_draws(
    game_id: UUID = Query(...),
    mode: str = Query("preview", regex="^(preview|commit)$"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    service = ImportService(db)
    content = await file.read()
    result = await service.import_csv(game_id, content.decode("utf-8"), mode)
    return result


@router.get("/{import_id}", response_model=ImportStatusResponse)
async def get_import_status(import_id: UUID, db: Session = Depends(get_db)):
    from app.db.models import Import
    
    import_record = db.query(Import).filter(Import.id == import_id).first()
    if not import_record:
        raise HTTPException(status_code=404, detail="Import not found")
    
    return ImportStatusResponse(
        id=import_record.id,
        game_id=import_record.game_id,
        source=import_record.source,
        file_hash=import_record.file_hash,
        status=import_record.status,
        stats_json=import_record.stats_json,
        error_log=import_record.error_log,
        created_at=import_record.created_at,
    )
