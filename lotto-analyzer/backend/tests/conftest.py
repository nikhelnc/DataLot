import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime, timedelta

from app.db.database import Base, get_db
from app.db.models import Game, Draw
from app.main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_game(db_session):
    game = Game(
        name="Test_Lotto_5_49",
        description="Test game",
        rules_json={
            "numbers": {"count": 5, "min": 1, "max": 49, "unique": True, "sorted": True},
            "bonus": {"enabled": True, "min": 1, "max": 10},
            "calendar": {"expected_frequency": "weekly", "days": ["WED", "SAT"]},
        },
    )
    db_session.add(game)
    db_session.commit()
    db_session.refresh(game)
    return game


@pytest.fixture
def sample_draws(db_session, sample_game):
    draws = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(100):
        draw = Draw(
            game_id=sample_game.id,
            draw_date=start_date + timedelta(days=i * 3),
            numbers=[1 + (i % 5), 10 + (i % 10), 20 + (i % 15), 30 + (i % 10), 40 + (i % 9)],
            bonus=(i % 10) + 1,
        )
        draws.append(draw)
        db_session.add(draw)
    
    db_session.commit()
    return draws


@pytest.fixture
def sample_dataframe():
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(100):
        data.append({
            "draw_date": start_date + timedelta(days=i * 3),
            "numbers": [1 + (i % 5), 10 + (i % 10), 20 + (i % 15), 30 + (i % 10), 40 + (i % 9)],
            "bonus": (i % 10) + 1,
        })
    
    return pd.DataFrame(data)
