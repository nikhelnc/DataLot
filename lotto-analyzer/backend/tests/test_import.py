import pytest
from io import BytesIO


def test_import_preview_valid_csv(client, sample_game):
    csv_content = """draw_date;n1;n2;n3;n4;n5;bonus
2024-01-01;5;12;23;34;45;7
2024-01-04;3;15;27;38;42;2
2024-01-07;8;19;25;33;48;9"""
    
    response = client.post(
        f"/draws/import?game_id={sample_game.id}&mode=preview",
        files={"file": ("test.csv", BytesIO(csv_content.encode()), "text/csv")},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["mode"] == "preview"
    assert data["total_rows"] == 3
    assert data["valid_rows"] == 3
    assert data["invalid_rows"] == 0
    assert len(data["preview_rows"]) == 3
    assert len(data["errors"]) == 0


def test_import_preview_invalid_csv(client, sample_game):
    csv_content = """draw_date;n1;n2;n3;n4;n5;bonus
2024-01-01;5;12;23;34;45;7
2024-01-04;3;15;27;38;60;2
2024-01-07;8;19;25;33;48;9"""
    
    response = client.post(
        f"/draws/import?game_id={sample_game.id}&mode=preview",
        files={"file": ("test.csv", BytesIO(csv_content.encode()), "text/csv")},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_rows"] == 3
    assert data["valid_rows"] == 2
    assert data["invalid_rows"] == 1
    assert len(data["errors"]) == 1


def test_import_commit(client, sample_game):
    csv_content = """draw_date;n1;n2;n3;n4;n5;bonus
2024-01-01;5;12;23;34;45;7
2024-01-04;3;15;27;38;42;2"""
    
    response = client.post(
        f"/draws/import?game_id={sample_game.id}&mode=commit",
        files={"file": ("test.csv", BytesIO(csv_content.encode()), "text/csv")},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["mode"] == "commit"
    assert data["valid_rows"] == 2
    
    draws_response = client.get(f"/draws?game_id={sample_game.id}")
    assert draws_response.status_code == 200
    draws = draws_response.json()
    assert len(draws) == 2
