import csv
import hashlib
from datetime import datetime
from io import StringIO
from typing import List
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.db.models import Game, Draw, Import
from app.schemas.import_schema import ImportResponse, ImportPreviewRow, ImportError


class ImportService:
    def __init__(self, db: Session):
        self.db = db

    async def import_csv(self, game_id: UUID, content: str, mode: str) -> ImportResponse:
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")

        rules = game.rules_json
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Remove UTF-8 BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]

        reader = csv.DictReader(StringIO(content), delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)
        
        print(f"[IMPORT DEBUG] CSV fieldnames: {fieldnames}")
        if rows:
            print(f"[IMPORT DEBUG] First row keys: {list(rows[0].keys())}")
            print(f"[IMPORT DEBUG] First row draw_number value: '{rows[0].get('draw_number', 'NOT FOUND')}'")

        valid_rows = []
        invalid_rows = []
        errors: List[ImportError] = []
        preview_rows: List[ImportPreviewRow] = []

        # Extract rules - use 'drawn' for import (how many numbers are drawn in the lottery)
        # 'pick' is how many the player chooses, 'drawn' is how many are drawn
        main_rules = rules.get("main", {})
        numbers_rules = rules.get("numbers", {})
        # For import, we need the number of drawn numbers (not player picks)
        main_drawn = main_rules.get("drawn", main_rules.get("pick", numbers_rules.get("count", 7)))
        main_min = main_rules.get("min", numbers_rules.get("min", 1))
        main_max = main_rules.get("max", numbers_rules.get("max", 49))
        
        bonus_rules = rules.get("bonus", {})
        bonus_enabled = bonus_rules.get("enabled", False)
        # For bonus, use 'drawn' (supplementary numbers drawn) not 'pick' (player's choice)
        bonus_drawn = bonus_rules.get("drawn", bonus_rules.get("pick", bonus_rules.get("count", 1))) if bonus_enabled else 0
        bonus_min = bonus_rules.get("min", main_min)
        bonus_max = bonus_rules.get("max", main_max)

        # Check if CSV has draw_number column (use fieldnames captured before list())
        has_draw_number_column = "draw_number" in fieldnames if fieldnames else False
        print(f"[IMPORT DEBUG] has_draw_number_column: {has_draw_number_column}")
        
        # Get the max existing draw_number for this game to auto-generate if needed
        max_existing_draw_number = 0
        if not has_draw_number_column:
            from sqlalchemy import func
            result = self.db.query(func.max(Draw.draw_number)).filter(Draw.game_id == game_id).scalar()
            max_existing_draw_number = result if result else 0

        for idx, row in enumerate(rows, start=1):
            try:
                # Parse draw_number if present, otherwise auto-generate
                draw_number = None
                draw_number_str = row.get("draw_number", "").strip()
                if draw_number_str:
                    draw_number = int(draw_number_str)
                else:
                    # Auto-generate draw_number based on row index
                    draw_number = max_existing_draw_number + idx
                
                draw_date_str = row.get("draw_date", "").strip()
                draw_date = datetime.fromisoformat(draw_date_str)

                numbers = []
                for i in range(1, main_drawn + 1):
                    num_str = row.get(f"n{i}", "").strip()
                    if not num_str:
                        raise ValueError(f"Missing n{i}")
                    num = int(num_str)
                    if num < main_min or num > main_max:
                        raise ValueError(f"n{i} out of range ({main_min}-{main_max})")
                    numbers.append(num)

                if len(numbers) != len(set(numbers)):
                    raise ValueError("Duplicate numbers in draw")

                numbers.sort()

                bonus_numbers = []
                if bonus_enabled and bonus_drawn > 0:
                    for i in range(1, bonus_drawn + 1):
                        bonus_str = row.get(f"bonus{i}", "").strip()
                        if bonus_str:
                            bonus_num = int(bonus_str)
                            if bonus_num < bonus_min or bonus_num > bonus_max:
                                raise ValueError(f"Bonus{i} out of range ({bonus_min}-{bonus_max})")
                            bonus_numbers.append(bonus_num)

                # V2.0: Parse emission order (order in which balls were drawn)
                emission_order = []
                for i in range(1, main_drawn + 1):
                    order_str = row.get(f"order{i}", "").strip()
                    if order_str:
                        order_num = int(order_str)
                        if order_num < main_min or order_num > main_max:
                            raise ValueError(f"order{i} out of range ({main_min}-{main_max})")
                        emission_order.append(order_num)
                
                # If emission order provided, validate it matches the numbers
                if emission_order:
                    if len(emission_order) != main_drawn:
                        raise ValueError(f"Emission order must have {main_drawn} numbers")
                    if set(emission_order) != set(numbers):
                        raise ValueError("Emission order numbers must match draw numbers")
                
                # V2.0: Parse bonus emission order
                bonus_emission_order = []
                if bonus_enabled and bonus_drawn > 0:
                    for i in range(1, bonus_drawn + 1):
                        order_str = row.get(f"bonus_order{i}", "").strip()
                        if order_str:
                            order_num = int(order_str)
                            bonus_emission_order.append(order_num)
                
                # V2.0: Parse jackpot information
                jackpot_amount = None
                jackpot_str = row.get("jackpot", "").strip()
                if jackpot_str:
                    # Remove currency symbols and spaces, handle comma as decimal
                    jackpot_clean = jackpot_str.replace("€", "").replace("$", "").replace(" ", "").replace(",", ".")
                    jackpot_amount = float(jackpot_clean)
                
                jackpot_rollover = False
                rollover_str = row.get("rollover", "").strip().lower()
                if rollover_str in ("true", "1", "yes", "oui"):
                    jackpot_rollover = True
                
                jackpot_consecutive_rollovers = 0
                consec_str = row.get("consecutive_rollovers", "").strip()
                if consec_str:
                    jackpot_consecutive_rollovers = int(consec_str)
                
                must_be_won = False
                mbw_str = row.get("must_be_won", "").strip().lower()
                if mbw_str in ("true", "1", "yes", "oui"):
                    must_be_won = True
                
                n_winners_div1 = None
                winners_str = row.get("winners_div1", "").strip()
                if winners_str:
                    n_winners_div1 = int(winners_str)

                valid_rows.append({
                    "draw_number": draw_number, 
                    "draw_date": draw_date, 
                    "numbers": numbers, 
                    "bonus_numbers": bonus_numbers,
                    "emission_order": emission_order if emission_order else None,
                    "bonus_emission_order": bonus_emission_order if bonus_emission_order else None,
                    "jackpot_amount": jackpot_amount,
                    "jackpot_rollover": jackpot_rollover,
                    "jackpot_consecutive_rollovers": jackpot_consecutive_rollovers,
                    "must_be_won": must_be_won,
                    "n_winners_div1": n_winners_div1,
                    "raw": row
                })

                if len(preview_rows) < 10:
                    preview_rows.append(
                        ImportPreviewRow(
                            draw_number=draw_number, 
                            draw_date=draw_date_str, 
                            numbers=numbers, 
                            bonus_numbers=bonus_numbers,
                            emission_order=emission_order if emission_order else None,
                            bonus_emission_order=bonus_emission_order if bonus_emission_order else None,
                            jackpot_amount=jackpot_amount,
                            jackpot_rollover=jackpot_rollover,
                            must_be_won=must_be_won,
                            n_winners_div1=n_winners_div1
                        )
                    )

            except Exception as e:
                invalid_rows.append(row)
                errors.append(ImportError(row=idx, field=None, message=str(e)))

        import_id = uuid4()

        if mode == "commit":
            for valid_row in valid_rows:
                existing = (
                    self.db.query(Draw)
                    .filter(
                        Draw.game_id == game_id,
                        Draw.draw_date == valid_row["draw_date"],
                        Draw.numbers == valid_row["numbers"],
                    )
                    .first()
                )
                if not existing:
                    draw = Draw(
                        game_id=game_id,
                        draw_number=valid_row["draw_number"],
                        draw_date=valid_row["draw_date"],
                        numbers=valid_row["numbers"],
                        bonus_numbers=valid_row["bonus_numbers"],
                        # V2.0: New columns
                        emission_order=valid_row.get("emission_order"),
                        bonus_emission_order=valid_row.get("bonus_emission_order"),
                        jackpot_amount=valid_row.get("jackpot_amount"),
                        jackpot_rollover=valid_row.get("jackpot_rollover", False),
                        jackpot_consecutive_rollovers=valid_row.get("jackpot_consecutive_rollovers", 0),
                        must_be_won=valid_row.get("must_be_won", False),
                        n_winners_div1=valid_row.get("n_winners_div1"),
                        raw_payload=valid_row["raw"],
                    )
                    self.db.add(draw)

            import_record = Import(
                id=import_id,
                game_id=game_id,
                source="upload",
                file_hash=file_hash,
                status="committed",
                stats_json={
                    "total_rows": len(rows),
                    "valid_rows": len(valid_rows),
                    "invalid_rows": len(invalid_rows),
                },
                error_log=[e.dict() for e in errors],
            )
            self.db.add(import_record)
            self.db.commit()
        else:
            import_record = Import(
                id=import_id,
                game_id=game_id,
                source="upload",
                file_hash=file_hash,
                status="preview",
                stats_json={
                    "total_rows": len(rows),
                    "valid_rows": len(valid_rows),
                    "invalid_rows": len(invalid_rows),
                },
                error_log=[e.dict() for e in errors],
            )
            self.db.add(import_record)
            self.db.commit()

        return ImportResponse(
            import_id=import_id,
            mode=mode,
            total_rows=len(rows),
            valid_rows=len(valid_rows),
            invalid_rows=len(invalid_rows),
            preview_rows=preview_rows,
            errors=errors,
        )
