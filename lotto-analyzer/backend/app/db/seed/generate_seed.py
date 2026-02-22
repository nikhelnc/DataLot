import argparse
import csv
import random
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np


class SeedGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.n_min = 1
        self.n_max = 49
        self.n_count = 5
        self.bonus_min = 1
        self.bonus_max = 10

    def generate_uniform_draw(self) -> Tuple[List[int], int]:
        numbers = sorted(random.sample(range(self.n_min, self.n_max + 1), self.n_count))
        bonus = random.randint(self.bonus_min, self.bonus_max)
        return numbers, bonus

    def generate_biased_draw(self, bias_numbers: List[int], weight: float) -> Tuple[List[int], int]:
        weights = []
        for num in range(self.n_min, self.n_max + 1):
            if num in bias_numbers:
                weights.append(weight)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        numbers = []
        available = list(range(self.n_min, self.n_max + 1))
        available_weights = weights.copy()
        
        for _ in range(self.n_count):
            idx = np.random.choice(len(available), p=available_weights / available_weights.sum())
            numbers.append(available[idx])
            available.pop(idx)
            available_weights = np.delete(available_weights, idx)
        
        numbers.sort()
        bonus = random.randint(self.bonus_min, self.bonus_max)
        return numbers, bonus

    def generate_outlier_draw(self, target_sum: str) -> Tuple[List[int], int]:
        max_attempts = 1000
        
        for _ in range(max_attempts):
            numbers = sorted(random.sample(range(self.n_min, self.n_max + 1), self.n_count))
            draw_sum = sum(numbers)
            
            if target_sum == "low" and draw_sum < 60:
                bonus = random.randint(self.bonus_min, self.bonus_max)
                return numbers, bonus
            elif target_sum == "high" and draw_sum > 200:
                bonus = random.randint(self.bonus_min, self.bonus_max)
                return numbers, bonus
        
        return self.generate_uniform_draw()

    def generate_dataset(self, output_path: str):
        start_date = datetime(2022, 1, 5)
        draws = []
        
        current_date = start_date
        draw_count = 0
        
        print("Generating S0: Normal uniform draws (2 years)...")
        while current_date < datetime(2024, 1, 1):
            numbers, bonus = self.generate_uniform_draw()
            draws.append((current_date, numbers, bonus))
            
            current_date += timedelta(days=3 if current_date.weekday() == 2 else 4)
            draw_count += 1
        
        print(f"Generated {draw_count} normal draws")
        
        print("Generating S1: Drift (3 months, weight 1.2)...")
        drift_start = len(draws)
        drift_numbers = [7, 13, 42]
        drift_count = 0
        
        while current_date < datetime(2024, 4, 1):
            numbers, bonus = self.generate_biased_draw(drift_numbers, 1.2)
            draws.append((current_date, numbers, bonus))
            
            current_date += timedelta(days=3 if current_date.weekday() == 2 else 4)
            drift_count += 1
        
        print(f"Generated {drift_count} drift draws")
        
        print("Generating S2: Rupture (1 month, weight 1.8)...")
        rupture_start = len(draws)
        rupture_count = 0
        
        while current_date < datetime(2024, 5, 1):
            numbers, bonus = self.generate_biased_draw(drift_numbers, 1.8)
            draws.append((current_date, numbers, bonus))
            
            current_date += timedelta(days=3 if current_date.weekday() == 2 else 4)
            rupture_count += 1
        
        print(f"Generated {rupture_count} rupture draws")
        
        print("Generating normal draws until present...")
        while current_date < datetime(2024, 12, 1):
            numbers, bonus = self.generate_uniform_draw()
            draws.append((current_date, numbers, bonus))
            
            current_date += timedelta(days=3 if current_date.weekday() == 2 else 4)
        
        print("Injecting S4: Outliers...")
        outlier_indices = random.sample(range(len(draws)), 20)
        outlier_count = 0
        
        for idx in outlier_indices[:10]:
            numbers, bonus = self.generate_outlier_draw("low")
            draws[idx] = (draws[idx][0], numbers, bonus)
            outlier_count += 1
        
        for idx in outlier_indices[10:]:
            numbers, bonus = self.generate_outlier_draw("high")
            draws[idx] = (draws[idx][0], numbers, bonus)
            outlier_count += 1
        
        print(f"Injected {outlier_count} outliers")
        
        print("Injecting S3: Data quality issues...")
        quality_indices = random.sample(range(len(draws)), 45)
        
        for idx in quality_indices[:20]:
            date, numbers, _ = draws[idx]
            draws[idx] = (date, numbers, None)
        
        for idx in quality_indices[20:30]:
            date, numbers, bonus = draws[idx]
            invalid_numbers = numbers.copy()
            invalid_numbers[0] = 0
            draws[idx] = (date, invalid_numbers, bonus)
        
        duplicate_base = draws[50]
        for i in range(15):
            draws.append(duplicate_base)
        
        print(f"Injected 20 missing bonus, 10 invalid numbers, 15 duplicates")
        
        draws.sort(key=lambda x: x[0])
        
        print(f"Writing {len(draws)} draws to {output_path}...")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["draw_date", "n1", "n2", "n3", "n4", "n5", "bonus"])
            
            for date, numbers, bonus in draws:
                row = [date.strftime("%Y-%m-%d")] + numbers
                if bonus is not None:
                    row.append(bonus)
                else:
                    row.append("")
                writer.writerow(row)
        
        print(f"✓ Dataset generated successfully: {output_path}")
        print(f"  Total draws: {len(draws)}")
        print(f"  Date range: {draws[0][0].date()} to {draws[-1][0].date()}")
        print(f"  Seed: {self.seed}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic lottery dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="seed_lotto_5_49_bonus.csv", help="Output file path")
    
    args = parser.parse_args()
    
    generator = SeedGenerator(seed=args.seed)
    generator.generate_dataset(args.out)


if __name__ == "__main__":
    main()
