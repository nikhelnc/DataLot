# Lotto Analyzer — Spécification dataset de seed (synthétique)

## 1) Objectif
Fournir un dataset réaliste pour :
- tests unitaires / intégration
- démonstration UI
- validation du pipeline d’alerting

Le dataset est **majoritairement uniforme** + anomalies injectées contrôlées.

## 2) Format
CSV : `draw_date;n1;n2;n3;n4;n5;bonus` (exemple 5/49 + bonus 1/10)

Exemple :
```csv
draw_date;n1;n2;n3;n4;n5;bonus
2024-01-03;4;11;19;33;47;8
2024-01-06;2;16;21;27;41;1
```

## 3) Scénarios injectés
- **S0 Normal** : tirage uniforme (2 ans)
- **S1 Drift léger** : sur 3 mois, poids léger sur (7,13,42), ex: w=1.2
- **S2 Rupture** : sur 1 mois, poids plus fort sur (7,13,42), ex: w=1.8
- **S3 Data quality** :
  - 20 lignes bonus manquant
  - 10 lignes hors bornes (0 ou 50)
  - 15 doublons exacts
- **S4 Outliers** :
  - 10 tirages somme très basse
  - 10 tirages somme très haute

## 4) Génération (spécification)
1. Générer calendrier selon rules_json (`calendar.days`).
2. Pour chaque date :
   - tirer k numéros sans remise sur [min,max]
   - tirer bonus si activé
3. Injecter S1/S2 :
   - pondérer quelques numéros via sampling biaisé (mais sans remise)
4. Injecter S4 :
   - resampler jusqu’à respecter contrainte somme < seuil ou > seuil
5. Injecter S3 :
   - post-process sur un sous-ensemble (invalid, missing, duplicates)
6. Export CSV + seed reproductible.

## 5) Livrables attendus
- `backend/app/db/seed/seed_lotto_5_49_bonus.csv`
- `backend/app/db/seed/generate_seed.py` (CLI `--seed 42 --out ...`)
- Script d’insertion DB (optionnel) au démarrage dev
