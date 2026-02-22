# Lotto Analyzer — Spécification API (v1)

## 1) Principes
- API REST JSON (FastAPI)
- Réponses versionnées (inclure `code_version` et `dataset_hash` dans les analyses)
- Pagination sur listes volumineuses
- Erreurs structurées (`error_code`, `message`, `details`)

---

## 2) Modèles (schemas)

### 2.1 Game
```json
{
  "id": "uuid",
  "name": "Lotto_5_49",
  "description": "Exemple 5/49 + bonus",
  "rules_json": { },
  "version": 1,
  "created_at": "iso-datetime"
}
```

### 2.2 Draw
```json
{
  "id": "uuid",
  "game_id": "uuid",
  "draw_date": "YYYY-MM-DD",
  "numbers": [4, 11, 19, 33, 47],
  "bonus": 8,
  "created_at": "iso-datetime"
}
```

### 2.3 Import
```json
{
  "id": "uuid",
  "game_id": "uuid",
  "source": "upload",
  "file_hash": "sha256:...",
  "status": "preview|committed|failed",
  "stats_json": { },
  "error_log": [ ],
  "created_at": "iso-datetime"
}
```

### 2.4 Analysis (standard)
```json
{
  "analysis_id": "uuid",
  "game_id": "uuid",
  "name": "randomness_tests_v1",
  "dataset_hash": "sha256:...",
  "code_version": "v1.0.0",
  "params": { },
  "results": { },
  "created_at": "iso-datetime"
}
```

### 2.5 Alert
```json
{
  "id": "uuid",
  "game_id": "uuid",
  "analysis_id": "uuid",
  "severity": "low|medium|high",
  "score": 0,
  "message": "string",
  "evidence_json": { },
  "created_at": "iso-datetime"
}
```

---

## 3) Endpoints

### 3.1 Health
- `GET /health`
Response:
```json
{ "status": "ok", "version": "v1.0.0" }
```

### 3.2 Games
- `POST /games`
- `GET /games`
- `GET /games/{id}`

### 3.3 Draws
- `GET /draws?game_id=&from=&to=&page=&page_size=`
- `GET /draws/{id}` *(inclure métriques dérivées optionnelles)*

### 3.4 Import
- `POST /draws/import?mode=preview|commit`
  - multipart file + (optionnel) mapping colonnes
Response (preview/commit):
```json
{
  "import_id": "uuid",
  "mode": "preview",
  "total_rows": 1000,
  "valid_rows": 985,
  "invalid_rows": 15,
  "preview_rows": [ { "draw_date": "...", "numbers": [...], "bonus": 1 } ],
  "errors": [ { "row": 12, "field": "n3", "message": "out_of_range" } ]
}
```

- `GET /imports/{id}`
- `GET /imports/{id}/errors`

### 3.5 Analyses
- `POST /analyses/run`
Request:
```json
{
  "game_id": "uuid",
  "analysis_name": "forecast_probabilities_v1",
  "params": { }
}
```
- `GET /analyses/{id}`
- `GET /analyses/{id}/export.csv`
- `GET /analyses/{id}/report.html`

### 3.6 Alerts
- `GET /alerts?game_id=&from=&to=&severity=`
- `GET /alerts/{id}`

---

## 4) Conventions résultats d’analyse (results_json)

### 4.1 Champs attendus
- `summary`
- `warnings`
- `metrics`
- `tests`
- `anomalies`
- `probabilities`
- `evaluation`

### 4.2 Probabilities (standard)
```json
{
  "method_id": "M1_dirichlet",
  "number_probs": { "1": 0.020, "2": 0.019, "...": 0.021 },
  "baseline_probs": { "1": 0.0204, "...": 0.0204 },
  "top_numbers": [7, 13, 42, 5, 19],
  "uncertainty": { "credible_interval_95": { "7": [0.015, 0.030] } },
  "warnings": ["No statistically significant lift vs baseline"]
}
```
