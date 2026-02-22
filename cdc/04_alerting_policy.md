# Lotto Analyzer — Politique d’alerting (sévérité, seuils, evidence)

## 1) Niveaux
- **LOW** : signal faible, surveillance
- **MEDIUM** : signal persistant, investigation
- **HIGH** : signal fort + cohérent, investigation prioritaire

## 2) Catégories
- DATA_QUALITY
- DRIFT (PSI/KL)
- CHANGE_POINT (ruptures)
- OUTLIER (MAD z-score)
- MODEL_DIAGNOSTIC (Brier/ECE)

## 3) Seuils par défaut
### PSI
- LOW ≥ 0.10
- MEDIUM ≥ 0.25
- HIGH ≥ 0.50

### KL normalisé (z_KL via Monte Carlo)
- LOW z ≥ 2
- MEDIUM z ≥ 3
- HIGH z ≥ 4

### Outliers robustes (MAD z-score)
- LOW |z| ≥ 3
- MEDIUM |z| ≥ 4
- HIGH |z| ≥ 5

### Change points (ruptures)
- Méthode : PELT
- Penalty : 5 (faible sensibilité) / 3 (moyenne) / 2 (forte)
- HIGH si confirmé par ≥2 métriques et persistance ≥2 fenêtres

### Model diagnostic
- MEDIUM : ΔBrier > +1% (relatif) **ou** ECE > 0.05
- HIGH : ΔBrier > +3% **ou** ECE > 0.10

## 4) Agrégation multi-signaux (anti faux positifs)
- **MEDIUM** : 1 signal MEDIUM + 1 signal LOW cohérent
- **HIGH** : 2 signaux MEDIUM ou 1 signal HIGH + persistance

## 5) evidence_json (structure standard)
```json
{
  "type": "DRIFT",
  "metric": "PSI",
  "window": 200,
  "value": 0.31,
  "thresholds": { "low": 0.10, "medium": 0.25, "high": 0.50 },
  "period": { "from": "2025-03-01", "to": "2025-06-01" },
  "support": { "series_points": [["2025-01-01",0.05],["2025-02-01",0.12],["2025-03-01",0.28]] },
  "interpretation": "Dérive modérée persistante. Vérifier imports et règles.",
  "actions": ["Vérifier imports", "Comparer à simulation H0", "Inspecter période"]
}
```
