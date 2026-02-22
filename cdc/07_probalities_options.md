# Lotto Analyzer — Option “Probabilités” : Présentation des résultats (spécification UI/outputs)

> **But de ce document** : décrire, de façon “pro” et exploitable par WindSurf, **comment présenter** les résultats de chaque méthode (M0→M10) via **tableaux, graphiques, indicateurs, messages d’interprétation** et formats d’export.  
> **Important** : je ne peux pas fournir de “numéros recommandés à jouer” pour un prochain tirage. En revanche, la fonctionnalité peut afficher des **classements analytiques** (“Top N par probabilité estimée **sur un dataset**”) **à des fins de diagnostic**, avec baseline uniforme, incertitude et scoring — sans promesse d’avantage.

---

## 0) Principes d’UI/Reporting (non négociables)

### 0.1 Encarts obligatoires (UI + rapport)

- **Baseline uniforme** visible partout où une probabilité est affichée.
- **Avertissement** : “Les tirages sont supposés aléatoires. Ces probabilités sont des estimations statistiques et peuvent ne pas être meilleures que l’uniforme.”
- **Scores et calibration** : ne pas afficher des probabilités sans **Brier/ECE** (ou mention “non calculable” si dataset trop petit).
- **Intervalles d’incertitude** (IC/credible intervals) affichés si dispo.
- **Pas de wording incitatif** (“jouez”, “gagnez”, “numéros sûrs”).

### 0.2 Normalisation de sortie

Toutes les méthodes doivent produire un bloc standard :

- `number_probs` : probabilité estimée par numéro
- `baseline_probs` : baseline uniforme
- `delta_vs_baseline` : différence ou ratio vs baseline
- `uncertainty` : IC/credible intervals (quand possible)
- `evaluation` : Brier, ECE, lift, IC bootstrap (si backtest)
- `warnings` : (dataset petit, pas de gain significatif, etc.)
- `explain` : explication courte “mécanique de la méthode” (pour UI)

---

## 1) Écrans (UI) — structure recommandée

### 1.1 Page “Probabilités” (v1.1)

**A. Sélecteur**

- Game (format)
- Période (from/to)
- Paramètres : window_size, alpha, lambda, n_sim, backtest start, etc.
- Méthodes activées : M0/M1/M2 (+ M5/M6/M9/M10 en v2)

**B. Résumé exécutif (card)**

- Verdict : “Aucun gain vs baseline” / “Gain faible (non significatif)” / “Gain significatif (rare, à vérifier)”
- Tableau : scores par méthode (Brier, ECE, lift, IC)
- Warnings

**C. Onglets**

1) **Classement & distribution**
2) **Évaluation (backtest)**
3) **Calibration**
4) **Comparaison méthodes**
5) **Exports**

### 1.2 Rapport HTML imprimable

- Résumé (méthodes, paramètres)
- Graphiques clés
- Annexes : tables complètes, p-values (si liées), configuration, seeds/versions

---

## 2) Présentations “pro” : tableaux + graphiques (catalogue)

### 2.1 Tableaux standards

#### T1 — Leaderboard méthodes (obligatoire)

| Méthode           | Brier (↓) | ΔBrier vs M0 | IC 95% ΔBrier      | ECE (↓) | Lift vs M0 | Statut        |
| ----------------- | --------: | -----------: | ------------------ | ------: | ---------: | ------------- |
| M0 Uniforme       |    0.0204 |       0.0000 | —                  |   0.010 |       1.00 | Référence     |
| M1 Dirichlet      |    0.0203 |      -0.0001 | [-0.0004; +0.0002] |   0.011 |       1.01 | Non concluant |
| M2 Fenêtre+shrink |    0.0205 |      +0.0001 | [-0.0002; +0.0004] |   0.014 |       0.99 | Non concluant |

> Remarques : chiffres **exemples fictifs** (maquette).

#### T2 — Top N “probabilité estimée” (diagnostic, non incitatif)

| Rang | Numéro |  p_est | p_uniforme |       Δ | IC 95%         | Notes                         |
| ---: | -----: | -----: | ---------: | ------: | -------------- | ----------------------------- |
|    1 |      7 | 0.0240 |     0.0204 | +0.0036 | [0.018; 0.030] | Exemple (dataset synthétique) |
|    2 |     13 | 0.0235 |     0.0204 | +0.0031 | [0.017; 0.029] | Exemple                       |
|    3 |     42 | 0.0230 |     0.0204 | +0.0026 | [0.016; 0.028] | Exemple                       |
|    4 |      5 | 0.0220 |     0.0204 | +0.0016 | [0.015; 0.027] | Exemple                       |
|    5 |     19 | 0.0218 |     0.0204 | +0.0014 | [0.014; 0.027] | Exemple                       |

> **IMPORTANT** : ce tableau est un **classement analytique** et **n’est pas** une recommandation de jeu.

#### T3 — Distribution complète (export)

CSV : `numero, p_est, p_uniforme, delta, lower95, upper95`

### 2.2 Graphiques standards

- **G1** Bar chart : `p_est` vs `p_uniforme` (Top 20)
- **G2** Delta plot : Δ vs baseline sur 1..N
- **G3** IC whiskers sur Top N
- **G4** Score over time (walk-forward) : Brier glissant
- **G5** Calibration plot : reliability diagram + ECE
- **G6** Comparaison méthodes : mini-bars par score

---

## 3) Spécification par méthode : résultats attendus & visualisations

### M0 — Baseline uniforme

- Résultats : baseline `p(i)=1/N`
- Présentation : sert de référence partout (ligne/colonne baseline)

### M1 — Dirichlet (Bayésien)

- Résultats : `number_probs` lissées + `credible_interval_95`
- Présentation : T2 + G1/G3 + G5

### M2 — Fenêtre glissante + shrinkage

- Résultats : params `window_size`, `lambda`
- Présentation : tableau sensibilité window/λ + G4 + G5

### M3 — Diagnostic features (parité/somme/suites)

- Résultats : probas sur propriétés (pas forcément par numéro)
- Présentation : tableau “propriétés du prochain tirage” + bar chart

### M4 — ML multi-label (expérimental)

- Résultats : `number_probs` + calibration obligatoire + warnings anti-overfit
- Présentation : leaderboard + G4/G5 + checks sur-ajustement

---

## 4) Méthodes avancées (v2/v3) — sorties “parlantes”

### M5 — Co-occurrence + tests vs attendu

- Sorties : matrices obs/exp/delta + p-values + FDR
- Présentation : heatmap obs + heatmap delta + table top paires (FDR)

### M6 — Gaps & streaks

- Sorties : distributions gaps/runs + tests vs Monte Carlo
- Présentation : histogram gaps (numéro sélectionné) + table numéros atypiques

### M7 — BOCPD

- Sorties : `P(change_point|data)` par date
- Présentation : courbe prob rupture + table top dates + métriques impactées

### M8 — Noncentral Hypergeometric (Wallenius/Fisher)

- Sorties : poids estimés `w_i` + incertitude + test global
- Présentation : bar chart poids vs 1.0 + top/bottom weights

### M9 — Meta-test p-values

- Sorties : QQ plot data + KS test + drift par périodes
- Présentation : QQ plot + verdict “batterie OK / trop significative / non concluant”

### M10 — Ensemble / stacking calibré

- Sorties : poids du stacking + probas combinées + calibration + IC
- Présentation : table poids + leaderboard + calibration + score time

---

## 5) Exports (pour WindSurf)

### 5.1 CSV

- `leaderboard.csv` : scores par méthode
- `probabilities_<method>.csv` : distribution complète
- `predictions_log.csv` : walk-forward (date, proba, observed)

### 5.2 HTML

- Rapport imprimable : résumé + graphiques + annexes
- Inclure `dataset_hash`, `code_version`, `seed`, paramètres

---

## 6) Contrats API (résumé)

### Request

```json
{
  "game_id": "uuid",
  "analysis_name": "forecast_probabilities_v1",
  "params": {
    "from": "2020-01-01",
    "to": "2025-12-31",
    "methods": ["M0","M1","M2","M10"],
    "m1": { "alpha": 1.0 },
    "m2": { "window_size": 200, "lambda": 0.3 },
    "backtest": { "enabled": true, "start": "2021-01-01", "bootstrap": 500, "seed": 42 },
    "calibration": { "bins": 10 },
    "monte_carlo": { "enabled": true, "n_sim": 2000, "seed": 42 }
  }
}
```

### Response (structure)

```json
{
  "analysis_id": "uuid",
  "dataset_hash": "sha256:...",
  "code_version": "v1.1.0",
  "results": {
    "summary": { "verdict": "non_conclusive", "notes": ["..."] },
    "leaderboard": [ ... ],
    "methods": {
      "M0": { "number_probs": {...}, "evaluation": {...}, "warnings": [...] },
      "M1": { "number_probs": {...}, "uncertainty": {...}, "evaluation": {...} }
    },
    "charts": {
      "prob_bar_top20": { "labels": [...], "series": [...] },
      "delta_all_numbers": { "x": [...], "y": [...] },
      "calibration": { "bin_centers": [...], "pred": [...], "obs": [...] },
      "score_over_time": { "dates": [...], "brier_M0": [...], "brier_M1": [...] }
    }
  }
}
```

---

## 7) Clarification sur “numéros recommandés”

Je ne peux pas fournir de numéros à jouer ni de recommandations de mise.  
**Pour le développement**, l’app peut afficher un “Top N par probabilité estimée” **uniquement** comme sortie analytique (diagnostic), avec baseline + incertitude + scores + avertissement explicite.

