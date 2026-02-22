# Lotto Analyzer — Cahier des charges & spécification méthodologique (WindSurf)

> Document de référence (Markdown) pour thèse + application.  
> Objectif : ingestion, contrôle qualité, analyses statistiques, détection d’anomalies, modèles probabilistes **comparés à un baseline uniforme**, avec résultats **reproductibles** et **responsables** (pas de promesse de gain).

---

## 1) Vision et principes

### 1.1 Finalité
- **Explorer** des historiques de tirages (Lotto-like) multi-jeux.
- Produire des **indicateurs** et **visualisations**.
- Exécuter des **tests de hasard** (uniformité, indépendance, stabilité).
- **Détecter des anomalies** (données, ruptures, drift, outliers).
- Proposer des **modèles probabilistes** pour le prochain tirage **à valeur scientifique** : *probabilités estimées*, *incertitude*, *calibration*, *scores* vs baseline.

### 1.2 Garde-fous (obligatoires)
- Loterie saine ≈ processus **i.i.d. uniforme sans remise** (H0).
- Toute “tendance” apparente peut être due au hasard + biais de sélection + multiples tests.
- L’application **doit** :
  - toujours afficher le **baseline uniforme**,
  - appliquer des **corrections multi-tests**,
  - fournir **intervalles d’incertitude**,
  - évaluer les modèles avec **walk-forward** (pas de fuite temporelle),
  - interdire tout wording “garanti”, “gagner”, “numéros gagnants”.

---

## 2) Périmètre

### 2.1 Inclus
- Import CSV (JSON optionnel), validation, normalisation, audit.
- Gestion de jeux (rules_json versionné).
- Moteur d’analyse : métriques, tests, anomalies, modèles probabilistes, reporting.
- UI : import, dashboard, tests, alertes, probabilités, exports.
- API + DB + Docker Compose + tests unitaires.

### 2.2 Exclu (par défaut)
- “Conseils de jeu” / “numéros à jouer”.
- Données externes non publiques (machine, opérateur, etc.) sauf extension v2/v3.

---

## 3) Parties prenantes / Personas
- **Chercheur/doctorant** : lance analyses, compare méthodes, exporte pour thèse.
- **Data analyst** : explore, contrôle qualité, édite rapports.
- **Admin technique** : gère règles, supervision, déploiement.

---

## 4) Données, formats et validation

### 4.1 Format CSV (v1)
Colonnes minimales : `draw_date; n1; n2; ...; nk; bonus(optional)`

- `draw_date` : format ISO recommandé (`YYYY-MM-DD`)
- Séparateur : `;` par défaut (configurable)
- Encodage : UTF-8 (détection/override possible)

### 4.2 Règles d’un Game (`rules_json`)
Exemple 5/49 + bonus 1/10 :
```json
{
  "numbers": { "count": 5, "min": 1, "max": 49, "unique": true, "sorted": true },
  "bonus": { "enabled": true, "min": 1, "max": 10 },
  "calendar": { "expected_frequency": "weekly", "days": ["WED", "SAT"] }
}
```

### 4.3 Validation stricte
- Nombre de boules exact `k`
- Unicité intra-tirage
- Bornes min/max
- Tri/normalisation selon règle
- Gestion des doublons : (game_id, date, numbers_sorted, bonus)
- Audit import : `file_hash`, mapping, stats, erreurs ligne/champ

---

## 5) Architecture cible (WindSurf)

### 5.1 Stack (v1)
- Backend : Python 3.12 + FastAPI
- Analyse : pandas, numpy, scipy, statsmodels, ruptures  
  *(optionnel : scikit-learn ; avancé : PyMC/Stan)*
- DB : PostgreSQL + SQLAlchemy + Alembic
- Front : React + Vite + TypeScript + Recharts/Chart.js
- Infra : Docker Compose (api + db + front)
- Qualité : pytest, ruff/black, (optionnel mypy), pre-commit
- Observabilité : logs structurés + `/health`

### 5.2 Arborescence recommandée
```
lotto-analyzer/
  backend/
    app/
      api/ (routes)
      services/ (use-cases)
      analysis/ (moteur)
        metrics.py
        randomness.py
        anomalies.py
        prob_models/
        evaluation/
        reporting/
      db/ (models+migrations)
      schemas/
      utils/
    tests/
  frontend/
    src/ (pages, components, api)
  infra/ (docker-compose, Dockerfiles)
  docs/ (méthodologie, limites)
  README.md
```

---

## 6) Modèle de données (minimum)
- `games(id, name, rules_json, version, created_at)`
- `draws(id, game_id, draw_date, numbers_int[], bonus_int, raw_payload, created_at)`
- `imports(id, game_id, source, file_hash, status, stats_json, error_log, created_at)`
- `analyses(id, game_id, name, params_json, results_json, dataset_hash, code_version, created_at)`
- `alerts(id, game_id, analysis_id, severity, score, message, evidence_json, created_at)`

---

## 7) Indicateurs (catalogue complet)

> Tous les indicateurs doivent être calculables : global, par période, par fenêtre glissante, exportables.

### A) Fréquences & écarts à l’uniforme
- Fréquence par numéro / bonus
- Expected count vs observed
- z-score (écart standardisé)
- Écart relatif (observé/attendu)
- Entropie (Shannon)
- Divergence KL (observé vs uniforme) *(normalisée via simulation si possible)*
- PSI (stabilité population) entre périodes/fenêtres

### B) Structure des tirages
- Somme, moyenne, variance
- Pair/impair ; bas/haut (seuil médian)
- Modulo classes (ex: mod 5)
- Suites (consecutifs) : nb, max length
- Range (max-min), spacing (écart min/max)
- Intersection avec tirage précédent (nb communs)
- Gaps par numéro : temps entre occurrences (moyenne, médiane, distribution)

### C) Co-occurrences
- Matrice NxN de co-occurrence (numéros apparaissant ensemble)
- Mesures : lift, z-score vs attendu analytique/simulé

### D) Qualité des données
- Taux invalides/doublons/hors bornes
- Dates manquantes/irrégularités
- Changements de format (détection)

### E) Modèles & évaluation
- Brier score (multi-label par numéro)
- Log-loss (si formulation adaptée)
- Calibration : reliability diagram + ECE
- Lift vs baseline + IC (bootstrap)

---

## 8) Méthodes d’analyse — baseline + avancées

### 8.1 Descriptif (obligatoire)
- Graphs : histogrammes fréquences, séries (entropie/KL/PSI), distribution somme, parité, heatmap
- Comparaisons : périodes (avant/après), fenêtres glissantes

### 8.2 Tests de hasard (obligatoires)
- Uniformité : χ² sur fréquences (numéros, bonus)
- Indépendance :
  - runs tests (pair/impair, haut/bas, etc.)
  - autocorrélation sur séries dérivées
  - permutation/Monte Carlo (référence robuste)
- Corrections multi-tests : FDR (BH) par familles
- Méta-test : distribution des p-values (doivent être uniformes sous H0)

### 8.3 Détection d’anomalies (obligatoire)
- Drift : PSI/KL (avec normalisation via simulation)
- Change points : ruptures (PELT/Binseg) sur entropie/KL/PSI/fréquences glissantes
- Outliers : robust z-score (MAD) sur somme/suites/intersections/range
- Production d’alertes : score, sévérité, evidence (graph + explication)

---

## 9) Fonction “Probabilités prochain tirage” — familles de modèles

> Sortie standard : distribution `numero -> proba` + baseline + incertitude + scoring + calibration + avertissements.  
> Important : **le modèle doit être évalué** en walk-forward. Si pas de gain vs baseline → le dire.

### Standard de sortie (extrait)
- `method_id`, `number_probs`, `top_numbers` *(optionnel)*  
- `uncertainty` : IC/credible intervals  
- `evaluation` : Brier, ECE, lift, IC bootstrap  
- `warnings` : données insuffisantes, gain non significatif, etc.

---

## 10) Méthodes “M0 → M10” (comparables)

### M0 — Baseline uniforme (référence)
- p(i) = 1/N pour chaque numéro ; tirage sans remise.
- Sert de référence **obligatoire**.

### M1 — Bayésien Dirichlet–Multinomial (lissage)
- Prior α ; posterior α+counts → probas lissées.
- Sortie : probas + intervalles crédibles.

### M2 — Fenêtre glissante + shrinkage vers uniforme
- p = λ * p_window + (1-λ) * p_uniforme  
- Param : window_size, λ.

### M3 — Modèles sur “features” (diagnostic)
- Prédire des propriétés (parité, somme, suites…) ; utile pour détecter biais structurels.
- Peut rester au niveau “propriétés” sans prétendre mapper sur numéros.

### M4 — ML expérimental (multi-label)
- Modèles (logistic/GBM/forest) sur features glissantes ; sortie proba par numéro.
- Exigences : walk-forward, calibration (Platt/isotonic), régularisation, interprétation prudente.

---

## 11) Méthodes avancées “publiables / thèse-friendly”

### M5 — Co-occurrence analytique + test vs attendu (sans remise)
- Construire la matrice co-occurrence NxN.
- Comparer au modèle attendu d’un tirage sans remise (attendu analytique ou Monte Carlo).
- Détecter des sur/sous-cooccurrences (avec FDR).

**Complexité dev** : moyenne.

### M6 — Gaps & streaks (temps entre occurrences)
- Distribution des gaps et streaks par numéro.
- Comparaison vs simulation H0, p-values + FDR.

**Complexité dev** : moyenne.

### M7 — BOCPD (Bayesian Online Change Point Detection)
- Détection en ligne de ruptures (posterior de change points).
- Appliqué aux séries entropie/KL/PSI.

**Complexité dev** : élevée.

### M8 — Noncentral Hypergeometric (Wallenius / Fisher)
- Modèle formel d’un tirage biaisé sans remise via poids par numéro.
- Estimation de poids + tests.

**Complexité dev** : élevée.

### M9 — Meta-test des p-values
- KS/QQ sur p-values de batterie ; drift dans le temps.
- Excellent garde-fou anti “faux signal”.

**Complexité dev** : faible/moyenne.

### M10 — Ensemble / stacking calibré
- Combiner M0/M1/M2/(M4) via optimisation walk-forward (min Brier).
- Calibration finale + IC bootstrap.

**Complexité dev** : moyenne.

---

## 12) Protocole d’évaluation (obligatoire)

### 12.1 Walk-forward backtesting
- Train jusqu’à t → prédire t+1 → mise à jour → avancer.
- Stocker `predictions` : date, proba par numéro, numéros réellement tirés.

### 12.2 Scoring
- Brier multi-label
- Calibration : reliability diagram + ECE
- Lift vs baseline : Δscore + IC bootstrap

### 12.3 Simulations Monte Carlo
- Datasets simulés sous H0 pour normaliser KL et certaines statistiques.
- Stocker seed + n_sim.

---

## 13) Politique anti-faux positifs
- Familles de tests + FDR.
- Toujours afficher p-values brutes et corrigées.
- Résultat “non concluant” si données insuffisantes.

---

## 14) Alerting : sévérité + seuils par défaut
- PSI : LOW 0.10, MEDIUM 0.25, HIGH 0.50
- KL : z_KL via simulation (LOW 2, MEDIUM 3, HIGH 4)
- Outliers robust z(MAD) : 3/4/5
- Diagnostics modèles : ΔBrier + ECE
- Evidence JSON standardisée (voir section 14.3 du CDC complet).

---

## 15) API (minimum v1)
- Games : POST/GET
- Import : POST preview/commit
- Draws : GET filtré
- Analyses : POST run + GET results + exports
- Alertes : GET
- Health : GET

---

## 16) UI (v1)
- Import
- Dashboard
- Tests & interprétation
- Alertes
- Probabilités (comparaison méthodes + scoring + calibration + limites)

---

## 17) Roadmap
- v0.1 : ingestion + descriptif
- v1.0 : tests + anomalies + alertes + report HTML
- v1.1 : proba M0/M1/M2 + évaluation + calibration
- v2.0 : M5/M6/M9/M10
- v3.0 : M7/M8 (+ ML renforcé)

---

## 18) Dataset de seed synthétique (dev/test)
- Normal + drift + rupture + data quality + outliers
- Livrables : `seed_lotto_5_49_bonus.csv` + `generate_seed.py` reproductible.

---

*Fin du document.*
