# Lotto Analyzer — Catalogue des modèles (M0 → M10)

## Standardisation des sorties
Chaque méthode doit produire :
- `number_probs` : dict numero->proba
- `baseline_probs` : baseline uniforme
- `uncertainty` : IC/credible intervals (si possible)
- `evaluation` : Brier, ECE, lift, IC bootstrap (si backtest)
- `warnings` : limites, non-significativité, dataset petit

---

## M0 — Baseline uniforme
- p(i)=1/N, sélection sans remise
- Sert de référence

## M1 — Bayésien Dirichlet–Multinomial
- prior α, posterior α+counts
- donne un lissage + intervalles crédibles

## M2 — Fenêtre glissante + shrinkage
- p = λ p_window + (1-λ) p_uniforme
- évite sur-ajustement

## M3 — Diagnostic sur features
- prédit propriétés (parité, somme, suites…)
- objectif : détecter biais structurels

## M4 — ML multi-label (expérimental)
- logistic/GBM/forest sur features glissantes
- obligations : walk-forward + calibration + régularisation

---

## M5 — Co-occurrence + tests vs attendu (sans remise)
- matrice NxN co-occurrences
- comparer à attendu analytique/simulé
- FDR sur p-values

## M6 — Gaps & streaks
- distribution des temps entre occurrences et runs
- test vs simulation H0

## M7 — BOCPD (Bayesian Online Change Point Detection)
- change point “en ligne” sur entropie/KL/PSI
- posterior de rupture

## M8 — Noncentral hypergeometric (Wallenius/Fisher)
- modèle formel d’un tirage biaisé sans remise
- estimation des poids + tests

## M9 — Meta-test des p-values
- QQ plot / KS sur p-values globales
- drift des p-values dans le temps

## M10 — Ensemble / stacking calibré
- combiner M0/M1/M2/(M4) pour min Brier
- calibration finale + IC bootstrap
