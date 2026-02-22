# Lotto Analyzer — Méthodologie (thèse + application)

## 1) Hypothèse nulle et cadre scientifique
### Hypothèse nulle (H0)
Pour une loterie correctement opérée, le processus attendu est :
- **Uniforme** : chaque combinaison valide est équiprobable
- **Sans remise** : pas de répétition d’un numéro dans un même tirage
- **Indépendant** : les tirages successifs sont indépendants (i.i.d.)

Conséquence : la meilleure “prédiction” **a priori** est le **baseline uniforme**.

### Risques d’interprétation
- **Biais de fenêtre** : sur une fenêtre courte, des écarts apparaissent naturellement
- **Multiples tests** : plus tu testes, plus tu trouveras des “signaux” par hasard
- **P-hacking** : ajuster les métriques jusqu’à obtenir un résultat “significatif”

Garde-fou : regrouper les tests en **familles**, corriger par **FDR**, et utiliser des **simulations Monte Carlo** comme référence.

---

## 2) Protocole d’évaluation standard

### 2.1 Découpage temporel (walk-forward)
Interdiction de fuite temporelle :
1. Entraîner sur les tirages jusqu’à la date *t*
2. Prédire la date *t+1*
3. Mettre à jour l’historique (inclure le vrai tirage de *t+1*)
4. Répéter

Ce protocole produit un journal `predictions` exploitable pour scores et calibration.

### 2.2 Métriques d’évaluation des probabilités
- **Brier score (multi-label)** : mesure la qualité des probabilités par numéro
- **ECE (Expected Calibration Error)** : mesure l’écart calibration (probabilité vs fréquence)
- **Lift vs baseline** : Δscore (modèle − baseline) + intervalle de confiance (bootstrap)

**Important** : présenter les scores avec incertitude (IC bootstrap) et conclure “aucun gain” si Δ non significatif.

### 2.3 Bootstrap / incertitude
- Bootstrap sur les pas de walk-forward (en respectant l’ordre si nécessaire)
- Produire IC 95% sur ΔBrier / ECE / lift

### 2.4 Monte Carlo (références sous H0)
Utiliser des tirages simulés sous H0 pour :
- normaliser des statistiques (ex: KL → z_KL)
- estimer distributions de test quand l’analytique est complexe
- tester la robustesse (faux positifs attendus)

Stocker `seed`, `n_sim`, version des règles, et paramètres de simulation.

---

## 3) Batterie de tests (recommandations)
### 3.1 Uniformité
- χ² sur fréquences numéros et bonus
- Segmentation temporelle (périodes / fenêtres)

### 3.2 Indépendance
- runs tests (pair/impair, haut/bas, etc.)
- autocorrélation sur métriques dérivées (entropie/KL/PSI)
- permutation tests quand l’approximation asymptotique est douteuse

### 3.3 Méta-tests
- Distribution des p-values (QQ plot, KS test)
- Permet de détecter une batterie “trop significative” (faux positifs cumulés)

---

## 4) Gouvernance “anti sur-ajustement”
- Pré-enregistrer (dans le code) les familles de tests et les métriques (éviter bricolage ad hoc)
- Versionner : `code_version`, `dataset_hash`, `params_json`
- Séparer :
  - exploration (E) : libre mais marquée comme exploratoire
  - confirmation (C) : protocole strict, seuils fixés à l’avance

---

## 5) Reporting scientifique
Un rapport doit contenir :
- Contexte et dataset (période, N tirages, règles, version)
- Résultats descriptifs
- Tests (p-values brutes + corrigées FDR + méta-tests)
- Anomalies (drift/change points/outliers) avec evidence
- Modèles probabilistes (baseline + scores + calibration + IC)
- Section **Limites & interprétation** obligatoire

---

## 6) Menaces à la validité (à documenter)
- Changements de règles, source, calendrier
- Données manquantes / erreurs d’import
- Taille d’échantillon insuffisante
- Multiples comparaisons et biais de sélection
