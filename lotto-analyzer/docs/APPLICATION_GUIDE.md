# 📖 Guide Complet de l'Application Lotto Analyzer

**Version** : 1.0  
**Date** : Février 2026  
**Objectif** : Analyse statistique et modélisation probabiliste des tirages de loterie

---

## 📋 Table des Matières

1. [Introduction](#1-introduction)
2. [Architecture Technique](#2-architecture-technique)
3. [Fonctionnalités Principales](#3-fonctionnalités-principales)
4. [Modèles de Prédiction](#4-modèles-de-prédiction)
5. [Système de Backtest](#5-système-de-backtest)
6. [Modèles Anti-Consensus](#6-modèles-anti-consensus)
7. [Métriques et Évaluation](#7-métriques-et-évaluation)
8. [Analyse Statistique](#8-analyse-statistique)
9. [Détection d'Anomalies](#9-détection-danomalies)
10. [Limitations et Avertissements](#10-limitations-et-avertissements)

---

## 1. Introduction

### 1.1 Objectif de l'Application

**Lotto Analyzer** est une application d'analyse statistique avancée conçue pour :

- **Analyser** les données historiques de tirages de loterie
- **Tester** l'hypothèse de randomisation (uniformité, indépendance)
- **Évaluer** différents modèles probabilistes via backtesting
- **Détecter** les anomalies et dérives statistiques
- **Générer** des rapports scientifiques rigoureux

### 1.2 Philosophie

L'application repose sur une approche **scientifique rigoureuse** :

- **Hypothèse nulle (H0)** : Les tirages suivent un processus uniforme i.i.d. sans remplacement
- **Correction multi-tests** : Benjamini-Hochberg pour contrôler le taux de faux positifs
- **Évaluation walk-forward** : Aucune fuite temporelle, entraînement uniquement sur le passé
- **Comparaison baseline** : Tous les modèles sont comparés à la distribution uniforme (M0)

### 1.3 Avertissement Important

> ⚠️ **Cette application est destinée à l'analyse statistique et à la recherche uniquement.**
> 
> - Aucun modèle ne peut prédire les tirages futurs d'une loterie équitable
> - Les performances passées ne garantissent pas les performances futures
> - L'application ne fournit PAS de "numéros gagnants"

---

## 2. Architecture Technique

### 2.1 Stack Technologique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| **Backend** | Python / FastAPI | 3.12+ |
| **Frontend** | React / TypeScript | 18+ |
| **Base de données** | PostgreSQL | 16+ |
| **Conteneurisation** | Docker / Docker Compose | - |
| **Machine Learning** | TensorFlow (optionnel) | 2.16+ |

### 2.2 Structure du Projet

```
lotto-analyzer/
├── backend/
│   ├── app/
│   │   ├── api/              # Routes FastAPI
│   │   ├── services/         # Logique métier
│   │   ├── analysis/         # Moteur d'analyse
│   │   │   ├── metrics.py        # Métriques statistiques
│   │   │   ├── randomness.py     # Tests de randomisation
│   │   │   ├── anomalies.py      # Détection d'anomalies
│   │   │   ├── backtest.py       # Système de backtest
│   │   │   ├── prob_models/      # Modèles M0-M15
│   │   │   ├── evaluation/       # Walk-forward, Brier, ECE
│   │   │   └── reporting/        # Exports CSV, HTML
│   │   ├── db/               # Modèles de données
│   │   └── schemas/          # Schémas Pydantic
│   └── tests/                # Tests unitaires
├── frontend/
│   ├── src/
│   │   ├── pages/            # Pages de l'application
│   │   ├── api/              # Client API
│   │   └── components/       # Composants React
│   └── package.json
├── docs/                     # Documentation
└── docker-compose.yml
```

### 2.3 Déploiement

```bash
# Démarrage avec Docker Compose
docker-compose up -d

# Accès
# - Frontend : http://localhost:5173
# - Backend API : http://localhost:8000
# - Documentation API : http://localhost:8000/docs
```

---

## 3. Fonctionnalités Principales

### 3.1 Gestion des Jeux

L'application supporte **plusieurs jeux de loterie** avec des règles configurables :

```json
{
  "numbers": {
    "count": 5,        // Nombre de numéros à tirer
    "min": 1,          // Numéro minimum
    "max": 49,         // Numéro maximum
    "unique": true,    // Numéros uniques
    "sorted": true     // Triés par ordre croissant
  },
  "bonus": {
    "enabled": true,   // Bonus activé
    "min": 1,          // Min du bonus
    "max": 12,         // Max du bonus
    "separate_pool": true  // Pool séparé pour le bonus
  }
}
```

**Exemples de jeux supportés** :
- Lotto 5/49 avec bonus 1-10
- EuroMillions 5/50 + 2 étoiles (1-12)
- Oz Lotto 7/45 + 2 supplémentaires

### 3.2 Import de Données

**Format CSV attendu** :
```csv
draw_date;n1;n2;n3;n4;n5;bonus
2024-01-03;4;11;19;33;47;8
2024-01-06;2;16;21;27;41;1
```

**Processus d'import** :
1. **Prévisualisation** : Validation du format et détection des erreurs
2. **Validation** : Vérification des règles du jeu
3. **Commit** : Insertion en base de données
4. **Audit** : Traçabilité complète des imports

### 3.3 Analyses Statistiques

- **Métriques descriptives** : Fréquences, entropie, divergence KL, PSI
- **Tests de randomisation** : χ² uniformité, tests de runs, indépendance
- **Correction FDR** : Benjamini-Hochberg pour tests multiples
- **Méta-tests** : Analyse de la distribution des p-values

### 3.4 Backtest des Modèles

Évaluation rigoureuse des modèles via **walk-forward validation** :
- Entraînement sur données passées uniquement
- Test sur données futures
- Calcul du lift vs baseline aléatoire

### 3.5 Exports et Rapports

- **CSV** : Métriques détaillées exportables
- **HTML** : Rapports imprimables avec section limitations obligatoire
- **Graphiques** : Visualisations interactives (Recharts)

---

## 4. Modèles de Prédiction

L'application implémente **18 modèles** de prédiction, chacun basé sur une approche statistique différente.

### 4.1 Tableau Récapitulatif

| Modèle | Nom | Type | Description |
|--------|-----|------|-------------|
| **M0** | Baseline | Aléatoire | Sélection uniforme (référence) |
| **M1** | Dirichlet | Bayésien | Prior Dirichlet-Multinomial |
| **M2** | Windowed | Fenêtre glissante | Pondération récente + shrinkage |
| **M3** | Exponential Decay | Temporel | Décroissance exponentielle |
| **M4** | HMM | États cachés | Hidden Markov Model |
| **M5** | Co-occurrence | Paires | Analyse des co-occurrences |
| **M6** | Gaps & Streaks | Écarts | Numéros "en retard" |
| **M7** | Entropy | Information | Sélection entropique |
| **M8** | Changepoint | Ruptures | Détection de changements |
| **M9** | Bayesian Network | Réseau | Dépendances conditionnelles |
| **M10** | Ensemble | Méta-modèle | Combinaison M0+M1+M2 |
| **M11** | LSTM Hybrid | Deep Learning | LSTM + Attention |
| **M12** | Mixture Dirichlet | Mélange | Multi-composantes Dirichlet |
| **M13** | Spectral | Fourier | Détection de périodicités |
| **M14** | Copula | Dépendances | Copules gaussiennes |
| **M15** | Thompson Sampling | Bandit | Exploration/exploitation |
| **ANTI** | Anti-Consensus | Contrarian | Numéros NON prédits |
| **ANTI2** | Anti-Consensus v2 | Contrarian+ | ANTI + diversité |

### 4.2 Modèles Statistiques Classiques

#### M0 - Baseline (Référence)

**Principe** : Sélection aléatoire uniforme
```
P(numéro i) = 1 / N
```
- Sert de **référence** pour tous les autres modèles
- Tout modèle performant doit faire mieux que M0

#### M1 - Dirichlet (Bayésien)

**Principe** : Estimation bayésienne avec prior Dirichlet
```
P(numéro i) = (count_i + α) / (total + N × α)
```
- **α** : Paramètre de concentration (défaut: 1)
- Robuste avec peu de données
- Évite les probabilités nulles

#### M2 - Windowed (Fenêtre Glissante)

**Principe** : Pondération des tirages récents avec shrinkage
```
P(numéro i) = λ × P_global(i) + (1-λ) × P_fenêtre(i)
```
- **window_size** : Taille de la fenêtre (défaut: 50)
- **λ** : Facteur de shrinkage (défaut: 0.1)
- Capture les tendances récentes

#### M3 - Exponential Decay

**Principe** : Décroissance exponentielle des poids temporels
```
w(t) = exp(-λ × (T - t))
P(numéro i) = Σ w(t) × I(i ∈ tirage_t) / Σ w(t)
```
- **λ** : Taux de décroissance (défaut: 0.02)
- Adaptatif aux changements récents

### 4.3 Modèles à États et Régimes

#### M4 - HMM (Hidden Markov Model)

**Principe** : Modélise différents "régimes" latents
```
P(numéro i) = Σ P(état_k | obs) × P(numéro i | état_k)
```
- **n_states** : Nombre d'états cachés (défaut: 3)
- Capture les changements de régime
- **Dépendance** : `hmmlearn` (optionnel)

#### M8 - Changepoint Detection

**Principe** : Détection de ruptures structurelles (algorithme PELT)
- Détecte les points de rupture dans les séries
- Recalcule les probabilités sur le segment post-rupture
- Ignore les données obsolètes

### 4.4 Modèles d'Analyse de Patterns

#### M5 - Co-occurrence

**Principe** : Analyse des paires de numéros sur-représentées
```
Delta = Fréquence_observée - Fréquence_attendue
```
- Identifie les corrélations entre numéros
- Sélectionne les numéros des top 20 paires

#### M6 - Gaps & Streaks

**Principe** : Numéros "en retard" (overdue)
```
Delta_gap = Écart_actuel - Écart_moyen
```
- Sélectionne les numéros avec le plus grand delta positif
- Basé sur la loi des grands nombres

#### M7 - Entropy-Based Selection

**Principe** : Sélection basée sur l'entropie de Shannon
```
H_i = -p_i × log2(p_i) - (1-p_i) × log2(1-p_i)
score_i = H0 - H_i
```
- Favorise les numéros avec comportement prévisible
- Basé sur la théorie de l'information

### 4.5 Modèles Avancés

#### M9 - Bayesian Network

**Principe** : Réseau bayésien pour dépendances conditionnelles
- Modélise les dépendances via un DAG
- Apprend la structure par Hill Climbing
- **Dépendance** : `pgmpy` (optionnel)

#### M10 - Ensemble (Stacking)

**Principe** : Méta-modèle combinant M0, M1, M2
- Combine les prédictions avec poids optimisés
- Réduit la variance des prédictions individuelles

#### M11 - LSTM Hybrid (Deep Learning)

**Architecture** :
```
┌─────────────────────────────────────────────────────────┐
│  BLOC 1: Ingestion                                      │
│  ├── Input: 50 derniers tirages × N numéros            │
│  ├── Embeddings: dim=32                                │
│  └── Méta-Features: somme, écart-type, pairs...        │
├─────────────────────────────────────────────────────────┤
│  BLOC 2: Cerveau Temporel                               │
│  ├── LSTM Bidirectionnel (64 unités)                   │
│  ├── Multi-Head Attention (2 têtes)                    │
│  └── Layer Normalization + Résiduel                    │
├─────────────────────────────────────────────────────────┤
│  BLOC 3: Tête de Prédiction                            │
│  ├── Dense (128) + Dropout (0.3)                       │
│  ├── Dense (64) + Dropout (0.3)                        │
│  └── Sortie Sigmoid (N probabilités)                   │
└─────────────────────────────────────────────────────────┘
```
- Capture les patterns temporels complexes
- **Dépendance** : TensorFlow (optionnel)

#### M12 - Mixture of Dirichlet

**Principe** : Mélange de distributions Dirichlet
```
P(numéro i) = Σ π_k × E[θ_i | α_k]
```
- Capture l'hétérogénéité via plusieurs composantes
- Estimation par algorithme EM

#### M13 - Spectral / Fourier

**Principe** : Analyse spectrale pour périodicités
```
X_i(f) = FFT(x_i(t))
PSD_i(f) = |X_i(f)|² / N
```
- Détecte les cycles cachés
- Test de Fisher pour significativité

#### M14 - Copula Model

**Principe** : Modélisation des dépendances par copules
- Sépare marginales et structure de dépendance
- Copule gaussienne pour corrélations
- **Dépendance** : `copulas` (optionnel)

#### M15 - Thompson Sampling

**Principe** : Bandit multi-bras bayésien
```
θ_i ~ Beta(α_i + succès_i, β_i + échecs_i)
P(numéro i) ∝ E[θ_i]
```
- Équilibre exploration et exploitation
- Converge vers les vraies probabilités

---

## 5. Système de Backtest

### 5.1 Méthodologie Walk-Forward

Le backtest utilise une validation **walk-forward** rigoureuse :

```
┌─────────────────────────────────────────────────────────────┐
│  Données historiques                                        │
│  ════════════════════════════════════════════════════════   │
│                                                             │
│  Tirage 1 ─────────────────────────────────────► Tirage N   │
│                                                             │
│  Pour chaque tirage T à tester :                           │
│  ┌──────────────────────┬─────┐                            │
│  │   TRAIN (1 à T-1)    │TEST │                            │
│  │   Entraînement       │ T   │                            │
│  └──────────────────────┴─────┘                            │
│                                                             │
│  → Aucune fuite temporelle                                 │
│  → Entraînement uniquement sur le passé                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Paramètres du Backtest

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| **test_draws** | Nombre de tirages à tester | 100 |
| **n_combinations** | Combinaisons par tirage (ANTI/ANTI2) | 10 |
| **max_common_main** | Max numéros principaux en commun (ANTI2) | 2 |
| **max_common_bonus** | Max bonus en commun (ANTI2) | 0 |

### 5.3 Processus d'Exécution

1. **Sélection des modèles** à tester
2. **Configuration** des paramètres
3. **Exécution** :
   - Pour chaque tirage de test
   - Entraîner chaque modèle sur l'historique
   - Générer les prédictions
   - Comparer aux résultats réels
4. **Calcul des métriques** :
   - Taux de réussite (hit rate)
   - Lift vs baseline
   - Divisions de prix atteintes

---

## 6. Modèles Anti-Consensus

### 6.1 ANTI - Anti-Consensus

**Principe** : Stratégie contrariante qui prédit les numéros **NON prédits** par les autres modèles.

**Hypothèse** : Si tous les modèles se trompent systématiquement, les numéros ignorés ont plus de chances.

**Modèles utilisés pour le consensus** :
- M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M12, M13, M14, M15

**Algorithme** :
```
1. Collecter toutes les prédictions des autres modèles
2. Identifier les numéros NON prédits par aucun modèle
3. Générer N combinaisons aléatoires à partir de ces numéros
```

### 6.2 ANTI2 - Anti-Consensus v2 (avec Diversité)

**Principe** : Même logique que ANTI, mais avec une **contrainte de diversité** entre les combinaisons générées.

**Paramètres de diversité** :
- **max_common_main** : Maximum de numéros principaux identiques entre deux combinaisons (défaut: 2)
- **max_common_bonus** : Maximum de bonus identiques (défaut: 0 = tous différents)

**Algorithme** :
```
1. Collecter toutes les prédictions des autres modèles
2. Identifier les numéros NON prédits
3. Pour chaque combinaison à générer :
   a. Générer une combinaison candidate
   b. Vérifier la contrainte de diversité avec les combinaisons existantes
   c. Si OK → ajouter ; sinon → réessayer (max 100 tentatives)
4. Si max_common_bonus = 0 → chaque combinaison a un bonus unique
```

**Exemple avec max_common_main=2, max_common_bonus=0** :
```
Combo #1: [5, 12, 23, 34, 45] + [7]
Combo #2: [8, 12, 19, 34, 41] + [3]   ← max 2 en commun (12, 34), bonus différent
Combo #3: [3, 15, 27, 38, 49] + [11]  ← bonus unique
```

### 6.3 Prédiction pour le Prochain Tirage

L'application peut générer une **prédiction ANTI2** pour le prochain tirage :

1. Utilise **TOUT l'historique** comme données d'entraînement
2. Collecte les prédictions de tous les modèles (M1-M15 sauf M0, M11)
3. Génère N combinaisons diversifiées à partir des numéros ignorés
4. Affiche les numéros exclus (prédits par consensus)

---

## 7. Métriques et Évaluation

### 7.1 Taux de Réussite (Hit Rate)

```
Hit Rate = Numéros corrects prédits / Numéros à deviner
```

**Exemple** : Si on prédit 5 numéros et 2 sont corrects → Hit Rate = 40%

### 7.2 Taux de Réussite Maximum (Max Hit Rate)

Le taux de réussite le plus élevé obtenu sur un tirage individuel.

### 7.3 Lift vs Random

```
Lift = Hit Rate du modèle / Hit Rate attendu (aléatoire)
```

| Lift | Interprétation |
|------|----------------|
| < 1 | Pire que le hasard |
| = 1 | Équivalent au hasard |
| > 1 | Meilleur que le hasard |

**Attention** : Un lift > 1 sur un petit échantillon peut être dû au hasard.

### 7.4 Divisions de Prix

Le système calcule les divisions atteintes selon les règles du jeu :

| Division | Condition typique |
|----------|-------------------|
| Div 1 | 5 principaux + bonus |
| Div 2 | 5 principaux |
| Div 3 | 4 principaux + bonus |
| Div 4 | 4 principaux |
| ... | ... |

### 7.5 Brier Score

Mesure la précision des probabilités prédites :
```
Brier = (1/N) × Σ (p_i - o_i)²
```
- **p_i** : Probabilité prédite
- **o_i** : Résultat observé (0 ou 1)
- Plus bas = meilleur

### 7.6 ECE (Expected Calibration Error)

Mesure la calibration des probabilités :
- Un modèle bien calibré prédit 30% pour des événements qui se produisent 30% du temps

---

## 8. Analyse Statistique

### 8.1 Tests de Randomisation

| Test | Objectif | H0 |
|------|----------|-----|
| **χ² Uniformité** | Vérifier distribution uniforme | Tous les numéros équiprobables |
| **Test de Runs** | Vérifier indépendance séquentielle | Pas de patterns séquentiels |
| **Test d'Indépendance** | Vérifier indépendance entre numéros | Pas de corrélation |

### 8.2 Métriques Descriptives

- **Fréquences** : Comptage de chaque numéro
- **Entropie** : Mesure du désordre/uniformité
- **Divergence KL** : Distance à la distribution uniforme
- **PSI** : Population Stability Index (stabilité temporelle)

### 8.3 Correction Multi-Tests

**Problème** : Avec N tests, le risque de faux positif augmente.

**Solution** : Correction Benjamini-Hochberg (FDR)
- Contrôle le taux de fausses découvertes
- Plus puissant que Bonferroni

---

## 9. Détection d'Anomalies

### 9.1 Détection de Drift

**PSI (Population Stability Index)** :
| Seuil | Interprétation |
|-------|----------------|
| < 0.10 | Pas de changement significatif |
| 0.10 - 0.25 | Changement modéré |
| > 0.25 | Changement significatif |

**KL Divergence** : Normalisée par simulation Monte Carlo

### 9.2 Points de Rupture

**Algorithme PELT** (Pruned Exact Linear Time) :
- Détecte les changements structurels dans les séries
- Pénalité BIC/AIC pour éviter le sur-ajustement

### 9.3 Outliers

**Z-scores MAD** (Median Absolute Deviation) :
- Robuste aux valeurs extrêmes
- Seuil typique : |z| > 3

### 9.4 Alertes Automatiques

Le système génère des alertes avec niveaux de sévérité :
- **INFO** : Observation notable
- **WARNING** : Anomalie potentielle
- **CRITICAL** : Anomalie significative

---

## 10. Limitations et Avertissements

### 10.1 Limitations Fondamentales

> ⚠️ **IMPORTANT : Cette section est OBLIGATOIRE dans tous les rapports**

1. **Aucune garantie de prédiction**
   - Une loterie équitable est fondamentalement aléatoire
   - Aucun modèle ne peut prédire les tirages futurs

2. **Comparaison baseline obligatoire**
   - Tous les modèles doivent être comparés à M0 (uniforme)
   - Un lift > 1 peut être dû au hasard

3. **Problème des tests multiples**
   - Les patterns apparents peuvent être des faux positifs
   - La correction FDR est appliquée mais ne garantit rien

4. **Taille d'échantillon**
   - Les petits datasets réduisent la puissance statistique
   - Les résultats sur peu de tirages ne sont pas fiables

5. **Changements temporels**
   - Les règles du jeu peuvent changer
   - La qualité des données peut varier

### 10.2 Interprétation Correcte

**Nécessite** :
- Compréhension des tests d'hypothèses statistiques
- Conscience du problème des comparaisons multiples
- Reconnaissance que les tirages passés n'influencent pas les futurs
- Évaluation critique du lift vs baseline

### 10.3 Ce que l'Application NE FAIT PAS

❌ Ne fournit PAS de "numéros gagnants"  
❌ Ne garantit PAS de gains  
❌ Ne prédit PAS les tirages futurs  
❌ Ne conseille PAS de jouer  

### 10.4 Ce que l'Application FAIT

✅ Analyse statistique rigoureuse des données historiques  
✅ Évaluation objective des modèles probabilistes  
✅ Détection d'anomalies et de dérives  
✅ Génération de rapports scientifiques  
✅ Outil éducatif et de recherche  

---

## Annexes

### A. Dépendances Optionnelles

| Package | Modèles concernés | Fallback |
|---------|-------------------|----------|
| `tensorflow` | M11 (LSTM) | Estimation fréquentielle |
| `hmmlearn` | M4 (HMM) | Estimation fréquentielle |
| `pgmpy` | M9 (Bayesian Network) | Corrélations simples |
| `copulas` | M14 (Copula) | Corrélations simples |

### B. Format des Données

**CSV d'import** :
```csv
draw_date;n1;n2;n3;n4;n5;bonus
2024-01-03;4;11;19;33;47;8
```

**Règles** :
- Délimiteur : `;`
- Date : ISO 8601 (YYYY-MM-DD)
- Numéros : Entiers triés
- Encodage : UTF-8

### C. API Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/games` | GET/POST | Gestion des jeux |
| `/draws/import` | POST | Import de données |
| `/analyses/run` | POST | Lancer une analyse |
| `/analyses/backtest` | POST | Lancer un backtest |
| `/analyses/{id}/export.csv` | GET | Export CSV |

### D. Références Scientifiques

- Benjamini-Hochberg (1995) - Correction FDR
- Brier (1950) - Score de probabilité
- Killick et al. (2012) - Algorithme PELT
- Kullback-Leibler (1951) - Divergence KL

---

*Document généré pour Lotto Analyzer v1.0 - Février 2026*
