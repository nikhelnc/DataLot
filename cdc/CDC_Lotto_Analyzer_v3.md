# Cahier des Charges — Lotto Analyzer v2.0
## Évolutions Applicatives — Axes Modélisation Probabiliste, Forensique du Générateur & Détection de Fraude

---

**Document** : CDC-LOTTO-ANALYZER-V2  
**Version** : 1.0  
**Date** : Février 2026  
**Auteur** : Projet Thèse — Analyse Statistique des Loteries  
**Statut** : Draft v1 — Pour validation  
**Application cible** : Lotto Analyzer v1.0 (existante)  
**Contexte** : Thèse en statistiques appliquées — Loteries australiennes (Oz Lotto, TattsLotto, Powerball AU)

---

## Table des Matières

1. [Contexte et Objectifs](#1-contexte-et-objectifs)
2. [Périmètre des Évolutions](#2-périmètre-des-évolutions)
3. [Nouvelles Sources de Données](#3-nouvelles-sources-de-données)
4. [Évolutions du Modèle de Données](#4-évolutions-du-modèle-de-données)
5. [Nouveaux Modèles de Prédiction](#5-nouveaux-modèles-de-prédiction)
6. [Module Forensique du Générateur](#6-module-forensique-du-générateur)
7. [Module Détection de Fraude](#7-module-détection-de-fraude)
8. [Module Analyse des Jackpots](#8-module-analyse-des-jackpots)
9. [Évolutions de l'Interface Utilisateur](#9-évolutions-de-linterface-utilisateur)
10. [Évolutions de l'API Backend](#10-évolutions-de-lapi-backend)
11. [Exigences Non-Fonctionnelles](#11-exigences-non-fonctionnelles)
12. [Architecture Cible](#12-architecture-cible)
13. [Roadmap et Livrables](#13-roadmap-et-livrables)
14. [Critères d'Acceptance](#14-critères-dacceptance)
15. [Limites et Avertissements](#15-limites-et-avertissements)

---

## 1. Contexte et Objectifs

### 1.1 Rappel de l'Existant

L'application **Lotto Analyzer v1.0** dispose actuellement des fonctionnalités suivantes :

- Import de données CSV et saisie manuelle des tirages
- Gestion multi-jeux (Oz Lotto, TattsLotto, Powerball AU)
- 18 modèles de prédiction (M0 à M15, ANTI, ANTI2)
- Système de backtest en walk-forward validation
- Tests de randomisation de base (χ², runs, indépendance)
- Détection de changepoints (PELT)
- Correction multi-tests Benjamini-Hochberg
- Exports CSV et rapports HTML
- API FastAPI + Frontend React/TypeScript

### 1.2 Nouvelles Données Disponibles

Suite à une analyse approfondie des données accessibles, deux nouvelles sources d'information sont désormais intégrables :

- **L'ordre d'émission physique des boules** : séquence temporelle réelle de sortie (avant tri)
- **Les montants de cagnotte** par tirage avec indicateur de report

Ces données élèvent significativement le potentiel scientifique de l'étude et permettent des tests forensiques impossibles à réaliser avec les seuls numéros triés.

### 1.3 Objectifs de la v2.0

| Objectif | Description | Priorité |
|----------|-------------|----------|
| **Forensique du générateur** | Déterminer si le générateur (physique ou logiciel) est conforme à un processus i.i.d. uniforme | HAUTE |
| **Modélisation probabiliste avancée** | Ajouter des modèles exploitant l'ordre d'émission et le contexte jackpot | HAUTE |
| **Détection de fraude** | Détecter les déviations statistiquement incompatibles avec un système équitable | HAUTE |
| **Analyse comportementale** | Modéliser l'influence du jackpot sur les numéros tirés et joués | MOYENNE |
| **Score de conformité global** | Produire un score unique auditable avec intervalles de confiance | HAUTE |
| **Rapports thèse** | Générer des rapports scientifiques complets au format académique | MOYENNE |

### 1.4 Contrainte Scientifique Fondamentale

> ⚠️ **Rappel impératif** : L'application ne prédit pas les tirages futurs d'un système équitable. Elle évalue statistiquement si un système EST équitable. Toute communication de résultats doit respecter cette distinction.

---

## 2. Périmètre des Évolutions

### 2.1 Ce qui est Dans le Périmètre (IN SCOPE)

- Modification du schéma de base de données pour intégrer les nouvelles colonnes
- Nouveaux modèles de prédiction M16 à M20
- Module complet de forensique du générateur (tests NIST adaptés, tests physiques, tests RNG)
- Module de détection de fraude avec score de risque
- Module d'analyse jackpot avec économétrie comportementale
- Nouvelles visualisations (heatmaps, matrices de transition, RDD)
- Calculateur de puissance statistique
- Comparateur inter-loteries
- Export rapport académique enrichi
- Nouveaux endpoints API

### 2.2 Ce qui est Hors Périmètre (OUT OF SCOPE)

- Refonte de l'architecture existante (backend Python/FastAPI, frontend React)
- Modification des modèles M0 à M15, ANTI, ANTI2 existants
- Changement de la base de données (PostgreSQL maintenu)
- Application mobile native
- Intégration de sources de données externes automatiques (scraping)
- Module de prédiction en temps réel (live draws)

---

## 3. Nouvelles Sources de Données

### 3.1 Ordre d'Émission des Boules

#### 3.1.1 Description

L'ordre d'émission désigne la **séquence temporelle réelle** dans laquelle les boules sortent physiquement de la machine, avant tout tri numérique.

**Exemple TattsLotto (6/45 + 2 complémentaires)** :
```
Numéros triés (existant)   : [3, 11, 22, 31, 38, 42]
Ordre d'émission (nouveau) : [22, 3, 42, 11, 38, 31]
Complémentaires (triés)    : [18, 27]
Complémentaires (émission) : [27, 18]
```

#### 3.1.2 Format d'Import CSV Étendu

Le format CSV doit être enrichi pour supporter l'ordre d'émission :

**Oz Lotto (7 principaux + 3 complémentaires)** :
```csv
draw_number;draw_date;n1;n2;n3;n4;n5;n6;n7;bonus1;bonus2;bonus3;e1;e2;e3;e4;e5;e6;e7;be1;be2;be3;jackpot_amount;jackpot_rollover
1234;2025-01-21;3;7;20;35;37;41;42;18;30;33;42;3;20;37;41;7;35;30;18;33;15000000;false
```

Où `e1..e7` = ordre d'émission des principaux, `be1..be3` = ordre d'émission des complémentaires.

**Règles de validation** :
- Les valeurs `e1..ek` doivent être une permutation exacte de `n1..nk`
- Les valeurs `be1..bem` doivent être une permutation exacte de `bonus1..bonusm`
- Aucun doublon dans la séquence d'émission

#### 3.1.3 Saisie Manuelle

Le formulaire de saisie manuelle doit proposer deux modes :

- **Mode Standard** : saisie des numéros triés uniquement (comportement actuel)
- **Mode Étendu** : saisie des numéros triés + ordre d'émission dans des champs séparés avec validation en temps réel

### 3.2 Montants de Cagnotte

#### 3.2.1 Description

Le montant du jackpot Div1 annoncé avant chaque tirage, exprimé en dollars australiens (AUD).

#### 3.2.2 Données Associées

| Champ | Type | Description | Obligatoire |
|-------|------|-------------|-------------|
| `jackpot_amount` | DECIMAL(15,2) | Montant Div1 en AUD | Non |
| `jackpot_rollover` | BOOLEAN | Report du jackpot précédent | Non |
| `jackpot_consecutive_rollovers` | INTEGER | Nombre de reports consécutifs | Non |
| `must_be_won` | BOOLEAN | Tirage "Must Be Won" officiel | Non |
| `n_winners_div1` | INTEGER | Nombre de gagnants Div1 | Non |

#### 3.2.3 Format CSV Jackpot

```csv
draw_number;draw_date;...numéros...;jackpot_amount;jackpot_rollover;must_be_won
1234;2025-01-21;...;15000000;false;false
1235;2025-01-28;...;20000000;true;false
1236;2025-02-04;...;100000000;true;true
```

---

## 4. Évolutions du Modèle de Données

### 4.1 Modifications de la Table `draws`

```sql
-- Ajout des colonnes d'ordre d'émission
ALTER TABLE draws 
    ADD COLUMN emission_order INTEGER[],
    ADD COLUMN bonus_emission_order INTEGER[],
    ADD COLUMN jackpot_amount DECIMAL(15,2),
    ADD COLUMN jackpot_rollover BOOLEAN DEFAULT FALSE,
    ADD COLUMN jackpot_consecutive_rollovers INTEGER DEFAULT 0,
    ADD COLUMN must_be_won BOOLEAN DEFAULT FALSE,
    ADD COLUMN n_winners_div1 INTEGER;

-- Contrainte : emission_order doit être une permutation des numbers
-- (vérifiée applicativement, pas en SQL pour flexibilité)

-- Index pour les analyses temporelles avec jackpot
CREATE INDEX idx_draws_jackpot ON draws(game_id, jackpot_amount) 
    WHERE jackpot_amount IS NOT NULL;

CREATE INDEX idx_draws_emission ON draws(game_id) 
    WHERE emission_order IS NOT NULL;
```

### 4.2 Nouvelle Table `generator_profiles`

```sql
CREATE TABLE generator_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Période d'analyse
    period_start DATE,
    period_end DATE,
    n_draws INTEGER NOT NULL,
    
    -- Score global
    conformity_score FLOAT CHECK (conformity_score BETWEEN 0 AND 1),
    conformity_ci_low FLOAT,
    conformity_ci_high FLOAT,
    conformity_n_simulations INTEGER DEFAULT 1000,
    
    -- Type de générateur détecté
    generator_type VARCHAR(20) CHECK (generator_type IN ('physical', 'rng', 'hybrid', 'unknown')),
    
    -- Résultats détaillés par catégorie de tests
    standard_tests JSONB,        -- χ², runs, indépendance
    nist_tests JSONB,            -- Fréquence monobit, runs NIST, sériel, blocs
    physical_tests JSONB,        -- Biais de poids, position, adjacence, dérive thermique
    rng_tests JSONB,             -- LSB bias, modulo bias, périodicité, birthday paradox
    structural_tests JSONB,      -- Statistiques d'ordre, distribution des sommes
    
    -- Reproductibilité
    dataset_hash VARCHAR(64) NOT NULL,
    app_version VARCHAR(20),
    params JSONB,
    seed INTEGER,
    
    CONSTRAINT uq_generator_profile UNIQUE (game_id, dataset_hash)
);
```

### 4.3 Nouvelle Table `changepoints` (Extension)

```sql
-- Enrichissement de la table changepoints existante
ALTER TABLE changepoints 
    ADD COLUMN affected_numbers INTEGER[],
    ADD COLUMN magnitude FLOAT,
    ADD COLUMN context_note TEXT,
    ADD COLUMN is_validated BOOLEAN DEFAULT FALSE,
    ADD COLUMN validation_note TEXT,
    ADD COLUMN jackpot_at_changepoint DECIMAL(15,2),
    ADD COLUMN emission_position_affected INTEGER;
```

### 4.4 Nouvelle Table `fraud_alerts`

```sql
CREATE TABLE fraud_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Classification
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'HIGH', 'CRITICAL')),
    signal_type VARCHAR(50) NOT NULL,
    category VARCHAR(30) CHECK (category IN ('generator', 'data_quality', 'behavioral', 'structural')),
    
    -- Description
    title VARCHAR(200) NOT NULL,
    description TEXT,
    statistical_evidence JSONB,  -- p-values, statistiques, intervalles de confiance
    
    -- Tirages concernés
    draw_ids UUID[],
    period_start DATE,
    period_end DATE,
    
    -- Workflow de traitement
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN' 
        CHECK (status IN ('OPEN', 'INVESTIGATING', 'CLOSED', 'FALSE_POSITIVE')),
    assigned_to VARCHAR(100),
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    
    -- Reproductibilité
    analysis_id UUID REFERENCES analyses(id)
);

CREATE INDEX idx_fraud_alerts_open ON fraud_alerts(game_id, severity) 
    WHERE status = 'OPEN';
CREATE INDEX idx_fraud_alerts_game_date ON fraud_alerts(game_id, created_at DESC);
```

### 4.5 Nouvelle Table `jackpot_analyses`

```sql
CREATE TABLE jackpot_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Résultats des tests jackpot
    generator_independence_test JSONB,  -- Corrélation jackpot/numéros tirés
    player_bias_analysis JSONB,         -- Modèle de sélection des joueurs
    threshold_effect JSONB,             -- Regression Discontinuity Design
    must_be_won_analysis JSONB,         -- Analyse des tirages Must Be Won
    jackpot_vs_emission JSONB,          -- Jackpot vs position d'émission
    
    -- Méta
    n_draws_analyzed INTEGER,
    jackpot_range_min DECIMAL(15,2),
    jackpot_range_max DECIMAL(15,2),
    dataset_hash VARCHAR(64)
);
```

### 4.6 Vue Matérialisée `draws_positional`

```sql
CREATE MATERIALIZED VIEW draws_positional AS
SELECT
    d.id,
    d.game_id,
    d.draw_date,
    d.numbers,
    d.emission_order,
    d.jackpot_amount,
    d.jackpot_rollover,
    d.must_be_won,
    -- Positions d'émission individuelles (pour requêtes analytiques rapides)
    d.emission_order[1] AS emit_pos1,
    d.emission_order[2] AS emit_pos2,
    d.emission_order[3] AS emit_pos3,
    d.emission_order[4] AS emit_pos4,
    d.emission_order[5] AS emit_pos5,
    d.emission_order[6] AS emit_pos6,
    d.emission_order[7] AS emit_pos7,
    -- Dérivées analytiques
    (SELECT SUM(n) FROM UNNEST(d.numbers) n) AS numbers_sum,
    (SELECT COUNT(*) FROM UNNEST(d.numbers) n WHERE n <= 31) AS count_lte31,
    (SELECT COUNT(*) FROM UNNEST(d.numbers) n WHERE n % 2 = 0) AS count_even,
    -- Jackpot normalisé (z-score sur la fenêtre glissante)
    (d.jackpot_amount - AVG(d.jackpot_amount) OVER (
        PARTITION BY d.game_id 
        ORDER BY d.draw_date 
        ROWS BETWEEN 52 PRECEDING AND CURRENT ROW
    )) / NULLIF(STDDEV(d.jackpot_amount) OVER (
        PARTITION BY d.game_id 
        ORDER BY d.draw_date 
        ROWS BETWEEN 52 PRECEDING AND CURRENT ROW
    ), 0) AS jackpot_zscore_52w
FROM draws d;

CREATE UNIQUE INDEX idx_draws_positional_id ON draws_positional(id);
CREATE INDEX idx_draws_positional_game_date ON draws_positional(game_id, draw_date);
```

---

## 5. Nouveaux Modèles de Prédiction

### 5.1 Vue d'Ensemble des Nouveaux Modèles

| Modèle | Nom | Données requises | Innovation |
|--------|-----|-----------------|------------|
| **M16** | Order Statistics Baseline | Numéros triés | Référentiel analytique exact par position |
| **M17** | Logistic Temporal | Numéros + dates | Features temporelles + interprétabilité |
| **M18** | Dirichlet Process | Numéros | Clustering non-paramétrique de numéros |
| **M19** | Emission Order Model | Ordre d'émission | Exploitation de l'ordre physique |
| **M20** | Jackpot Context Model | Numéros + jackpot | Ajustement par contexte jackpot |

### 5.2 M16 — Order Statistics Baseline

**Objectif** : Établir le référentiel analytique exact basé sur les distributions théoriques des statistiques d'ordre pour un tirage sans remise.

**Principe** :
Pour un tirage k/N, la k-ième statistique d'ordre (k-ième plus petit numéro) a une distribution analytiquement connue. Ce modèle sert de **baseline théorique parfaite** pour mesurer l'écart des données réelles.

**Formules** :
```
E[X_(r)] = r(N+1) / (k+1)
Var[X_(r)] = r(k-r+1)(N+1)(N-k) / [(k+1)²(k+2)]
```

**Paramètres** : Aucun (modèle purement théorique)

**Valeur thèse** : Toute déviation significative entre données réelles et ce modèle prouve formellement un biais mécanique, sans ambiguïté statistique.

**Exigences de mise en œuvre** :
- Calcul exact des distributions par position (r=1 à k)
- Test KS par position (observé vs théorique)
- Heatmap des déviations position × numéro
- Export des paramètres théoriques vs observés par position

### 5.3 M17 — Logistic Temporal

**Objectif** : Modèle de régression logistique binaire par numéro intégrant des features temporelles interprétables.

**Features** :
- Fréquences glissantes multi-fenêtres : W = {10, 30, 100, all}
- Gap depuis dernière apparition (normalisé par gap moyen)
- Position dans le mois (1-4 pour saisonnalité)
- Numéro de semaine dans l'année
- Jackpot normalisé (si disponible)
- Interaction : gap × fréquence_globale

**Paramètres configurables** :
- `regularization` : L1 (Lasso, sélection de features) ou L2 (Ridge)
- `cv_folds` : Nombre de folds pour la cross-validation temporelle (défaut: 5)

**Valeur thèse** : Les coefficients sont interprétables. Si tous ≈ 0 → aucune feature n'est prédictive → système équitable. Permet un test d'hypothèse formel sur chaque feature (test de Wald).

**Livrables spécifiques** :
- Tableau des coefficients avec intervalles de confiance à 95%
- Test de Wald pour chaque coefficient
- Feature importance globale (somme des |coefficients| normalisés)

### 5.4 M18 — Dirichlet Process

**Objectif** : Clustering non-paramétrique des numéros via un prior Dirichlet Process, pour détecter si des groupes de numéros partagent un comportement statistique commun.

**Principe** :
Si des clusters correspondent à des plages de numéros (ex. {1-15}, {16-30}, {31-45}), cela suggère un biais lié à la construction physique de l'urne (position des boules, poids de l'encre par plage).

**Valeur thèse** : Preuve indirecte de biais mécanique structuré sans hypothèse a priori sur la nature du biais.

**Dépendance** : `scikit-learn` (BayesianGaussianMixture)

### 5.5 M19 — Emission Order Model

**Objectif** : Exploiter l'ordre physique d'émission pour prédire les numéros restants étant donné les premiers émis.

**Modes d'utilisation** :
- **Mode pré-tirage** : prédire les k numéros sans aucun contexte d'ordre
- **Mode partiel** : après p boules émises, prédire les k-p restantes

**Algorithme** :
```
1. Apprendre la matrice de transition n × n (probabilité que numéro j 
   soit émis directement après numéro i)
2. Apprendre les biais par position (distribution empirique par position d'émission)
3. Pour la prédiction : combiner probabilité uniforme + correction transition 
   + correction position
4. Pondération des corrections proportionnelle à leur significativité statistique
```

**Paramètres** :
- `w_base` : Poids probabilité uniforme (défaut: 0.6)
- `w_transition` : Poids biais de transition (défaut: 0.2)
- `w_position` : Poids biais de position (défaut: 0.2)
- `min_draws_for_bias` : Seuil minimum de tirages pour activer les corrections (défaut: 100)

**Exigences** : Uniquement disponible si les données d'ordre d'émission sont présentes pour ≥ `min_draws_for_bias` tirages.

**Métrique clé** : `lift_by_context_size` — lift du modèle vs M0 pour chaque taille de contexte (0 à k-1 boules connues). Graphique principal pour la thèse.

### 5.6 M20 — Jackpot Context Model

**Objectif** : Modèle de régression logistique ajustant les probabilités en fonction du montant du jackpot annoncé.

**Principe** :
Pour chaque numéro i, ajuster la probabilité d'apparition via :
```
P(numéro i | jackpot) = sigmoid(β₀ᵢ + β₁ᵢ × log(jackpot))
```

**Combinaison** :
```
P_final = w_base × P_fréquentiste + w_jackpot × P_logistique
```

**Paramètres** :
- `w_jackpot` : Poids de la correction jackpot (défaut: 0.3, adaptatif selon significativité)
- `min_jackpot_variance` : Variance minimale des jackpots pour activer le modèle

**Valeur thèse** : Si le lift de M20 > lift de M1 de façon significative, cela prouve que le montant du jackpot est informatif sur les numéros tirés — signal majeur de non-indépendance.

**Exigences** : Uniquement disponible si les données de jackpot couvrent ≥ 50 tirages avec variance > 0.

---

## 6. Module Forensique du Générateur

### 6.1 Vue d'Ensemble

Le module forensique est la contribution principale de la thèse. Il évalue si le générateur (physique ou logiciel) est conforme à un processus i.i.d. uniforme via une batterie de tests organisés en quatre catégories.

### 6.2 Catégorie 1 — Tests NIST Adaptés

Adaptation des tests NIST SP 800-22 (standard de validation des générateurs cryptographiques) aux séquences de loterie.

**Transformation des données** :
Les tirages sont convertis en séquences binaires via un encodage one-hot concaténé :
- Chaque tirage → vecteur binaire de taille N (1 si numéro présent, 0 sinon)
- Concaténation de tous les tirages → séquence de longueur T × N bits

| Test | Description | H0 |
|------|-------------|-----|
| **Fréquence Monobit** | Proportion de 1 dans la séquence binaire | 50% de 0 et 50% de 1 |
| **Runs** | Longueur et distribution des séquences de 0 ou 1 consécutifs | Conforme à l'aléatoire |
| **Blocs de Fréquence** | Uniformité dans des blocs de M bits | Chaque bloc est uniforme |
| **Test Sériel** | Uniformité des paires et triplets de bits | Motifs de longueur m équiprobables |
| **Entropie Approximative** | Régularité des motifs de longueur m vs m+1 | Complexité maximale |

**Paramètres** :
- `block_size_M` : Taille des blocs pour le test des blocs (défaut: 128)
- `serial_pattern_length` : Longueur des motifs pour le test sériel (défaut: 3)
- `alpha` : Niveau de signification (défaut: 0.01, conforme NIST)

### 6.3 Catégorie 2 — Tests de Biais Physiques

Tests ciblant les anomalies mécaniques documentées dans les machines à boules.

#### 6.3.1 Test de Biais de Poids

**Hypothèse** : Les boules légères (numéros élevés = moins d'encre) sortent plus souvent dans les machines physiques.

**Méthode** : Régression linéaire `fréquence_i ~ numéro_i`

**Indicateurs** :
- Pente de régression avec intervalle de confiance
- p-value (H0 : pente = 0)
- Coefficient de détermination R²
- Direction du biais (numéros élevés ou bas favorisés)

#### 6.3.2 Test de Biais de Position d'Émission

**Hypothèse** : La distribution d'un numéro varie selon sa position d'émission (1ère, 2ème, ..., kème boule).

**Propriété théorique clé** : Par échangeabilité de De Finetti, dans un tirage sans remise parfait, la distribution marginale de chaque position est **identique** (uniforme sur {1..N}). Toute déviation est donc un biais mécanique réel.

**Méthode** : Test χ² par position d'émission (nécessite les données d'ordre)

**Livrables** :
- Heatmap `[position_émission × numéro]` avec ratio observé/attendu
- Boxplot des numéros émis par position
- Test KS par position vs distribution uniforme

#### 6.3.3 Test de Dérive Thermique

**Hypothèse** : La machine chauffe pendant le tirage, modifiant les probabilités au fil des émissions.

**Méthode** : Régression linéaire `numéro_émis ~ position_émission` + test de permutation (10 000 permutations)

**Indicateurs** :
- Pente avec intervalle de confiance
- p-value paramétrique et par permutation
- Visualisation : numéro moyen émis par position dans le temps

#### 6.3.4 Test d'Évitement d'Adjacence

**Hypothèse** : Les boules physiquement proches (numéros adjacents) ont tendance à s'attirer ou se repousser.

**Méthode** : Comparaison de la fréquence des paires adjacentes `(i, i+1)` émises consécutivement vs attendu théorique par simulation Monte Carlo (50 000 simulations).

#### 6.3.5 Test de Dépendance Intra-Tirage (Information Mutuelle)

**Hypothèse** : La boule émise en position p est influencée par la boule en position p-1, au-delà de la contrainte sans remise.

**Méthode** : Information mutuelle normalisée entre positions consécutives, avec distribution nulle par permutation (1 000 permutations).

### 6.4 Catégorie 3 — Tests de Générateur Logiciel (RNG)

Tests ciblant les vulnérabilités classiques des générateurs pseudo-aléatoires.

| Test | Description | Signal de fraude |
|------|-------------|-----------------|
| **Biais LSB** | Les bits de poids faible sont-ils uniformes ? | PRNG mal conçu |
| **Biais Modulo** | Biais pour M = 2^k non multiple de N | Troncature naïve |
| **Périodicité** | Séquences répétées toutes les P périodes | Période trop courte |
| **Birthday Paradox** | Collision de tirages identiques | Espace d'états trop petit |
| **Transition Matrix** | Dépendances séquentielles entre tirages | PRNG avec mémoire |

**Paramètres** :
- `period_candidates` : Périodes à tester [52, 104, 260, 520] (en nombre de tirages)
- `n_permutations_lsb` : Permutations pour le test LSB (défaut: 10 000)

### 6.5 Catégorie 4 — Tests Structurels

| Test | Description | Distribution théorique |
|------|-------------|----------------------|
| **Distribution des sommes** | Test KS sur S = n1+...+nk | Normale approchée (TCL) avec μ et σ² analytiques |
| **Statistiques d'ordre** | Test KS par rang vs distribution exacte | Calculée analytiquement |
| **Corrélation inter-positions** | Matrice Spearman des rangs par position | Corrélation attendue = -1/(k-1) |
| **Distribution des gaps minimaux** | Écart minimum entre numéros consécutifs | Simulée par Monte Carlo |

### 6.6 Score de Conformité Global

#### 6.6.1 Calcul du Score

```
Score = Σ (w_i × pass_i) / Σ w_i
```

Où `w_i` est le poids associé à chaque test selon sa pertinence forensique.

| Catégorie | Poids total | Justification |
|-----------|-------------|---------------|
| Tests NIST | 25% | Standard international |
| Tests physiques | 35% | Priorité (machine physique) |
| Tests RNG | 25% | Pertinent pour les generateurs logiciels |
| Tests structurels | 15% | Complémentaire |

#### 6.6.2 Intervalles de Confiance par Monte Carlo

Le score est calibré par simulation :
1. Générer 1 000 datasets synthétiques de T tirages parfaitement aléatoires
2. Calculer le score de conformité sur chaque dataset
3. En déduire `[CI_2.5%, CI_97.5%]` de l'intervalle attendu
4. Comparer le score observé à cet intervalle

**Interprétation** :
- Score dans `[CI_2.5%, CI_97.5%]` → Générateur conforme
- Score < `CI_2.5%` → Moins aléatoire qu'attendu (biais potentiel)
- Score > `CI_97.5%` → Plus aléatoire qu'attendu (excès d'uniformité, suspect)

#### 6.6.3 Évolution Temporelle du Score

Le score doit être calculable sur des **fenêtres glissantes** (ex. score annuel glissant) pour détecter une dérive progressive du générateur dans le temps.

---

## 7. Module Détection de Fraude

### 7.1 Définition et Cadre Épistémologique

> **Formulation académique obligatoire** : Ce module détecte des déviations statistiquement incompatibles avec un processus aléatoire équitable. Il ne prouve pas une fraude, qui nécessiterait des preuves externes (accès aux systèmes, témoins, etc.).

### 7.2 Taxonomie des Signaux de Fraude

```
Signaux détectables
├── Anomalies de qualité des données
│   ├── Tirages dupliqués
│   ├── Numéros hors plage
│   ├── Dates invalides ou anachroniques
│   └── Hash de tirage dupliqué sur des dates différentes
│
├── Anomalies du générateur
│   ├── Sous-dispersion (données trop uniformes = lissage artificiel)
│   ├── Sur-dispersion (biais de fréquence)
│   ├── Périodicité courte du générateur
│   ├── Biais de position d'émission
│   └── Biais LSB / Modulo
│
├── Anomalies comportementales
│   ├── Corrélation jackpot → numéros tirés
│   ├── Clustering temporel (tirages similaires proches)
│   ├── Effet de seuil (distribution change au-dessus d'un montant)
│   └── Anomalie Must-Be-Won (distribution différente sur ces draws)
│
└── Anomalies structurelles
    ├── Violation de la loi de Benford (données manipulées)
    ├── Ruptures non annoncées (changepoints)
    └── Jackpot vs position d'émission (le plus grave)
```

### 7.3 Tests de Détection

#### 7.3.1 Test de Sous/Sur-Dispersion

**Objectif** : Détecter si la variance des fréquences est anormalement faible (données lissées artificiellement) ou élevée (biais fort).

**Méthode** : Test χ² sur la variance observée vs variance binomiale attendue

**Seuils d'alerte** :
- Ratio `var_observée / var_attendue` < 0.7 → UNDER-DISPERSION (CRITICAL)
- Ratio > 1.5 → OVER-DISPERSION (WARNING)

#### 7.3.2 Test de Loi de Benford

**Objectif** : Vérifier que la distribution des premiers chiffres des numéros tirés est cohérente avec la distribution théorique attendue pour ce pool.

**Note** : La distribution de Benford est adaptée pour chaque jeu (ex: pool 1-45 vs 1-47 vs 1-35).

#### 7.3.3 Test de Clustering Temporel

**Objectif** : Vérifier que les tirages similaires ne se concentrent pas dans le temps.

**Méthode** : Similarité de Jaccard entre tirages consécutifs, comparée à la distribution théorique hypergéométrique.

#### 7.3.4 Test de Jackpot vs Position d'Émission (Test Ultime)

**Objectif** : Vérifier que la position d'émission d'un numéro est indépendante du montant du jackpot.

**Interprétation** : Si la corrélation est significative, cela signifie que le mécanisme physique est influencé par les enjeux financiers — matériellement impossible sans manipulation.

**Méthode** : Corrélation de Spearman `rang(jackpot) ~ position_émission(numéro_i)` pour chaque numéro.

### 7.4 Score de Risque de Fraude

Le score est calculé comme une somme pondérée des signaux détectés.

| Signal | Poids | Sévérité déclenchée |
|--------|-------|---------------------|
| Sous-dispersion | 3 | CRITICAL |
| Périodicité RNG | 3 | CRITICAL |
| Jackpot vs émission | 3 | CRITICAL |
| Violation Benford | 2 | HIGH |
| Biais de position | 2 | HIGH |
| Clustering temporel | 2 | HIGH |
| Biais de poids | 1 | WARNING |
| Anomalie Must-Be-Won | 2 | HIGH |

**Niveaux de risque** :

| Score normalisé | Niveau | Interprétation |
|----------------|--------|----------------|
| 0.0 – 0.2 | LOW | Aucun signal significatif |
| 0.2 – 0.4 | MODERATE | Quelques anomalies isolées, surveillance recommandée |
| 0.4 – 0.6 | HIGH | Plusieurs signaux convergents, investigation recommandée |
| 0.6 – 1.0 | CRITICAL | Signaux multiples et forts, investigation externe requise |

### 7.5 Workflow de Gestion des Alertes

```
Alerte générée (status: OPEN)
        │
        ▼
Examen initial par l'analyste
        │
   ┌────┴────┐
   ▼         ▼
Faux positif  Signal à investiguer
(FALSE_POSITIVE)   │
                   ▼
           status: INVESTIGATING
           + note d'investigation
                   │
             ┌─────┴─────┐
             ▼           ▼
           CLOSED    Escalade externe
           + résolution    (hors scope)
```

---

## 8. Module Analyse des Jackpots

### 8.1 Test d'Indépendance Jackpot/Générateur

**Objectif** : Vérifier formellement que le montant du jackpot n'influence pas les numéros tirés.

**Méthode** : Corrélation point-bisériale pour chaque numéro avec correction Benjamini-Hochberg sur N tests multiples.

**Résultat attendu pour un système équitable** : Aucune corrélation significative après correction FDR.

### 8.2 Modèle de Biais de Sélection des Joueurs

**Objectif** : Caractériser les biais de sélection des joueurs via les proxies disponibles dans les données de tirage.

**Proxies utilisés** :
- Distribution des sommes des numéros tirés (les joueurs choisissent des numéros groupés)
- Ratio de numéros ≤ 31 (anniversaires sur-joués)
- Ratio de numéros pairs
- Étendue des numéros (max - min)

**Note** : Ces proxies mesurent indirectement le comportement des joueurs, pas les numéros tirés. Si des corrélations apparaissent, elles signalent un biais du GÉNÉRATEUR, pas des joueurs.

### 8.3 Regression Discontinuity Design (RDD)

**Objectif** : Détecter si la distribution des tirages change brusquement à un seuil de jackpot.

**Méthode** : Identification du seuil optimal par analyse des percentiles [25, 50, 75, 90] du jackpot, puis test de Mann-Whitney entre distributions sous/sur le seuil.

**Valeur thèse** : Le RDD est une méthode économétrique robuste standard — son application aux loteries est originale et publiable.

### 8.4 Analyse des Tirages Must-Be-Won

**Objectif** : Vérifier que les tirages officiellement annoncés "Must Be Won" produisent les mêmes distributions que les tirages normaux.

**Hypothèse H0** : La distribution des numéros tirés est identique pour les tirages Must-Be-Won et normaux.

**Si H0 rejetée** → Le générateur est paramétré différemment pour ces tirages → Signal de fraude CRITICAL.

---

## 9. Évolutions de l'Interface Utilisateur

### 9.1 Nouvelles Pages

#### 9.1.1 Page "Forensique du Générateur" — `/games/{id}/forensics`

**Sections** :
1. **En-tête** : Score de conformité global avec jauge visuelle et intervalle de confiance Monte Carlo
2. **Résultats par catégorie** : Accordéon avec les 4 catégories de tests (NIST / Physiques / RNG / Structurels)
3. **Heatmap Position × Numéro** : Visualisation principale (si données d'ordre disponibles)
4. **Matrice de transition** : Heatmap N×N des transitions entre émissions
5. **Évolution temporelle du score** : Graphique linéaire du score sur fenêtres glissantes
6. **Changepoints annotés** : Timeline avec les ruptures détectées
7. **Bouton "Exporter rapport forensique"** → PDF académique

#### 9.1.2 Page "Détection de Fraude" — `/games/{id}/fraud`

**Sections** :
1. **Score de risque** : Gauge 0-100% avec niveau de couleur (vert/orange/rouge)
2. **Signaux détectés** : Liste des alertes actives avec sévérité et description
3. **Tableau des alertes** : Avec colonnes type / sévérité / date / statut / actions
4. **Workflow** : Interface de gestion des alertes (changer statut, ajouter note)
5. **Historique** : Alertes résolues ou classées faux positif

#### 9.1.3 Page "Analyse Jackpots" — `/games/{id}/jackpot`

**Sections** :
1. **Graphique jackpot dans le temps** : Avec marqueurs de must-be-won et changepoints
2. **Test d'indépendance** : Résultats numéro par numéro avec graphique à barres (corrélations)
3. **Scatter jackpot × somme** : Avec droite de régression et intervalle de confiance
4. **RDD graphique** : Regression Discontinuity à la manière économétrique standard
5. **Tableau Must-Be-Won** : Comparaison statistique avec tirages normaux

#### 9.1.4 Page "Comparateur Inter-Loteries" — `/compare`

**Fonctionnalités** :
- Sélection de 2 à 4 jeux à comparer
- Tableau comparatif des scores de conformité avec intervalles de confiance
- Test d'homogénéité entre loteries (même opérateur → même RNG ?)
- Graphique radar multi-jeux

#### 9.1.5 Page "Calculateur de Puissance" — `/power-analysis`

**Fonctionnalités** :
- Saisie : jeu sélectionné, nombre de tirages disponibles, biais à détecter (%)
- Affichage : puissance statistique pour chaque test de la batterie
- Graphique : courbe puissance vs taille d'échantillon
- Recommandation : "Pour atteindre 80% de puissance sur le test X, il faudrait Y tirages"

### 9.2 Évolutions des Pages Existantes

#### 9.2.1 Import de Données

- Ajout de la détection automatique des colonnes d'ordre d'émission dans le CSV
- Mode "Ordre d'émission" dans la saisie manuelle avec validation en temps réel
- Ajout des champs jackpot dans le formulaire de saisie
- Indicateur visuel : "Ordre d'émission : ✅ Disponible pour 412/520 tirages"

#### 9.2.2 Page d'Analyse Statistique Existante

- Ajout d'une section "Analyse Positionnelle" (si données d'ordre disponibles)
- Ajout du méta-test de la distribution des p-values (test des tests)
- Indicateur de puissance statistique contextuel pour chaque test affiché

#### 9.2.3 Page de Backtest

- Ajout de M16, M17, M18, M19, M20 dans la liste des modèles sélectionnables
- M19 grisé si les données d'ordre ne sont pas disponibles
- M20 grisé si les données de jackpot ne sont pas disponibles
- Graphique "Lift par taille de contexte" pour M19

### 9.3 Composants Réutilisables à Créer

| Composant | Description |
|-----------|-------------|
| `ConformityGauge` | Jauge circulaire avec score + intervalle de confiance |
| `TestResultBadge` | Badge vert/rouge avec p-value et interprétation |
| `HeatmapChart` | Heatmap générique (position × numéro, transitions) |
| `FraudRiskGauge` | Gauge de risque avec niveaux de couleur |
| `AlertCard` | Carte d'alerte avec workflow intégré |
| `PowerCurve` | Courbe de puissance statistique interactive |
| `RDDChart` | Graphique Regression Discontinuity |
| `TimelineAnnotated` | Timeline avec changepoints annotés |

---

## 10. Évolutions de l'API Backend

### 10.1 Nouveaux Endpoints

#### Module Forensique

```
GET  /api/games/{id}/forensics
     → Retourne le dernier profil forensique calculé

POST /api/games/{id}/forensics/run
     Body: { "period_start": "...", "period_end": "...", "n_simulations": 1000 }
     → Lance le calcul du profil forensique (tâche asynchrone)
     → Retourne: { "task_id": "uuid" }

GET  /api/games/{id}/forensics/{profile_id}/heatmap
     → Retourne les données de la heatmap position × numéro

GET  /api/games/{id}/forensics/{profile_id}/transition-matrix
     → Retourne la matrice de transition entre émissions

GET  /api/games/{id}/forensics/history
     → Historique des scores de conformité dans le temps
```

#### Module Fraude

```
GET  /api/games/{id}/fraud/score
     → Retourne le score de risque de fraude actuel

GET  /api/games/{id}/fraud/alerts
     Query params: ?status=OPEN&severity=HIGH
     → Liste des alertes avec filtres

PUT  /api/games/{id}/fraud/alerts/{alert_id}
     Body: { "status": "INVESTIGATING", "note": "..." }
     → Met à jour le statut d'une alerte

POST /api/games/{id}/fraud/run
     → Lance l'analyse de détection de fraude complète
```

#### Module Jackpot

```
GET  /api/games/{id}/jackpot/analysis
     → Retourne les résultats de l'analyse jackpot

POST /api/games/{id}/jackpot/run
     → Lance l'analyse jackpot

GET  /api/games/{id}/jackpot/independence-test
     → Test d'indépendance jackpot/numéros détaillé par numéro
```

#### Module Comparaison

```
POST /api/games/compare
     Body: { "game_ids": ["uuid1", "uuid2"], "metrics": ["conformity", "fraud_risk"] }
     → Retourne la comparaison inter-loteries

GET  /api/games/compare/{comparison_id}
     → Résultats d'une comparaison sauvegardée
```

#### Module Puissance Statistique

```
POST /api/power-analysis
     Body: { "game_id": "uuid", "n_draws": 520, "bias_size": 0.08 }
     → Retourne la puissance pour chaque test avec le dataset fourni
```

#### Nouveaux Modèles (Backtest)

```
POST /api/analyses/backtest
     Body: { "models": ["M16", "M17", "M19", "M20"], ... }
     → Extension du backtest existant avec les nouveaux modèles

GET  /api/analyses/{id}/m19/lift-by-context
     → Graphique lift M19 par taille de contexte
```

### 10.2 WebSocket pour Tâches Longues

```
WS   /ws/tasks/{task_id}/progress
     → Stream de progression pour les calculs forensiques longs
     Message format: {
         "progress": 0.42,
         "eta_seconds": 180,
         "current_step": "nist_tests",
         "completed_steps": ["standard_tests"]
     }
```

### 10.3 Endpoint d'Export Académique

```
GET  /api/games/{id}/report/academic
     Query params: ?format=pdf&include_forensics=true&include_fraud=true
     → Génère et retourne un rapport PDF académique complet

GET  /api/games/{id}/report/academic/latex
     → Retourne le source LaTeX du rapport (pour soumission académique)
```

---

## 11. Exigences Non-Fonctionnelles

### 11.1 Reproductibilité Scientifique

- **Seed globale** : Toute analyse stochastique utilise une seed configurable (défaut: 42)
- **Hash du dataset** : SHA-256 du jeu de données utilisé stocké avec chaque résultat
- **Hash des paramètres** : SHA-256 de la configuration d'analyse
- **Version applicative** : Numéro de version stocké avec chaque résultat
- **Traçabilité complète** : Toute analyse est reproductible à l'identique en relançant avec les mêmes paramètres et la même seed

```python
# Chaque analyse stocke :
{
    "dataset_hash": "sha256:abc123...",
    "params_hash": "sha256:def456...",
    "seed": 42,
    "app_version": "2.0.0",
    "python_version": "3.12.x",
    "package_versions": {
        "scipy": "1.12.x",
        "numpy": "1.26.x",
        "statsmodels": "0.14.x"
    },
    "timestamp_utc": "2026-02-21T10:30:00Z"
}
```

### 11.2 Performance

| Opération | Cible | Contexte |
|-----------|-------|---------|
| Calcul profil forensique complet | < 60 secondes | 520 tirages, 1 000 simulations MC |
| Calcul score fraude | < 10 secondes | 520 tirages |
| Analyse jackpot | < 15 secondes | 520 tirages |
| Backtest M17 | < 30 secondes | 100 tirages de test |
| Backtest M19 | < 45 secondes | 100 tirages de test |
| Heatmap position × numéro | < 5 secondes | 520 tirages |
| Rafraîchissement vue matérialisée | < 30 secondes | Post-import |

### 11.3 Qualité des Tests Statistiques

- Tous les tests doivent rapporter : statistique de test, p-value, degrés de liberté, seuil de rejet, conclusion
- Les corrections multi-tests (Benjamini-Hochberg) sont appliquées systématiquement dès que N ≥ 3 tests
- La puissance statistique est calculée et affichée pour chaque test principal
- Les intervalles de confiance sont systématiquement à 95% sauf mention contraire

### 11.4 Sécurité des Données

- Aucune donnée personnelle traitée (les tirages sont des données publiques)
- Hash SHA-256 des fichiers importés pour détection de tampering
- Log d'audit complet sur toutes les modifications de données
- Les rapports PDF incluent un watermark "Analyse académique — Ne pas jouer"

### 11.5 Qualité du Code

- Couverture de tests unitaires ≥ 80% pour tous les nouveaux modules d'analyse
- Chaque nouvelle fonction statistique doit être validée sur des données synthétiques dont le résultat attendu est connu analytiquement
- Documentation de chaque test statistique : hypothèses, formules, limites, références bibliographiques

### 11.6 Accessibilité Scientifique

- Chaque test doit afficher une description en langage naturel interprétable par un non-statisticien
- Les termes techniques sont systématiquement accompagnés d'une infobulle explicative
- Les résultats sont présentés à deux niveaux : vulgarisé (interface) et technique (export)

---

## 12. Architecture Cible

### 12.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React/TypeScript)                   │
│                                                                         │
│  Pages existantes       Nouvelles pages                                 │
│  (enrichies)            ├── /forensics      ├── /fraud                 │
│  ├── /draws             ├── /jackpot         ├── /compare              │
│  ├── /analyses          └── /power-analysis                            │
│  └── /backtest                                                          │
│                                                                         │
│  Nouveaux composants : ConformityGauge, HeatmapChart, FraudRiskGauge  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP/WebSocket
┌───────────────────────────────▼─────────────────────────────────────────┐
│                          BACKEND (Python/FastAPI)                       │
│                                                                         │
│  api/                    services/              analysis/               │
│  ├── forensics.py        ├── forensics_svc.py   ├── nist_tests.py      │
│  ├── fraud.py            ├── fraud_svc.py        ├── physical_tests.py │
│  ├── jackpot.py          ├── jackpot_svc.py      ├── rng_tests.py      │
│  └── compare.py          └── power_svc.py        ├── fraud_engine.py   │
│                                                   ├── jackpot_models.py│
│                          prob_models/             └── power_analysis.py│
│                          ├── m16_order_stats.py                        │
│                          ├── m17_logistic.py                           │
│                          ├── m18_dirichlet_process.py                  │
│                          ├── m19_emission_order.py                     │
│                          └── m20_jackpot_context.py                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                          PostgreSQL 16+                                 │
│                                                                         │
│  Tables modifiées : draws (+ emission_order, jackpot_amount, ...)      │
│                                                                         │
│  Nouvelles tables :                                                     │
│  ├── generator_profiles    ├── fraud_alerts                            │
│  ├── changepoints (ext.)   └── jackpot_analyses                        │
│                                                                         │
│  Nouvelles vues matérialisées : draws_positional, jackpot_analysis     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Organisation des Nouveaux Modules Python

```
backend/app/analysis/
├── forensics/
│   ├── __init__.py
│   ├── nist_tests.py           # Tests NIST adaptés
│   ├── physical_tests.py       # Biais de poids, position, dérive thermique
│   ├── rng_tests.py            # LSB, modulo, périodicité, birthday
│   ├── structural_tests.py     # Statistiques d'ordre, distribution sommes
│   ├── conformity_score.py     # Score global + Monte Carlo CI
│   └── generator_profile.py   # Orchestrateur du profil complet
│
├── fraud/
│   ├── __init__.py
│   ├── dispersion_tests.py     # Sous/sur-dispersion
│   ├── benford_tests.py        # Loi de Benford adaptée
│   ├── clustering_tests.py     # Clustering temporel
│   ├── jackpot_fraud_tests.py  # Jackpot vs émission
│   ├── fraud_score.py          # Score de risque agrégé
│   └── alert_manager.py        # Création et gestion des alertes
│
├── jackpot/
│   ├── __init__.py
│   ├── independence_test.py    # Corrélation jackpot/numéros
│   ├── player_bias.py          # Modèle de biais de sélection
│   ├── rdd_analysis.py         # Regression Discontinuity Design
│   └── must_be_won.py          # Analyse tirages spéciaux
│
├── power/
│   ├── __init__.py
│   └── power_calculator.py    # Calculateur de puissance statistique
│
└── prob_models/
    ├── m16_order_statistics.py
    ├── m17_logistic_temporal.py
    ├── m18_dirichlet_process.py
    ├── m19_emission_order.py
    └── m20_jackpot_context.py
```

---

## 13. Roadmap et Livrables

### 13.1 Phase 1 — Fondations & Données (Semaines 1–4)

**Objectif** : Préparer l'infrastructure pour les nouvelles données.

| Tâche | Responsable | Durée |
|-------|-------------|-------|
| Migration schéma BDD (nouvelles colonnes) | Dev Backend | 2 jours |
| Création des nouvelles tables | Dev Backend | 1 jour |
| Création des vues matérialisées | Dev Backend | 1 jour |
| Mise à jour du parser CSV (ordre + jackpot) | Dev Backend | 3 jours |
| Mise à jour du formulaire de saisie manuelle | Dev Frontend | 2 jours |
| Tests d'import avec données réelles | Analyste | 2 jours |
| Migration des données existantes | Dev Backend | 1 jour |

**Livrable Phase 1** : Application v1.0 avec support des nouvelles colonnes, import CSV étendu fonctionnel, données 10 ans importées.

### 13.2 Phase 2 — Forensique du Générateur (Semaines 5–10)

**Objectif** : Implémenter et valider le module forensique complet.

| Tâche | Responsable | Durée |
|-------|-------------|-------|
| Implémentation tests NIST adaptés | Data Scientist | 5 jours |
| Implémentation tests physiques (biais poids, position, thermique) | Data Scientist | 5 jours |
| Implémentation tests RNG (LSB, modulo, périodicité, birthday) | Data Scientist | 4 jours |
| Implémentation tests structurels (statistiques d'ordre, sommes) | Data Scientist | 3 jours |
| Score de conformité global + Monte Carlo CI | Data Scientist | 3 jours |
| Validation sur données synthétiques connues | Data Scientist | 3 jours |
| API endpoints forensiques | Dev Backend | 3 jours |
| Page UI Forensique (score + tests + heatmap) | Dev Frontend | 5 jours |
| Tests unitaires (couverture ≥ 80%) | Dev Backend | 3 jours |

**Livrable Phase 2** : Module forensique complet, validé, avec résultats sur les 3 loteries australiennes. Premier chapitre thèse rédigeable.

### 13.3 Phase 3 — Détection de Fraude & Jackpot (Semaines 11–16)

**Objectif** : Implémenter les modules fraude et jackpot.

| Tâche | Responsable | Durée |
|-------|-------------|-------|
| Moteur de détection de fraude (5 tests) | Data Scientist | 5 jours |
| Score de risque fraude agrégé | Data Scientist | 2 jours |
| Système d'alertes avec workflow | Dev Backend | 3 jours |
| Analyse jackpot (indépendance, RDD, Must-Be-Won) | Data Scientist | 5 jours |
| API endpoints fraude et jackpot | Dev Backend | 3 jours |
| Page UI Fraude | Dev Frontend | 4 jours |
| Page UI Jackpot | Dev Frontend | 4 jours |
| Tests unitaires | Dev Backend | 3 jours |

**Livrable Phase 3** : Modules fraude et jackpot opérationnels. Deuxième chapitre thèse rédigeable.

### 13.4 Phase 4 — Nouveaux Modèles & Backtest (Semaines 17–22)

**Objectif** : Implémenter les 5 nouveaux modèles de prédiction.

| Tâche | Responsable | Durée |
|-------|-------------|-------|
| M16 Order Statistics Baseline | Data Scientist | 3 jours |
| M17 Logistic Temporal | Data Scientist | 4 jours |
| M18 Dirichlet Process | Data Scientist | 3 jours |
| M19 Emission Order Model | Data Scientist | 5 jours |
| M20 Jackpot Context Model | Data Scientist | 4 jours |
| Intégration dans le backtest existant | Dev Backend | 3 jours |
| Comparateur inter-loteries | Dev Backend + Frontend | 4 jours |
| Calculateur de puissance statistique | Data Scientist + Frontend | 3 jours |
| Tests de validation sur données synthétiques | Data Scientist | 3 jours |

**Livrable Phase 4** : 5 nouveaux modèles intégrés au backtest. Résultats comparatifs complets.

### 13.5 Phase 5 — Rapports & Finalisation (Semaines 23–26)

**Objectif** : Export académique, documentation, déploiement.

| Tâche | Responsable | Durée |
|-------|-------------|-------|
| Export rapport académique PDF | Dev Backend | 4 jours |
| Export source LaTeX | Dev Backend | 2 jours |
| Documentation API complète | Dev Backend | 2 jours |
| Documentation utilisateur | Analyste | 3 jours |
| Tests d'intégration end-to-end | Dev Backend | 3 jours |
| Optimisation performance | Dev Backend | 3 jours |
| Déploiement et mise à jour Docker Compose | Dev Backend | 2 jours |

**Livrable Phase 5** : Application v2.0 complète, documentée, deployée.

---

## 14. Critères d'Acceptance

### 14.1 Critères Fonctionnels

| Critère | Test de validation |
|---------|-------------------|
| Import CSV avec ordre d'émission | Importer 100 tirages avec ordre → vérifier intégrité en BDD |
| Import CSV avec jackpot | Importer 100 tirages avec jackpot → vérifier cohérence |
| Score de conformité reproductible | Lancer 2× la même analyse → scores identiques |
| Score de conformité calibré | Sur données synthétiques i.i.d. → score dans [CI_2.5%, CI_97.5%] |
| Tests NIST : faux positifs contrôlés | Sur 1 000 datasets synthétiques → ≤ 5% de rejets à α=0.05 |
| Détection de sous-dispersion | Sur données lissées artificiellement → CRITICAL détecté |
| M19 disponible si ordre présent | Si 0 tirage avec ordre → M19 grisé dans le backtest |
| M20 disponible si jackpot présent | Si 0 tirage avec jackpot → M20 grisé dans le backtest |
| Alertes workflow fonctionnel | Créer alerte → changer statut → résoudre |
| Export PDF académique | Générer rapport → vérifier présence section limitations |

### 14.2 Critères Statistiques

| Critère | Valeur cible |
|---------|-------------|
| Puissance test χ² sur biais 10% avec 520 tirages | ≥ 80% |
| Faux positifs des tests NIST sur données i.i.d. | ≤ 5% |
| Couverture tests unitaires nouveaux modules | ≥ 80% |
| Temps de calcul profil forensique complet | ≤ 60 secondes |
| Précision score Monte Carlo (CI) | n_simulations ≥ 1 000 |

### 14.3 Critères de Qualité Scientifique

- Chaque test référence au moins une source bibliographique
- Toute anomalie signale sa taille d'effet (pas seulement le p-value)
- La puissance statistique est affichée pour chaque test principal
- Les rapports PDF contiennent obligatoirement la section "Limitations"
- Le watermark "Analyse académique — Ne pas jouer" est présent sur tous les exports

---

## 15. Limites et Avertissements

### 15.1 Limites Fondamentales de l'Étude

> ⚠️ **Ces limites doivent figurer dans TOUS les rapports générés par l'application.**

1. **Puissance statistique limitée** : Avec 520 tirages par loterie, seuls les biais supérieurs à ~8% sont détectables avec 80% de puissance. Les biais plus faibles peuvent exister sans être détectés.

2. **Aucune preuve de fraude** : Une déviation statistique, même significative, ne prouve pas une fraude intentionnelle. Elle peut résulter d'un défaut mécanique naturel, d'un biais de données, ou d'une coïncidence statistique.

3. **Tests multiples** : Avec une batterie de 15+ tests, le risque de faux positifs augmente. La correction Benjamini-Hochberg est appliquée mais ne garantit pas l'absence de faux positifs.

4. **Absence de métadonnées** : Sans accès aux données techniques des opérateurs (type de machine, numéros de série, maintenances), les changepoints détectés ne peuvent pas être contextualisés.

5. **Aucune prédiction** : Aucun modèle de l'application ne peut prédire les tirages futurs d'un système équitable. Le lift > 1 observé en backtest peut résulter de l'overfitting ou du hasard.

### 15.2 Ce que l'Application Peut Conclure

✅ "Le générateur présente une déviation de X% sur la position p (p=0.003, puissance=82%)"  
✅ "Un changepoint est détecté au tirage T, compatible avec un changement de matériel"  
✅ "Les données sont statistiquement compatibles avec un processus i.i.d. uniforme"  
✅ "Le montant du jackpot n'est pas significativement corrélé aux numéros tirés"  

### 15.3 Ce que l'Application Ne Peut Pas Conclure

❌ "Le système est frauduleux"  
❌ "Ces numéros ont plus de chances de sortir"  
❌ "Jouer améliorera vos gains"  
❌ "Le biais détecté est intentionnel"  

---

## Annexes

### A. Références Bibliographiques

| Référence | Utilisation dans l'application |
|-----------|--------------------------------|
| NIST SP 800-22 (2010) | Tests de générateurs pseudo-aléatoires |
| Benjamini & Hochberg (1995) | Correction FDR multi-tests |
| Killick et al. (2012) | Algorithme PELT pour changepoints |
| Kullback & Leibler (1951) | Divergence KL |
| Brier (1950) | Score de probabilité |
| De Finetti (1931) | Échangeabilité et tirage sans remise |
| Benford (1938) | Loi du premier chiffre |
| Lo & MacKinlay (1988) | Variance Ratio Test |
| Thistlethwaite & Campbell (1960) | Regression Discontinuity Design |
| Nigrini (2012) | Forensique statistique par la loi de Benford |

### B. Glossaire

| Terme | Définition |
|-------|------------|
| **Ordre d'émission** | Séquence temporelle réelle de sortie des boules, avant tri numérique |
| **Statistique d'ordre** | k-ième plus petite valeur dans un échantillon aléatoire |
| **Échangeabilité** | Propriété selon laquelle la distribution jointe est invariante par permutation des indices |
| **Must-Be-Won** | Tirage officiel dont le jackpot Div1 doit impérativement être gagné, avec redistribution si non |
| **Biais modulo** | Biais introduit quand un RNG génère dans [0, M[ avec M non multiple de N |
| **RDD** | Regression Discontinuity Design — méthode économétrique de détection d'effets de seuil |
| **Walk-forward** | Méthode de backtest sans fuite temporelle (entraînement sur le passé uniquement) |
| **Score de Brier** | Mesure de la précision des prédictions probabilistes |
| **ECE** | Expected Calibration Error — mesure de la calibration d'un modèle probabiliste |
| **Lift** | Ratio hit rate observé / hit rate attendu aléatoirement |
| **FDR** | False Discovery Rate — taux de fausses découvertes contrôlé par BH |

### C. Environnement de Développement

```bash
# Dépendances Python supplémentaires pour la v2.0
pip install scipy statsmodels scikit-learn ruptures

# Dépendances optionnelles (déjà dans v1.0)
pip install tensorflow hmmlearn pgmpy copulas

# Nouvelles dépendances optionnelles v2.0
pip install pingouin  # Tests statistiques avancés
pip install linearmodels  # RDD et économétrie
```

### D. Variables d'Environnement Supplémentaires

```env
# Reproductibilité
GLOBAL_RANDOM_SEED=42

# Performance forensique
FORENSICS_MC_SIMULATIONS=1000
FORENSICS_PERMUTATION_TESTS=10000
MAX_CONCURRENT_FORENSICS=2

# Alertes
FRAUD_ALERT_AUTO_NOTIFY=false
FRAUD_ALERT_CRITICAL_THRESHOLD=0.6

# Rapports
REPORT_WATERMARK_TEXT="Analyse académique - Ne pas jouer"
REPORT_INCLUDE_LIMITATIONS=true
```

---

*Document : CDC-LOTTO-ANALYZER-V2 — Version 1.0 — Février 2026*  
*Projet Thèse : Analyse Statistique, Forensique et Détection d'Anomalies dans les Loteries Australiennes*  
*Toute évolution de ce document doit être validée avant implémentation.*