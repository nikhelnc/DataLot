# 📊 Modèles de Prédiction - Lotto Analyzer

Ce document décrit les différents modèles statistiques utilisés pour l'analyse et la simulation des tirages de loterie.

---

## 🎯 Vue d'ensemble

Le système utilise plusieurs modèles de prédiction, chacun basé sur une approche statistique différente. Ces modèles sont utilisés dans le module de **backtest** pour évaluer leur performance historique et générer des prédictions pour les prochains tirages.

| Modèle | Nom | Type | Description courte |
|--------|-----|------|-------------------|
| M0 | Baseline | Aléatoire | Sélection aléatoire uniforme |
| M1 | Dirichlet | Bayésien | Estimation bayésienne des probabilités |
| M2 | Windowed | Fenêtre glissante | Pondération récente avec shrinkage |
| M3 | Exponential Decay | Pondération temporelle | Décroissance exponentielle des fréquences |
| M4 | HMM | Modèle à états cachés | Hidden Markov Model pour régimes |
| M5 | Co-occurrence | Analyse de paires | Paires de numéros sur-représentées |
| M6 | Gaps & Streaks | Analyse des écarts | Numéros "en retard" |
| M7 | Entropy | Sélection entropique | Basé sur l'entropie locale |
| M8 | Changepoint | Détection de ruptures | Analyse post-rupture structurelle |
| M9 | Bayesian Network | Réseau bayésien | Dépendances conditionnelles |
| M10 | Ensemble | Méta-modèle | Combinaison de M0, M1, M2 |
| M11 | LSTM Hybrid | Deep Learning | LSTM + Attention + Embeddings |
| M12 | Mixture Dirichlet | Mélange bayésien | Mélange de distributions Dirichlet |
| M13 | Spectral | Analyse de Fourier | Détection de périodicités |
| M14 | Copula | Modèle de dépendance | Copules pour corrélations |
| M15 | Thompson Sampling | Bandit multi-bras | Exploration/exploitation bayésienne |
| M16 | Gradient Boosting | Machine Learning | XGBoost avec features engineered |
| M17 | Autoencoder Anomaly | Deep Learning | Détection d'anomalies par autoencoder |
| M18 | Graph Neural Network | Deep Learning | GNN sur graphe de co-occurrences |
| M19 | Temporal Fusion | Multi-échelle | Fusion de features multi-temporelles |
| M20 | Meta-Learner | Méta-apprentissage | Sélection dynamique des meilleurs modèles |
| ANTI | Anti-Consensus | Contrarian | Numéros NON prédits par les autres modèles |
| ANTI2 | Anti-Consensus v2 | Contrarian + Diversité | ANTI avec contrainte de diversité |

---

## 📈 Modèles Détaillés

### M0 - Baseline (Aléatoire)

**Type** : Sélection aléatoire uniforme

**Principe** :
- Sélectionne les numéros de manière complètement aléatoire
- Chaque numéro a une probabilité égale d'être choisi
- Sert de **référence** pour comparer les autres modèles

**Utilité** :
- Établir un taux de réussite de base (baseline)
- Tout modèle performant doit faire mieux que M0 sur le long terme

**Formule** :
```
P(numéro i) = 1 / N  (où N = nombre total de numéros)
```

---

### M1 - Dirichlet (Bayésien)

**Type** : Estimation bayésienne avec prior Dirichlet

**Principe** :
- Utilise un prior Dirichlet pour estimer les probabilités de chaque numéro
- Le prior permet de "lisser" les estimations quand on a peu de données
- Les numéros fréquemment tirés ont une probabilité plus élevée

**Paramètres** :
- `alpha` : Paramètre de concentration du prior (défaut: 1 = prior uniforme)

**Avantages** :
- Robuste avec peu de données historiques
- Évite les probabilités nulles pour les numéros jamais tirés

**Formule** :
```
P(numéro i) = (count_i + alpha) / (total_tirages + N * alpha)
```

---

### M2 - Windowed (Fenêtre Glissante)

**Type** : Estimation avec fenêtre temporelle et shrinkage

**Principe** :
- Donne plus de poids aux tirages récents
- Utilise une fenêtre glissante pour capturer les tendances récentes
- Applique un "shrinkage" (λ) vers la moyenne globale pour éviter le surapprentissage

**Paramètres** :
- `window_size` : Taille de la fenêtre (défaut: 50 tirages)
- `lambda_shrink` : Facteur de shrinkage (défaut: 0.1)

**Avantages** :
- Capture les tendances récentes
- Équilibre entre données récentes et historique complet

**Formule** :
```
P(numéro i) = λ * P_global(i) + (1-λ) * P_fenêtre(i)
```

---

### M5 - Co-occurrence (Analyse de Paires)

**Type** : Analyse des paires de numéros

**Principe** :
- Identifie les paires de numéros qui apparaissent ensemble plus souvent que prévu
- Calcule le "delta" entre fréquence observée et fréquence attendue
- Sélectionne les numéros présents dans les paires les plus sur-représentées

**Méthode** :
1. Calculer la fréquence de chaque paire (i, j)
2. Calculer la fréquence attendue : `P(i) * P(j) * n_tirages`
3. Delta = Observé - Attendu
4. Sélectionner les numéros des top 20 paires avec le plus grand delta

**Avantages** :
- Capture les corrélations entre numéros
- Peut détecter des patterns non évidents

---

### M6 - Gaps & Streaks (Analyse des Écarts)

**Type** : Analyse des écarts et séries

**Principe** :
- Identifie les numéros "en retard" (overdue) qui n'ont pas été tirés depuis longtemps
- Compare l'écart actuel à l'écart moyen historique
- Sélectionne les numéros avec le plus grand écart positif (delta_gap)

**Méthode** :
1. Pour chaque numéro, calculer l'écart actuel (tirages depuis dernière apparition)
2. Calculer l'écart moyen historique
3. Delta = Écart actuel - Écart moyen
4. Sélectionner les numéros avec le plus grand delta positif

**Hypothèse** :
- Les numéros "en retard" ont une probabilité accrue d'apparaître (loi des grands nombres)

**Note** : Cette hypothèse est controversée (gambler's fallacy), mais peut capturer des patterns réels dans certains systèmes.

---

### M10 - Ensemble (Stacking)

**Type** : Méta-modèle combinant plusieurs modèles

**Principe** :
- Combine les prédictions de M0, M1 et M2
- Utilise une technique de stacking pour pondérer les modèles
- Produit une prédiction consensus

**Méthode** :
1. Obtenir les probabilités de M0, M1, M2
2. Combiner avec des poids optimisés
3. Sélectionner les numéros avec la probabilité combinée la plus élevée

**Avantages** :
- Réduit la variance des prédictions individuelles
- Plus robuste que les modèles individuels

---

### M11 - LSTM Hybrid (Deep Learning)

**Type** : Réseau de neurones récurrent avec attention

**Principe** :
- Utilise un **LSTM Bidirectionnel** pour capturer la séquentialité temporelle des tirages
- Intègre un **mécanisme d'Attention Multi-Head** pour identifier les tirages passés les plus pertinents
- Emploie des **Embeddings** pour apprendre les relations latentes entre numéros
- Génère une **carte de chaleur de probabilité** pour tous les numéros possibles

**Architecture** :
```
┌─────────────────────────────────────────────────────────┐
│  BLOC 1: Ingestion                                      │
│  ├── Input Séquence: (50 derniers tirages × N numéros) │
│  ├── Embeddings: Projection en espace vectoriel (dim=32)│
│  └── Méta-Features: Somme, écart-type, pairs, etc.     │
├─────────────────────────────────────────────────────────┤
│  BLOC 2: Cerveau Temporel                               │
│  ├── LSTM Bidirectionnel (64 unités)                   │
│  ├── Multi-Head Attention (2 têtes)                    │
│  └── Layer Normalization + Résiduel                    │
├─────────────────────────────────────────────────────────┤
│  BLOC 3: Tête de Prédiction                            │
│  ├── Dense (128) + Dropout (0.3)                       │
│  ├── Dense (64) + Dropout (0.3)                        │
│  └── Sortie Sigmoid (N probabilités indépendantes)     │
└─────────────────────────────────────────────────────────┘
```

**Paramètres** :
- `sequence_length` : Nombre de tirages passés utilisés (défaut: 50)
- `embedding_dim` : Dimension des embeddings (défaut: 32)
- `lstm_units` : Unités LSTM (défaut: 64)
- `attention_heads` : Têtes d'attention (défaut: 2)
- `dropout_rate` : Taux de dropout (défaut: 0.3)
- `epochs` : Époques d'entraînement (défaut: 30)

**Méta-Features calculées** :
1. Somme des numéros (normalisée)
2. Écart-type (normalisé)
3. Ratio de numéros pairs
4. Ratio de numéros bas (< médiane)
5. Étendue (max - min, normalisée)

**Fonction de coût** :
- `binary_crossentropy` : Traite le problème comme N classifications binaires indépendantes

**Avantages** :
- Capture les patterns temporels complexes
- L'attention permet de "voir" des motifs à longue distance
- Les embeddings peuvent révéler des biais mécaniques (si le jeu n'est pas parfaitement aléatoire)
- Fallback automatique vers estimation fréquentielle si TensorFlow non disponible

**Hypothèse testable** :
Si le générateur de nombres est parfait, les embeddings resteront orthogonaux. S'ils se regroupent (clustering), cela pourrait indiquer un biais.

**Note** : Ce modèle est plus lent à entraîner que les autres. Il nécessite TensorFlow.

---

### M3 - Exponential Decay

**Type** : Pondération temporelle avec décroissance exponentielle

**Principe** :
- Pondère les tirages récents plus fortement que les anciens
- Utilise une fonction de décroissance exponentielle : `w(t) = exp(-λ * (T - t))`
- Les numéros fréquents dans les tirages récents ont une probabilité plus élevée

**Paramètres** :
- `lambda_decay` : Taux de décroissance (défaut: 0.02)
- `normalize` : Normaliser les poids (défaut: true)

**Formule** :
```
P(numéro i) = Σ w(t) * I(i ∈ tirage_t) / Σ w(t)
où w(t) = exp(-λ * (T - t))
```

**Avantages** :
- Adaptatif aux changements récents
- Simple et interprétable
- Pas d'hypothèse sur la stationnarité

---

### M4 - HMM (Hidden Markov Model)

**Type** : Modèle à états cachés

**Principe** :
- Modélise les tirages comme provenant de différents "régimes" latents
- Chaque état a sa propre distribution de probabilité sur les numéros
- Prédit l'état suivant et utilise sa distribution d'émission

**Paramètres** :
- `n_states` : Nombre d'états cachés (défaut: 3)
- `n_iter` : Itérations max pour Baum-Welch (défaut: 100)

**Formule** :
```
P(numéro i) = Σ P(état_k | observations) * P(numéro i | état_k)
```

**Avantages** :
- Capture les changements de régime
- Modèle probabiliste complet
- Fallback disponible si hmmlearn non installé

**Dépendance** : `hmmlearn` (optionnel)

---

### M7 - Entropy-Based Selection

**Type** : Sélection basée sur l'entropie locale

**Principe** :
- Calcule l'entropie de Shannon pour chaque numéro sur une fenêtre glissante
- Favorise les numéros avec une entropie plus faible (comportement plus prévisible)
- Compare à l'entropie théorique sous H0

**Paramètres** :
- `window_size` : Taille de la fenêtre (défaut: 30)
- `selection_mode` : 'low_entropy' ou 'high_entropy'
- `temperature` : Température softmax (défaut: 1.0)

**Formule** :
```
H_i = -p_i * log2(p_i) - (1-p_i) * log2(1-p_i)
score_i = H0 - H_i
P(i) ∝ softmax(score_i / temperature)
```

**Avantages** :
- Détecte les numéros avec comportement anormal
- Basé sur la théorie de l'information

---

### M8 - Changepoint Detection

**Type** : Détection de ruptures structurelles

**Principe** :
- Détecte les points de rupture dans les séries de fréquences
- Utilise l'algorithme PELT (Pruned Exact Linear Time)
- Recalcule les probabilités uniquement sur le segment post-rupture

**Paramètres** :
- `model_type` : Modèle de coût ('l1', 'l2', 'rbf')
- `min_segment_size` : Taille minimale de segment (défaut: 20)
- `penalty` : Pénalité PELT ('bic', 'aic')

**Avantages** :
- Adaptatif aux changements structurels
- Ignore les données obsolètes
- Détection automatique des ruptures

**Dépendance** : `ruptures`

---

### M9 - Bayesian Network

**Type** : Réseau bayésien pour dépendances conditionnelles

**Principe** :
- Modélise les dépendances entre numéros via un DAG
- Apprend la structure du réseau par Hill Climbing
- Calcule les probabilités marginales par inférence

**Paramètres** :
- `structure_algo` : Algorithme de structure ('hc', 'pc')
- `max_parents` : Parents max par nœud (défaut: 3)
- `n_top_numbers` : Nombre de numéros à modéliser (défaut: 15)

**Avantages** :
- Capture les corrélations entre numéros
- Interprétable (visualisation du DAG)
- Fallback par corrélations si pgmpy non disponible

**Dépendance** : `pgmpy` (optionnel)

---

### M12 - Mixture of Dirichlet

**Type** : Mélange de distributions Dirichlet

**Principe** :
- Utilise plusieurs composantes Dirichlet pour capturer l'hétérogénéité
- Chaque composante représente un "mode" potentiel
- Estimation par algorithme EM

**Paramètres** :
- `n_components` : Nombre de composantes (défaut: 2)
- `alpha_prior` : Prior de concentration (défaut: 1.0)
- `n_iter` : Itérations EM max (défaut: 100)

**Formule** :
```
P(numéro i) = Σ π_k * E[θ_i | α_k]
où π_k sont les poids du mélange
```

**Avantages** :
- Plus flexible qu'un simple Dirichlet
- Capture les modes multiples

---

### M13 - Spectral / Fourier Analysis

**Type** : Analyse spectrale pour détection de périodicités

**Principe** :
- Applique la FFT sur les séries binaires de présence/absence
- Détecte les fréquences significatives (test de Fisher)
- Extrapole les harmoniques pour prédire

**Paramètres** :
- `min_frequency` : Fréquence minimale à analyser
- `n_harmonics` : Nombre d'harmoniques à retenir (défaut: 3)
- `detrend` : Retirer la tendance linéaire (défaut: true)

**Formule** :
```
X_i(f) = FFT(x_i(t))
PSD_i(f) = |X_i(f)|² / N
```

**Avantages** :
- Détecte les cycles cachés
- Basé sur l'analyse de Fourier classique

---

### M14 - Copula Model

**Type** : Modélisation des dépendances par copules

**Principe** :
- Sépare les distributions marginales de la structure de dépendance
- Utilise une copule gaussienne pour modéliser les corrélations
- Simule pour estimer les probabilités

**Paramètres** :
- `copula_type` : Type de copule ('gaussian')
- `n_simulations` : Nombre de simulations (défaut: 10000)
- `n_groups` : Groupes de numéros (défaut: 5)

**Avantages** :
- Modélise les dépendances non-linéaires
- Flexible sur les marginales
- Fallback par corrélations si copulas non disponible

**Dépendance** : `copulas` (optionnel)

---

### M15 - Thompson Sampling

**Type** : Bandit multi-bras bayésien

**Principe** :
- Traite chaque numéro comme un bras de bandit
- Maintient une distribution Beta(α, β) pour chaque numéro
- Échantillonne pour équilibrer exploration/exploitation

**Paramètres** :
- `alpha_prior` : Prior α (défaut: 1.0)
- `beta_prior` : Prior β (défaut: 1.0)
- `n_samples` : Échantillons Thompson (défaut: 1000)

**Formule** :
```
θ_i ~ Beta(α_i + succès_i, β_i + échecs_i)
P(numéro i) ∝ E[θ_i]
```

**Avantages** :
- Équilibre exploration et exploitation
- Approche bayésienne naturelle
- Converge vers les vraies probabilités

---

### ANTI - Anti-Consensus

**Type** : Stratégie contrariante

**Principe** :
- Identifie les numéros **NON prédits** par les autres modèles (M0, M1, M2, M5, M6, M10)
- Génère des combinaisons à partir de ces numéros "ignorés"
- Hypothèse : si tous les modèles se trompent, les numéros ignorés ont plus de chances

**Paramètres** :
- `n_combinations` : Nombre de combinaisons à générer par tirage (défaut: 10)

**Méthode** :
1. Collecter tous les numéros prédits par les autres modèles
2. Identifier les numéros restants (non prédits)
3. Générer N combinaisons aléatoires à partir de ces numéros

**Avantages** :
- Diversification par rapport aux autres modèles
- Peut capturer des numéros systématiquement sous-estimés

---

### ANTI2 - Anti-Consensus v2 (avec Diversité)

**Type** : Stratégie contrariante avec contrainte de diversité

**Principe** :
- Même logique que ANTI, mais avec une contrainte supplémentaire
- Les combinaisons générées doivent être **diversifiées** entre elles
- Évite d'avoir trop de numéros en commun entre les combinaisons

**Paramètres** :
- `n_combinations` : Nombre de combinaisons à générer par tirage (défaut: 10)
- `max_common_main` : Nombre maximum de numéros principaux identiques entre deux combinaisons (défaut: 2)
- `max_common_bonus` : Nombre maximum de numéros bonus identiques entre deux combinaisons (défaut: 0)

**Méthode** :
1. Collecter tous les numéros prédits par les autres modèles
2. Identifier les numéros restants (non prédits)
3. Pour chaque combinaison à générer :
   - Générer une combinaison candidate
   - Vérifier qu'elle ne partage pas trop de numéros avec les combinaisons déjà générées
   - Si OK, l'ajouter ; sinon, réessayer (max 100 tentatives)
4. Si `max_common_bonus = 0`, chaque combinaison aura un numéro bonus unique

**Avantages** :
- Maximise la couverture des numéros possibles
- Avec `max_common_bonus = 0`, garantit des bonus tous différents
- Meilleure diversification du portefeuille de combinaisons

**Exemple** :
Avec `max_common_main = 2` et `max_common_bonus = 0` :
```
Combo #1: [5, 12, 23, 34, 45] + [7]
Combo #2: [8, 12, 19, 34, 41] + [3]   ← max 2 numéros en commun (12, 34), bonus différent
Combo #3: [3, 15, 27, 38, 49] + [11]  ← bonus unique
```

---

### M16 - Gradient Boosting Ensemble

**Type** : Machine Learning (XGBoost/LightGBM)

**Principe** :
- Utilise le gradient boosting pour prédire les probabilités de chaque numéro
- Ingénierie de features avancée : fréquences, gaps, co-occurrences, statistiques temporelles
- Entraînement walk-forward pour éviter le surapprentissage

**Features calculées** :
1. Fréquence globale, fenêtrée, et avec décroissance exponentielle
2. Gap actuel, gap moyen, ratio de gap
3. Statistiques des tirages récents (somme, écart-type, étendue)
4. Position du numéro (bas/haut), parité

**Paramètres** :
- `n_estimators` : Nombre d'arbres (défaut: 100)
- `max_depth` : Profondeur maximale (défaut: 6)
- `learning_rate` : Taux d'apprentissage (défaut: 0.1)
- `window_size` : Fenêtre pour les features (défaut: 50)

**Avantages** :
- Capture des interactions non-linéaires entre features
- Importance des features interprétable
- Robuste au bruit

**Dépendance** : `xgboost` ou `lightgbm` (optionnel)

---

### M17 - Autoencoder Anomaly

**Type** : Deep Learning (Détection d'anomalies)

**Principe** :
- Entraîne un autoencoder à reconstruire les patterns "normaux" de tirages
- Les numéros avec une erreur de reconstruction élevée sont considérés anomaux
- Hypothèse : les anomalies peuvent indiquer des numéros plus susceptibles d'apparaître

**Architecture** :
```
Encoder: Input → Dense(32) → Dense(16) → Latent(16)
Decoder: Latent(16) → Dense(16) → Dense(32) → Output
```

**Paramètres** :
- `encoding_dim` : Dimension de l'espace latent (défaut: 16)
- `hidden_layers` : Couches cachées (défaut: [32, 16])
- `epochs` : Époques d'entraînement (défaut: 50)
- `sequence_length` : Longueur des séquences (défaut: 20)

**Avantages** :
- Détecte des patterns subtils non visibles par les méthodes classiques
- L'espace latent peut révéler des structures cachées
- Applicable à la détection d'anomalies dans les tirages

**Dépendance** : `tensorflow` (optionnel)

---

### M18 - Graph Neural Network

**Type** : Deep Learning (Réseaux de graphes)

**Principe** :
- Modélise les numéros comme des nœuds dans un graphe
- Les arêtes représentent les co-occurrences entre numéros
- Utilise le message passing pour apprendre des embeddings de numéros

**Architecture** :
```
Nodes: Numéros de loterie (1 à N)
Edges: Co-occurrences pondérées
GNN: 2 couches de message passing + sortie sigmoïde
```

**Paramètres** :
- `embedding_dim` : Dimension des embeddings (défaut: 32)
- `hidden_dim` : Dimension cachée (défaut: 64)
- `n_layers` : Nombre de couches GNN (défaut: 2)
- `epochs` : Époques d'entraînement (défaut: 100)

**Avantages** :
- Capture les relations structurelles entre numéros
- Les embeddings peuvent être analysés (clustering, visualisation)
- Approche innovante pour l'analyse de loteries

**Dépendance** : `torch` (optionnel)

---

### M19 - Temporal Fusion

**Type** : Multi-échelle temporelle

**Principe** :
- Combine des informations de plusieurs échelles temporelles
- Court terme (10 tirages), moyen terme (30 tirages), long terme (100+ tirages)
- Utilise un mécanisme d'attention pour pondérer les échelles

**Échelles** :
1. **Court terme** : Capture les tendances très récentes
2. **Moyen terme** : Capture les patterns mensuels
3. **Long terme** : Baseline historique stable

**Paramètres** :
- `short_window` : Fenêtre court terme (défaut: 10)
- `medium_window` : Fenêtre moyen terme (défaut: 30)
- `long_window` : Fenêtre long terme (défaut: 100)
- `temperature` : Température softmax pour l'attention (défaut: 1.0)

**Avantages** :
- Adaptatif : donne plus de poids aux échelles performantes
- Pas de dépendances externes
- Interprétable : on peut voir les poids de chaque échelle

---

### M20 - Meta-Learner Adaptive

**Type** : Méta-apprentissage

**Principe** :
- Maintient un pool de modèles de base (M1-M19)
- Évalue la performance récente de chaque modèle
- Combine dynamiquement les prédictions avec des poids adaptatifs

**Fonctionnement** :
1. Évalue chaque modèle sur une fenêtre de validation
2. Calcule des poids via softmax des scores de performance
3. Combine les prédictions des top-N modèles
4. S'adapte au fil du temps

**Paramètres** :
- `validation_window` : Fenêtre de validation (défaut: 20)
- `n_top_models` : Nombre de modèles à combiner (défaut: 5)
- `temperature` : Température pour le softmax (défaut: 1.0)
- `decay_factor` : Facteur d'oubli (défaut: 0.95)

**Avantages** :
- Sélection automatique des meilleurs modèles
- Robuste : combine plusieurs approches
- Adaptatif : s'ajuste aux changements de régime
- Fournit un diagnostic sur la performance relative des modèles

---

## 📊 Métriques d'Évaluation

### Taux de Réussite (Hit Rate)

```
Hit Rate = Numéros corrects / Numéros à deviner
```

### Lift vs Random

```
Lift = Hit Rate du modèle / Hit Rate attendu (aléatoire)
```

Un lift > 1 indique que le modèle fait mieux que le hasard.

### Divisions de Prix

Le système calcule également les divisions de prix atteintes par chaque combinaison, basées sur les règles du jeu (nombre de numéros principaux + bonus corrects).

---

## 🔧 Utilisation dans le Backtest

1. **Sélectionner les modèles** à tester dans l'interface
2. **Configurer les paramètres** :
   - Nombre de tirages à tester
   - Nombre de combinaisons (pour ANTI/ANTI2)
   - Contraintes de diversité (pour ANTI2)
3. **Lancer le backtest**
4. **Analyser les résultats** :
   - Taux de réussite par modèle
   - Évolution temporelle
   - Divisions de prix atteintes

---

## 📝 Notes Importantes

1. **Aucun modèle ne peut prédire les tirages** - Les loteries sont des systèmes aléatoires
2. **Les performances passées ne garantissent pas les performances futures**
3. **Ces modèles sont à but éducatif et d'analyse statistique**
4. **Le lift > 1 peut être dû au hasard** sur un petit échantillon

---

*Document généré pour Lotto Analyzer v2.0*
