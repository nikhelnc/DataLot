# CAHIER DES CHARGES — Modèles de Prédiction Statistique

## Lotto Analyzer — Extension des modèles probabilistes

**Application Python pour l'analyse et la simulation de loteries**

> Version 2.0 — Février 2026
> Auteur : Nicolas — Calédonienne des Eaux / Doctorat
> *Confidentiel — Usage académique*

---

## Table des matières

1. [Introduction et contexte](#1-introduction-et-contexte)
2. [Interface commune des modèles (ModelBase)](#2-interface-commune-des-modèles-modelbase)
3. [Modèles existants (rappel synthétique)](#3-modèles-existants-rappel-synthétique)
4. [Nouveaux modèles à implémenter](#4-nouveaux-modèles-à-implémenter)
   - [M3 — Exponential Decay](#m3--exponential-decay)
   - [M4 — Hidden Markov Model (HMM)](#m4--hidden-markov-model-hmm)
   - [M7 — Entropy-Based Selection](#m7--entropy-based-selection)
   - [M8 — Changepoint Detection](#m8--changepoint-detection)
   - [M9 — Bayesian Network](#m9--bayesian-network)
   - [M12 — Mixture of Dirichlet](#m12--mixture-of-dirichlet)
   - [M13 — Spectral / Fourier Analysis](#m13--spectral--fourier-analysis)
   - [M14 — Copula Model](#m14--copula-model)
   - [M15 — Thompson Sampling](#m15--thompson-sampling)
5. [Tableau récapitulatif complet](#5-tableau-récapitulatif-complet)

---

## 1. Introduction et contexte

### 1.1 Objectif du document

Ce cahier des charges décrit l'ensemble des modèles statistiques à implémenter dans l'application Lotto Analyzer v2.0. Il couvre les modèles existants (M0 à M10, ANTI, ANTI2) ainsi que les nouveaux modèles proposés (M3, M4, M7, M8, M9, M15, M12, M13, M14). Chaque modèle est spécifié avec ses fondements théoriques, ses paramètres, ses formules, et ses contraintes d'implémentation en Python.

L'objectif final est double : évaluer rigoureusement ces modèles sur des données historiques de loteries australiennes (Powerball, TattsLotto, Oz Lotto) via un protocole de backtest walk-forward, et produire des distributions de probabilité estimées pour les prochains tirages.

### 1.2 Loteries cibles

| Loterie | Grille principale | Bonus | Complémentaires |
|---------|------------------|-------|-----------------|
| **Powerball (AU)** | 7 parmi 1-35 | 1 parmi 1-20 | — |
| **TattsLotto** | 6 parmi 1-45 | — | 2 (7ème + 8ème) |
| **Oz Lotto** | 7 parmi 1-47 | — | 3 (8/9/10ème) |

### 1.3 Architecture applicative

L'application est développée en Python et s'articule autour des modules suivants :

- **core/models/** : un fichier Python par modèle, respectant l'interface commune ModelBase.
- **core/backtest.py** : moteur de backtest walk-forward, appelle chaque modèle de manière standardisée.
- **core/metrics.py** : calcul de Brier score, ECE, Lift, hit rate, divisions de prix.
- **core/montecarlo.py** : simulations Monte Carlo sous H0 pour référencement.
- **api/** : endpoints pour l'interface web (FastAPI ou Flask).
- **data/** : historiques CSV des tirages, metadata des règles par loterie.

---

## 2. Interface commune des modèles (ModelBase)

Tous les modèles doivent implémenter l'interface suivante pour être compatibles avec le moteur de backtest et l'interface utilisateur :

```python
class ModelBase(ABC):
    model_id: str          # ex: 'M3'
    model_name: str         # ex: 'Exponential Decay'
    model_type: str         # ex: 'Pondération temporelle'

    @abstractmethod
    def fit(self, draws: List[Draw], game_rules: GameRules) -> None:
        """Entraîne le modèle sur l'historique."""

    @abstractmethod
    def predict_proba(self) -> Dict[str, np.ndarray]:
        """Retourne les probabilités par numéro.
        Returns: {'main': array[N], 'bonus': array[B]}"""

    @abstractmethod
    def generate_combinations(self, n: int = 1) -> List[Combination]:
        """Génère n combinaisons pondérées."""

    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres pour reproductibilité."""
```

Chaque modèle produit obligatoirement un vecteur de probabilités normalisé (somme = 1) pour les numéros principaux et, le cas échéant, pour les numéros bonus.

### 2.1 Structure de données Draw

```python
@dataclass
class Draw:
    date: datetime
    main_numbers: List[int]   # numéros principaux triés
    bonus_numbers: List[int]   # bonus / complémentaires
    draw_number: int            # identifiant séquentiel
```

### 2.2 Structure GameRules

```python
@dataclass
class GameRules:
    name: str                   # 'powerball_au', 'tattslotto', 'ozlotto'
    main_range: Tuple[int, int]  # (1, 35) pour Powerball
    main_pick: int               # 7 pour Powerball
    bonus_range: Tuple[int, int]  # (1, 20) pour Powerball
    bonus_pick: int               # 1 pour Powerball
    supplementary_count: int      # 0, 2 ou 3
```

---

## 3. Modèles existants (rappel synthétique)

Les modèles suivants sont déjà implémentés et servent de référence. Ils sont décrits ici de manière synthétique pour complétude.

| ID | Nom | Type | Principe |
|----|-----|------|----------|
| **M0** | Baseline | Aléatoire | Sélection uniforme. Référence obligatoire pour tout benchmark. |
| **M1** | Dirichlet | Bayésien | Prior Dirichlet + fréquences observées. Lissage naturel. |
| **M2** | Windowed | Fenêtre glissante | Pondération récente avec shrinkage vers la moyenne globale. |
| **M5** | Co-occurrence | Paires | Détection de paires sur-représentées (delta obs-attendu). |
| **M6** | Gaps & Streaks | Écarts | Numéros « en retard » vs écart moyen historique. |
| **M10** | Ensemble | Méta-modèle | Stacking de M0 + M1 + M2 avec poids optimisés. |
| **ANTI** | Anti-Consensus | Contrarian | Numéros non prédits par les autres modèles. |
| **ANTI2** | Anti-Consensus v2 | Contrarian+ | ANTI avec contrainte de diversité inter-combinaisons. |

---

## 4. Nouveaux modèles à implémenter

Cette section constitue le cœur du cahier des charges. Chaque modèle est spécifié avec son identifiant, son fondement théorique, ses paramètres configurables, ses formules, et ses recommandations d'implémentation.

---

### M3 — Exponential Decay

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M3 |
| **Nom complet** | Exponential Decay |
| **Type** | Pondération temporelle exponentielle |
| **Librairies Python** | numpy |
| **Complexité** | O(N·K) où N=tirages, K=numéros |

#### Principe

Contrairement à M2 qui utilise une fenêtre fixe (les N derniers tirages comptent également, les précédents sont ignorés), M3 attribue un poids à chaque tirage passé selon une décroissance exponentielle. Cela élimine l'effet de bord brutal de la fenêtre et crée une transition douce entre tirages récents et anciens.

Le paramètre lambda contrôle la vitesse de décroissance : un lambda élevé donne plus de poids aux tirages très récents, un lambda faible se rapproche du modèle uniforme.

Ce modèle est particulièrement pertinent pour détecter des dérives lentes dans les fréquences, comme celles qui pourraient résulter d'un changement d'équipement progressif.

#### Paramètres

- **lambda_decay** (float, défaut: 0.02) : Taux de décroissance exponentielle. Optimisable par validation croisée walk-forward.
- **min_weight** (float, défaut: 1e-6) : Seuil minimum de poids. Les tirages en dessous sont ignorés pour performance.

#### Formule / Pseudo-code

```
w(t) = exp(-λ * (T - t))
P(i) = Σ_t [ w(t) * 𝟙{i ∈ tirage_t} ] / Σ_t w(t)
Normalisation : P = P / sum(P)
```

#### Avantages

- Élimine l'artefact de bord de la fenêtre fixe (M2).
- Paramètre unique et interprétable.
- Transition douce entre régime « mémoire courte » et « mémoire longue ».
- Compatible nativement avec le walk-forward (pas de fenêtre à définir).

#### Notes d'implémentation

Implémenter la recherche du lambda optimal par grid search ou Brent optimization sur le Brier score du backtest. Pour éviter les problèmes numériques, travailler en log-espace pour les poids.

---

### M4 — Hidden Markov Model (HMM)

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M4 |
| **Nom complet** | Hidden Markov Model |
| **Type** | Modèle à états cachés |
| **Librairies Python** | hmmlearn, numpy, scikit-learn |
| **Complexité** | O(N·S²) par itération EM, S=états |

#### Principe

M4 modélise l'hypothèse que les tirages peuvent provenir de plusieurs « régimes » latents, chacun avec sa propre distribution de probabilité sur les numéros. Les transitions entre régimes suivent une chaîne de Markov.

Cas d'usage : si une loterie change de machine de tirage, de procédure, ou de conditions environnementales, le HMM peut capturer ces transitions à travers ses états cachés.

Pour chaque état caché, le modèle apprend une distribution multinomiale sur les numéros. La prédiction est la distribution émise par l'état le plus probable au temps t+1.

Le nombre d'états est un hyperparamètre critique : trop d'états entraînent du surapprentissage, trop peu ne capturent pas les régimes.

#### Paramètres

- **n_states** (int, défaut: 3) : Nombre d'états cachés. Tester 2 à 5, sélectionner par BIC.
- **n_iter** (int, défaut: 100) : Itérations maximum pour l'algorithme Baum-Welch (EM).
- **tol** (float, défaut: 1e-4) : Critère de convergence de la log-vraisemblance.
- **random_state** (int) : Seed pour reproductibilité de l'initialisation.

#### Formule / Pseudo-code

```
# Matrice de transition : A[i,j] = P(state_t+1=j | state_t=i)
# Émission : B[s] = Multinomial(K numéros) pour chaque état s
# Entraînement : Baum-Welch (EM)
# Prédiction : α_T · A -> distribution sur états t+1
#              P(num) = Σ_s P(state=s|t+1) * B[s](num)
```

#### Avantages

- Capture des changements de régime non détectables par les modèles stationnaires.
- Cadre probabiliste rigoureux avec vraisemblance.
- Sélection de modèle par BIC/AIC.
- Forte valeur académique pour la thèse (test de l'hypothèse de régimes).

#### Notes d'implémentation

Utiliser hmmlearn (GaussianHMM ou MultinomialHMM). Si MultinomialHMM n'est pas disponible, encoder les tirages comme vecteurs binaires et utiliser GaussianHMM ou implémenter un HMM custom.

Attention : la convergence de Baum-Welch est sensible à l'initialisation. Lancer 10 inits aléatoires et garder celle avec la meilleure log-vraisemblance.

Pour le walk-forward, réentraîner le HMM à chaque pas (coûteux) ou mettre à jour de manière incrémentale le forward pass.

---

### M7 — Entropy-Based Selection

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M7 |
| **Nom complet** | Entropy-Based Selection |
| **Type** | Sélection par entropie locale |
| **Librairies Python** | numpy, scipy |
| **Complexité** | O(N·K·W) |

#### Principe

M7 calcule l'entropie de Shannon sur des fenêtres glissantes pour chaque numéro individuellement. L'intuition est que si un numéro a un comportement localement plus « régulier » (entropie basse), il pourrait indiquer une anomalie exploitable.

Pour chaque numéro i, on construit la série binaire x_i(t) = 1 si le numéro i a été tiré au tirage t, 0 sinon. Sur une fenêtre glissante, on calcule la fréquence p et l'entropie H = -p·log(p) - (1-p)·log(1-p).

Les numéros dont l'entropie locale est significativement plus basse que l'entropie théorique sous H0 sont favorisés.

#### Paramètres

- **window_size** (int, défaut: 30) : Taille de la fenêtre pour le calcul d'entropie locale.
- **alpha_threshold** (float, défaut: 0.05) : Seuil de significativité pour l'écart à H0.
- **selection_mode** (str, défaut: 'low_entropy') : Stratégie : favoriser les numéros à faible entropie ('low_entropy') ou forte entropie ('high_entropy').

#### Formule / Pseudo-code

```
x_i(t) = 1{numéro i tiré au tirage t}
p_i = mean(x_i[t-W:t])
H_i = -p_i * log2(p_i) - (1-p_i) * log2(1-p_i)
H0_theorique = -p0 * log2(p0) - (1-p0) * log2(1-p0)
   où p0 = K/N (ex: 7/35 pour Powerball)
score_i = H0_theorique - H_i  (positif = plus régulier qu'attendu)
P(i) ∝ softmax(score_i / temperature)
```

#### Avantages

- Détecte les numéros dont le comportement dévie de l'aléatoire pur.
- Interprétable : l'entropie est une mesure d'information standard.
- Complémentaire aux modèles fréquentistes (M1, M2) car capture la régularité, pas la fréquence.
- Bon signal d'alerte pour la thèse (détection d'anomalies).

#### Notes d'implémentation

Implémenter via scipy.stats.entropy ou calcul direct numpy. Comparer H_i à la distribution de H sous Monte Carlo (H0) pour obtenir une p-value par numéro. Convertir les scores en probabilités via softmax avec un paramètre de température.

---

### M8 — Changepoint Detection

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M8 |
| **Nom complet** | Changepoint Detection |
| **Type** | Détection de ruptures (CUSUM/PELT) |
| **Librairies Python** | ruptures, numpy |
| **Complexité** | O(N·K) pour PELT (quasi-linéaire) |

#### Principe

M8 identifie les points de rupture structurelle dans les séries temporelles de fréquence de chaque numéro. Après détection d'une rupture, le modèle recalcule les probabilités uniquement sur le segment post-rupture.

L'hypothèse sous-jacente est que certains changements (machine, procédure, opérateur) peuvent modifier la distribution des tirages. Si un point de rupture est détecté, les données antérieures ne sont plus pertinentes pour la prédiction.

L'algorithme PELT (Pruned Exact Linear Time) est utilisé pour sa performance sur de longues séries. En complément, un CUSUM global peut servir de détection en ligne.

Ce modèle a une forte valeur pour la thèse car il teste directement l'hypothèse de stationnarité.

#### Paramètres

- **model_type** (str, défaut: 'l2') : Modèle de coût pour PELT : 'l1', 'l2', 'rbf', 'normal'.
- **min_segment_size** (int, défaut: 20) : Taille minimale d'un segment entre deux ruptures.
- **penalty** (str/float, défaut: 'bic') : Pénalité pour PELT : 'bic', 'aic', ou valeur numérique.
- **cusum_threshold** (float, défaut: 4.0) : Seuil pour la détection CUSUM en ligne (en écarts-types).

#### Formule / Pseudo-code

```
# Détection offline (PELT) :
breakpoints = ruptures.Pelt(model=model_type, min_size=min_seg)
                      .fit(freq_series).predict(pen=penalty)

# Segment courant : données depuis le dernier breakpoint
current_segment = draws[last_breakpoint:]
P(i) = freq_i(current_segment) / sum(freqs)

# Détection online (CUSUM) :
S(t) = max(0, S(t-1) + (x(t) - mu0) - k)
Alarme si S(t) > h
```

#### Avantages

- Test direct de l'hypothèse de stationnarité.
- Élimine les données obsolètes après un changement de régime.
- Algorithme PELT en temps quasi-linéaire.
- Applicable à chaque numéro indépendamment ou globalement.
- Valeur académique très élevée pour la thèse.

#### Notes d'implémentation

Utiliser la librairie ruptures pour PELT. Implémenter CUSUM manuellement (quelques lignes). Si aucune rupture détectée, fallback sur M1 (Dirichlet global). Stocker les breakpoints détectés dans les métadonnées du modèle pour audit. Corréler les ruptures détectées avec des informations externes (changements de règles, machines) pour validation.

---

### M9 — Bayesian Network

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M9 |
| **Nom complet** | Bayesian Network |
| **Type** | Réseau bayésien de dépendances |
| **Librairies Python** | pgmpy, numpy, networkx |
| **Complexité** | O(K²·N) pour structure learning |

#### Principe

M9 modélise les dépendances conditionnelles entre numéros via un graphe acyclique dirigé (DAG). Contrairement à M5 qui analyse les co-occurrences par paires de manière « plate », un réseau bayésien capture les dépendances conditionnelles (ex: le numéro A est corrélé à B seulement quand C est présent).

La structure du réseau est apprise à partir des données via l'algorithme PC (tests d'indépendance conditionnelle) ou un score BIC/BDeu. L'inférence produit des probabilités mises à jour étant donné l'évidence partielle.

Pour la prédiction du prochain tirage, le modèle utilise la propagation de croyances (belief propagation) pour estimer la distribution marginale de chaque numéro.

#### Paramètres

- **structure_algo** (str, défaut: 'hc') : Algorithme d'apprentissage de structure : 'hc' (Hill Climbing), 'pc', 'mmhc'.
- **scoring** (str, défaut: 'bic') : Score pour la recherche de structure : 'bic', 'bdeu', 'k2'.
- **max_parents** (int, défaut: 3) : Nombre maximum de parents par nœud (régularisation).
- **significance_level** (float, défaut: 0.05) : Seuil pour les tests d'indépendance (algo PC).

#### Formule / Pseudo-code

```
# Apprentissage de structure :
dag = HillClimbSearch(data).estimate(
        scoring_method=BicScore(data), max_parents=3)

# Estimation des paramètres :
model = BayesianNetwork(dag.edges())
model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')

# Inférence :
inference = VariableElimination(model)
P(i) = inference.query(['num_i'], evidence={})
```

#### Avantages

- Capture des dépendances conditionnelles impossibles à voir avec de simples co-occurrences.
- Structure apprise interprétable visuellement (graphe).
- Cadre bayésien avec gestion naturelle de l'incertitude.
- L'absence de liens dans le DAG confirme l'indépendance (support à H0).

#### Notes d'implémentation

Utiliser pgmpy pour la construction et l'inférence. Encoder les tirages comme une matrice binaire (K colonnes, une par numéro). Le DAG appris doit être stocké et visualisé (networkx/graphviz) pour l'analyse. Attention : avec K=35-47 numéros, l'espace de recherche est large. Limiter max_parents et utiliser des heuristiques.

---

### M12 — Mixture of Dirichlet

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M12 |
| **Nom complet** | Mixture of Dirichlet |
| **Type** | Mélange de distributions Dirichlet |
| **Librairies Python** | numpy, scipy |
| **Complexité** | O(N·K·C) par itération EM, C=composantes |

#### Principe

Extension de M1 : au lieu d'un seul prior Dirichlet représentant un régime unique, M12 utilise un mélange de K distributions Dirichlet. Chaque composante du mélange représente un « mode » potentiel de la loterie.

L'estimation se fait par l'algorithme EM : E-step assigne chaque tirage à une composante (soft assignment), M-step met à jour les paramètres de chaque Dirichlet.

La prédiction est la mixture pondérée des K composantes. Le nombre de composantes K est sélectionné par BIC ou WAIC.

Ce modèle est plus expressif que M1 et complémentaire à M4 (HMM) : M4 capture les transitions temporelles entre régimes, M12 capture l'hétérogénéité non temporelle.

#### Paramètres

- **n_components** (int, défaut: 2) : Nombre de composantes du mélange. Tester 2 à 5, sélectionner par BIC.
- **alpha_prior** (float, défaut: 1.0) : Prior de concentration pour chaque composante.
- **n_iter** (int, défaut: 100) : Itérations EM maximum.
- **tol** (float, défaut: 1e-4) : Convergence EM.

#### Formule / Pseudo-code

```
# Mélange :
P(x) = Σ_k π_k * Dirichlet(x | α_k)

# E-step : responsabilité de chaque composante
r_ik = π_k * Dir(x_i | α_k) / Σ_j π_j * Dir(x_i | α_j)

# M-step : mise à jour des paramètres
π_k = Σ_i r_ik / N
α_k = MLE ou fixed-point iterations

# Prédiction :
P(num) = Σ_k π_k * E[Dir_k](num)
```

#### Avantages

- Plus expressif qu'un seul Dirichlet (M1) car capture l'hétérogénéité.
- Sélection de modèle rigoureuse par BIC.
- Complémentaire à M4 (HMM) sans l'aspect temporel.
- Si K=1 optimal, confirme que M1 suffit (support à H0).

#### Notes d'implémentation

Implémenter avec scipy.special (digamma, gammaln) pour les calculs Dirichlet. L'estimation MLE des paramètres Dirichlet nécessite des fixed-point iterations (méthode de Minka). Initialiser les composantes par K-means sur les vecteurs de fréquences. Attention : la log-vraisemblance Dirichlet peut être instable si α → 0.

---

### M13 — Spectral / Fourier Analysis

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M13 |
| **Nom complet** | Spectral / Fourier Analysis |
| **Type** | Analyse spectrale des périodicités |
| **Librairies Python** | numpy, scipy.signal, scipy.fft |
| **Complexité** | O(N·K·log(N)) |

#### Principe

M13 applique la transformée de Fourier rapide (FFT) sur la série binaire de présence/absence de chaque numéro pour détecter des périodicités éventuelles.

Si des cycles existent (liés au calendrier des tirages, à la rotation de machines, ou à d'autres facteurs systématiques), le spectre de puissance les révélera comme des pics significatifs.

La prédiction extrapole les composantes fréquentielles significatives au temps t+1 par synthèse harmonique.

Même si aucune périodicité n'est trouvée (ce qui est attendu sous H0), le test spectral a une valeur diagnostique importante pour la thèse.

#### Paramètres

- **min_frequency** (float, défaut: 0.01) : Fréquence minimale à analyser (inverse de la période maximale).
- **significance_threshold** (float, défaut: 0.01) : Seuil de significativité pour la détection de pics spectraux.
- **n_harmonics** (int, défaut: 3) : Nombre de composantes fréquentielles retenues pour la prédiction.
- **detrend** (bool, défaut: True) : Retirer la tendance linéaire avant la FFT.

#### Formule / Pseudo-code

```
# Série binaire :
x_i(t) = 1{numéro i tiré au tirage t}

# FFT :
X_i(f) = FFT(x_i(t))
PSD_i(f) = |X_i(f)|² / N

# Détection de pics (Fisher's g-test) :
g = max(PSD) / sum(PSD)
p_value = P(g > g_obs | H0)

# Prédiction par synthèse harmonique :
x_pred_i(t+1) = Σ_h A_h * cos(2π*f_h*(t+1) + φ_h)
P(i) ∝ softmax(x_pred_i(t+1))
```

#### Avantages

- Détection de périodicités impossibles à voir autrement.
- Fondement mathématique solide (théorie spectrale).
- Le Fisher's g-test fournit une p-value exacte pour chaque fréquence.
- Valeur diagnostique forte même si aucun signal n'est trouvé (confirme H0).

#### Notes d'implémentation

Utiliser numpy.fft.rfft pour la FFT et scipy.signal pour le PSD. Appliquer le g-test de Fisher pour la significativité des pics (implem custom, ~20 lignes). Si aucun pic significatif, retourner la distribution uniforme (fallback M0). Stocker le spectre complet pour visualisation dans les rapports.

---

### M14 — Copula Model

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M14 |
| **Nom complet** | Copula Model |
| **Type** | Modélisation de dépendances par copules |
| **Librairies Python** | copulas ou pyvinecopulib, numpy, scipy |
| **Complexité** | O(N·K²) ajustement + O(S·K) simulation |

#### Principe

M14 modélise les dépendances entre numéros en séparant les distributions marginales (fréquence individuelle de chaque numéro) de la structure de dépendance (copule). Cela permet une modélisation plus flexible que les corrélations linéaires utilisées par M5.

Le modèle estime d'abord les marginales empiriques de chaque numéro, les transforme en distributions uniformes [0,1] via la CDF empirique, puis ajuste une copule (gaussienne ou vine) sur les données transformées.

Pour la prédiction, on échantillonne depuis la copule ajustée et on retransforme vers l'espace original pour obtenir des probabilités jointes.

Particulièrement utile si les dépendances entre numéros sont non linéaires ou asymétriques (dépendance dans les queues).

#### Paramètres

- **copula_type** (str, défaut: 'gaussian') : Type de copule : 'gaussian', 'student_t', 'vine', 'clayton', 'gumbel'.
- **n_simulations** (int, défaut: 10000) : Nombre de simulations pour estimer les probabilités jointes.
- **marginal_method** (str, défaut: 'empirical') : Méthode d'estimation des marginales : 'empirical', 'beta', 'kde'.
- **selection_criterion** (str, défaut: 'aic') : Critère de sélection de la copule : 'aic', 'bic'.

#### Formule / Pseudo-code

```
# 1. Marginales empiriques :
u_i = F_hat_i(x_i) = rang(x_i) / (N+1)

# 2. Ajustement de la copule :
C(u_1, ..., u_K ; θ) = copule ajustée sur [u_1...u_K]

# 3. Simulation :
[v_1...v_K] ~ C(θ)  (n_simulations fois)
x_sim_i = F_hat_i^{-1}(v_i)

# 4. Probabilités :
P(i) = mean(x_sim_i > seuil) ∀ simulations
```

#### Avantages

- Capture des dépendances non linéaires et asymétriques.
- Séparation propre entre structure marginale et structure de dépendance.
- Plusieurs types de copules testables, sélection par AIC/BIC.
- Si la copule gaussienne est optimale avec corrélation nulle, cela confirme l'indépendance (H0).

#### Notes d'implémentation

Utiliser pyvinecopulib ou copulas (pip) pour l'ajustement. Pour les loteries avec K=35-47 numéros, une copule gaussienne complète est impraticable (matrice K×K). Utiliser une copule vine (par paires) ou réduire la dimensionnalité par PCA. Alternative : travailler sur des groupes de numéros (bas/haut, pair/impair) plutôt que sur chaque numéro individuellement. La dimension élevée est le défi principal de ce modèle.

---

### M15 — Thompson Sampling

| Champ | Valeur |
|-------|--------|
| **Identifiant** | M15 |
| **Nom complet** | Thompson Sampling |
| **Type** | Bandit multi-bras bayésien |
| **Librairies Python** | numpy |
| **Complexité** | O(N·K) pour fit, O(K) pour predict |

#### Principe

M15 traite chaque numéro comme un « bras » d'un problème de bandit multi-bras. Pour chaque numéro, on maintient une distribution Beta(α, β) mise à jour à chaque tirage : α augmente quand le numéro est tiré, β augmente quand il ne l'est pas.

La sélection se fait par échantillonnage : on tire un θ_i de chaque Beta(α_i, β_i) et on sélectionne les K numéros avec les plus grands θ.

L'avantage de Thompson Sampling est son équilibre naturel exploration/exploitation : un numéro peu observé a une distribution large (forte incertitude), donc il peut être sélectionné par surprise. Cela donne de la diversité aux combinaisons générées.

Pour la production de probabilités (predict_proba), on utilise la moyenne de la distribution Beta : E[θ_i] = α_i / (α_i + β_i).

#### Paramètres

- **prior_alpha** (float, défaut: 1.0) : Prior α initial (1.0 = prior uniforme / non informatif).
- **prior_beta** (float, défaut: 1.0) : Prior β initial.
- **decay_factor** (float, défaut: 1.0) : Facteur de décroissance appliqué à α et β à chaque pas (1.0 = pas de décroissance, 0.99 = oubli progressif).
- **n_samples** (int, défaut: 1000) : Nombre d'échantillonnages pour estimer les probabilités moyennes.

#### Formule / Pseudo-code

```
# Initialisation :
α_i = prior_alpha, β_i = prior_beta  ∀i

# Mise à jour après chaque tirage :
α_i *= decay_factor ; β_i *= decay_factor
α_i += 1 si i tiré, sinon β_i += 1

# Génération de combinaisons :
θ_i ~ Beta(α_i, β_i)  ∀i
Combos = top-K(θ)

# Probabilités :
P(i) = E[θ_i] = α_i / (α_i + β_i), normalisé
```

#### Avantages

- Équilibre exploration/exploitation naturel et théoriquement fondé.
- Chaque appel à generate_combinations produit une combinaison différente (stochastique).
- Le decay_factor permet d'oublier progressivement l'historique ancien.
- Cadre décisionnel bien étudié avec regret bounds théoriques.
- Implémentation très simple et rapide.

#### Notes d'implémentation

Implémentation directe avec numpy (np.random.beta). Le decay_factor est critique : sans decay, les distributions deviennent très concentrées après de nombreux tirages et l'exploration disparaît. Pour le backtest, stocker la série complète des (α_i, β_i) pour analyse.

---

## 5. Tableau récapitulatif complet

Ce tableau synthétise tous les modèles (existants + nouveaux) avec leur positionnement.

| ID | Nom | Approche | Hypothèse testée | Librairie | Statut |
|----|-----|----------|-------------------|-----------|--------|
| **M0** | Baseline | Uniforme | Référence (H0) | numpy | ✅ Existant |
| **M1** | Dirichlet | Bayésien | Biais de fréquence | numpy | ✅ Existant |
| **M2** | Windowed | Fenêtre fixe | Tendance récente | numpy | ✅ Existant |
| **M3** | Exp. Decay | Pondération exp. | Dérive lente | numpy | 🔶 À impl. |
| **M4** | HMM | États cachés | Régimes multiples | hmmlearn | 🔶 À impl. |
| **M5** | Co-occurrence | Paires | Corrélations | numpy | ✅ Existant |
| **M6** | Gaps & Streaks | Écarts | Retard / rattrapage | numpy | ✅ Existant |
| **M7** | Entropy | Entropie locale | Prévisibilité locale | scipy | 🔶 À impl. |
| **M8** | Changepoint | PELT/CUSUM | Stationnarité | ruptures | 🔶 À impl. |
| **M9** | Bayes Net | DAG + inférence | Dépendances cond. | pgmpy | 🔶 À impl. |
| **M10** | Ensemble | Stacking | Consensus | numpy | ✅ Existant |
| **M15** | Thompson | Bandit | Explore/exploit | numpy | 🔶 À impl. |
| **M12** | Mix Dirichlet | Mélange EM | Hétérogénéité | scipy | 🔶 À impl. |
| **M13** | Spectral | FFT | Périodicités | numpy/scipy | 🔶 À impl. |
| **M14** | Copula | Copules | Dép. non linéaires | copulas | 🔶 À impl. |
| **ANTI** | Anti-Consensus | Contrarian | Sous-estimation | numpy | ✅ Existant |
| **ANTI2** | Anti-Cons. v2 | Contrarian+ | Diversité | numpy | ✅ Existant |

---

*— Fin du cahier des charges —*
