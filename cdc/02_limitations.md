# Lotto Analyzer — Limites, responsabilité et non-conclusions

## 1) Ce que l’outil peut faire
- Décrire des historiques (fréquences, distributions, métriques dérivées)
- Tester des hypothèses de hasard (uniformité/indépendance/stabilité) **avec rigueur**
- Détecter des anomalies **statistiques** et **de qualité de données**
- Comparer des modèles probabilistes **au baseline uniforme**

## 2) Ce que l’outil ne peut pas conclure (sans données externes)
- **Prouver une fraude** à partir des seuls tirages historiques
- **Prédire** des numéros gagnants de manière fiable sur une loterie saine
- Démontrer un “avantage” sans protocole d’évaluation robuste et significatif

## 3) Pourquoi “probabilités prochain tirage” ≠ “gagner”
Même si l’outil produit des probabilités estimées (M1/M2/…),
- une loterie saine doit rester **indistinguable** d’un baseline uniforme,
- tout gain apparent peut venir du hasard, d’un biais de fenêtre, ou du sur-ajustement.

## 4) Recommandations de communication (UI / rapports)
- Toujours afficher baseline + scores + incertitude
- Utiliser les termes :
  - “distribution estimée”, “diagnostic”, “comparaison”, “non concluant”
- Éviter :
  - “numéros à jouer”, “meilleures chances”, “garanti”, “vous devriez”

## 5) Jeu responsable
Si l’application est diffusée :
- afficher un encart jeu responsable
- éviter incitations (CTA, gamification)
- privilégier l’usage scientifique (thèse, audit data, pédagogie)

## 6) Conditions minimales pour interpréter un “signal”
- persistance temporelle (plusieurs fenêtres)
- cohérence sur plusieurs métriques
- correction multi-tests
- comparaison à Monte Carlo sous H0
- idéalement : information externe (changement machine/règles/source)
