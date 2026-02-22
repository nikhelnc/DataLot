# Configuration des Lottos Australiens

Ce document décrit les configurations `rules_json` pour les différents types de lottos australiens supportés par l'application.

## Concepts clés

- **pick** : Nombre de numéros que le **joueur choisit** sur son ticket
- **drawn** : Nombre de numéros **tirés** lors du tirage officiel

Pour Oz Lotto par exemple :
- Le joueur choisit 7 numéros (pick=7)
- Au tirage, 10 numéros sont tirés : 7 gagnants (drawn=7) + 3 complémentaires (bonus.drawn=3)

## Structure des règles

```json
{
  "main": {
    "pick": <nombre de numéros choisis par le joueur>,
    "drawn": <nombre de numéros gagnants tirés>,
    "min": <numéro minimum>,
    "max": <numéro maximum>
  },
  "bonus": {
    "enabled": <true/false>,
    "pick": <nombre de bonus choisis par le joueur (0 si non applicable)>,
    "drawn": <nombre de bonus/complémentaires tirés>,
    "separate_pool": <true/false>,
    "min": <numéro minimum bonus (si separate_pool)>,
    "max": <numéro maximum bonus (si separate_pool)>
  },
  "calendar": {
    "expected_frequency": "weekly",
    "days": ["TUE", "WED", "SAT", ...]
  }
}
```

---

## 1. Oz Lotto

**Règles du jeu :**
- Le joueur choisit 7 numéros parmi 1 à 47
- Au tirage : 7 numéros gagnants + 3 complémentaires (du même pool)

**Configuration :**
```json
{
  "main": {
    "pick": 7,
    "drawn": 7,
    "min": 1,
    "max": 47
  },
  "bonus": {
    "enabled": true,
    "pick": 0,
    "drawn": 3,
    "separate_pool": false
  },
  "calendar": {
    "expected_frequency": "weekly",
    "days": ["TUE"]
  }
}
```

---

## 2. TattsLotto / Saturday Lotto

**Règles du jeu :**
- Le joueur choisit 6 numéros parmi 1 à 45
- Au tirage : 6 numéros gagnants + 2 complémentaires (du même pool)

**Configuration :**
```json
{
  "main": {
    "pick": 6,
    "drawn": 6,
    "min": 1,
    "max": 45
  },
  "bonus": {
    "enabled": true,
    "pick": 0,
    "drawn": 2,
    "separate_pool": false
  },
  "calendar": {
    "expected_frequency": "weekly",
    "days": ["SAT"]
  }
}
```

---

## 3. Powerball (Australie)

**Règles du jeu :**
- Le joueur choisit 7 numéros parmi 1 à 35 + 1 Powerball parmi 1 à 20
- Au tirage : 7 numéros gagnants + 1 Powerball (pools séparés)

**Configuration :**
```json
{
  "main": {
    "pick": 7,
    "drawn": 7,
    "min": 1,
    "max": 35
  },
  "bonus": {
    "enabled": true,
    "pick": 1,
    "drawn": 1,
    "separate_pool": true,
    "min": 1,
    "max": 20
  },
  "calendar": {
    "expected_frequency": "weekly",
    "days": ["THU"]
  }
}
```

---

## Différences clés

| Lotto | Joueur choisit | Tirage | Bonus tirés | Pool séparé |
|-------|----------------|--------|-------------|-------------|
| **Oz Lotto** | 7 sur 1-47 | 7 gagnants | 3 complémentaires | Non |
| **TattsLotto** | 6 sur 1-45 | 6 gagnants | 2 complémentaires | Non |
| **Powerball** | 7 + 1 PB | 7 + 1 PB | - | **Oui** |

## Format CSV pour l'import

Le format CSV doit correspondre au nombre de numéros **tirés** (drawn), pas au nombre choisi par le joueur.

**Oz Lotto (7 main + 3 bonus) :**
```
draw_number;draw_date;n1;n2;n3;n4;n5;n6;n7;bonus1;bonus2;bonus3
1234;2025-01-21;3;7;20;35;37;41;42;18;30;33
```

**Powerball (7 main + 1 bonus) :**
```
draw_number;draw_date;n1;n2;n3;n4;n5;n6;n7;bonus1
1234;2025-01-23;5;12;18;25;31;35;38;15
```

Note: `draw_number` est optionnel mais recommandé pour identifier les tirages.

## Impact sur les prédictions

### Pool commun (Oz Lotto, TattsLotto)
- Les modèles prédisent les numéros les plus probables du pool principal
- Les bonus sont les numéros suivants dans le classement de probabilité

### Pool séparé (Powerball)
- Les modèles prédisent séparément pour chaque pool
- Le Powerball est prédit indépendamment des numéros principaux
- Les statistiques de fréquence sont calculées séparément pour chaque pool

## Mise à jour des règles d'un jeu existant

Pour mettre à jour les règles d'un jeu existant via l'API :

```bash
curl -X PUT "http://localhost:8000/api/games/{game_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "rules_json": {
      "main": {"pick": 7, "min": 1, "max": 47},
      "bonus": {"enabled": true, "pick": 3, "separate_pool": false},
      "calendar": {"expected_frequency": "weekly", "days": ["TUE"]}
    }
  }'
```
