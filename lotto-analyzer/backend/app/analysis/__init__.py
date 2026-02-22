from typing import Dict, Tuple


def get_main_rules(rules: Dict) -> Tuple[int, int, int]:
    """
    Extract main number rules from either old or new structure.
    
    Returns:
        Tuple of (n_min, n_max, n_count)
    """
    main_rules = rules.get("main", rules.get("numbers", {}))
    n_min = main_rules.get("min", 1)
    n_max = main_rules.get("max", 49)
    # For count, use 'pick' (new) or 'count' (old)
    n_count = main_rules.get("pick", main_rules.get("count", 6))
    return n_min, n_max, n_count


def get_bonus_rules(rules: Dict) -> Tuple[bool, int, int, int]:
    """
    Extract bonus number rules from either old or new structure.
    
    Returns:
        Tuple of (enabled, n_min, n_max, n_count)
    """
    bonus_rules = rules.get("bonus", {})
    enabled = bonus_rules.get("enabled", False)
    if not enabled:
        return False, 1, 1, 0
    
    # For bonus pool, check if separate_pool
    main_rules = rules.get("main", rules.get("numbers", {}))
    if bonus_rules.get("separate_pool", False):
        n_min = bonus_rules.get("min", 1)
        n_max = bonus_rules.get("max", 20)
    else:
        n_min = main_rules.get("min", 1)
        n_max = main_rules.get("max", 49)
    
    # For count, use 'pick' (new) or 'count' (old)
    n_count = bonus_rules.get("pick", bonus_rules.get("count", 1))
    return enabled, n_min, n_max, n_count