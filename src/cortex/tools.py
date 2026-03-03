"""
Property name resolution utility used by extract_node.

The @tool functions (get_property_pnl, compare_properties, etc.) have been
replaced by the NL→SQL pipeline (plan_sql_node → validate_node → execute_node).
"""

import difflib

from cortex.config import PROPERTY_METADATA, PROPERTY_NAMES


def resolve_property(name: str) -> str:
    """
    Return the canonical property name closest to `name`, or raise ValueError.

    Matches against building names ('Building 120') AND street addresses
    ('120 Harbor Boulevard') from PROPERTY_METADATA.
    """
    normalised = name.strip().lower()

    candidates: dict[str, str] = {p.lower(): p for p in PROPERTY_NAMES}
    for canonical, meta in PROPERTY_METADATA.items():
        addr = meta["address"].lower()
        candidates[addr] = canonical
        for token in addr.split():
            if len(token) > 3 and token not in candidates:
                candidates[token] = canonical

    matches = difflib.get_close_matches(normalised, candidates.keys(), n=1, cutoff=0.4)
    if matches:
        return candidates[matches[0]]

    for key, canonical in candidates.items():
        if normalised in key or key in normalised:
            return canonical

    raise ValueError(
        f"Unknown property '{name}'. Available: {', '.join(PROPERTY_NAMES)}"
    )
