"""
Phase 2, Step 3: Candidate term library.

Deliberately generic: a "term" is just a tuple of field names to be
multiplied together, e.g.:

    ()              -> constant 1
    ("u",)          -> u
    ("u_x",)        -> u_x
    ("u", "u")      -> u^2
    ("u", "u_x")    -> u*u_x

This means the SAME code works for u_t = D*u_xx (diffusion),
u_t = -c*u_x (advection), u_t = D*u_xx - k*u (diffusion+decay), or any
other combination of the available fields -- you only change the list
of term specs (in config.yaml), not this file. That's the
"IMPORTANT ARCHITECTURAL REQUIREMENT" from the Phase 2 spec: this file
has no idea what the "correct" equation is.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

TermSpec = Tuple[str, ...]  # e.g. () , ("u",), ("u", "u_x")


def term_label(spec: TermSpec) -> str:
    if len(spec) == 0:
        return "1"
    counts: Dict[str, int] = {}
    for name in spec:
        counts[name] = counts.get(name, 0) + 1
    parts = [name if cnt == 1 else f"{name}^{cnt}" for name, cnt in counts.items()]
    return "*".join(parts)


def build_library(
    fields: Dict[str, np.ndarray], term_specs: Sequence[TermSpec]
) -> Tuple[np.ndarray, List[str]]:
    """
    fields: dict mapping field name -> 1D array of length n_samples
            (e.g. {"u": ..., "u_x": ..., "u_xx": ..., "u_xxx": ...})
    term_specs: list of tuples of field names, see module docstring.

    Returns (Theta, names):
        Theta: array of shape (n_samples, n_terms)
        names: human-readable label for each column, e.g. ["1","u","u_xx",...]
    """
    n_samples = len(next(iter(fields.values())))
    columns = []
    names = []
    for spec in term_specs:
        spec = tuple(spec)
        if len(spec) == 0:
            col = np.ones(n_samples)
        else:
            col = np.ones(n_samples)
            for name in spec:
                if name not in fields:
                    raise KeyError(
                        f"Term spec {spec} references unknown field '{name}'. "
                        f"Available fields: {list(fields.keys())}"
                    )
                col = col * fields[name]
        columns.append(col)
        names.append(term_label(spec))
    Theta = np.column_stack(columns)
    return Theta, names


def parse_term_specs(raw_specs: Sequence[Sequence[str]]) -> List[TermSpec]:
    """Convert config.yaml lists (e.g. [[], ["u"], ["u","u_x"]]) into
    tuples usable by build_library."""
    return [tuple(spec) for spec in raw_specs]
