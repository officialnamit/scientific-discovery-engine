"""
Phase 7: generic observation description for ANY (x, t, U) dataset.

Generalizes data/problem_description.py (which is tied to the Phase 1 noisy
file). ONE fixed template is used for every dataset; the only things that vary
are measured numbers and qualitative words chosen by fixed thresholds on those
numbers. No mechanism name ever appears, so the text cannot steer retrieval
toward the answer by phrasing. Thresholds are stated here, fixed before any
benchmark result was seen.
"""

import numpy as np

AMP_CHANGE_TOL = 0.10     # |relative amplitude change| below this -> "roughly constant"
DRIFT_TOL = 0.05          # centroid moved less than 5% of domain length -> "did not move appreciably"
WIDTH_TOL = 0.10          # relative change in spatial spread


def _centroid(x, u):
    w = np.abs(u)
    return float(np.sum(x * w) / np.sum(w)) if w.sum() > 0 else float("nan")


def _spread(x, u):
    c = _centroid(x, u)
    w = np.abs(u)
    return float(np.sqrt(np.sum(w * (x - c) ** 2) / np.sum(w))) if w.sum() > 0 else float("nan")


def _n_peaks(u, rel=0.05):
    thr = rel * np.max(np.abs(u))
    return int(np.sum((u[1:-1] > u[:-2]) & (u[1:-1] > u[2:]) & (u[1:-1] > thr)))


def observation_stats(x, t, U):
    early, late = U[0], U[-1]
    a0, a1 = float(np.max(np.abs(early))), float(np.max(np.abs(late)))
    L = float(x[-1] - x[0])
    return {
        "amp_early": a0, "amp_late": a1, "amp_rel_change": (a1 - a0) / a0 if a0 else 0.0,
        "centroid_early": _centroid(x, early), "centroid_late": _centroid(x, late),
        "drift_frac": (_centroid(x, late) - _centroid(x, early)) / L,
        "spread_early": _spread(x, early), "spread_late": _spread(x, late),
        "peaks_early": _n_peaks(early), "peaks_late": _n_peaks(late),
        "edge_max": float(max(np.max(np.abs(U[:, 0])), np.max(np.abs(U[:, -1])))),
        "x_range": (float(x[0]), float(x[-1])), "t_range": (float(t[0]), float(t[-1])),
    }


def describe_observations(x, t, U) -> str:
    s = observation_stats(x, t, U)
    rc = s["amp_rel_change"]
    amp = ("decreased substantially" if rc < -AMP_CHANGE_TOL else
           "increased substantially" if rc > AMP_CHANGE_TOL else "remained roughly constant")
    df = s["drift_frac"]
    drift = ("did not move appreciably (little evidence of bulk translation)" if abs(df) < DRIFT_TOL else
             f"moved by {df * 100:+.0f}% of the domain length (the profile translates across the domain)")
    sp = (s["spread_late"] - s["spread_early"]) / s["spread_early"] if s["spread_early"] else 0.0
    spread = ("broadened (the profile spreads out)" if sp > WIDTH_TOL else
              "narrowed" if sp < -WIDTH_TOL else "stayed roughly the same width (shape preserved)")
    peaks = ("the number of local peaks fell from {a} to {b} (fine structure smooths out)"
             if s["peaks_late"] < s["peaks_early"] else
             "the number of local peaks stayed at {a}").format(a=s["peaks_early"], b=s["peaks_late"])
    return (
        f"We observe a scalar field u(x,t) on x in [{s['x_range'][0]:.1f},{s['x_range'][1]:.1f}] over "
        f"t in [{s['t_range'][0]:.1f},{s['t_range'][1]:.1f}].\n"
        f"- Peak amplitude went from {s['amp_early']:.3f} to {s['amp_late']:.3f}: it {amp}.\n"
        f"- The amplitude-weighted spatial centroid went from x~{s['centroid_early']:.2f} to "
        f"x~{s['centroid_late']:.2f}: it {drift}.\n"
        f"- The spatial spread {spread}.\n"
        f"- Over time {peaks}.\n"
        f"- At the domain edges the field stays near zero (max |u| there ~{s['edge_max']:.3f}).\n"
        f"Propose plausible governing mechanisms for this field's evolution."
    )
