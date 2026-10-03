"""
Shared by experiments/06_hypothesis_generation.py (Phase 4) and
experiments/07_scientific_rag_kg.py (Phase 5). Factored out here
(rather than duplicated, or imported from a numerically-prefixed
experiment filename, which Python can't import as a module) since both
phases need the exact same evidence-based problem description.

Builds a scientific-problem description from ACTUAL computed
statistics of the noisy observations -- no equation, no D, no mention
of any mechanism name. This is also used verbatim as the RAG retrieval
query in Phase 5.
"""

import numpy as np

from pinn.data_utils import load_observational_data


def build_problem_description(out_dir: str) -> str:
    noisy = load_observational_data(out_dir, "noisy")
    x, t, u = noisy["x"], noisy["t"], noisy["u"]

    def amp_near(t_target, width=0.05):
        mask = np.abs(t - t_target) < width
        return float(np.max(np.abs(u[mask]))) if mask.any() else float("nan")

    amp_early = amp_near(t.min() + 0.05)
    amp_late = amp_near(t.max() - 0.05)

    def centroid_x_near(t_target, width=0.05):
        """Amplitude-weighted centroid of |u| in x -- more robust to sparse-sample
        noise than argmax, and more directly diagnostic of bulk translation
        (advection) vs. in-place decay (diffusion) than a single peak location,
        which can jump between comparable local humps."""
        mask = np.abs(t - t_target) < width
        if not mask.any():
            return float("nan")
        w = np.abs(u[mask])
        if w.sum() == 0:
            return float("nan")
        return float(np.sum(x[mask] * w) / np.sum(w))

    centroid_early = centroid_x_near(t.min() + 0.05)
    centroid_late = centroid_x_near(t.max() - 0.05)

    edge_mask = (x < 0.03) | (x > 0.97)
    edge_amplitude = float(np.max(np.abs(u[edge_mask]))) if edge_mask.any() else float("nan")

    return f"""We observe a scalar field u(x,t) on a 1D spatial domain x in [0,1] over a time \
window t in [0,1], measured via {len(u)} sparse, noisy point samples (x_i, t_i, u_i) -- no \
dense grid or simulation is provided to you.

Observed facts (computed directly from the data):
- Near t={t.min()+0.05:.2f}, the field's peak absolute amplitude is approximately {amp_early:.3f}.
- Near t={t.max()-0.05:.2f}, the field's peak absolute amplitude is approximately {amp_late:.3f} \
(i.e. the amplitude has decreased substantially over the observation window).
- Near the domain edges (x<0.03 or x>0.97), the field stays very close to zero at essentially \
all observed times (max |u| there is approximately {edge_amplitude:.4f}).
- The amplitude-weighted spatial centroid of |u| is approximately x~{centroid_early:.2f} near \
t={t.min()+0.05:.2f} and x~{centroid_late:.2f} near t={t.max()-0.05:.2f} (i.e. little evidence of \
bulk translation of the field across the domain -- compare this to the amplitude decay above).
- The spatial profile appears to smooth out over time: early-time samples show more \
fine-grained spatial structure (multiple local peaks) than late-time samples.

Propose plausible governing mechanisms for this field's evolution."""
