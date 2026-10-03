"""
Phase 3: generic training loop.

This is the ONE trainer used for every experiment in this phase --
forward diffusion, inverse diffusion, H1 diffusion validation, H2
advection validation. What changes between runs is only:
  - which residual_fn is passed in (pinn/residuals.py),
  - whether the physics parameters are trainable (forward vs inverse),
  - which data/collocation/BC/IC points are passed in (e.g. for the
    OOD experiment, training data is restricted to t <= t_cutoff).

Nothing about this file is diffusion-specific.
"""

from typing import Callable, Dict, List, Optional

import torch

from pinn.derivatives import compute_derivatives
from pinn.losses import bc_loss, data_loss, ic_loss, physics_loss, total_loss
from pinn.residuals import ResidualFn, make_params


class PINNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        residual_fn: ResidualFn,
        params: Dict[str, torch.Tensor],
        weights: Dict[str, float],
    ):
        self.model = model
        self.residual_fn = residual_fn
        self.params = params
        self.weights = weights

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = list(self.model.parameters())
        for p in self.params.values():
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                params.append(p)
        return params

    def compute_losses(self, data, collocation, bc, ic):
        derivs_f = compute_derivatives(self.model, collocation["x"], collocation["t"])
        residual = self.residual_fn(derivs_f, self.params)

        components = {
            "data": data_loss(self.model, data["x"], data["t"], data["u"]),
            "physics": physics_loss(residual),
            "bc": bc_loss(self.model, bc["x"], bc["t"], bc["u"]),
            "ic": ic_loss(self.model, ic["x"], ic["t"], ic["u"]),
        }
        total = total_loss(components, self.weights)
        return total, components, residual

    def train(
        self,
        data, collocation, bc, ic,
        adam_epochs: int = 3000,
        adam_lr: float = 1e-3,
        lbfgs_epochs: int = 300,
        log_every: int = 500,
        resample_collocation_fn: Optional[Callable[[], Dict]] = None,
        resample_every: int = 200,
    ) -> List[Dict]:
        history = []
        optimizer = torch.optim.Adam(self.trainable_parameters(), lr=adam_lr)

        for epoch in range(adam_epochs):
            if resample_collocation_fn is not None and epoch % resample_every == 0:
                collocation = resample_collocation_fn()

            optimizer.zero_grad()
            total, components, _ = self.compute_losses(data, collocation, bc, ic)
            total.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == adam_epochs - 1:
                history.append(self._record(epoch, total, components))

        if lbfgs_epochs > 0:
            optimizer2 = torch.optim.LBFGS(
                self.trainable_parameters(), max_iter=lbfgs_epochs, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer2.zero_grad()
                total, _, _ = self.compute_losses(data, collocation, bc, ic)
                total.backward()
                return total

            optimizer2.step(closure)
            total, components, _ = self.compute_losses(data, collocation, bc, ic)
            history.append(self._record("lbfgs_final", total, components))

        return history

    def _record(self, epoch, total, components) -> Dict:
        rec = {"epoch": epoch, "total": float(total.item())}
        rec.update({k: float(v.item()) for k, v in components.items()})
        rec.update({k: float(v.item()) for k, v in self.params.items()})
        return rec


def build_and_train(
    hypothesis_cfg: Dict,
    param_init: Dict[str, float],
    trainable: bool,
    architecture_cfg: Dict,
    weights: Dict[str, float],
    data, collocation, bc, ic,
    training_cfg: Dict,
    resample_collocation_fn: Optional[Callable[[], Dict]] = None,
    seed: int = 0,
):
    """Shared entry point for both forward.py (trainable=False) and
    inverse.py (trainable=True)."""
    from pinn.network import PINN

    torch.manual_seed(seed)
    model = PINN(hidden_dims=architecture_cfg["hidden_dims"], activation=architecture_cfg["activation"])
    params = make_params(param_init, trainable=trainable)

    trainer = PINNTrainer(model, hypothesis_cfg["fn"], params, weights)
    history = trainer.train(
        data, collocation, bc, ic,
        adam_epochs=training_cfg["adam_epochs"],
        adam_lr=training_cfg["adam_lr"],
        lbfgs_epochs=training_cfg["lbfgs_epochs"],
        resample_collocation_fn=resample_collocation_fn,
    )
    return trainer, history
