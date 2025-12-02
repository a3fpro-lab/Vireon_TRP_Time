# src/vireon_trp/quantum/kl_leash.py
#
# VIREON-Q KL-Leash: quantum-compatible entropy constraint
# for hidden states, implemented with NumPy only.
#
# We treat consecutive hidden-state snapshots as Gaussian
# "mixed states" and enforce a per-step KL bound:
#
#     D_KL(N(mu_{t+1}, Σ) || N(mu_t, Σ)) <= eps_kl.
#
# When the bound is exceeded, we damp the step via linear
# interpolation:
#
#     h_out = h_prev + s * (h_t - h_prev),
#
# with s ∈ (0, 1] chosen so that the effective Δμ is shrunk.
#
# This implementation is framework-agnostic: callers using
# PyTorch / JAX / TF can convert tensors to np.ndarray before
# calling this primitive, and convert back afterward.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class VireonQLConfig:
    """Configuration for the Vireon-Q KL-Leash.

    Attributes
    ----------
    eps_kl : float
        Maximum allowed per-step KL divergence (ε_Vireon).
        Interpreted as an entropy budget per step.
    min_scale : float
        Lower bound on damping scale when KL is exceeded.
        0 < min_scale <= 1.0.
    max_scale : float
        Upper bound on damping scale (usually 1.0).
    cov_reg : float
        Regularization added to covariance trace to keep
        the Gaussian proxy well-conditioned.
    """

    eps_kl: float = 0.012
    min_scale: float = 0.10
    max_scale: float = 1.00
    cov_reg: float = 1e-4


class VireonQLLeash:
    """Vireon-Q KL-Leash for hidden states (NumPy implementation).

    Hidden states are treated as samples from a Gaussian proxy
    N(mu_t, Σ). For consecutive steps, we bound

        D_KL(N(mu_{t+1}, Σ) || N(mu_t, Σ)) <= eps_kl.

    If the KL exceeds eps_kl, we damp the step by interpolating
    between h_prev and h_t.

    This class is intentionally NumPy-only so the core repo has
    no heavy ML framework dependency. To use with PyTorch, call:

        h_t_np = h_t.detach().cpu().numpy()
        h_prev_np = h_prev.detach().cpu().numpy() if h_prev is not None else None
        h_out_np, stats = leash(h_t_np, h_prev_np)
        h_out = torch.from_numpy(h_out_np).to(h_t.device)

    """

    def __init__(self, cfg: VireonQLConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _as_2d(self, h) -> np.ndarray:
        """Ensure input is a 2D np.ndarray [batch, dim]."""
        h_arr = np.asarray(h, dtype=float)
        if h_arr.ndim == 1:
            h_arr = h_arr[None, :]
        elif h_arr.ndim != 2:
            raise ValueError(
                f"Expected h to have shape [batch, dim] or [dim], "
                f"got shape {h_arr.shape}"
            )
        return h_arr

    def _estimate_cov_trace(self, h_arr: np.ndarray) -> float:
        """Estimate covariance trace proxy from hidden states.

        Parameters
        ----------
        h_arr : np.ndarray
            Array of shape [batch, dim].

        Returns
        -------
        float
            Trace of covariance (sum of variances) with a minimum
            regularization floor.
        """
        # Center per batch and compute variance per dimension
        centered = h_arr - h_arr.mean(axis=0, keepdims=True)  # [batch, dim]
        var = np.mean(centered**2, axis=0)  # [dim]
        trace = float(np.sum(var))
        return max(trace, self.cfg.cov_reg)

    def _gaussian_kl(
        self,
        mu_new: np.ndarray,
        mu_old: np.ndarray,
        trace_cov: float,
    ) -> float:
        """Approximate KL between two Gaussians with shared Σ.

        For N(mu_new, Σ) and N(mu_old, Σ) with equal Σ:

            D_KL = 0.5 * Δμ^T Σ^{-1} Δμ.

        We approximate Σ^{-1} using a scalar variance estimate
        trace_cov / d, where d is the hidden dimension.

        Parameters
        ----------
        mu_new : np.ndarray
            New hidden batch [batch, dim] or [1, dim].
        mu_old : np.ndarray
            Previous hidden batch [batch, dim] or [1, dim].
        trace_cov : float
            Estimated trace of Σ.

        Returns
        -------
        float
            Approximate KL divergence.
        """
        if mu_new.shape != mu_old.shape:
            raise ValueError(
                "mu_new and mu_old must have the same shape, "
                f"got {mu_new.shape} and {mu_old.shape}"
            )

        # Use batch-mean difference as Δμ
        delta = np.mean(mu_new - mu_old, axis=0, keepdims=True)  # [1, dim]
        d = delta.shape[-1]

        sigma2 = trace_cov / float(d)
        inv_sigma = 1.0 / (sigma2 + self.cfg.cov_reg)

        quad = float(np.sum(delta**2) * inv_sigma)
        kl = 0.5 * quad
        return float(kl)

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------
    def __call__(
        self,
        h_t,
        h_prev: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply the Vireon-Q KL-Leash to a hidden state snapshot.

        Parameters
        ----------
        h_t :
            Current hidden state (array-like), shape [batch, dim]
            or [dim].
        h_prev : np.ndarray or None
            Previous hidden state, shape [batch, dim] or [dim],
            or None if this is the first step in a sequence.

        Returns
        -------
        h_out : np.ndarray
            Leashed hidden state [batch, dim].
        stats : dict
            Dictionary with:
              - 'kl': approximate KL before damping
              - 'scale': damping scale s applied
        """
        h_t_arr = self._as_2d(h_t)

        if h_prev is None:
            return h_t_arr, {"kl": 0.0, "scale": 1.0}

        h_prev_arr = self._as_2d(h_prev)

        if h_t_arr.shape != h_prev_arr.shape:
            raise ValueError(
                "h_t and h_prev must have the same shape, "
                f"got {h_t_arr.shape} and {h_prev_arr.shape}"
            )

        trace_cov = self._estimate_cov_trace(h_t_arr)
        kl = self._gaussian_kl(h_t_arr, h_prev_arr, trace_cov)

        if kl <= self.cfg.eps_kl:
            scale = 1.0
            h_out = h_t_arr
        else:
            # Since KL ≈ ||Δμ||², scale ~ sqrt(eps_kl / kl)
            raw_scale = (self.cfg.eps_kl / (kl + 1e-12)) ** 0.5
            scale = max(self.cfg.min_scale, min(self.cfg.max_scale, raw_scale))

            h_out = h_prev_arr + scale * (h_t_arr - h_prev_arr)

        return h_out, {"kl": float(kl), "scale": float(scale)}
