# src/vireon_trp/quantum/kl_leash.py
#
# VIREON-Q KL-Leash: quantum-compatible entropy constraint
# for transformer (or generic sequence) hidden states.
#
# This is the first Vireon-Q primitive: we treat consecutive
# hidden states as Gaussian "mixed states" and enforce a
# per-step KL bound:
#
#     D_KL(N(mu_{t+1}, Σ) || N(mu_t, Σ)) <= eps_kl.
#
# When the bound is exceeded, we damp the step via a linear
# interpolation between h_prev and h_t, approximating a
# Lindblad-style drift constraint in representation space.
#
# This slots conceptually into the TRP law:
#   - R: structure / alignment quality (external)
#   - P: gain / precision (internal)
#   - T = R × P: effective subjective time
#
# Here, eps_kl is the "yellow zone" drift bound in the
# representation manifold: a quantum-compatible KL-Leash.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class VireonQLConfig:
    """Configuration for the Vireon-Q KL-Leash.

    Attributes
    ----------
    eps_kl : float
        Maximum allowed per-step KL divergence (ε_Vireon).
        Interpreted as an entropy budget per token / step.
    min_scale : float
        Lower bound on damping scale when KL is exceeded.
        Must satisfy 0 < min_scale <= 1.0.
    max_scale : float
        Upper bound on damping scale (usually 1.0).
    cov_reg : float
        Diagonal regularization added to the covariance
        estimate to keep the Gaussian proxy well-conditioned.
    """

    eps_kl: float = 0.012
    min_scale: float = 0.10
    max_scale: float = 1.00
    cov_reg: float = 1e-4


class VireonQLLeash:
    """Vireon-Q KL-Leash for transformer hidden states.

    We interpret last-layer hidden states h_t as samples
    from a Gaussian proxy N(mu_t, Σ). For consecutive steps,
    we bound

        D_KL(N(mu_{t+1}, Σ) || N(mu_t, Σ)) <= eps_kl.

    If the KL exceeds eps_kl, we damp the step by linearly
    interpolating between h_prev and h_t:

        h_out = h_prev + s * (h_t - h_prev),

    where s ∈ (0, 1] is chosen so that the effective Δμ
    is shrunk (KL scales roughly with ||Δμ||²). This is a
    discrete-time analog of a Lindblad drift bound in the
    representation manifold.

    Usage (per sequence):

        from vireon_trp.quantum import VireonQLConfig, VireonQLLeash

        cfg = VireonQLConfig(eps_kl=0.012)
        leash = VireonQLLeash(cfg)

        prev_state = None
        for t in range(T):
            h_t = last_layer_hidden(...)  # [batch, dim]
            h_t_leashed, stats = leash(h_t, prev_state)
            prev_state = h_t_leashed.detach()
            # use h_t_leashed for logits / decoding / RL

    """

    def __init__(self, cfg: VireonQLConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _estimate_cov_trace(self, h: torch.Tensor) -> float:
        """Estimate a covariance trace proxy from hidden states.

        Parameters
        ----------
        h : torch.Tensor
            Tensor of shape [batch, dim].

        Returns
        -------
        float
            Trace of covariance (sum of variances) with a minimum
            regularization floor.
        """
        if h.ndim != 2:
            raise ValueError(
                f"Expected h to have shape [batch, dim], "
                f"got {tuple(h.shape)}"
            )

        # Center per batch and compute variance per dimension
        centered = h - h.mean(dim=0, keepdim=True)  # [batch, dim]
        var = centered.pow(2).mean(dim=0)  # [dim]
        trace = var.sum().item()
        return max(trace, self.cfg.cov_reg)

    def _gaussian_kl(
        self,
        mu_new: torch.Tensor,
        mu_old: torch.Tensor,
        trace_cov: float,
    ) -> float:
        """Approximate KL between two Gaussians with shared Σ.

        For N(mu_new, Σ) and N(mu_old, Σ) with equal Σ,

            D_KL = 0.5 * Δμ^T Σ^{-1} Δμ.

        We approximate Σ^{-1} using a scalar variance estimate
        trace_cov / d, where d is the hidden dimension.

        Parameters
        ----------
        mu_new : torch.Tensor
            New hidden state batch [batch, dim].
        mu_old : torch.Tensor
            Previous hidden state batch [batch, dim].
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
                f"got {tuple(mu_new.shape)} and {tuple(mu_old.shape)}"
            )

        # Use batch-mean difference as Δμ
        delta = (mu_new - mu_old).mean(dim=0, keepdim=True)  # [1, dim]
        d = delta.shape[-1]

        sigma2 = trace_cov / float(d)
        inv_sigma = 1.0 / (sigma2 + self.cfg.cov_reg)

        quad = (delta.pow(2).sum() * inv_sigma).item()
        kl = 0.5 * quad
        return float(kl)

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------
    def __call__(
        self,
        h_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply the Vireon-Q KL-Leash to a hidden state.

        Parameters
        ----------
        h_t : torch.Tensor
            Current hidden state, shape [batch, dim].
        h_prev : torch.Tensor or None
            Previous hidden state, shape [batch, dim], or None
            if this is the first step in a sequence.

        Returns
        -------
        h_out : torch.Tensor
            Leashed hidden state [batch, dim].
        stats : dict
            Dictionary with at least:
              - 'kl': approximate KL before damping
              - 'scale': applied damping scale s
        """
        if h_prev is None:
            # First step: nothing to compare against.
            return h_t, {"kl": 0.0, "scale": 1.0}

        if h_t.shape != h_prev.shape:
            raise ValueError(
                "h_t and h_prev must have the same shape, "
                f"got {tuple(h_t.shape)} and {tuple(h_prev.shape)}"
            )

        # Estimate covariance trace from current states (cheap proxy)
        trace_cov = self._estimate_cov_trace(h_t.detach())
        kl = self._gaussian_kl(h_t.detach(), h_prev.detach(), trace_cov)

        if kl <= self.cfg.eps_kl:
            # Within budget: no damping
            scale = 1.0
            h_out = h_t
        else:
            # Compute damping scale s ∈ (0, 1] such that effective KL
            # is roughly scaled down toward eps_kl. Since KL ≈ ||Δμ||²
            # in this approximation, we use:
            #
            #   s_raw = sqrt(eps_kl / kl).
            #
            raw_scale = (self.cfg.eps_kl / (kl + 1e-12)) ** 0.5
            scale = float(
                max(self.cfg.min_scale, min(self.cfg.max_scale, raw_scale))
            )

            # Interpolate between old and new state
            h_out = h_prev + scale * (h_t - h_prev)

        return h_out, {"kl": float(kl), "scale": float(scale)}
