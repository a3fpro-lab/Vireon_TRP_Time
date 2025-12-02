# tests/test_quantum_kl_leash.py
#
# Basic sanity checks for Vireon-Q KL-Leash.
#
# We only test local behavior and shapes here so that CI
# remains fast and deterministic.

import torch

from vireon_trp.quantum import VireonQLConfig, VireonQLLeash


def test_first_step_passthrough():
    cfg = VireonQLConfig(eps_kl=0.012)
    leash = VireonQLLeash(cfg)

    h_t = torch.zeros(4, 16)  # [batch, dim]
    h_out, stats = leash(h_t, h_prev=None)

    assert h_out.shape == h_t.shape
    assert torch.allclose(h_out, h_t)
    assert stats["kl"] == 0.0
    assert stats["scale"] == 1.0


def test_leash_shape_and_scale_when_kl_exceeds():
    torch.manual_seed(0)
    cfg = VireonQLConfig(eps_kl=0.001)  # very small budget to force damping
    leash = VireonQLLeash(cfg)

    batch, dim = 4, 32
    h_prev = torch.zeros(batch, dim)
    # Large step to trigger KL > eps_kl
    h_t = torch.randn(batch, dim) * 5.0

    h_out, stats = leash(h_t, h_prev)

    assert h_out.shape == h_t.shape
    # Scale should be <= 1 and >= min_scale
    assert cfg.min_scale <= stats["scale"] <= cfg.max_scale
    # Output should lie between h_prev and h_t
    diff_prev = (h_out - h_prev).norm()
    diff_full = (h_t - h_prev).norm()
    assert diff_prev <= diff_full + 1e-6


def test_no_damping_when_kl_small():
    torch.manual_seed(1)
    cfg = VireonQLConfig(eps_kl=10.0)  # very large budget
    leash = VireonQLLeash(cfg)

    batch, dim = 4, 32
    h_prev = torch.zeros(batch, dim)
    h_t = torch.randn(batch, dim) * 0.1  # small step

    h_out, stats = leash(h_t, h_prev)

    assert h_out.shape == h_t.shape
    # With huge eps_kl, scale should remain 1.0
    assert stats["scale"] == 1.0
    assert torch.allclose(h_out, h_t, atol=1e-6)
