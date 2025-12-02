# src/vireon_trp/quantum/__init__.py
#
# Quantum-cognition extensions for VIREON TRP Time.
#
# This namespace houses Vireon-Q primitives (von Neumann-style
# structure metrics, Lindblad-like drift bounds, etc.).
#
# First primitive shipped:
#   - VireonQLConfig
#   - VireonQLLeash
#
# which interpret sequence hidden states as mixed states and
# apply a quantum-compatible KL-Leash.

from .kl_leash import VireonQLConfig, VireonQLLeash

__all__ = [
    "VireonQLConfig",
    "VireonQLLeash",
]
