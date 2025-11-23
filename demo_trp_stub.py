import numpy as np
from trp_math import dt_eff, subjective_time

# Toy example: external structure R_t increases,
# perception P_t is fixed.
R = np.linspace(0, 2, 1000)
P = np.ones_like(R) * 3.0

T_subj = subjective_time(P, R)
print("Subjective time:", T_subj)
print("First dt_eff:", dt_eff(P[0], R[0]))
