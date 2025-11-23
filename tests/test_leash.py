import numpy as np
from vireon_trp import KLLeash

def test_kl_zero_when_same():
    leash = KLLeash()
    S0 = np.array([1, 2, 3])
    St = np.array([1, 2, 3])
    assert abs(leash.kl(St, S0)) < 1e-9

def test_zone_logic():
    leash = KLLeash(yellow_days=2, red_days=3)
    ks = [0.1, 0.2, 0.2, 0.2]
    assert leash.zone(ks, yellow_thr=0.15, red_thr=0.5) == "YELLOW"
