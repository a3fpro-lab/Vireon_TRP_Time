import numpy as np
from vireon_trp import PEstimator, REstimator, TEstimator

def test_p_estimator_ratio():
    pe = PEstimator()
    base = np.array([10, 20, 30])
    cur  = np.array([20, 40, 60])
    assert abs(pe.compute(base, cur) - 2.0) < 1e-9

def test_r_entropy_mode_bounds():
    re = REstimator(mode="entropy")
    sig = np.array([1, 1, 1, 1])
    r = re.compute(env_signal=sig)
    assert 0.0 <= r <= 1.0

def test_t_estimator_sum():
    te = TEstimator()
    R = np.array([1, 1, 1])
    P = np.array([2, 2, 2])
    assert te.compute(R, P, dt=1.0) == 6.0
