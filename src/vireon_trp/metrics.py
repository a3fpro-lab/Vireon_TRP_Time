import numpy as np

class PEstimator:
    """
    Perception gain estimator.

    Inputs:
      proxies_baseline: array shape (k,) baseline averages (P=1)
      proxies_t:        array shape (k,) current averages

    Output:
      P_t = mean(proxies_t / proxies_baseline)
    """
    def __init__(self, eps=1e-9):
        self.eps = eps

    def compute(self, proxies_baseline, proxies_t):
        b = np.asarray(proxies_baseline, dtype=float)
        x = np.asarray(proxies_t, dtype=float)
        ratio = x / (b + self.eps)
        return float(np.mean(ratio))


class REstimator:
    """
    Reality bandwidth estimator.

    mode="given": user supplies R_t directly.
    mode="entropy": estimate R_t as 1 - normalized entropy of environment signal.
    """
    def __init__(self, mode="given", eps=1e-9):
        self.mode = mode
        self.eps = eps

    def compute(self, R_t=None, env_signal=None):
        if self.mode == "given":
            if R_t is None:
                raise ValueError("R_t required for mode='given'")
            return float(R_t)

        if self.mode == "entropy":
            if env_signal is None:
                raise ValueError("env_signal required for mode='entropy'")
            s = np.asarray(env_signal, dtype=float)
            s = s - s.min()
            p = s / (s.sum() + self.eps)
            H = -(p * np.log(p + self.eps)).sum()
            Hmax = np.log(len(p) + self.eps)
            return float(1.0 - H / (Hmax + self.eps))

        raise ValueError("Unknown REstimator mode.")


class TEstimator:
    """
    Effective time integration:
      dT/dtau = R(tau) * P(tau)
    discrete:
      T = sum_t R_t * P_t * dt
    """
    def compute(self, R_series, P_series, dt=1.0):
        R = np.asarray(R_series, dtype=float)
        P = np.asarray(P_series, dtype=float)
        if R.shape != P.shape:
            raise ValueError("R and P must have same shape")
        return float(np.sum(R * P) * dt)
