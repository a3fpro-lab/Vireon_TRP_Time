import numpy as np

class KLLeash:
    """
    KL divergence leash between current state proxies S(t)
    and baseline stable state S0.

    D_KL = sum p * log(p/q)
    We normalize proxies into distributions.
    """

    def __init__(self, eps=1e-9, yellow_days=2, red_days=3):
        self.eps = eps
        self.yellow_days = yellow_days
        self.red_days = red_days

    def _to_dist(self, v):
        v = np.asarray(v, dtype=float)
        v = np.clip(v, self.eps, None)
        return v / v.sum()

    def kl(self, S_t, S0):
        p = self._to_dist(S_t)
        q = self._to_dist(S0)
        return float(np.sum(p * np.log((p + self.eps)/(q + self.eps))))

    def zone(self, kl_series, yellow_thr, red_thr):
        """
        Zone logic:
          - yellow if >= yellow_thr for yellow_days
          - red if >= red_thr for red_days or any point >= 2*red_thr
        """
        ks = np.asarray(kl_series, dtype=float)

        if np.any(ks >= 2.0 * red_thr):
            return "RED"

        yellow_run = np.convolve((ks >= yellow_thr).astype(int),
                                np.ones(self.yellow_days, dtype=int),
                                mode="valid")
        red_run = np.convolve((ks >= red_thr).astype(int),
                              np.ones(self.red_days, dtype=int),
                              mode="valid")

        if np.any(red_run >= self.red_days):
            return "RED"
        if np.any(yellow_run >= self.yellow_days):
            return "YELLOW"
        return "GREEN"
