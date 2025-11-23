import numpy as np

class TRPToyModel:
    """
    Minimal TRP toy:

    Environment has structure R_t in [0,1].
    Agent has perception gain P_t >= 0.

    Training increases P_t by lever u_t,
    but divergence cost increases with (P_t - 1)^2.

    Simulate:
      P_{t+1} = P_t + u_t - gamma*(P_t-1)
      D_{t+1} = D_t + alpha*(P_t-1)^2
      T = sum R_t * P_t
    """
    def __init__(self, gamma=0.05, alpha=0.02, seed=0):
        self.gamma = gamma
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def run(self, steps=200, u=0.02, R_mode="constant"):
        P = np.ones(steps)
        D = np.zeros(steps)
        R = np.zeros(steps)

        for t in range(steps-1):
            if R_mode == "constant":
                R[t] = 1.0
            elif R_mode == "noisy":
                R[t] = np.clip(self.rng.normal(0.7, 0.1), 0, 1)
            else:
                raise ValueError("Unknown R_mode")

            P[t+1] = max(0.0, P[t] + u - self.gamma*(P[t]-1.0))
            D[t+1] = D[t] + self.alpha*(P[t]-1.0)**2

        R[-1] = R[-2]
        return R, P, D
