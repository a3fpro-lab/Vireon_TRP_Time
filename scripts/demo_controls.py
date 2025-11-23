#!/usr/bin/env python3
from vireon_trp import TRPToyModel, TEstimator, shuffle_proxies, poissonize
import numpy as np

def main():
    model = TRPToyModel(seed=2)

    # Real run
    R, P, D = model.run(steps=300, u=0.02, R_mode="constant")
    T_real = TEstimator().compute(R, P)

    # Control 1: shuffle P (destroys temporal structure)
    P_shuf = shuffle_proxies(P, seed=3)
    T_shuf = TEstimator().compute(R, P_shuf)

    # Control 2: poissonize P (null with same mean)
    P_pois = poissonize(P, seed=4)
    T_pois = TEstimator().compute(R, P_pois)

    print("T real:", T_real)
    print("T shuffled-P control:", T_shuf)
    print("T poisson-P control:", T_pois)
    print("Means: real {:.3f} vs shuf {:.3f} vs pois {:.3f}".format(
        np.mean(P), np.mean(P_shuf), np.mean(P_pois)
    ))

if __name__ == "__main__":
    main()
