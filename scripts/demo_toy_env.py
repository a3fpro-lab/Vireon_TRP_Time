#!/usr/bin/env python3
from vireon_trp import TRPToyModel, TEstimator
import numpy as np

def main():
    model = TRPToyModel(gamma=0.05, alpha=0.02, seed=1)
    R, P, D = model.run(steps=300, u=0.02, R_mode="constant")

    T = TEstimator().compute(R, P, dt=1.0)

    print("Final P:", P[-1])
    print("Final divergence D:", D[-1])
    print("Effective time T:", T)
    print("Baseline (P=1) time would be:", np.sum(R))

if __name__ == "__main__":
    main()
