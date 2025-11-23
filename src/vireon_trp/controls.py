import numpy as np

def shuffle_proxies(proxies, seed=0):
    rng = np.random.default_rng(seed)
    p = np.asarray(proxies).copy()
    rng.shuffle(p)
    return p

def poissonize(series, seed=0):
    """
    Replace series with Poisson noise matching its mean.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    lam = float(np.mean(x))
    return rng.poisson(lam, size=len(x)).astype(float)
