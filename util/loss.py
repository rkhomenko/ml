import numpy as np


def empirical_risk_cls(ys, ys_exact):
    assert ys.shape == ys_exact.shape
    return 1.0 / np.size(ys, axis=0) * np.sum(ys != ys_exact)
