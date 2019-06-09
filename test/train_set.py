import numpy as np
from sklearn.model_selection import train_test_split


def gauss_cls(n, size, test_size, mus, sigmas):
    xs = []
    ys = []
    for i in range(n):
        xs.append(np.random.normal(mus[i], sigmas[i], (size, 2)))
        ys.append(np.full(size, i))

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return train_test_split(xs, ys, test_size=test_size)
