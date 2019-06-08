import numpy as np


def generic(n, k, xs, ys, u, metric, w):
    dist = np.fromiter(map(lambda x: metric(x, u), xs), dtype=np.float)

    # [index, (dist, ys)]
    dist = np.array(list(enumerate(zip(dist, ys))))

    dist_val = dist[dist[:, 1].argsort()]

    results = np.zeros(n, dtype=np.float)
    for y in range(0, n):
        res = 0
        for i in range(0, k):
            res += int(dist_val[i][1][1] == y) * w(i, xs[dist_val[i][0]])
        results[y] = res

    return np.argmax(results), results


def knn(n, k, xs, ys, u, metric):
    w = lambda i, u: int(i < k)

    return generic(n, k, xs, ys, u, metric, w)
