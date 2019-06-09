import numpy as np


def knn_generic(n, k, xs, ys, u, metric, w):
    assert k <= np.size(xs, axis=0)

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

    return knn_generic(n, k, xs, ys, u, metric, w)


def leave_one_out(n, max_k, a, xs, ys):
    assert max_k <= np.size(xs, axis=0)

    size = np.size(xs, axis=0)
    indexes = np.arange(0, size, 1, dtype=np.int)
    sums = np.zeros(max_k - 1, dtype=np.int)

    for k in range(1, max_k):
        f = np.vectorize(lambda i: a(n, k,
                                     xs[indexes != i, :],
                                     ys[indexes != i],
                                     xs[i, :])[0])
        res = f(indexes) != ys
        sums[k - 1] = np.sum(res.astype(np.int))

    return np.argmin(sums) + 1, sums
