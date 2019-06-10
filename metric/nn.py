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


def knn_parzen(n, k, xs, ys, u, metric):
    assert k + 1 <= np.size(xs, axis=0)

    dist = np.fromiter(map(lambda x: metric(x, u), xs), dtype=np.float)

    # [index, (dist, ys)]
    dist = np.array(list(enumerate(zip(dist, ys))))

    dist_val = dist[dist[:, 1].argsort()]

    results = np.zeros(n, dtype=np.float)
    for y in range(0, n):
        res = 0
        for i in range(0, k):
            res += int(dist_val[i][1][1] == y) * \
                    3.0 / 4 * (1 - (dist_val[i][1][0] / dist_val[k][1][0]) ** 2)
        results[y] = res

    return np.argmax(results), results


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


def margin(n, a, xs, ys):
    size = np.size(xs, axis=0)
    margins = np.zeros(size, dtype=np.float)
    indexes = np.arange(0, n, 1, dtype=np.int)

    for i in range(size):
        cls, results = a(xs[i])
        margins[i] = results[ys[i]] - np.max(results[indexes != ys[i]])

    return margins


def margin_stolp(n, k, a, train_xs, train_ys, xs, ys):
    size = np.size(xs, axis=0)
    margins = np.zeros(size, dtype=np.float)
    indexes = np.arange(0, n, 1, dtype=np.int)

    for i in range(size):
        cls, results = a(xs[i], k, train_xs, train_ys)
        margins[i] = results[ys[i]] - np.max(results[indexes != ys[i]])

    return margins


def stolp(n, k, delta, l0, a, xs, ys):
    size = np.size(xs, axis=0)
    margins = margin_stolp(n, k, a,
                           train_xs=xs, train_ys=ys,
                           xs=xs, ys=ys)

    old_xs = np.copy(xs)
    old_ys = np.stack((np.arange(0, size, 1), ys), axis=-1)

    # Удаление шума
    ids = margins >= delta
    margins = margins[ids]

    xs = old_xs[ids, :]
    ys = old_ys[ids, :]

    # Размер выборки после удаления шума
    size = np.size(xs, axis=0)

    # Генерация массива "отступ-индекс-класс"
    mc = np.stack((margins, ys[:, 0], ys[:, 1]), axis=-1)
    mc_margin = 0
    mc_idx = 1
    mc_class = 2

    es = np.full(n, np.finfo(np.float).min, dtype=np.float)
    omega_ids = np.zeros(n, dtype=np.int)

    # Поиск эталонов
    for i in range(size):
        cls = int(mc[i][mc_class])
        if mc[i][mc_margin] > es[cls]:
            es[cls] = mc[i][mc_margin]
            omega_ids[cls] = mc[i][mc_idx]

    current_size = n
    while current_size != size:
        n_xs = np.delete(old_xs, omega_ids, axis=0)
        n_ys = np.delete(old_ys, omega_ids, axis=0)
        omega_xs = old_xs[omega_ids, :]
        omega_ys = old_ys[omega_ids, :]

        k = np.size(omega_xs, axis=0) - 1
        margins = margin_stolp(n, k, a,
                               train_xs=omega_xs, train_ys=omega_ys[:, 1:],
                               xs=n_xs, ys=n_ys[:, 1:])

        ids = margins < 0
        margins = margins[ids]
        err_ys = n_ys[ids, :]

        if np.size(err_ys, axis=0) < l0:
            break

        min_idx = err_ys[np.argmin(margins)][0]
        omega_ids = np.append(omega_ids, min_idx)

        current_size += 1

    return old_xs[omega_ids, :], old_ys[omega_ids, 1]
