#!/usr/bin/env python
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import numpy as np
import numpy.testing as npt
import time


def to_int_keys_np(x, maxnum=None):
    if maxnum is None:
        maxnum = int(x.max() + 1)  # Determines extent of indexing array
    p = np.zeros(maxnum, dtype=bool)
    p[x] = 1

    p2 = np.empty(maxnum, dtype=np.uint64)
    c = p.sum()
    p2[p] = np.arange(c)
    out = p2[x]
    return out


def sparse_to_int_keys_np(x):
    start = time.time()
    sparse_p = coo_matrix(
        (
            np.ones(x.shape[0], dtype=np.uint64),
            (np.zeros(x.shape[0], dtype=np.uint64), x),
        ),
        shape=(1, x.shape[0]),
        dtype=bool,
    ).tocsc()
    print("csc1", time.time()-start)

    start = time.time()
    sparse_c = sparse_p.count_nonzero()
    print("count_nonzero", time.time()-start)

    start = time.time()
    sparse_p2 = csc_matrix(
        (np.arange(sparse_c, dtype=np.uint64) + 1, sparse_p.nonzero()), dtype=np.uint64
    )
    print("csc2", time.time()-start)

    return sparse_p2[0, x].data - 1


if __name__ == "__main__":
    np.random.seed(7)
    for i in range(1000):
        print(i)
        inp = np.random.randint(10000, size=100_000_000)
        # inp = np.random.randint(10, size=10)
        start = time.time()
        left = to_int_keys_np(inp)
        print("np", time.time() - start)
        
        start = time.time()
        right = sparse_to_int_keys_np(inp)
        print("sparse", time.time() - start)
        start = time.time()
        npt.assert_almost_equal(left, right)

        _, truth = np.unique(inp, return_inverse=True)
        print("truth", time.time() - start)

        npt.assert_almost_equal(left, truth)
        npt.assert_almost_equal(right, truth)
