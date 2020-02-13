#!/usr/bin/env python

from itertools import zip_longest, islice
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import numpy.testing as npt
import time
# from numba import njit,

def to_int_keys_best(l):
    """
    l: iterable of keys
    returns: a list with integer keys
    """
    ls = list(dict.fromkeys(l))
    ls.sort()
    index = {v: i for i, v in enumerate(ls)}
    return [index[v] for v in l]


def to_int_keys_np(l):
    indices = np.unique(l, return_inverse=True)[1]
    return indices

def to_int_keys_sparse(x):
    print(x.shape)
    print(x.min(), x.max())
    start = time.time()
    sparse_p = coo_matrix(
        (
            np.ones(x.shape[0], dtype=np.uint64),
            (np.zeros(x.shape[0], dtype=np.uint64), x),
        ),
        # shape=(x.shape[0], 1),
        dtype=bool,
    ).tocsc()
    print("csc1", time.time()-start)

    start = time.time()
    sparse_c = sparse_p.count_nonzero()
    print("count_nonzero", time.time()-start)

    start = time.time()
    sparse_p2 = coo_matrix(
        (np.arange(sparse_c, dtype=np.uint64) + 1, sparse_p.nonzero()), dtype=np.uint64
    ).tocsc()
    print("csc2", time.time()-start)

    return sparse_p2[0, x].data - 1

def suffix_array_best(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys_best(s)
    while max(line) < n - 1:
        line = to_int_keys_best(
            [a * (n + 1) + b + 1
             for (a, b) in
             zip_longest(line, islice(line, k, None),
                         fillvalue=-1)])
        k <<= 1
    return line

def suffix_array_np(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    # start = time.time()
    n = len(s)
    k = 1
    # print(to_int_keys_best(s))
    line = to_int_keys_np(s)
    # print(line)
    # print(to_int_keys_sparse(s))
    # line = to_int_keys_sparse(s)
    # print("HERE")
    # test = to_int_keys_np_2(s)
    # npt.assert_almost_equal(line, test)
    tmp_line = np.ones(n, dtype=np.int64)
    # print(time.time() - start)
    while max(line) < n - 1:
        # start = time.time()
        tmp_line[:] = -1
        tmp_line[:-k] = line[k:]

        # line[:] = to_int_keys_np((n + 1) * line + tmp_line + 1)
        line[:] = (n + 1) * line + tmp_line + 1
        # print("here", time.time() - start)

        # start = time.time()
        # test = to_int_keys_np_2(line)
        # line[:] = to_int_keys_best(line)
        # print(to_int_keys_best(line))
        # print(to_int_keys_sparse(line))
        line[:] = to_int_keys_np(line)
        # print(line)
        # line[:] = to_int_keys_sparse(line)
        # print(line)
        # npt.assert_almost_equal(line, test)
        # print(time.time() - start)

        k <<= 1
        
    return line

def inverse_array(l):
    n = len(l)
    ans = [0] * n
    for i in range(n):
        ans[l[i]] = i
    return ans

def kasai(s, sa=None):
    """
    constructs the lcp array
    O(n)
    s: string
    sa: suffix array
    from https://www.hackerrank.com/topics/lcp-array
    """
    if sa is None:
        sa = suffix_array(s)
    n = len(s)
    k = 0
    lcp = [0] * n
    pos = inverse_array(sa)
    for i in range(n):
        if sa[i] == n - 1:
            k = 0
            continue
        j = pos[sa[i] + 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[sa[i]] = k
        if k:
            k -= 1
    return lcp

if __name__ == '__main__':
    word = 'one$banana$phone$'
    # word = np.array([6, 5, 3, 0, 2, 1, 5, 1, 5, 1, 0, 7, 4, 6, 5, 3, 0])
    # word = "ABABBAB"
    # word = "banana"
    sarray = suffix_array_best(word)
    # print(to_int_keys_np(np.array(list(word))))
    print(sarray)
    # print(suffix_array_np(np.array(list(word))))
    for i in np.argsort(sarray):
        print(i, word[i:])
    print(kasai(word, sarray))
    exit()

    # print(suffix_array_best([2,1,3,1,3,1]))
    # print(suffix_array_best(np.array([2,1,3,1,3,1])))
    seq_length = 10
    # n_seqs = 1_000_000
    n_seqs = 1_000_000
    inp = np.random.randint(1000, size=seq_length*n_seqs)

    start = time.time()
    # sa = suffix_array_best(inp)
    print("suffix_array_best", time.time() - start)

    start = time.time()
    sa_np = suffix_array_np(inp)
    print("sparse", time.time() - start)

    # npt.assert_almost_equal(np.array(sa), sa_np)

    lcp_array = kasai(inp, sa_np)

