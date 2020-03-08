#!/usr/bin/env python

from itertools import zip_longest, islice
import numpy as np

# import numpy.testing as npt
import time
import numba
from numpy.lib.stride_tricks import as_strided


@numba.njit(fastmath=True)
def numba_unique(arr):
    return np.array(list(set(arr)))


def to_int_keys_best(l):
    """
    l: iterable of keys
    returns: a list with integer keys
    """
    ls = list(dict.fromkeys(l))
    ls.sort()
    index = {k: v for v, k in enumerate(ls)}
    return [index[k] for k in l]


def to_int_keys_np(l):
    indices = np.unique(l, return_inverse=True)[1]
    return indices


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
            [
                a * (n + 1) + b + 1
                for (a, b) in zip_longest(line,
                                          islice(line, k, None),
                                          fillvalue=-1
                                          )
            ]
        )
        k <<= 1

    return inverse_array_np(line)


def suffix_array_np(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys_np(s)
    tmp_line = np.ones(n, dtype=np.int64)
    while max(line) < n - 1:
        tmp_line[:] = -1
        tmp_line[:-k] = line[k:]

        line[:] = (n + 1) * line + tmp_line + 1
        line[:] = to_int_keys_np(line)

        k <<= 1

    return inverse_array_np(line)


def inverse_array(l):
    n = len(l)
    ans = [0] * n
    for i in range(n):
        ans[l[i]] = i
    return ans


def inverse_array_np(l):
    return np.argsort(l)


def kasai(s, sa=None):
    """
    constructs the lcp array
    O(n)
    s: string
    sa: suffix array
    from https://www.hackerrank.com/topics/lcp-array
    """
    if sa is None:
        sa = suffix_array_best(s)
    n = len(s)
    k = 0
    lcp = [0] * n
    pos = sa
    sa = inverse_array(sa)
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
    return np.array(lcp)


@numba.njit(fastmath=True)
def kasai_numba(s, sa, stop=None):
    """
    constructs the lcp array
    O(n)
    s: string
    sa: suffix array
    from https://www.hackerrank.com/topics/lcp-array
    """
    n = s.shape[0]
    k = 0
    lcp = np.zeros(n, dtype=np.int64)
    pos = sa
    sa = np.argsort(sa)
    for i in numba.prange(n):
        if sa[i] == n - 1:
            k = 0
        else:
            j = pos[sa[i] + 1]
            while (
                i + k < n
                and j + k < n
                and s[i + k] != stop
                and s[j + k] != stop
                and s[i + k] == s[j + k]
            ):
                k = k + 1
            lcp[sa[i]] = k
            if k > 0:
                k = k - 1
    return lcp


def distance_to_sentinel(s, sentinel):
    mask_z = s == sentinel
    idx_z = np.flatnonzero(mask_z)

    # Cover for the case when there's no 0 left to the right
    if s[-1] != 0:
        idx_z = np.r_[idx_z, len(s)]

    out = idx_z[np.r_[False, mask_z[:-1]].cumsum()] - np.arange(len(s))

    return out


def get_runs(x, min_run=0):
    r = np.full(len(x), 2)
    d = np.diff(x) == 1
    r[1:] -= d
    r[:-1] -= d
    out = np.repeat(x, r).reshape(-1, 2)
    out = out[(out[:, 1] - out[:, 0]) >= min_run]
    out[:, 1] += 1
    return out


def get_dtype(n):
    if n < np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif n < np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif n < np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return dtype


def get_overlaps(x):
    a = np.unique(x[x >= 2])  # np.arange(2, x.max()+1)

    b = x >= a[:, None]

    c = np.argwhere(b)
    c[:, 0] = a[c[:, 0]]
    c = np.pad(c, ((1, 1), (0, 0)), "symmetric")

    d = np.where(np.diff(c[:, 1]) != 1)[0]

    e = as_strided(d, shape=(len(d) - 1, 2), strides=(8, 8))
    e = e[(np.diff(e, axis=1) >= 1).flatten()]
    e[:, 0] = e[:, 0] + 1

    f = np.hstack([c[:, 0][e[:, 0, None]], c[:, 1][e]])

    f[:, 2] += 1

    return f


if __name__ == "__main__":
    word = "one$banana$phone$"
    word = "a$banana$and$a$bandana$"
    word = "a$banana#and@a*bandana+"
    # word = np.array([6, 5, 3, 0, 2, 1, 5, 1, 5, 1, 0, 7, 4, 6, 5, 3, 0])
    # word = 'mississippi$'
    # word = "ABABBAB"
    # word = "banana"
    # word = "banana$"
    word = np.array(list(word))
    print("Input Word")
    print(to_int_keys_np(word))
    # sarray = suffix_array_best(word)
    sarray = suffix_array_np(word)
    print()
    print("Suffix Array")
    print(sarray)
    # print(inverse_array_np(sarray))
    # print(suffix_array_np(np.array(list(word))))

    for i in sarray:
        print(i, word[i:])

    print()
    print("LCP Array")
    lcp_array = kasai(word, sarray)
    print(lcp_array)
    lcp_array_numba = kasai_numba(word, sarray)
    print(lcp_array_numba)
    # print(distance_to_sentinel(word, '$'))
    # lcp_array_numba = kasai_numba(word, sarray, '$')
    # print(lcp_array_numba)

    print()
    print("Overlap")

    overlap_array = get_overlaps(lcp_array_numba)
    print(overlap_array)
    for min_overlap, start_inx, stop_inx in overlap_array:
        for i in range(start_inx, stop_inx + 1):
            word_start_inx = sarray[i]
            word_stop_inx = word_start_inx + min_overlap
            print(i, word[word_start_inx:word_stop_inx])
        print()
    # exit()

    print()
    print("Large Input Array")
    # print(suffix_array_best([2,1,3,1,3,1]))
    # print(suffix_array_best(np.array([2,1,3,1,3,1])))
    # for n_seqs in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    # for n_seqs in [10_000_000, 20_000_000, 30_000_000, 40_000_000]:
    for n_seqs in [30_000_000]:
        np.random.seed(7)
        seq_length = 10
        n_states = 1000
        dtype = get_dtype(n_states)
        inp = np.random.randint(n_states,
                                size=seq_length * n_seqs,
                                dtype=dtype
                                )

        # start = time.time()
        # # sa = suffix_array_best(inp)
        # print("suffix_array_best", seq_length*n_seqs, time.time() - start)

        start = time.time()
        sa_np = suffix_array_np(inp)
        print("suffix_array_np", seq_length * n_seqs, time.time() - start)

        # npt.assert_almost_equal(np.array(sa), sa_np)
        # npt.assert_almost_equal(np.array(sa), sa_opt)

        # start = time.time()
        # lcp_array = kasai(inp, sa_np)
        # print("lcp", time.time() - start)

        start = time.time()
        lcp_array = kasai_numba(inp, sa_np)
        print("lcp", time.time() - start)

        start = time.time()
        overlap_array = get_overlaps(lcp_array)
        print("overlap", time.time() - start)
