#!/usr/bin/env python

from itertools import zip_longest, islice
import numpy as np
import numpy.testing as npt
import time
import numba
import pandas as pd

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


def to_int_keys_optimized(x):
    sorted_x_unique_series = pd.Series(np.sort(numba_unique(x)))
    mask = pd.Series(sorted_x_unique_series.index.values, index=sorted_x_unique_series)
    return mask.loc[x].values


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
        
    return line

def suffix_array_optimized(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1    
    line = to_int_keys_optimized(s)
    tmp_line = np.ones(n, dtype=np.int64)
    while max(line) < n - 1:
        tmp_line[:] = -1
        tmp_line[:-k] = line[k:]

        line[:] = (n + 1) * line + tmp_line + 1
        line[:] = to_int_keys_optimized(line)

        k <<= 1
        
    return line

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
    return np.array(lcp)


@numba.njit()
def kasai_numba(s, sa, pos, stop=None):
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
    for i in numba.prange(n):
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


def get_runs(x, min_run=0):
    r = np.full(len(x),2)
    d = np.diff(x)==1
    r[1:]-=d
    r[:-1]-=d 
    out = np.repeat(x, r).reshape(-1,2)
    out = out[(out[:, 1] - out[:,0]) >= min_run]
    out[:, 1] += 1
    return out

if __name__ == '__main__':
    word = 'one$banana$phone$'
    # word = np.array([6, 5, 3, 0, 2, 1, 5, 1, 5, 1, 0, 7, 4, 6, 5, 3, 0])
    # word = 'mississippi$'
    # word = "ABABBAB"
    # word = "banana"
    word = np.array(list(word))
    # sarray = suffix_array_best(word)
    sarray = suffix_array_np(word)
    # print(sarray)
    # print(inverse_array(sarray))
    # print(suffix_array_np(np.array(list(word))))
    for i in inverse_array_np(sarray):
        print(i, word[i:])
    lcp_array = kasai(word, sarray)
    print(lcp_array)
    lcp_array_numba = kasai_numba(word, sarray, inverse_array_np(sarray))
    print(lcp_array_numba)
    overlap = 2
    overlap_array = np.argwhere(lcp_array_numba >= overlap).flatten()
    print(overlap_array)
    runs_array = get_runs(overlap_array)
    min_count = 0
    for start_inx, stop_inx in runs_array:
        if stop_inx - start_inx > min_count:
            min_overlap = lcp_array_numba[start_inx:stop_inx].min()
            print("min overlap", min_overlap)
            for i in range(start_inx, stop_inx + 1):
                word_start_inx = inverse_array_np(sarray)[i]
                word_stop_inx = word_start_inx + min_overlap
                print(i, word[word_start_inx:word_stop_inx])
            print()
    # exit()

    # print(suffix_array_best([2,1,3,1,3,1]))
    # print(suffix_array_best(np.array([2,1,3,1,3,1])))
    # for n_seqs in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
    # for n_seqs in [10_000_000, 20_000_000, 30_000_000, 40_000_000]:
    for n_seqs in [20_000_000]:
        seq_length = 10
        inp = np.random.randint(1000, size=seq_length*n_seqs)

        # start = time.time()
        # # sa = suffix_array_best(inp)
        # print("suffix_array_best", seq_length*n_seqs, time.time() - start)

        start = time.time()
        sa_np = suffix_array_np(inp)
        print("suffix_array_np", seq_length*n_seqs, time.time() - start)

        # start = time.time()
        # sa_opt = suffix_array_optimized(inp)
        # print("suffix_array_optimized", seq_length*n_seqs, time.time() - start)

        # npt.assert_almost_equal(np.array(sa), sa_np)
        # npt.assert_almost_equal(np.array(sa), sa_opt)

        start = time.time()
        lcp_array = kasai_numba(inp, sa_np)
        print("lcp", time.time() - start)

        start = time.time()
        overlap = 2
        overlapping_indices = np.argwhere(lcp_array > overlap).flatten()
        runs_array = get_runs(overlapping_indices)
        print("index start/stop", time.time() - start)

