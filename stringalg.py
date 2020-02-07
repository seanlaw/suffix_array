from collections import defaultdict
from itertools import zip_longest, islice


def to_int_keys(l):
    """
    l: iterable of keys
    returns: a list with integer keys
    """
    seen = set()
    ls = []
    for e in l:
        if not e in seen:
            ls.append(e)
            seen.add(e)
    ls.sort()
    index = {v: i for i, v in enumerate(ls)}
    return [index[v] for v in l]


def suffix_matrix(s):
    """
    suffix matrix of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys(s)
    ans = [line]
    while max(line) < n - 1:
        line = to_int_keys(
            [a * (n + 1) + b + 1
             for (a, b) in
             zip_longest(line, islice(line, k, None),
                         fillvalue=-1)])
        ans.append(line)
        k <<= 1
    return ans


def suffix_array(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys(s)
    while max(line) < n - 1:
        line = to_int_keys(
            [a * (n + 1) + b + 1
             for (a, b) in
             zip_longest(line, islice(line, k, None),
                         fillvalue=-1)])
        k <<= 1
    return line


def lcp(sm, i, j):
    """
    longest common prefix
    O(log(n))

    sm: suffix matrix
    """
    n = len(sm[-1])
    if i == j:
        return n - i
    k = 1 << (len(sm) - 2)
    ans = 0
    for line in sm[-2::-1]:
        if i >= n or j >= n:
            break
        if line[i] == line[j]:
            ans ^= k
            i += k
            j += k
        k >>= 1
    return ans


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


def BWT(s, sa=None):
    if sa is None:
        sa = suffix_array(s)
    pos = inverse_array(sa)
    return [s[-1]] + [s[pos[i] - 1] if pos[i] > 0 else '' for i in range(len(s))]


def iBWT(b):
    n = len(b)
    last = list(zip(b, range(n)))
    ilast = {}
    for i, e in enumerate(last):
        ilast[e] = i
    first = sorted(last)
    ans = []
    c = first[0]
    while len(ans) < n:
        ans.append(c[0])
        c = first[ilast[c]]
    return ''.join(ans)
        
    


# TODO: suffix tree, burrows-wheeler matching, kmp
