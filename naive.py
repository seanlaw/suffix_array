import numpy as np

def naive_suffix_array(inp):
    suffixes = [None] * inp.shape[0]
    for i in range(inp.shape[0]):
        suffixes[i] = {"index": i, "suffix": inp[i:].tolist()}

    suffixes = sorted(suffixes, key=lambda x: x["suffix"])
    suffix_array = np.array([x["index"] for x in suffixes])

    return suffix_array


def naive_lcp_array(inp, suffix_array):
    lcp_array = np.zeros(inp.shape[0], dtype=np.int)
    n = inp.shape[0] - 1
    for idx, (i, j) in enumerate(zip(suffix_array[:-1], suffix_array[1:])):
        max_k = min(inp.shape[0] - i, inp.shape[0] - j)
        overlap = 0
        for k in range(max_k):
            if inp[i + k] != inp[j + k]:
                break
            overlap += 1
        lcp_array[idx] = overlap

    return lcp_array


def naive_overlap_array(lcp_array):
    overlap_list = []
    for val in np.unique(lcp_array[lcp_array >= 2]):
        mask = lcp_array >= val

        start = []
        stop = []
        if mask[0]:
            start.append(0)
        for i in range(len(mask)):
            if mask[i] and not mask[i - 1]:
                start.append(i)
            if i > 0 and (not mask[i] and mask[i - 1]):
                stop.append(i)  # Exclusive stop index
        if mask[len(mask) - 1]:
            stop.append(len(mask))  # Exclusive stop index

        steps = [val] * len(start)
        overlap_list.extend(list(map(list, zip(steps, start, stop))))

    overlap_array = np.array(overlap_list, dtype=np.int64)

    return overlap_array
