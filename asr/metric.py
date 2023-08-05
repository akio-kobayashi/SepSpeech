import torch
import numpy as np

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n :
        ref, hyp = hyp, ref
        m, n = n, m

    distance = np.zeros((2, n+1), dtype=np.int32)

    for j in range(0, n+1):
        distance[0][j] = j

    for i in range(1, m+1):
        prev_row_idx = (i-1)%2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j-1]
            else:
                s_num = distance[prev_row_idx][j-1]+1
                i_num = distance[cur_row_idx][j-1]+1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)
    return distance[m%2][n]

def cer(ref, hyp, max_ignore_index=7):
    removed = []
    prv_id=-1
    for h in hyp:
        if prv_id == id:
            continue
        prv_id = id
        if id <= max_ignore_index:
            continue
        removed.append(h)

    edit_distance = _levenshtein_distance(ref, removed)

    if len(ref) == 0:
        raise ValueError("Length of reference should be greater than 0")
    cer = float(edit_distance)/len(ref)
    return cer

def counts(ref, hyp):
    edit_distance = _levenshtein_distance(ref, hyp)

    if len(ref) == 0:
        raise ValueError("Length of reference should be greater than 0")
    return edit_distance, len(ref)
