"""Slot ↔ GT 对象最优分配（Hungarian / Kuhn-Munkres）。

训练时预测的是一组无序 slot，但辅助损失需要把"第 k 个预测 slot"对应到
"第 g 个 GT 对象"。由于可微渲染损失在图像层面是置换不变的，固定按 slot
下标对齐 GT（targets.py 里按面积排序）会与渲染损失冲突，导致 slot 定位
坍缩。用 Hungarian 在每一步做最优一一匹配即可消除该冲突。

只支持方阵（NUM_SLOTS × NUM_SLOTS），规模极小（8×8），O(n^3) 够用。
"""
from __future__ import annotations

import numpy as np


def hungarian(cost: np.ndarray):
    """返回 row_assign: len=n，row_assign[i] = 与第 i 行匹配的列下标。

    cost 为 (n, n) numpy 数组，最小化总代价。标准 O(n^3) 实现。
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n == m, "Hungarian 当前仅支持方阵"
    INF = 1e18
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)      # p[j] = 匹配到列 j 的行（1-indexed）
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    row_assign = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            row_assign[p[j] - 1] = j - 1
    return row_assign


def match_slots(cost: np.ndarray):
    """cost: (n_pred, n_gt) → 返回 list，assign[g] = 匹配到的预测 slot 下标。

    用于把每个 GT 对象 g 对应到一个预测 slot k。当 pred 多于 gt 时，多余
    的 pred 不进入 assign（训练时会被推向 invalid）。
    """
    n_pred, n_gt = cost.shape
    if n_pred <= n_gt:
        # 行少：补齐成方阵（虚拟代价列）
        padded = np.full((n_gt, n_gt), 1e9)
        padded[:n_pred, :n_gt] = cost
        ra = hungarian(padded)
        assign = [-1] * n_gt
        for col in range(n_gt):
            row = ra[col]
            if row < n_pred:
                assign[col] = int(row)
        return assign
    else:
        padded = np.full((n_pred, n_pred), 1e9)
        padded[:n_pred, :n_gt] = cost
        ra = hungarian(padded)
        assign = [-1] * n_gt
        for col in range(n_gt):
            row = ra[col]
            if row < n_pred:
                assign[col] = int(row)
        return assign


if __name__ == "__main__":
    # 自检：已知最优解
    c = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    ra = hungarian(c)
    print("row_assign", ra, "total", sum(c[i, ra[i]] for i in range(3)))
    assert ra.tolist() == [1, 0, 2], ra
    print("Hungarian self-test OK")
