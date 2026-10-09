"""Модуль Boids — формулы (6) работы [1]:
  разделение  F^S_i = -sum_{j in N_i} f_s(r_ij) * (p_j - p_i) / r_ij
  выравнивание F^A_i = k_a * (mean_{j in N_i} v_j - v_i)
  сплочение   F^C_i = k_c * (mean_{j in N_i} p_j - p_i)
Соседство N_i — аппараты ближе neighbor_radius (радиус обзора — параметр стенда).
Разделение по шагу 5 алгоритма (с. 7) действует только при r_ij < d_min.
"""
import numpy as np


def boids_accel(p, v, cfg):
    n = p.shape[0]
    diff = p[None, :, :] - p[:, None, :]          # diff[i, j] = p_j - p_i
    r = np.linalg.norm(diff, axis=-1)              # r[i, j]
    mask = (r > 0) & (r < cfg.neighbor_radius)     # соседи (без себя)
    cnt = mask.sum(axis=1)
    has = cnt > 0
    u = np.zeros_like(p)

    # Разделение
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = np.where(r[..., None] > 0, diff / r[..., None], 0.0)
    with np.errstate(divide="ignore"):
        inv = np.where(r > 0, cfg.k_sep / r, 0.0)
    if cfg.sep_law == "paper":
        # Статья, с. 7, шаг 5: F_S = −k_s Σ_{j: r_ij < d_min} (p_j − p_i) / r_ij²
        sep_mask = (r > 0) & (r < cfg.d_min)
    elif cfg.sep_law == "inverse":
        # Статья, с. 6: f_s = k_s / r_ij по всем соседям N_i
        rs = cfg.neighbor_radius if cfg.sep_radius is None else cfg.sep_radius
        sep_mask = (r > 0) & (r < rs)
    else:
        raise ValueError(cfg.sep_law)
    u += -(np.where(sep_mask, inv, 0.0)[..., None] * unit).sum(axis=1)

    # Выравнивание и сплочение
    m = mask[..., None]
    v_mean = np.where(has[:, None], (m * v[None, :, :]).sum(axis=1) / np.maximum(cnt, 1)[:, None], v)
    p_mean = np.where(has[:, None], (m * p[None, :, :]).sum(axis=1) / np.maximum(cnt, 1)[:, None], p)
    u += cfg.k_ali * (v_mean - v)
    u += cfg.k_coh * (p_mean - p)
    return u


def post_separate(p, dist, iters=50):
    """Геометрическое разведение: пары ближе dist раздвигаются вдоль линии, соединяющей их, поровну.
    Не физика: положение меняется без ускорения. Повторяется, пока все пары не дальше dist или iters."""
    p = p.copy()
    n = p.shape[0]
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = p[j] - p[i]
                r = float(np.hypot(*d))
                if r < dist:
                    e = d / r if r > 1e-9 else np.array([1.0, 0.0])
                    s = 0.5 * (dist - r)
                    p[i] -= s * e
                    p[j] += s * e
                    moved = True
        if not moved:
            break
    return p
