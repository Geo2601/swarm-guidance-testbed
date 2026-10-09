"""Законы наведения. Каждый возвращает «наводящее» ускорение u_guid для всех аппаратов;
модуль Boids (если включён) добавляется к нему одинаково для всех законов,
затем сумма ограничивается a_max. Это и делает сравнение абляционным:
меняется только наводящая часть, локальный модуль один и тот же.
"""
import numpy as np


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _vel_tracking(v_des, v, cfg):
    """Ускорение, переводящее текущую скорость в желаемую за один шаг."""
    return (v_des - v) / cfg.dt


# ---------- GWO как закон наведения (по [1]) ----------

def gwo(p, v, q, vq, k, rng, cfg):
    n, d = p.shape
    a = 2.0 * (1.0 - k / cfg.gwo_k_max)
    if cfg.gwo_a_clamp:
        a = max(a, 0.0)
    dist = np.linalg.norm(p - q, axis=1)
    order = np.argsort(dist)

    if cfg.gwo_leaders == "centroids":
        # Статья, с. 5 (шаг 4) и с. 7 (п. 4.3): «добыча» — центроиды целей c_1, c_2, c_3;
        # при одной цели они совпадают с её положением.
        leaders = [q, q, q]
    elif cfg.gwo_leaders == "agents":
        leaders = [p[order[0]], p[order[1]], p[order[2 % n]]]
    else:
        raise ValueError(cfg.gwo_leaders)

    x_new = np.zeros_like(p)
    for XL in leaders:
        r1 = rng.random((n, d))
        r2 = rng.random((n, d))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * XL[None, :] - p)
        x_new += XL[None, :] - A * D
    x_gwo = x_new / 3.0

    w_g = cfg.gwo_weight_scale * a
    return w_g * (x_gwo - p)


# ---------- Классические законы ----------

def pursuit(p, v, q, vq, k, rng, cfg):
    """Метод погони (прямое наведение): скорость на текущее положение цели."""
    v_des = cfg.v_max * _unit(q[None, :] - p)
    return _vel_tracking(v_des, v, cfg)


def intercept_point(p, v, q, vq, cfg):
    """Желаемая скорость на упреждённую точку встречи (|v| = v_max).
    Если встреча невозможна — None для этого аппарата."""
    r = q[None, :] - p
    V = cfg.v_max
    a = vq @ vq - V * V
    b = 2 * (r @ vq)
    c = (r * r).sum(axis=1)
    v_des = np.empty_like(p)
    ok = np.ones(p.shape[0], dtype=bool)
    for i in range(p.shape[0]):
        if abs(a) < 1e-9:
            t = -c[i] / b[i] if b[i] < 0 else np.inf
        else:
            disc = b[i] * b[i] - 4 * a * c[i]
            if disc < 0:
                ok[i] = False
                continue
            sq = np.sqrt(disc)
            ts = [t for t in ((-b[i] - sq) / (2 * a), (-b[i] + sq) / (2 * a)) if t > 1e-9]
            t = min(ts) if ts else np.inf
        if not np.isfinite(t):
            ok[i] = False
            continue
        v_des[i] = (r[i] + vq * t) / t
    return v_des, ok


def parallel(p, v, q, vq, k, rng, cfg):
    """Параллельное сближение: курс на точку встречи (линия визирования сохраняет направление).
    Если встреча невозможна — откат к методу погони."""
    v_des, ok = intercept_point(p, v, q, vq, cfg)
    fallback = cfg.v_max * _unit(q[None, :] - p)
    v_des = np.where(ok[:, None], v_des, fallback)
    return _vel_tracking(v_des, v, cfg)


def prop_nav(p, v, q, vq, k, rng, cfg):
    """Пропорциональная навигация (2D): боковое ускорение N * Vc * dλ/dt, перпендикулярно скорости,
    плюс удержание скорости v_max вдоль текущего вектора скорости.
    Пока аппарат почти неподвижен, разгоняется методом погони."""
    r = q[None, :] - p
    rdot = vq[None, :] - v
    rn2 = np.maximum((r * r).sum(axis=1), 1e-9)
    lam_dot = (r[:, 0] * rdot[:, 1] - r[:, 1] * rdot[:, 0]) / rn2
    vc = -(r * rdot).sum(axis=1) / np.sqrt(rn2)
    speed = np.linalg.norm(v, axis=1)
    t_hat = _unit(v)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    a_lat = (cfg.pn_gain * vc * lam_dot)[:, None] * n_hat
    a_tan = ((cfg.v_max - speed) / cfg.dt)[:, None] * t_hat
    u = a_lat + a_tan
    slow = speed < 1.0
    if slow.any():
        u[slow] = pursuit(p[slow], v[slow], q, vq, k, rng, cfg)
    return u


LAWS = {
    "gwo": gwo,
    "pursuit": pursuit,
    "parallel": parallel,
    "pn": prop_nav,
}
