"""Одно испытание и серия испытаний.

Кинематика: ||u|| <= a_max, ||v|| <= v_max; порядок обновления — cfg.integrator (см. config.py).
Захват: первый шаг, на котором хотя бы один аппарат ближе r_capture к цели.
Нарушение безопасности («столкновение»): на каком-либо шаге D_min(k) < d_min (с. 4, условие (4)).
  collision     — включая шаг захвата;
  collision_pre — только шаги до захвата (считаются оба варианта).
Испытание длится не более t_max = k_max = 30 итераций (с. 8, шаг 7.2).
"""
import numpy as np
import pandas as pd

from .boids import boids_accel, post_separate
from .guidance import LAWS
from .scenario import make_initial


def _clip_norm(x, lim):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > lim, x * (lim / np.maximum(n, 1e-12)), x)


def _min_pair(p):
    if p.shape[0] < 2:
        return np.inf
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(p.shape[0], 1)
    return d[iu].min()


def seg_min_pair(p0, p1):
    """Минимальная дистанция между парами при линейном движении от p0 к p1 внутри шага (между отсчётами)."""
    n = p0.shape[0]
    if n < 2:
        return np.inf
    i, j = np.triu_indices(n, 1)
    a = p0[j] - p0[i]
    b = (p1[j] - p1[i]) - a
    bb = (b * b).sum(1)
    t = np.where(bb > 1e-12, np.clip(-(a * b).sum(1) / np.maximum(bb, 1e-12), 0, 1), 0.0)
    return float(np.linalg.norm(a + t[:, None] * b, axis=1).min())


def run_trial(cfg, law, use_boids, seed, record=False):
    """record=True — дополнительно вернуть траектории: P[k, i, :] аппаратов и Q[k, :] цели."""
    p, v, q, vq, rng = make_initial(cfg, seed)
    P, Q = [p.copy()], [q.copy()]
    guide = LAWS[law]
    min_pair = _min_pair(p)
    min_pair_pre = min_pair          # без учёта шага, на котором произошёл захват
    min_seg = min_pair               # с учётом движения между отсчётами
    last = min_pair
    for k in range(cfg.t_max + 1):
        if np.linalg.norm(p - q, axis=1).min() <= cfg.r_capture:
            out = dict(captured=True, t_cap=k, min_pair=min_pair,
                       collision=min_pair < cfg.d_min,
                       collision_pre=min_pair_pre < cfg.d_min,
                       collision_cont=min_seg < cfg.d_min)
            if record:
                out.update(P=np.array(P), Q=np.array(Q))
            return out
        if k == cfg.t_max:
            break
        u = guide(p, v, q, vq, k, rng, cfg)
        if use_boids:
            u = u + boids_accel(p, v, cfg)
        u = _clip_norm(u, cfg.a_max)
        p_prev = p
        if cfg.integrator == "semi_implicit":
            v = _clip_norm(v + cfg.dt * u, cfg.v_max)
            p = p + cfg.dt * v
        elif cfg.integrator == "explicit":
            p = p + cfg.dt * v
            v = _clip_norm(v + cfg.dt * u, cfg.v_max)
        else:
            raise ValueError(cfg.integrator)
        if use_boids and cfg.post_separate:
            p = post_separate(p, cfg.post_separate)
        min_seg = min(min_seg, seg_min_pair(p_prev, p))
        q = q + cfg.dt * vq
        if record:
            P.append(p.copy())
            Q.append(q.copy())
        min_pair_pre = min(min_pair_pre, last)
        last = _min_pair(p)
        min_pair = min(min_pair, last)
    min_pair_pre = min(min_pair_pre, last)
    out = dict(captured=False, t_cap=np.nan, min_pair=min_pair,
               collision=min_pair < cfg.d_min,
               collision_pre=min_pair_pre < cfg.d_min,
               collision_cont=min_seg < cfg.d_min)
    if record:
        out.update(P=np.array(P), Q=np.array(Q))
    return out


def run_series(cfg, law, use_boids, seeds):
    rows = []
    for s in seeds:
        r = run_trial(cfg, law, use_boids, s)
        r["seed"] = s
        rows.append(r)
    df = pd.DataFrame(rows)
    df["law"] = law
    df["boids"] = use_boids
    return df


def summarize(df):
    cap = df["captured"]
    t = df.loc[cap, "t_cap"]
    # «Цензурированная» медиана: незахваченные испытания засчитываются как t_max.
    tc = df["t_cap"].fillna(df.attrs.get("t_max", 30))
    return dict(
        median_censored=float(np.median(tc)),
        capture_pct=100.0 * cap.mean(),
        median=float(np.median(t)) if len(t) else np.nan,
        q25=float(np.percentile(t, 25)) if len(t) else np.nan,
        q75=float(np.percentile(t, 75)) if len(t) else np.nan,
        collisions_per_100=100.0 * df["collision"].mean(),
        # Нарушения строго до шага захвата: при R_cap = d_min = 5 м два аппарата,
        # одновременно вошедшие в зону захвата, почти неизбежно ближе d_min друг к другу.
        collisions_pre_per_100=100.0 * df["collision_pre"].mean(),
        # С учётом движения между отсчётами (линейная интерполяция внутри шага).
        collisions_cont_per_100=100.0 * df["collision_cont"].mean(),
        min_pair_median=float(df["min_pair"].median()),
        n=len(df),
    )
