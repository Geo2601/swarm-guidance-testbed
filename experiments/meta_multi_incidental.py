"""Расчёт 2 (проверка механизма). Захваты «попутно»: аппарат, назначенный на одну цель, пролетает в R_cap от другой.

Волновая модель (meta_multi_model) не учитывает таких захватов и при плотной группе целей (разброс 25 м, M ≥ 3)
занижает долю полного захвата на 9–26 п. п. Проверка: тот же стенд gwo_assign без Boids, но цель засчитывается только
аппарату, назначенному на неё на предыдущем шаге (assigned_only=True). Если гипотеза верна, при этом правиле стенд
должен сблизиться с волновой моделью. Контроль: assigned_only=False обязан совпасть с multi_sim1000 испытание в
испытание (тот же код, тот же генератор).
Результат: results/meta/multi_incidental.csv.
Запуск:  python -m experiments.meta_multi_incidental --jobs 2
"""
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from swarmsim.config import Config
from swarmsim.multi import _clip_norm, assign_sum, gwo_point, intercept_time, make_targets

OUT = "results/meta"
K = 30


def trial(cfg, seed, m, spread, assigned_only):
    p, v, q, vq, rng = make_targets(cfg, seed, m, spread, 10.0)
    n = p.shape[0]
    T_m = np.full(m, -1)
    g_prev = None
    incidental = 0
    for k in range(cfg.t_max + 1):
        dist = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
        near = dist <= cfg.r_capture                                   # N×M
        for j in np.where(T_m < 0)[0]:
            hitters = np.where(near[:, j])[0]
            if not len(hitters):
                continue
            own = g_prev is not None and np.any(g_prev[hitters] == j)
            if assigned_only and g_prev is not None and not own:
                continue
            if g_prev is not None and not own:
                incidental += 1
            T_m[j] = k
        if (T_m >= 0).all() or k == cfg.t_max:
            break
        alive = np.where(T_m < 0)[0]
        qa, va = q[alive], vq[alive]
        cost = intercept_time(p, qa, va, cfg.v_max)
        g = assign_sum(cost)
        g_prev = alive[g]
        leaders = np.repeat(qa[g][:, None, :], 3, axis=1)
        x, w = gwo_point(p, leaders, k, rng, cfg)
        u = _clip_norm(w * (x - p), cfg.a_max)
        p = p + cfg.dt * v
        v = _clip_norm(v + cfg.dt * u, cfg.v_max)
        q = q + cfg.dt * vq
    return bool((T_m >= 0).all()), incidental


def cell(args):
    n, m, spread, ao, seeds = args
    cfg = Config(n_agents=n, target_speed=0.0, t_max=K, gwo_k_max=K)
    return [(n, m, spread, ao, s, *trial(cfg, s, m, spread, ao)) for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1000)
    a = ap.parse_args()
    cells = [(n, m, sp, ao, range(a.seeds)) for n, m, sp, ao in
             itertools.product([4, 8], [2, 3, 5], [25.0, 60.0], [False, True])]
    with Pool(a.jobs) as pool:
        rows = [r for rr in pool.map(cell, cells, chunksize=1) for r in rr]
    df = pd.DataFrame(rows, columns=["n", "m", "spread", "assigned_only", "seed", "all_captured", "incidental"])
    ref = pd.read_csv(f"{OUT}/multi_sim1000.csv.gz")
    mod = pd.read_csv(f"{OUT}/multi_model.csv")
    out = []
    for (n, m, sp), g in df.groupby(["n", "m", "spread"]):
        base = g[~g.assigned_only].set_index("seed")
        r = ref[(ref.n == n) & (ref.m == m) & (ref.spread == sp)].set_index("seed")
        same = float((base.all_captured.astype(bool) == r.all_captured.astype(bool).loc[base.index]).mean() * 100)
        out.append(dict(n=n, m=m, spread=sp, sim=100 * base.all_captured.mean(),
                        incidental_per_trial=base.incidental.mean(),
                        sim_assigned_only=100 * g[g.assigned_only].all_captured.mean(),
                        wave=mod[(mod.n == n) & (mod.m == m) & (mod.spread == sp)].wave.iloc[0],
                        same_as_stand=same))
    s = pd.DataFrame(out)
    s.to_csv(f"{OUT}/multi_incidental.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
