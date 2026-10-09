"""Расчёт 2. Модель нескольких целей с перераспределением после захвата («волновая» модель без кооперации).

Статичный прогноз E4, ∏_m [1 − (1 − p₁)^{n_m}], занижает долю полного захвата в 2–7 раз: после захвата цели
освободившиеся аппараты переназначаются на оставшиеся цели и получают новую попытку. Модель ниже учитывает это,
но по-прежнему не содержит кооперации: каждая попытка — независимый исход одиночного аппарата.

Исходы попыток берутся из двух пулов одиночных прогонов (не из многоцелевых испытаний):
  первая попытка — аппарат стартует из покоя, как в многоцелевом сценарии (та же расстановка аппаратов и целей,
                   make_targets), цель — одна из целей расстановки; фиксируются дальность d, захват, итерация захвата;
  повторная      — аппарат у только что захваченной цели (в круге R_cap), скорость 7–10 м/с в случайном
                   направлении, итерация k₀ ∈ [8, 29]; новая цель — другая цель той же расстановки на расстоянии d.
Модель испытания (событийная):
  1) начальное назначение — как в стенде (сбалансированный венгерский метод по времени встречи);
  2) каждый аппарат получает исход первой попытки из пула с ближайшей дальностью (|Δd| ≤ 5 м);
  3) цель захвачена в момент самой ранней успешной попытки назначенных на неё аппаратов; эти аппараты свободны;
  4) свободные аппараты распределяются по оставшимся целям (к цели с наименьшим числом назначенных, при равенстве —
     к ближайшей) и получают исход повторной попытки из пула (|Δd| ≤ 5 м, тот же k₀);
  5) испытание успешно, если все цели захвачены к K.
Сравнение: стенд, gwo_assign без Boids, неподвижные цели, K = 30, N ∈ {4, 8}, M ∈ {1, 2, 3, 5}, разброс 25 и 60 м,
1000 испытаний на ячейку (пересчитано, раньше было 100).
Результат: results/meta/multi_model.csv, multi_pool_first.csv.gz, multi_pool_redirect.csv.gz, multi_sim1000.csv.gz.
Запуск:  python -m experiments.meta_multi_model --jobs 2
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401
from swarmsim.config import Config
from swarmsim.guidance import LAWS
from swarmsim.multi import assign_sum, intercept_time, make_targets, run_trial_multi

OUT = "results/meta"
K = 30


def _clip(x, lim):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > lim, x * (lim / np.maximum(n, 1e-12)), x)


def run_single(p, v, q, k0, cfg, rng):
    """Один аппарат, неподвижная цель q, закон gwo, итерации k0…K. Возвращает (захват, итерация захвата)."""
    p, v = p[None, :].copy(), v[None, :].copy()
    law = LAWS["gwo"]
    zero = np.zeros(2)
    for k in range(k0, cfg.t_max + 1):
        if np.linalg.norm(p[0] - q) <= cfg.r_capture:
            return True, k
        if k == cfg.t_max:
            break
        u = _clip(law(p, v, q, zero, k, rng, cfg), cfg.a_max)
        p = p + cfg.dt * v
        v = _clip(v + cfg.dt * u, cfg.v_max)
    return False, -1


def pool_first(args):
    spread, seeds = args
    cfg = Config(n_agents=1, t_max=K, gwo_k_max=K)
    rows = []
    for s in seeds:
        p, v, q, vq, rng = make_targets(cfg, s, 5, spread, 10.0)
        j = s % 5
        c, t = run_single(p[0], v[0], q[j], 0, cfg, rng)
        rows.append((spread, float(np.linalg.norm(p[0] - q[j])), c, t))
    return rows


def pool_redirect(args):
    spread, seeds = args
    cfg = Config(n_agents=1, t_max=K, gwo_k_max=K)
    rows = []
    for s in seeds:
        _, _, q, _, rng = make_targets(cfg, s, 5, spread, 10.0)
        r2 = np.random.default_rng([s, 99])
        old, new = q[0], q[1 + s % 4]
        rad, ang = 5.0 * np.sqrt(r2.random()), 2 * np.pi * r2.random()
        p = old + rad * np.array([np.cos(ang), np.sin(ang)])
        sp, hd = r2.uniform(7.0, 10.0), 2 * np.pi * r2.random()
        v = sp * np.array([np.cos(hd), np.sin(hd)])
        k0 = int(r2.integers(8, 30))
        c, t = run_single(p, v, new, k0, cfg, rng)
        rows.append((spread, float(np.linalg.norm(new - old)), k0, c, t - k0 if c else -1))
    return rows


def sim_cell(args):
    n, m, spread, seeds = args
    cfg = Config(n_agents=n, target_speed=0.0, t_max=K, gwo_k_max=K)
    return [(n, m, spread, s, run_trial_multi(cfg, "gwo_assign", False, s, m=m, spread=spread)["all_captured"])
            for s in seeds]


def _draw_first(pf, d, rng):
    g = pf[np.abs(pf[:, 0] - d) <= 5.0]
    if len(g) < 20:
        g = pf[np.argsort(np.abs(pf[:, 0] - d))[:20]]
    r = g[rng.integers(len(g))]
    return r[2] if r[1] > 0 else np.inf


def _draw_redirect(pr, d, k0, rng):
    g = pr[(pr[:, 1] == k0) & (np.abs(pr[:, 0] - d) <= 5.0)]
    if len(g) < 10:
        g = pr[pr[:, 1] == k0]
        g = g[np.argsort(np.abs(g[:, 0] - d))[:10]]
    r = g[rng.integers(len(g))]
    return k0 + r[3] if r[2] > 0 else np.inf


def wave_trial(n, m, spread, seed, pf, pr, rng):
    cfg = Config(n_agents=n, t_max=K, gwo_k_max=K)
    p, v, q, vq, _ = make_targets(cfg, seed, m, spread, 10.0)
    g = assign_sum(intercept_time(p, q, vq, cfg.v_max))
    s = np.array([_draw_first(pf, float(np.linalg.norm(p[i] - q[g[i]])), rng) for i in range(n)])
    alive = np.ones(m, dtype=bool)
    while alive.any():
        tj = np.array([s[g == j].min() if alive[j] and (g == j).any() else np.inf for j in range(m)])
        j = int(np.argmin(tj))
        t = tj[j]
        if not np.isfinite(t) or t > K:
            return False
        alive[j] = False
        if not alive.any():
            return True
        free = np.where(g == j)[0]
        for i in free:
            cnt = np.array([(g == jj).sum() if alive[jj] else 10 ** 6 for jj in range(m)])
            dist = np.linalg.norm(q - q[j], axis=1)
            cand = np.where(cnt == cnt.min())[0]
            new = cand[np.argmin(dist[cand])]
            g[i] = new
            k0 = int(min(max(t, 8), 29))
            s[i] = _draw_redirect(pr, float(dist[new]), k0, rng) if t < K else np.inf
    return True


def static_pred(n, m, spread, seeds, p1):
    cfg = Config(n_agents=n, t_max=K, gwo_k_max=K)
    out = []
    for sd in seeds:
        p, v, q, vq, _ = make_targets(cfg, sd, m, spread, 10.0)
        g = assign_sum(intercept_time(p, q, vq, cfg.v_max))
        cnt = np.bincount(g, minlength=m)
        out.append(np.prod(1 - (1 - p1) ** cnt))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = [(n, m, sp) for n, m, sp in itertools.product([4, 8], [1, 2, 3, 5], [25.0, 60.0]) if not (m == 1 and sp == 60.0)]
    chunks = lambda N: [range(i, min(i + 1000, N)) for i in range(0, N, 1000)]
    with Pool(a.jobs) as pool:
        pf = [r for rows in pool.map(pool_first, [(sp, c) for sp in (25.0, 60.0) for c in chunks(a.pool // 2)])
              for r in rows]
        pr = [r for rows in pool.map(pool_redirect, [(sp, c) for sp in (25.0, 60.0) for c in chunks(a.pool * 2)])
              for r in rows]
        sim = [r for rows in pool.map(sim_cell, [(n, m, sp, range(a.seeds)) for n, m, sp in cells]) for r in rows]
    pf = pd.DataFrame(pf, columns=["spread", "d", "captured", "t"])
    pr = pd.DataFrame(pr, columns=["spread", "d", "k0", "captured", "dt"])
    sim = pd.DataFrame(sim, columns=["n", "m", "spread", "seed", "all_captured"])
    pf.to_csv(f"{OUT}/multi_pool_first.csv.gz", index=False)
    pr.to_csv(f"{OUT}/multi_pool_redirect.csv.gz", index=False)
    sim.to_csv(f"{OUT}/multi_sim1000.csv.gz", index=False)
    p1 = pd.read_csv(f"{OUT}/redundancy_summary.csv").query("law == 'gwo' and vt == 0").p1.iloc[0] / 100
    rng = np.random.default_rng(0)
    rows = []
    for n, m, sp in cells:
        PF = pf[pf.spread == sp][["d", "captured", "t"]].to_numpy(float)
        PR = pr[pr.spread == sp][["d", "k0", "captured", "dt"]].to_numpy(float)
        if m == 1:   # одна цель в начале координат: первая попытка — пул E1 (дальность 85–110 м)
            PF = PF[(PF[:, 0] >= 80) & (PF[:, 0] <= 115)]
        wav = np.mean([wave_trial(n, m, sp, s, PF, PR, rng) for s in range(4000)])
        obs = sim[(sim.n == n) & (sim.m == m) & (sim.spread == sp)].all_captured.mean()
        k_obs = int(round(obs * a.seeds))
        rows.append(dict(n=n, m=m, spread=sp, sim=100 * obs,
                         sim_lo=100 * (obs - 1.96 * np.sqrt(obs * (1 - obs) / a.seeds)),
                         sim_hi=100 * (obs + 1.96 * np.sqrt(obs * (1 - obs) / a.seeds)),
                         wave=100 * wav, static=100 * static_pred(n, m, sp, range(1000), p1)))
    df = pd.DataFrame(rows)
    df["wave_diff"] = df.wave - df.sim
    df.to_csv(f"{OUT}/multi_model.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    print("первая попытка: p =", round(100 * pf.captured.mean(), 1), "%; повторная:", round(100 * pr.captured.mean(), 1), "%")


if __name__ == "__main__":
    main()
