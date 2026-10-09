"""Анализ расчёта 3 (чувствительность) и расчёта 4 (HHO).

Для каждой конфигурации (блок, Δt, K, R, профиль Boids, скорость цели):
  избыточность — F_1(k) по N = 1, прогноз F_N = 1 − (1 − F_1)^N, наблюдение с ДИ Уилсона, бутстреп-ДИ прогноза,
                 KS-расхождение кривых; в блоке safe N = 1 не прогоняется, F_1 берётся из блока rk (R = 5, K = 30,
                 без Boids — одиночному аппарату Boids не нужен);
  сдвиг        — доля захвата при сдвиге s против s = 0 на тех же сидах, точный критерий Мак-Немара, Холм.
Результат: results/meta/sens_redundancy.csv, sens_shift.csv, hho_redundancy.csv, hho_shift.csv.
Запуск:  python -m experiments.meta_sens_analyze
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from experiments.meta_analyze import holm, wilson

OUT = os.environ.get("META_OUT", "results/meta")


def cdf(t, K):
    t = np.asarray(t, float)
    return np.array([(t <= k).mean() for k in range(K + 1)])


def redundancy(g1, groups, K, keys, rng):
    """g1 — испытания N = 1; groups — iterable of ((n, boids), DataFrame)."""
    t1 = g1["t_cap"].fillna(np.inf).to_numpy()
    F1 = cdf(t1, K)
    boot = rng.integers(0, len(t1), (2000, len(t1)))
    p1b = (t1[boot] <= K).mean(axis=1)
    rows = []
    for (n, b), h in groups:
        if n == 1:
            continue
        to = h["t_cap"].fillna(np.inf).to_numpy()
        Fo, Fp = cdf(to, K), 1 - (1 - F1) ** n
        pb = 1 - (1 - p1b) ** n
        lo, hi = wilson(int((to <= K).sum()), len(to))
        rows.append(dict(**keys, n=n, boids=b, p1=100 * F1[K], pred=100 * Fp[K],
                         pred_lo=100 * np.percentile(pb, 2.5), pred_hi=100 * np.percentile(pb, 97.5),
                         obs=100 * Fo[K], obs_lo=lo, obs_hi=hi, diff=100 * (Fo[K] - Fp[K]),
                         ks=float(np.abs(Fo - Fp).max()), viol=100 * h["collision"].mean()))
    return rows


def shift_rows(g, keys):
    rows = []
    base = g[g["shift"] == 0].set_index("seed")
    for sh, h in g.groupby("shift"):
        h = h.set_index("seed").loc[base.index]
        a = base["captured"].astype(bool).to_numpy()
        c = h["captured"].astype(bool).to_numpy()
        b01, b10 = int((a & ~c).sum()), int((~a & c).sum())
        p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        rows.append(dict(**keys, shift=sh, cap0=100 * a.mean(), cap=100 * c.mean(), only_base=b01, only_shift=b10,
                         p=p, identical=100 * float((a == c).mean())))
    return rows


def sens():
    df = pd.read_csv(f"{OUT}/sens_trials.csv.gz")
    rng = np.random.default_rng(0)
    red, shf = [], []
    # dt: избыточность (сдвиг 0, без Boids и с Boids [1]) и сдвиг
    d = df[df.block == "dt"]
    for (law, dt), g in d.groupby(["law", "dt"]):
        K = int(g.K.iloc[0])
        g0 = g[g["shift"] == 0]
        red += redundancy(g0[(g0.n == 1)], g0.groupby(["n", "boids"]), K,
                          dict(block="dt", law=law, dt=dt, K=K, R=5.0, prof="paper", vt=0.0), rng)
        for (n, b), h in g.groupby(["n", "boids"]):
            if h["shift"].nunique() > 1:
                shf += shift_rows(h, dict(block="dt", law=law, dt=dt, K=K, prof="paper", n=n, boids=b, vt=0.0))
    # rk
    d = df[df.block == "rk"]
    for (law, R, K, vt), g in d.groupby(["law", "R", "K", "vt"]):
        red += redundancy(g[g.n == 1], g.groupby(["n", "boids"]), int(K),
                          dict(block="rk", law=law, dt=1.0, K=int(K), R=R, prof="paper", vt=vt), rng)
    # safe: F_1 — из rk (R = 5, K = 30)
    d = df[df.block == "safe"]
    rk = df[(df.block == "rk") & (df.R == 5.0) & (df.K == 30) & (df.n == 1)]
    for (law, vt), g in d.groupby(["law", "vt"]):
        g0 = g[g["shift"] == 0]
        if law == "gwo_rel":  # в блоке rk нет gwo_rel с N = 1 — прогноз не строится
            continue
        g1 = rk[(rk.law == law) & (rk.vt == vt)]
        red += redundancy(g1, g0.groupby(["n", "boids"]), 30,
                          dict(block="safe", law=law, dt=1.0, K=30, R=5.0, prof="paper_safe", vt=vt), rng)
        h = g[g.n == 4]
        if h["shift"].nunique() > 1:
            shf += shift_rows(h, dict(block="safe", law=law, dt=1.0, K=30, prof="paper_safe", n=4, boids=True, vt=vt))
    r, s = pd.DataFrame(red), pd.DataFrame(shf)
    m = s["shift"] > 0
    s.loc[m, "p_holm"] = holm(s.loc[m, "p"])
    r.to_csv(f"{OUT}/sens_redundancy.csv", index=False)
    s.to_csv(f"{OUT}/sens_shift.csv", index=False)
    return r, s


def hho():
    rng = np.random.default_rng(0)
    d = pd.read_csv(f"{OUT}/hho_redundancy_trials.csv.gz")
    red = []
    for (law, vt), g in d.groupby(["law", "vt"]):
        red += redundancy(g[g.n == 1], g.groupby(["n", "boids"]), 30, dict(law=law, vt=vt), rng)
    s = pd.read_csv(f"{OUT}/hho_shift_trials.csv.gz")
    shf = []
    for (law, n, vt), g in s.groupby(["law", "n", "vt"]):
        shf += shift_rows(g, dict(law=law, n=n, vt=vt))
    r, s = pd.DataFrame(red), pd.DataFrame(shf)
    m = s["shift"] > 0
    s.loc[m, "p_holm"] = holm(s.loc[m, "p"])
    r.to_csv(f"{OUT}/hho_redundancy.csv", index=False)
    s.to_csv(f"{OUT}/hho_shift.csv", index=False)
    return r, s


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    if os.path.exists(f"{OUT}/sens_trials.csv.gz"):
        r, s = sens()
        print(r.round(1).to_string(index=False))
        print(s.round(3).to_string(index=False))
    if os.path.exists(f"{OUT}/hho_shift_trials.csv.gz"):
        r, s = hho()
        print(r.round(1).to_string(index=False))
        print(s.round(3).to_string(index=False))
