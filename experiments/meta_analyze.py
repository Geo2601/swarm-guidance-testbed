"""Анализ E1 (избыточность) и E2 (сдвиг начала координат).

E1: для каждого закона и скорости цели по N = 1 оценивается F_1(k) — доля захватов к итерации k. Прогноз
    F_N(k) = 1 − (1 − F_1(k))^N. Сравниваются: доля захвата к k_max (наблюдение с ДИ Уилсона против прогноза с
    бутстреп-ДИ по N = 1) и вся кривая (максимум расхождения CDF, KS-статистика).
E2: доля захвата при сдвиге s против s = 0 на тех же сидах; точный критерий Мак-Немара, поправка Холма.

Запуск:  python -m experiments.meta_analyze
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

OUT = os.environ.get("META_OUT", "results/meta")
KMAX = 30


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def cdf(t):
    t = np.asarray(t, float)
    return np.array([(t <= k).mean() for k in range(KMAX + 1)])


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j])
        adj[j] = min(run, 1.0)
    return adj


def e1():
    df = pd.read_csv(f"{OUT}/redundancy_trials.csv.gz")
    df["t"] = df["t_cap"].fillna(np.inf)
    rng = np.random.default_rng(0)
    rows = []
    for (law, vt), g in df.groupby(["law", "vt"]):
        t1 = g[(g.n == 1)]["t"].to_numpy()
        F1 = cdf(t1)
        boot = rng.integers(0, len(t1), (2000, len(t1)))
        p1b = (t1[boot] <= KMAX).mean(axis=1)
        for (n, b), h in g.groupby(["n", "boids"]):
            obs_t = h["t"].to_numpy()
            Fo = cdf(obs_t)
            Fp = 1 - (1 - F1) ** n
            pb = 1 - (1 - p1b) ** n
            k_obs = int((obs_t <= KMAX).sum())
            lo, hi = wilson(k_obs, len(obs_t))
            fin = obs_t[np.isfinite(obs_t)]
            # медиана прогноза: первая итерация, где F_N ≥ 0,5 от всех испытаний
            kmed_p = next((k for k in range(KMAX + 1) if Fp[k] >= 0.5), np.nan)
            kmed_o = next((k for k in range(KMAX + 1) if Fo[k] >= 0.5), np.nan)
            rows.append(dict(law=law, vt=vt, n=n, boids=b, p1=100 * F1[KMAX],
                             pred=100 * Fp[KMAX], pred_lo=100 * np.percentile(pb, 2.5), pred_hi=100 * np.percentile(pb, 97.5),
                             obs=100 * Fo[KMAX], obs_lo=lo, obs_hi=hi,
                             diff=100 * (Fo[KMAX] - Fp[KMAX]), ks=np.abs(Fo - Fp).max(),
                             med_pred=kmed_p, med_obs=kmed_o,
                             viol=100 * h["collision"].mean()))
    s = pd.DataFrame(rows)
    s.to_csv(f"{OUT}/redundancy_summary.csv", index=False)
    return s


def e2():
    df = pd.read_csv(f"{OUT}/shift_trials.csv.gz")
    rows = []
    for (law, n, vt), g in df.groupby(["law", "n", "vt"]):
        base = g[g["shift"] == 0].set_index("seed")
        for sh, h in g.groupby("shift"):
            h = h.set_index("seed").loc[base.index]
            a = base["captured"].astype(bool).to_numpy()
            c = h["captured"].astype(bool).to_numpy()
            b01, b10 = int((a & ~c).sum()), int((~a & c).sum())
            p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
            same = float((a == c).mean())
            t_same = float((base["t_cap"].fillna(-1).to_numpy() == h["t_cap"].fillna(-1).to_numpy()).mean())
            rows.append(dict(law=law, n=n, vt=vt, shift=sh, cap=100 * c.mean(), cap0=100 * a.mean(),
                             only_base=b01, only_shift=b10, p=p, identical_outcome=100 * same,
                             identical_time=100 * t_same, viol=100 * h["collision"].mean()))
    s = pd.DataFrame(rows)
    m = s["shift"] > 0
    s.loc[m, "p_holm"] = holm(s.loc[m, "p"])
    s.to_csv(f"{OUT}/shift_summary.csv", index=False)
    return s


if __name__ == "__main__":
    import os
    pd.set_option("display.width", 200)
    if os.path.exists(f"{OUT}/redundancy_trials.csv.gz"):
        s = e1()
        print(s.round(1).to_string(index=False))
    if os.path.exists(f"{OUT}/shift_trials.csv.gz"):
        s = e2()
        print(s.round(3).to_string(index=False))
