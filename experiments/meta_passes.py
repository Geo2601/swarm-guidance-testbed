"""E3. Почему одиночный аппарат с метаэвристическим законом промахивается: разбор сближений.

Для N = 1 траектория относительного движения r_k = p_k − q_k разбивается на эпизоды сближения — локальные минимумы
непрерывной (линейно интерполированной внутри шага) дистанции. Для эпизода:
  b — промах (минимум интерполированной дистанции), L — относительное смещение за шаг около минимума,
  захват — хотя бы один из двух отсчётов, окружающих минимум, ближе R_cap.
Модель «окна дискретизации»: при случайной фазе отсчёта относительно пролёта
  P(захват | b, L) = min(1, 2√(R² − b²) / L) при b < R, иначе 0.
Проверяется калибровка модели и вклад промаха и окна в итоговую вероятность.

Запуск:  python -m experiments.meta_passes
"""
import os

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401
from swarmsim import sim
from swarmsim.config import SCENARIO_1, SCENARIO_2

OUT = os.environ.get("META_OUT", "results/meta")
LAWS = ["gwo", "gwo_rel", "woa", "pso", "attract", "parallel"]


def episodes(P, Q, R):
    r = P[:, 0, :] - Q
    d = np.linalg.norm(r, axis=1)
    out = []
    for k in range(len(r) - 1):
        a, b = r[k], r[k + 1] - r[k]
        bb = b @ b
        t = 0.0 if bb < 1e-12 else float(np.clip(-(a @ b) / bb, 0.0, 1.0))
        if 0.0 < t < 1.0 or (t == 1.0 and k + 2 < len(r) and d[k + 2] > d[k + 1]) or \
                (t == 1.0 and k + 2 == len(r)):
            dmin = float(np.linalg.norm(a + t * b))
            L = float(np.sqrt(bb))
            cap = bool(min(d[k], d[k + 1]) <= R)
            win = 0.0 if dmin >= R else min(1.0, 2.0 * np.sqrt(R * R - dmin * dmin) / max(L, 1e-9))
            out.append((k, dmin, L, cap, win))
    return out


def main(seeds=1000):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for law in LAWS:
        for vt in (0.0, 5.0):
            base = SCENARIO_1 if vt == 0 else SCENARIO_2.with_(target_speed=vt)
            cfg = base.with_(n_agents=1)
            for s in range(seeds):
                r = sim.run_trial(cfg, law, False, s, record=True)
                for k, dmin, L, cap, win in episodes(r["P"], r["Q"], cfg.r_capture):
                    rows.append((law, vt, s, k, dmin, L, cap, win, bool(r["captured"])))
    df = pd.DataFrame(rows, columns=["law", "vt", "seed", "k", "b", "L", "cap", "win", "trial_captured"])
    df.to_csv(f"{OUT}/passes.csv.gz", index=False)
    pd.set_option("display.width", 200)
    # калибровка окна
    near = df[df.b < 5.0].copy()
    near["bin"] = pd.cut(near["win"], [0, .2, .4, .6, .8, .999, 1.0], include_lowest=True)
    print("Калибровка (эпизоды с промахом < R): предсказание окна против доли захвата")
    print(near.groupby("bin", observed=True).agg(n=("cap", "size"), pred=("win", "mean"), obs=("cap", "mean")).round(3))
    # сводка по законам
    summ = []
    for (law, vt), g in df.groupby(["law", "vt"]):
        first = g.groupby("seed").head(100)
        n_tr = g["seed"].nunique()
        summ.append(dict(law=law, vt=vt, episodes_per_trial=len(g) / n_tr,
                         share_b_lt_R=100 * (g.b < 5).mean(), median_b=g.b.median(),
                         median_L_near=g[g.b < 5].L.median(),
                         cap_given_b_lt_R=100 * g[g.b < 5].cap.mean(),
                         mean_win_given_b_lt_R=100 * g[g.b < 5].win.mean()))
    s = pd.DataFrame(summ)
    s.to_csv(f"{OUT}/passes_summary.csv", index=False)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
