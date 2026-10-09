"""Компромисс «быстродействие — безопасность».

1) Аналитика: два аппарата сближаются со скоростью w; каждый тормозит не сильнее a_max, реакция
   запаздывает на шаг Δt (положение обновляется по скорости прошлого шага). Чтобы не нарушить d_min,
   разделение должно начинать действовать не позже дистанции
       R_need(w) = d_min + w·Δt + w² / (4·a_max).
   При w = 2·V_max = 20 м/с, a_max = 2 м/с², Δt = 1 с: R_need = 5 + 20 + 50 = 75 м.
   Базовое разделение [1] включается при r < d_min = 5 м, т. е. уже после нарушения.
2) Численно: скорость сближения пар в момент пересечения d_min (базовый сценарий).
3) Фронт Парето: перебор радиуса и силы разделения для GWO и упреждения.

Запуск:  python -m experiments.safety_front --seeds 100
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from swarmsim.config import SCENARIO_1, SCENARIO_2
from swarmsim.sim import run_trial, run_series, summarize


def r_need(w, cfg):
    return cfg.d_min + w * cfg.dt + w * w / (4 * cfg.a_max)


def closing_speeds(cfg, law, boids, seeds):
    """Скорость сближения пары на шаге, где дистанция впервые стала < d_min (по положениям)."""
    out = []
    for s in seeds:
        r = run_trial(cfg, law, boids, s, record=True)
        P = r["P"]
        n = P.shape[1]
        for i, j in itertools.combinations(range(n), 2):
            d = np.linalg.norm(P[:, i] - P[:, j], axis=1)
            idx = np.where(d < cfg.d_min)[0]
            if len(idx) and idx[0] > 0:
                k = idx[0]
                out.append((d[k - 1] - d[k]) / cfg.dt)   # среднее сближение на последнем шаге
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    seeds = range(a.seeds)
    os.makedirs(a.out, exist_ok=True)
    cfg = SCENARIO_1
    print("Необходимая дистанция срабатывания разделения R_need(w), м:")
    for w in (5, 10, 15, 20):
        print(f"  w = {w:>2} м/с → {r_need(w, cfg):.0f}")
    for name, c in [("сц. 1", SCENARIO_1), ("сц. 2, 5 м/с", SCENARIO_2.with_(target_speed=5.0))]:
        for law in ("gwo", "parallel"):
            w = closing_speeds(c, law, True, seeds)
            if len(w):
                print(f"{name}, {law}+Boids (базовый профиль): пересечений d_min {len(w)}, скорость сближения "
                      f"медиана {np.median(w):.1f} м/с, 90-й перцентиль {np.percentile(w, 90):.1f} м/с")
    rows = []
    for (scn, base), law, rs, ks in itertools.product(
            [("s1", SCENARIO_1), ("s2v5", SCENARIO_2.with_(target_speed=5.0))],
            ["gwo", "parallel"], [5.0, 10.0, 20.0, 40.0, 80.0], [10.0, 100.0, 1000.0]):
        cfg = base.with_(sep_law="inverse", sep_radius=rs, k_sep=ks)
        st = summarize(run_series(cfg, law, True, seeds))
        rows.append(dict(scenario=scn, law=law, sep_radius=rs, k_sep=ks, **st))
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/safety_front.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["scenario", "law", "sep_radius", "k_sep", "capture_pct", "median", "median_censored",
              "collisions_per_100", "collisions_pre_per_100"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
