"""E2. Зависимость закона наведения от выбора начала системы координат.

Вся сцена (аппараты и цель) сдвигается на s·(1, 1)/√2, геометрия и сиды не меняются. Для правила, эквивариантного
к сдвигу (gwo_rel, pso, attract, parallel), результат обязан совпасть с s = 0 испытание в испытание; для gwo и woa —
нет: в них есть член |C ⊙ q − p|, зависящий от абсолютных координат цели.

Сетка: закон × s ∈ {0, 10, 30, 100, 300, 1000, 10000} м × (N = 1 без Boids; N = 4 с Boids [1]) × скорость цели
{0, 5} м/с; 1000 испытаний на ячейку. Результат: results/meta/shift_trials.csv.gz.

Запуск:  python -m experiments.meta_shift --jobs 2
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401
from swarmsim import sim
from swarmsim.config import SCENARIO_1, SCENARIO_2
from swarmsim.scenario import make_initial

LAWS = ["gwo", "gwo_rel", "woa", "pso", "attract", "parallel"]
SHIFTS = [0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0]
OUT = os.environ.get("META_OUT", "results/meta")


def cell(args):
    law, shift, n, boids, vt, seeds = args
    off = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)

    def mk(cfg, seed):
        p, v, q, vq, rng = make_initial(cfg, seed)
        return p + off, v, q + off, vq, rng

    sim.make_initial = mk
    base = SCENARIO_1 if vt == 0 else SCENARIO_2.with_(target_speed=vt)
    cfg = base.with_(n_agents=n)
    rows = []
    for s in seeds:
        r = sim.run_trial(cfg, law, boids, s)
        rows.append((law, shift, n, boids, vt, s, r["captured"], r["t_cap"], r["collision"]))
    sim.make_initial = make_initial
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(a.seeds))
    tasks = [(law, s, n, b, vt, seeds) for law, s, (n, b), vt in
             itertools.product(LAWS, SHIFTS, [(1, False), (4, True)], [0.0, 5.0])]
    with Pool(a.jobs) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows],
                      columns=["law", "shift", "n", "boids", "vt", "seed", "captured", "t_cap", "collision"])
    df.to_csv(f"{OUT}/shift_trials.csv.gz", index=False)
    print("испытаний:", len(df))


if __name__ == "__main__":
    main()
