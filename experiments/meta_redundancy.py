"""E1. Закон избыточности для метаэвристических законов наведения.

Гипотеза-«ноль»: аппараты не кооперируются, группа ловит цель за счёт числа попыток:
    P_N(T ≤ k) = 1 − (1 − F_1(k))^N,
где F_1 — распределение времени захвата одиночным аппаратом. Без Boids аппараты с законами gwo/gwo_rel/woa/pso/
attract не взаимодействуют, и равенство — санитарная проверка; с Boids [1] отклонение измеряет вклад
взаимодействия (кооперации или помехи).

Сетка: закон × N ∈ {1, 2, 3, 4, 6, 8} × Boids {нет, [1]} × скорость цели {0, 5} м/с; 1000 испытаний на ячейку,
одинаковые сиды (парный дизайн). Результат: results/meta/redundancy_trials.csv.gz, redundancy_summary.csv.

Запуск:  python -m experiments.meta_redundancy --jobs 2
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401  (регистрирует законы)
from swarmsim import sim
from swarmsim.config import SCENARIO_1, SCENARIO_2

LAWS = ["gwo", "gwo_rel", "woa", "pso", "pso_hist", "attract", "parallel"]
NS = [1, 2, 3, 4, 6, 8]
OUT = os.environ.get("META_OUT", "results/meta")


def cell(args):
    law, n, boids, vt, seeds = args
    base = SCENARIO_1 if vt == 0 else SCENARIO_2.with_(target_speed=vt)
    cfg = base.with_(n_agents=n)
    rows = []
    for s in seeds:
        r = sim.run_trial(cfg, law, boids, s)
        rows.append((law, n, boids, vt, s, r["captured"], r["t_cap"], r["collision"], r["collision_cont"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(a.seeds))
    tasks = [(law, n, b, vt, seeds) for law, n, b, vt in itertools.product(LAWS, NS, [False, True], [0.0, 5.0])
             if not (n == 1 and b)]
    with Pool(a.jobs) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows],
                      columns=["law", "n", "boids", "vt", "seed", "captured", "t_cap", "collision", "collision_cont"])
    df.to_csv(f"{OUT}/redundancy_trials.csv.gz", index=False)
    print("испытаний:", len(df))


if __name__ == "__main__":
    main()
