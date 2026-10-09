"""E5. Масштаб эффекта сдвига и чувствительность.

Теория (для одного лидера q, покоординатно, C ~ U(0, 2)):  E[(C q − p)²] = (q − p)² + q²/3.
Шум точки GWO/WOA-окружения состоит из относительной части (∝ дистанции до цели) и абсолютной (∝ |q|, зависит от
начала координат). Вопрос: какой безразмерный параметр управляет потерей захвата — сдвиг относительно стартовой
дистанции D₀, относительно радиуса захвата R или абсолютный сдвиг.

Сетка: конфигурации (масштаб стартовой дистанции f, горизонт K = k_max, радиус захвата R) × сдвиг s = m · D̄,
D̄ = 97,5 f м (средняя стартовая дистанция), m ∈ {0; 0,1; 0,3; 1; 3; 10; 100} × законы {gwo, gwo_rel, woa,
woa_spiral, woa_encircle}; N = 1, цель неподвижна, 1000 испытаний. Результат: results/meta/scale_trials.csv.gz.

Запуск:  python -m experiments.meta_scale --jobs 2
"""
import os
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401
from swarmsim import sim
from swarmsim.config import SCENARIO_1
from swarmsim.scenario import make_initial

OUT = os.environ.get("META_OUT", "results/meta")

CONFIGS = {"A": (0.5, 30, 5.0), "B": (1.0, 30, 5.0), "C": (2.0, 60, 5.0),
           "D": (1.0, 30, 3.0), "E": (1.0, 30, 8.0), "F": (1.0, 60, 5.0)}
MULTS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
LAWS = ["gwo", "gwo_rel", "woa", "woa_spiral", "woa_encircle"]


def cell(args):
    name, m, law, seeds = args
    f, K, R = CONFIGS[name]
    shift = m * 97.5 * f
    off = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)

    def mk(cfg, seed):
        p, v, q, vq, rng = make_initial(cfg, seed)
        return p + off, v, q + off, vq, rng

    sim.make_initial = mk
    cfg = SCENARIO_1.with_(n_agents=1, start_dist_range=(85.0 * f, 110.0 * f), t_max=K, gwo_k_max=K,
                           r_capture=R)
    rows = [(name, f, K, R, m, shift, law, s, *(lambda r: (r["captured"], r["t_cap"]))(sim.run_trial(cfg, law, False, s)))
            for s in seeds]
    sim.make_initial = make_initial
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    tasks = [(c, m, law, seeds) for c, m, law in itertools.product(CONFIGS, MULTS, LAWS)]
    with Pool(a.jobs) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows],
                      columns=["config", "f", "K", "R", "m", "shift", "law", "seed", "captured", "t_cap"])
    df.to_csv(f"{OUT}/scale_trials.csv.gz", index=False)
    s = df.groupby(["config", "f", "K", "R", "law", "m"])["captured"].mean().mul(100).round(1).unstack("m")
    s.to_csv(f"{OUT}/scale_summary.csv")
    pd.set_option("display.width", 200)
    print(s.to_string())


if __name__ == "__main__":
    main()
