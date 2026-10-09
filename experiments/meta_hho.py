"""Расчёт 4. Harris hawks optimization (HHO) как закон наведения: тест на сдвиг и модель избыточности.

В HHO три ветви содержат абсолютные координаты: исследование (2 r2 X; точка области LB + r4 (UB − LB)), мягкое сжатие
(ΔX — вектор разности, подставленный как точка, и J X_rabbit), ветви с шагом Леви (J X_rabbit). Жёсткое сжатие
X_rabbit − E |ΔX| эквивариантна (закон hho_hard — контроль).

Сетка как в E1 и E2 (meta_redundancy, meta_shift):
  сдвиг — hho, hho_noclip, hho_hard × s ∈ {0, 10, 30, 100, 300, 1000, 10000} м × (N = 1 без Boids; N = 4 с Boids [1])
          × скорость цели {0, 5} м/с;
  избыточность — hho, hho_hard × N ∈ {1, 2, 3, 4, 6, 8} × Boids {нет, [1]} × скорость цели {0, 5} м/с.
1000 испытаний на ячейку. Результат: results/meta/hho_shift_trials.csv.gz, hho_redundancy_trials.csv.gz.
Запуск:  python -m experiments.meta_hho --jobs 2
"""
import argparse
import itertools
import os
from multiprocessing import Pool

import pandas as pd

from experiments import meta_redundancy as red
from experiments import meta_shift as sh

OUT = os.environ.get("META_OUT", "results/meta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    t_sh = [(law, s, n, b, vt, seeds) for law, s, (n, b), vt in
            itertools.product(os.environ.get("HHO_LAWS", "hho,hho_noclip,hho_hard").split(","), sh.SHIFTS, [(1, False), (4, True)], [0.0, 5.0])]
    t_red = [(law, n, b, vt, seeds) for law, n, b, vt in
             itertools.product(["hho", "hho_hard"], red.NS, [False, True], [0.0, 5.0]) if not (n == 1 and b)]
    with Pool(a.jobs) as pool:
        r1 = pool.map(sh.cell, t_sh, chunksize=1)
        r2 = pool.map(red.cell, t_red, chunksize=1)
    pd.DataFrame([r for rows in r1 for r in rows],
                 columns=["law", "shift", "n", "boids", "vt", "seed", "captured", "t_cap", "collision"]
                 ).to_csv(f"{OUT}/hho_shift_trials.csv.gz", index=False)
    pd.DataFrame([r for rows in r2 for r in rows],
                 columns=["law", "n", "boids", "vt", "seed", "captured", "t_cap", "collision", "collision_cont"]
                 ).to_csv(f"{OUT}/hho_redundancy_trials.csv.gz", index=False)
    print("готово")


if __name__ == "__main__":
    main()
