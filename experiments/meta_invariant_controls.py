"""Инвариантные контроли к расчёту 6: SPSO-2011 и изотропный GWO (gwo_iso).

Прогноз механизма (зависимость от осей создают покоординатные случайные множители): у законов с изотропным шумом
доля захвата не зависит ни от угла подхода φ, ни от сдвига начала координат, исходы при разных φ отличаются
только розыгрышем. Проверка: 48 углов × скорость цели {0, 5} м/с × 1000 испытаний (как meta_rotation) и сдвиг
{0, 100, 300, 1000, 10000} м при N = 1 и N = 4 с Boids [1] (как meta_shift).
Результат: results/meta/invctl_rotation_trials.csv.gz, invctl_shift_trials.csv.gz.
Запуск:  python -m experiments.meta_invariant_controls --jobs 2
"""
import os
import argparse
import itertools
from multiprocessing import Pool

import pandas as pd

from experiments import meta_rotation as rot
from experiments import meta_shift as sh

LAWS = ["spso2011", "gwo_iso"]
OUT = os.environ.get("META_OUT", "results/meta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1000)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    t_rot = [(law, phi, vt, seeds) for law, phi, vt in itertools.product(LAWS, rot.PHIS, [0.0, 5.0])]
    t_sh = [(law, s, n, b, vt, seeds) for law, s, (n, b), vt in
            itertools.product(LAWS, sh.SHIFTS if os.environ.get("INVCTL_FULL") else [0.0, 100.0, 300.0, 1000.0, 10000.0],
                              [(1, False), (4, True)], [0.0, 5.0])]
    with Pool(a.jobs) as pool:
        r1 = pool.map(rot.cell, t_rot, chunksize=1)
        r2 = pool.map(sh.cell, t_sh, chunksize=1)
    pd.DataFrame([r for rows in r1 for r in rows], columns=["law", "phi", "vt", "seed", "captured", "t_cap"]
                 ).to_csv(f"{OUT}/invctl_rotation_trials.csv.gz", index=False)
    pd.DataFrame([r for rows in r2 for r in rows],
                 columns=["law", "shift", "n", "boids", "vt", "seed", "captured", "t_cap", "collision"]
                 ).to_csv(f"{OUT}/invctl_shift_trials.csv.gz", index=False)
    print("готово")


if __name__ == "__main__":
    main()
