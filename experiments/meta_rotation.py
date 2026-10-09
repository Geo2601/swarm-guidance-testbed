"""Расчёт 6 (часть). Зависимость закона наведения от ориентации осей координат.

Закон наведения, имеющий физический смысл, не должен зависеть от выбора системы координат: ни от начала (E2),
ни от направления осей. В GWO, WOA, PSO и HHO случайные множители действуют покоординатно (C ⊙ q, A ⊙ |…|,
r1 ⊙ (q − p), покоординатный модуль |q − p| в спирали WOA), поэтому шум предложения выровнен по осям.

Опыт: один аппарат (N = 1, без Boids), цель неподвижна в начале координат (сдвиг 0), аппарат стартует на дальности
85–110 м под фиксированным углом φ к оси X. Повороту сцены на φ соответствует смена ориентации осей; закон,
инвариантный к повороту, даёт одинаковую вероятность захвата при всех φ. Перебирается полный круг φ = 0°, 7,5°, …, 352,5°:
спираль WOA (покоординатный модуль |q − p|) не симметрична даже к отражению осей.
Законы: gwo, gwo_rel, woa, woa_spiral, woa_encircle, pso, hho, hho_hard, attract, parallel; 1000 испытаний на ячейку,
одинаковые сиды при разных φ. Результат: results/meta/rotation_trials.csv.gz.
Запуск:  python -m experiments.meta_rotation --jobs 2
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

LAWS = ["gwo", "gwo_rel", "woa", "woa_spiral", "woa_encircle", "pso", "hho", "hho_hard", "attract", "parallel"]
PHIS = [7.5 * i for i in range(48)]  # полный круг: спираль WOA не симметрична к отражению осей
OUT = os.environ.get("META_OUT", "results/meta")
# ROT_COURSE=1 — поворачивать курс движущейся цели вместе с углом подхода (поворот всей сцены, а не только
# направления подхода); без него курс цели одинаков для всех φ одного испытания (исходный вариант опыта).
ROT_COURSE = bool(int(os.environ.get("ROT_COURSE", "0")))


def cell(args):
    law, phi, vt, seeds = args
    c, s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))

    def mk(cfg, seed):
        p, v, q, vq, rng = make_initial(cfg, seed)
        d = np.linalg.norm(p[0] - q)            # дальность из того же генератора, что и в E1
        p = q + d * np.array([[c, s]])
        if ROT_COURSE:                          # поворот всей сцены: курс цели поворачивается вместе с подходом
            vq = np.array([[c, -s], [s, c]]) @ vq
        return p, v, q, vq, rng

    sim.make_initial = mk
    base = SCENARIO_1 if vt == 0 else SCENARIO_2.with_(target_speed=vt)
    cfg = base.with_(n_agents=1)
    rows = []
    for sd in seeds:
        r = sim.run_trial(cfg, law, False, sd)
        rows.append((law, phi, vt, sd, r["captured"], r["t_cap"]))
    sim.make_initial = make_initial
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    tasks = [(law, phi, vt, seeds) for law, phi, vt in itertools.product(LAWS, PHIS, [0.0, 5.0])]
    with Pool(a.jobs) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows],
                      columns=["law", "phi", "vt", "seed", "captured", "t_cap"])
    df.to_csv(f"{OUT}/rotation_trials.csv.gz", index=False)
    print("испытаний:", len(df))


if __name__ == "__main__":
    main()
