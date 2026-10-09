"""Устойчивость выводов по нескольким целям к решениям, которые в [1] не заданы.

Ячейки: N ∈ {4, 8} × M ∈ {3, 5} (для правила дублирования — M = 2) × разброс 60 м × скорость {0, 5} м/с, k_max = 30.
Методы: GWO+Boids (формула 5), GWO+Boids (группы), венгерский, bottleneck. 100 сидов, одинаковые старты.

Запуск:  python -m experiments.robustness --seeds 100 --jobs 2   → results/robustness.csv, results/ring_sweep.csv
"""
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from swarmsim.config import Config
from swarmsim.multi import run_trial_multi

METHODS = [("gwo_avg", True), ("gwo_group", True), ("hungarian", False), ("bottleneck", False)]
VARIANTS = {
    "базовый": ({}, {}),
    "порядок как в тексте (v, затем p)": (dict(integrator="semi_implicit"), {}),
    "радиус соседства 10 м": (dict(neighbor_radius=10.0), {}),
    "радиус соседства 50 м": (dict(neighbor_radius=50.0), {}),
    "интервал целей 20 м": ({}, dict(min_sep=20.0)),
    "центроиды по незахваченным": ({}, dict(gwo_drop_captured=True)),
    "K-means: случайная инициализация": ({}, dict(km_init="random")),
    "гистерезис назначения 10 %": ({}, dict(hysteresis=0.1)),
    "гистерезис назначения 30 %": ({}, dict(hysteresis=0.3)),
    "старт с ненулевой скоростью": (dict(start_at_rest=False), {}),
}
DUP = {"дублирование [q1,q2,q1]": "first", "дублирование [q1,q2,q2]": "last", "контроль [q1,q2,середина]": "mean"}


def job(args):
    vname, ckw, tkw, n, m, vt, meth, b, seeds = args
    cfg = Config(n_agents=n, target_speed=vt, **ckw)
    R = [run_trial_multi(cfg, meth, b, s, m=m, spread=60.0, **tkw) for s in seeds]
    return dict(вариант=vname, N=n, M=m, v=vt, метод=meth,
                захват_всех=100 * np.mean([r["all_captured"] for r in R]),
                медиана=np.nanmedian([r["t_cap"] for r in R]) if any(r["all_captured"] for r in R) else np.nan,
                нарушения=100 * np.mean([r["collision"] for r in R]),
                смены_цели=np.mean([r["switch_rate"] for r in R]))


def ring_job(args):
    rr, n, m, vt, seeds = args
    R = [run_trial_multi(Config(n_agents=n, target_speed=vt), "ring", False, s, m=m, spread=60.0, ring_radius=rr) for s in seeds]
    return dict(радиус=rr, N=n, M=m, v=vt, захват_всех=100 * np.mean([r["all_captured"] for r in R]),
                медиана=np.nanmedian([r["t_cap"] for r in R]), нарушения=100 * np.mean([r["collision"] for r in R]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    work = [(vn, ckw, tkw, n, m, vt, meth, b, seeds) for vn, (ckw, tkw) in VARIANTS.items()
            for n, m, vt in itertools.product([4, 8], [3, 5], [0.0, 5.0]) for meth, b in METHODS]
    work += [(vn, {}, dict(dup_rule=r), n, 2, vt, meth, b, seeds) for vn, r in DUP.items()
             for n, vt in itertools.product([4, 8], [0.0, 5.0]) for meth, b in METHODS[:2]]
    ring = [(rr, n, m, vt, seeds) for rr in (10.0, 15.0, 25.0, 40.0) for n, m, vt in itertools.product([4, 8], [1, 3], [0.0, 5.0])]
    with Pool(a.jobs) as pool:
        rows = pool.map(job, work, chunksize=2)
        rrows = pool.map(ring_job, ring, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv("results/robustness.csv", index=False)
    rg = pd.DataFrame(rrows); rg.to_csv("results/ring_sweep.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    piv = df[df.M > 2].pivot_table(index="вариант", columns=["метод"], values="захват_всех", aggfunc=["min", "max"])
    print("Захват всех целей, % (min и max по 8 ячейкам N × M × скорость)\n", piv.round(0).to_string())
    print("\nПравило дублирования, M = 2\n", df[df.M == 2].pivot_table(index=["вариант", "N", "v"], columns="метод", values="захват_всех").round(0).to_string())
    print("\nДоля смен цели за шаг (базовый)\n", df[(df.вариант == "базовый") & df.метод.isin(["hungarian", "bottleneck"])]
          .pivot_table(index=["N", "M", "v"], columns="метод", values="смены_цели").round(3).to_string())
    print("\nКольцо: радиус\n", rg.pivot_table(index=["N", "M", "v"], columns="радиус", values=["захват_всех", "нарушения"]).round(0).to_string())


if __name__ == "__main__":
    main()
