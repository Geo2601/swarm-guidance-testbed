"""Расчёт 3. Чувствительность двух выводов — эффекта сдвига начала координат и модели избыточности
1 − (1 − p₁)^N — к параметрам, которые в исходной постановке зафиксированы.

Блоки сетки (по 1000 испытаний на ячейку, одинаковые сиды во всех ячейках — парный дизайн):

  dt    — шаг Δt ∈ {0,5; 1; 2} с при неизменном физическом горизонте 30 с: K = t_max = gwo_k_max = 30/Δt,
          так что график a(k) растягивается по тем же 30 с. Законы gwo, woa (сдвиг 0…1000 м), gwo_rel
          (контроль, сдвиг 0 и 1000 м), pso, attract (только сдвиг 0); N ∈ {1, 2, 4} без Boids и N = 4 с Boids [1].
  rk    — радиус захвата R ∈ {3; 5; 8} м и горизонт K ∈ {30; 60}; законы gwo, woa, pso; N ∈ {1, 2, 4, 8};
          Boids нет/[1]; скорость цели 0 и 5 м/с; без сдвига.
  safe  — Boids, удовлетворяющий условию безопасности из [1] (k_s = 100, отталкивание k_s/r по всем соседям, профиль
          paper_safe): законы gwo, woa, pso, gwo_rel; N ∈ {2, 4, 8}; скорость цели 0 и 5 м/с; для gwo и woa при N = 4
          ещё сдвиг 100, 300, 1000 м.

Результат: results/meta/sens_trials.csv.gz (одна строка — одно испытание).
Запуск:  python -m experiments.meta_sens --jobs 2
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import swarmsim.metaguide  # noqa: F401  (регистрирует законы)
from swarmsim import sim
from swarmsim.config import SCENARIO_1, SCENARIO_2
from swarmsim.scenario import make_initial

OUT = os.environ.get("META_OUT", "results/meta")
SAFE = dict(sep_law="inverse", k_sep=100.0)


def cell(args):
    block, law, kw, vt, shift, n, boids, seeds = args
    off = shift * np.array([1.0, 1.0]) / np.sqrt(2.0)

    def mk(cfg, seed):
        p, v, q, vq, rng = make_initial(cfg, seed)
        return p + off, v, q + off, vq, rng

    sim.make_initial = mk
    base = SCENARIO_1 if vt == 0 else SCENARIO_2.with_(target_speed=vt)
    cfg = base.with_(n_agents=n, **kw)
    rows = []
    for s in seeds:
        r = sim.run_trial(cfg, law, boids, s)
        rows.append((block, law, cfg.dt, cfg.t_max, cfg.r_capture, cfg.sep_law, vt, shift, n, boids, s,
                     r["captured"], r["t_cap"], r["collision"]))
    sim.make_initial = make_initial
    return rows


def tasks(seeds):
    t = []
    # dt
    for dt in (0.5, 1.0, 2.0):
        K = int(round(30 / dt))
        kw = dict(dt=dt, t_max=K, gwo_k_max=K)
        for n, b in ((1, False), (2, False), (4, False), (4, True)):
            for law in ("gwo", "woa"):
                for s in (0.0, 100.0, 300.0, 1000.0):
                    t.append(("dt", law, kw, 0.0, s, n, b, seeds))
            for s in (0.0, 1000.0):
                t.append(("dt", "gwo_rel", kw, 0.0, s, n, b, seeds))
            for law in ("pso", "attract"):
                t.append(("dt", law, kw, 0.0, 0.0, n, b, seeds))
    # rk
    for R in (3.0, 5.0, 8.0):
        for K in (30, 60):
            kw = dict(r_capture=R, t_max=K, gwo_k_max=K)
            for vt in (0.0, 5.0):
                for law in ("gwo", "woa", "pso"):
                    for n in (1, 2, 4, 8):
                        for b in ((False,) if n == 1 else (False, True)):
                            t.append(("rk", law, kw, vt, 0.0, n, b, seeds))
    # safe
    for vt in (0.0, 5.0):
        for law in ("gwo", "woa", "pso", "gwo_rel"):
            for n in (2, 4, 8):
                t.append(("safe", law, SAFE, vt, 0.0, n, True, seeds))
        for law in ("gwo", "woa"):
            for s in (100.0, 300.0, 1000.0):
                t.append(("safe", law, SAFE, vt, s, 4, True, seeds))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tk = tasks(list(range(a.seeds)))
    print("ячеек:", len(tk), flush=True)
    with Pool(a.jobs) as pool:
        res = pool.map(cell, tk, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows],
                      columns=["block", "law", "dt", "K", "R", "sep_law", "vt", "shift", "n", "boids", "seed",
                               "captured", "t_cap", "collision"])
    df.to_csv(f"{OUT}/sens_trials.csv.gz", index=False)
    print("испытаний:", len(df))


if __name__ == "__main__":
    main()
