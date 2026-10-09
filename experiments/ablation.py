"""Абляция «закон наведения × модуль Boids».

Варианты: {gwo, pursuit (прямое наведение / метод погони), parallel (параллельное сближение),
pn (пропорциональная навигация)} × {без Boids, с Boids}. Одинаковые сиды у всех вариантов —
парный дизайн. Главный контраст: gwo+Boids против pursuit+Boids.

Запуск:
  python -m experiments.ablation --scenario 1 --seeds 100 --profile paper
  python -m experiments.ablation --scenario 2 --seeds 100 --profile paper_safe --target-speed 5

Профили:
  paper      — базовая модель [1] (k_s = 0,1; отталкивание только ближе d_min);
  paper_safe — тот же GWO, но Boids с отталкиванием k_s/r по всем соседям и k_s = 100,
               что удовлетворяет условию безопасности из [1] (k_s > d_min² = 25).
               Позволяет сравнивать законы наведения при сильном модуле разделения.
"""
import argparse
import itertools
import os

import pandas as pd

from swarmsim.config import Config
from swarmsim.sim import run_series, summarize
from swarmsim.stats import paired_time_test, paired_binary_test, holm

PROFILES = {
    "paper": dict(),
    "paper_safe": dict(sep_law="inverse", k_sep=100.0),
}

LAWS = ["gwo", "pursuit", "parallel", "pn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--profile", default="paper", choices=list(PROFILES))
    ap.add_argument("--target-speed", type=float, default=None,
                    help="переопределить скорость цели (сценарий 2)")
    ap.add_argument("--target-heading", default="random")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    vt = 0.0 if a.scenario == 1 else 10.0
    if a.target_speed is not None:
        vt = a.target_speed
    cfg = Config(target_speed=vt, target_heading=a.target_heading,
                 n_agents=a.n_agents, **PROFILES[a.profile])
    seeds = list(range(a.seeds))

    series = {}
    for law, boids in itertools.product(LAWS, [False, True]):
        series[(law, boids)] = run_series(cfg, law, boids, seeds)

    # Сводка
    rows = []
    for (law, boids), df in series.items():
        s = summarize(df)
        rows.append(dict(variant=f"{law}{'+boids' if boids else ''}", **s))
    summary = pd.DataFrame(rows)

    # Парные сравнения: каждый вариант с Boids против gwo+boids; плюс вклад Boids внутри закона.
    ref = series[("gwo", True)]
    tests = []
    for law in ["pursuit", "parallel", "pn"]:
        df = series[(law, True)]
        t = paired_time_test(df, ref)
        c = paired_binary_test(df, ref, "collision")
        tests.append(dict(contrast=f"{law}+boids vs gwo+boids", measure="время захвата",
                          n=t["n_pairs"], effect=t["median_diff"], ci=t["ci"], p=t["p"]))
        tests.append(dict(contrast=f"{law}+boids vs gwo+boids", measure="столкновения",
                          n=a.seeds, effect=c["b"] - c["c"], ci=None, p=c["p"]))
        c = paired_binary_test(df, ref, "collision_pre")
        tests.append(dict(contrast=f"{law}+boids vs gwo+boids", measure="нарушения до захвата",
                          n=a.seeds, effect=c["b"] - c["c"], ci=None, p=c["p"]))
    for law in LAWS:
        c = paired_binary_test(series[(law, True)], series[(law, False)], "collision")
        tests.append(dict(contrast=f"{law}: с Boids vs без", measure="столкновения",
                          n=a.seeds, effect=c["b"] - c["c"], ci=None, p=c["p"]))
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].to_numpy())

    os.makedirs(a.out, exist_ok=True)
    tag = f"s{a.scenario}_{a.profile}_vt{vt:g}_n{a.n_agents}"
    pd.concat(series.values()).to_csv(f"{a.out}/ablation_{tag}_trials.csv", index=False)
    summary.to_csv(f"{a.out}/ablation_{tag}_summary.csv", index=False)
    tests.to_csv(f"{a.out}/ablation_{tag}_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Сценарий {a.scenario}, профиль {a.profile}, скорость цели {vt} м/с, аппаратов {a.n_agents}, сидов {a.seeds}")
    print(summary.round(1).to_string(index=False))
    print()
    print("Эффект времени: медиана разности (вариант − gwo+boids), итераций; отрицательная = быстрее gwo+boids.")
    print("Эффект столкновений: (испытаний, где столкнулся только вариант) − (только gwo+boids).")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
