"""Расчёт 1 (проверка). Сравнение полуаналитической модели вероятности захвата одним аппаратом (meta_model) со стендом.

Модель не подгоняется: в ней нет ни одного параметра, оцениваемого по результатам стенда. Сравниваются p₁ (N = 1,
неподвижная цель) в ячейках, уже посчитанных на стенде:
  base   — базовая постановка, законы gwo, woa, hho при сдвиге 0…10 км; gwo_rel, ветви WOA, hho_hard, attract;
  scale  — конфигурации A–F расчёта E5 (масштаб, R, K) для gwo при сдвиге 0, D₀, 3D₀, 10D₀;
  dt     — шаг Δt = 0,5 и 2 с (K = 30/Δt) для gwo, gwo_rel, woa, attract;
  rot    — фиксированный угол подхода φ (расчёт 6, meta_rotation) для gwo, gwo_rel, woa, woa_spiral, hho.
Результат: results/meta/model_vs_sim.csv.
Запуск:  python -m experiments.meta_model_check [--rot]
"""
import argparse
import os

import numpy as np
import pandas as pd

from experiments.meta_model import p1
from swarmsim.config import SCENARIO_1

OUT = os.environ.get("META_OUT", "results/meta")
PHIS = np.deg2rad(np.arange(0, 360, 15))
ND = int(os.environ.get("MODEL_ND", "10"))  # число начальных дальностей; при Δt = 2 с исход периодичен по D₀
DS = (np.arange(ND) + 0.5) / ND          # с периодом ≈ VΔt = 20 м, поэтому нужна мелкая сетка (≥ 100)


def model(law, cfg, shift, phis=PHIS):
    lo, hi = cfg.start_dist_range
    return 100 * p1(law, cfg, shift=shift, phis=phis, Ds=lo + DS * (hi - lo), n_mc=1000)


def base_rows():
    cfg = SCENARIO_1.with_(n_agents=1)
    sh = pd.read_csv(f"{OUT}/shift_summary.csv")
    hh = pd.read_csv(f"{OUT}/hho_shift.csv")
    sc = pd.read_csv(f"{OUT}/scale_summary.csv")
    rows = []
    for law in ("gwo", "woa", "hho"):
        src = hh if law == "hho" else sh
        for s in (0, 10, 30, 100, 300, 1000, 10000):
            sim = src[(src.law == law) & (src.n == 1) & (src.vt == 0) & (src["shift"] == s)].cap.iloc[0]
            rows.append(dict(block="base", law=law, cfg="B", shift=s, phi=np.nan, model=model(law, cfg, s), sim=sim))
    for law in ("woa_spiral", "woa_encircle"):
        r = sc[(sc.config == "B") & (sc.law == law)].iloc[0]
        for m in (0.0, 1.0, 3.0, 10.0):
            s = m * 97.5
            rows.append(dict(block="base", law=law, cfg="B", shift=s, phi=np.nan, model=model(law, cfg, s),
                             sim=float(r[str(m)])))
    red = pd.read_csv(f"{OUT}/redundancy_summary.csv")
    hred = pd.read_csv(f"{OUT}/hho_redundancy.csv")
    for law, src in (("gwo_rel", red), ("attract", red), ("hho_hard", hred)):
        sim = src[(src.law == law) & (src.vt == 0)].p1.iloc[0]
        rows.append(dict(block="base", law=law, cfg="B", shift=0, phi=np.nan, model=model(law, cfg, 0), sim=sim))
    return rows


def scale_rows():
    sc = pd.read_csv(f"{OUT}/scale_summary.csv")
    rows = []
    for _, r in sc[sc.law == "gwo"].iterrows():
        f, K, R = float(r.f), int(r.K), float(r.R)
        cfg = SCENARIO_1.with_(n_agents=1, start_dist_range=(85 * f, 110 * f), t_max=K, gwo_k_max=K, r_capture=R)
        for m in (0.0, 1.0, 3.0, 10.0):
            s = m * 97.5 * f
            rows.append(dict(block="scale", law="gwo", cfg=r.config, shift=s, phi=np.nan,
                             model=model("gwo", cfg, s), sim=float(r[str(m)])))
    return rows


def dt_rows():
    sr = pd.read_csv(f"{OUT}/sens_redundancy.csv")
    ss = pd.read_csv(f"{OUT}/sens_shift.csv")
    rows = []
    for dt in (0.5, 2.0):
        K = int(round(30 / dt))
        cfg = SCENARIO_1.with_(n_agents=1, dt=dt, t_max=K, gwo_k_max=K)
        for law in ("gwo", "gwo_rel", "woa", "attract"):
            sim = sr[(sr.block == "dt") & (sr.law == law) & (sr.dt == dt)].p1.iloc[0]
            rows.append(dict(block="dt", law=law, cfg=f"dt={dt}", shift=0, phi=np.nan, model=model(law, cfg, 0), sim=sim))
        for law in ("gwo", "woa"):
            sim = ss[(ss.block == "dt") & (ss.law == law) & (ss.dt == dt) & (ss.n == 1) & (ss["shift"] == 1000)].cap.iloc[0]
            rows.append(dict(block="dt", law=law, cfg=f"dt={dt}", shift=1000, phi=np.nan,
                             model=model(law, cfg, 1000), sim=sim))
    return rows


def rot_rows():
    d = pd.read_csv(f"{OUT}/rotation_trials.csv.gz")
    d = d[d.vt == 0]
    cfg = SCENARIO_1.with_(n_agents=1)
    rows = []
    for law in ("gwo", "gwo_rel", "woa", "woa_spiral", "hho"):
        g = d[d.law == law].groupby("phi").captured.mean() * 100
        for phi in np.arange(0, 360, 15.0):
            rows.append(dict(block="rot", law=law, cfg="B", shift=0, phi=phi,
                             model=model(law, cfg, 0, phis=np.deg2rad([phi])), sim=float(g.loc[phi])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot", action="store_true", help="только блок rot (дописать к готовому файлу)")
    a = ap.parse_args()
    path = os.environ.get("MODEL_OUT", f"{OUT}/model_vs_sim.csv")
    if a.rot:
        old = pd.read_csv(path)
        df = pd.concat([old[old.block != "rot"], pd.DataFrame(rot_rows())])
    else:
        df = pd.DataFrame(base_rows() + scale_rows() + dt_rows())
    df["diff"] = df.model - df.sim
    df.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    for b, g in df.groupby("block"):
        print(b, "MAE", round(g["diff"].abs().mean(), 1), "max", round(g["diff"].abs().max(), 1), "n", len(g))


if __name__ == "__main__":
    main()
