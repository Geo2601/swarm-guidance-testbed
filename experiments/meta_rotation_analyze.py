"""Анализ расчёта 6 (ориентация осей): доля захвата одиночным аппаратом в зависимости от угла подхода φ.

Для каждого закона и скорости цели: доля захвата при каждом φ (1000 испытаний), минимум и максимум по φ, значения
на оси (φ = 0°, 90°, 180°, 270°) и на диагонали (45°, 135°, 225°, 315°), критерий однородности χ² по всем 48 углам
(H₀: вероятность захвата не зависит от φ), поправка Холма по законам. Для детерминированных законов (attract,
parallel) дополнительно — совпадение исходов испытание в испытание при всех φ.
Результат: results/meta/rotation_summary.csv, rotation_by_phi.csv.
Запуск:  python -m experiments.meta_rotation_analyze
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from experiments.meta_analyze import holm

OUT = os.environ.get("META_OUT", "results/meta")
AXIS = [0.0, 90.0, 180.0, 270.0]
DIAG = [45.0, 135.0, 225.0, 315.0]


def main():
    d = pd.read_csv(f"{OUT}/rotation_trials.csv.gz")
    by = d.groupby(["law", "vt", "phi"]).captured.mean().mul(100).rename("cap").reset_index()
    by.to_csv(f"{OUT}/rotation_by_phi.csv", index=False)
    rows = []
    for (law, vt), g in d.groupby(["law", "vt"]):
        tab = g.groupby("phi").captured.agg(["sum", "count"])
        ct = np.stack([tab["sum"], tab["count"] - tab["sum"]], axis=1)
        p = chi2_contingency(ct)[1] if (ct.sum(axis=0) > 0).all() else 1.0
        cap = 100 * tab["sum"] / tab["count"]
        piv = g.pivot(index="seed", columns="phi", values="captured")
        same = float((piv.nunique(axis=1) == 1).mean() * 100)
        rows.append(dict(law=law, vt=vt, cap_min=cap.min(), phi_min=cap.idxmin(), cap_max=cap.max(),
                         phi_max=cap.idxmax(), axis=cap.loc[AXIS].mean(), diag=cap.loc[DIAG].mean(),
                         mean=cap.mean(), p_chi2=p, same_all_phi=same))
    s = pd.DataFrame(rows)
    s["p_holm"] = holm(s.p_chi2)
    s.to_csv(f"{OUT}/rotation_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
