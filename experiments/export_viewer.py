"""Выгрузка траекторий для интерактивного просмотрщика (viewer/data.json).

Траектории считаются этим же стендом, просмотрщик их только проигрывает — расхождений между
картинкой и числами быть не может.

Запуск:  python -m experiments.export_viewer --seeds 30 --stat-seeds 100
"""
import argparse
import json
import os

import numpy as np

from swarmsim.config import SCENARIO_1, SCENARIO_2
from swarmsim.sim import run_trial, run_series, summarize

SCENARIOS = {
    "s1": ("Сценарий 1 · цель неподвижна", SCENARIO_1),
    "s2v10": ("Сценарий 2 · цель 10 м/с", SCENARIO_2),
    "s2v5": ("Сценарий 2 · цель 5 м/с", SCENARIO_2.with_(target_speed=5.0)),
}
PROFILES = {
    "paper": ("Базовый Boids: k_s = 0,1, отталкивание ближе 5 м", {}),
    "safe": ("Сильный Boids: k_s = 100, отталкивание по всем соседям", dict(sep_law="inverse", k_sep=100.0)),
}
LAWS = [("gwo", "GWO"), ("parallel", "Упреждение"), ("pursuit", "Погоня"), ("pn", "Пропорц. навигация")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--stat-seeds", type=int, default=100)
    ap.add_argument("--out", default="viewer/data.json")
    a = ap.parse_args()

    data = dict(meta=dict(r_capture=SCENARIO_1.r_capture, d_min=SCENARIO_1.d_min, t_max=SCENARIO_1.t_max,
                          seeds=a.seeds, stat_seeds=a.stat_seeds, integrator=SCENARIO_1.integrator),
                scenarios={k: v[0] for k, v in SCENARIOS.items()},
                profiles={k: v[0] for k, v in PROFILES.items()},
                laws=dict(LAWS), runs={}, stats={})
    for sk, (_, base) in SCENARIOS.items():
        for pk, (_, kw) in PROFILES.items():
            cfg = base.with_(**kw)
            for law, _ in LAWS:
                for boids in (False, True):
                    if pk == "safe" and not boids:
                        continue  # без Boids профиль ничего не меняет
                    key = f"{sk}|{pk}|{law}|{int(boids)}"
                    trials = []
                    for s in range(a.seeds):
                        r = run_trial(cfg, law, boids, s, record=True)
                        trials.append(dict(
                            P=np.round(r["P"] * 10).astype(int).reshape(len(r["P"]), -1).tolist(),
                            Q=np.round(r["Q"] * 10).astype(int).tolist(),
                            t=None if not r["captured"] else int(r["t_cap"])))
                    data["runs"][key] = trials
                    st = summarize(run_series(cfg, law, boids, range(a.stat_seeds)))
                    data["stats"][key] = {k: (None if isinstance(v, float) and np.isnan(v) else round(float(v), 1))
                                          for k, v in st.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print("записано", a.out, os.path.getsize(a.out) // 1024, "КБ")

    # Собрать просмотрщик: данные встраиваются в страницу, открывается двойным щелчком без сервера.
    tpl_path = os.path.join(os.path.dirname(a.out), "template.html")
    if os.path.exists(tpl_path):
        tpl = open(tpl_path, encoding="utf-8").read()
        js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        body = tpl.replace("__DATA__", js)
        page = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                + body + "</body></html>")
        out_html = os.path.join(os.path.dirname(a.out), "viewer.html")
        open(out_html, "w", encoding="utf-8").write(page)
        print("просмотрщик:", out_html)


if __name__ == "__main__":
    main()
