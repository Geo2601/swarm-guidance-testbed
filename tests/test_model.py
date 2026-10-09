"""Проверки модели. Запуск: python -m pytest -q  (или python tests/test_model.py)"""
import numpy as np

from swarmsim.config import Config
from swarmsim.sim import run_trial, run_series, _clip_norm


def test_clip_norm():
    x = np.array([[3.0, 4.0], [0.3, 0.4]])
    y = _clip_norm(x, 2.0)
    assert np.allclose(np.linalg.norm(y, axis=1), [2.0, 0.5])


def test_seed_reproducibility():
    cfg = Config()
    a = run_series(cfg, "gwo", True, range(5))
    b = run_series(cfg, "gwo", True, range(5))
    assert a.equals(b)


def test_pursuit_kinematics_from_rest():
    """Один аппарат, неподвижная цель на 100 м, разгон с 2 м/с² до 10 м/с, шаг 1 с.
    Пройденный путь за k шагов: 2+4+6+8+10 = 30 м за 5 шагов, далее по 10 м.
    До радиуса захвата 5 м нужно пройти 95 м → 5 + ceil(65/10) = 12 шагов при порядке «скорость, затем
    положение»; при порядке «положение по старой скорости» первый шаг стоит на месте → 13."""
    for integ, expect in [("semi_implicit", 12), ("explicit", 13)]:
        cfg = Config(n_agents=1, start_mode="cluster", start_spread=0.0, start_min_sep=0.0,
                     start_distance=(100.0, 100.0), target_speed=0.0, integrator=integ)
        r = run_trial(cfg, "pursuit", False, seed=0)
        assert r["captured"] and r["t_cap"] == expect, (integ, r)


def test_parallel_beats_pursuit_on_crossing_target():
    cfg = Config(target_speed=5.0, target_heading="crossing")
    tp = run_series(cfg, "pursuit", False, range(30))["t_cap"].median()
    tq = run_series(cfg, "parallel", False, range(30))["t_cap"].median()
    assert tq <= tp


def test_surround_start_geometry():
    """Старт как на рис. 1: аппараты в 85–110 м от цели, цель в начале координат."""
    from swarmsim.scenario import make_initial
    for s in range(20):
        p, v, q, vq, _ = make_initial(Config(), s)
        d = np.linalg.norm(p - q, axis=1)
        assert np.all((d >= 85.0) & (d <= 110.0)) and np.allclose(q, 0)


def test_separation_paper_only_inside_dmin():
    """Базовое разделение [1, шаг 5]: отталкивание действует только при r < d_min."""
    from swarmsim.boids import boids_accel
    cfg = Config(k_ali=0.0, k_coh=0.0)
    p = np.array([[0.0, 0.0], [6.0, 0.0]]); v = np.zeros_like(p)
    assert np.allclose(boids_accel(p, v, cfg), 0.0)
    p = np.array([[0.0, 0.0], [4.0, 0.0]])
    u = boids_accel(p, v, cfg)
    assert np.isclose(u[0, 0], -cfg.k_sep / 4.0) and np.isclose(u[1, 0], cfg.k_sep / 4.0)


def test_gwo_centroids_converges_on_static_target():
    cfg = Config()
    df = run_series(cfg, "gwo", False, range(20))
    assert df["captured"].mean() >= 0.8  # 20 сидов; на 100 сидах ≈ 0,9


def test_multi_equals_single_for_one_target():
    """При одной цели модуль нескольких целей даёт ровно те же испытания, что одиночная модель."""
    from swarmsim.multi import run_trial_multi
    from swarmsim.config import SCENARIO_2
    for cfg in (Config(), SCENARIO_2.with_(target_speed=5.0)):
        for s in range(15):
            for meth, law in (("gwo_avg", "gwo"), ("hungarian", "parallel")):
                for b in (False, True):
                    a = run_trial(cfg, law, b, s)
                    c = run_trial_multi(cfg, meth, b, s, m=1)
                    assert a["captured"] == c["all_captured"] and abs(a["min_pair"] - c["min_pair"]) < 1e-9


def test_bottleneck_matches_brute_force():
    import itertools
    from swarmsim.multi import assign_bottleneck
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(2, 6)); m = int(rng.integers(1, n + 1))
        C = rng.random((n, m))
        g = assign_bottleneck(C)
        val = max(min(C[i, j] for i in range(n) if g[i] == j) for j in range(m))
        best = min(max(min(C[i, j] for i in range(n) if pm[i] == j) for j in range(m))
                   for pm in itertools.product(range(m), repeat=n) if len(set(pm)) == m)
        assert abs(val - best) < 1e-12


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
