"""Генерация начальных условий испытания. Один сид — одно испытание; одинаковые сиды
у разных алгоритмов дают одинаковые начальные условия (парный дизайн)."""
import numpy as np


def _place_with_min_sep(sample_one, n, min_sep, rng):
    pts = []
    for _ in range(n):
        for _try in range(10000):
            c = sample_one(rng)
            if not pts or np.linalg.norm(np.array(pts) - c, axis=1).min() >= min_sep:
                pts.append(c)
                break
        else:
            raise RuntimeError("не удалось расставить аппараты с заданным интервалом")
    return np.array(pts)


def make_initial(cfg, seed):
    rng = np.random.default_rng(seed)
    n, d = cfg.n_agents, cfg.dim

    if cfg.start_mode == "surround":
        # Цель в начале координат, аппараты вокруг неё со случайных направлений
        # ([1], с. 8: «сократить дистанцию с разных направлений»).
        q = np.zeros(d)
        lo, hi = cfg.start_dist_range

        def one(r):
            rad = r.uniform(lo, hi)
            phi = 2 * np.pi * r.random()
            return np.array([rad * np.cos(phi), rad * np.sin(phi)])

        p = _place_with_min_sep(one, n, cfg.start_min_sep, rng)
        los = None
    elif cfg.start_mode == "cluster":
        def one(r):
            rad = cfg.start_spread * np.sqrt(r.random())
            phi = 2 * np.pi * r.random()
            return np.array([rad * np.cos(phi), rad * np.sin(phi)])

        p = _place_with_min_sep(one, n, cfg.start_min_sep, rng)
        dist = rng.uniform(*cfg.start_distance)
        bearing = 2 * np.pi * rng.random()
        q = dist * np.array([np.cos(bearing), np.sin(bearing)])
        los = q / np.linalg.norm(q)
    else:
        raise ValueError(cfg.start_mode)

    v = np.zeros((n, d)) if cfg.start_at_rest else 0.5 * cfg.v_max * _unit(rng.normal(size=(n, d)))

    # Курс цели (прямолинейно и равномерно; направление задаётся cfg.target_heading).
    if los is None:
        los = _unit(q - p.mean(axis=0)) if np.linalg.norm(q - p.mean(axis=0)) > 1e-9 else np.array([1.0, 0.0])
    if cfg.target_heading == "random":
        a = 2 * np.pi * rng.random()
        heading = np.array([np.cos(a), np.sin(a)])
    elif cfg.target_heading == "crossing":
        perp = np.array([-los[1], los[0]]) * (1 if rng.random() < 0.5 else -1)
        jit = np.deg2rad(rng.uniform(-cfg.crossing_jitter_deg, cfg.crossing_jitter_deg))
        c, s = np.cos(jit), np.sin(jit)
        heading = np.array([c * perp[0] - s * perp[1], s * perp[0] + c * perp[1]])
    elif cfg.target_heading == "away":
        heading = los
    elif cfg.target_heading == "toward":
        heading = -los
    else:
        raise ValueError(cfg.target_heading)
    vq = cfg.target_speed * heading
    return p, v, q, vq, rng


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)
