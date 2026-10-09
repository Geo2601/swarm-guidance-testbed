"""Статистика сравнения вариантов. Дизайн парный: одинаковые сиды = одинаковые начальные условия.

- Время захвата: парный критерий Уилкоксона по испытаниям, где захватили оба варианта.
- Столкновения и неудачи (бинарные исходы): точный критерий Мак-Немара.
- Поправка Холма на множественные сравнения.
- Бутстреп-интервал для разности медиан времени захвата.
"""
import numpy as np
from scipy import stats


def paired_time_test(df_a, df_b):
    m = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"))
    both = m["captured_a"] & m["captured_b"]
    ta, tb = m.loc[both, "t_cap_a"].to_numpy(), m.loc[both, "t_cap_b"].to_numpy()
    if len(ta) < 5 or np.all(ta == tb):
        return dict(n_pairs=int(len(ta)), p=1.0, median_diff=0.0, ci=(0.0, 0.0))
    res = stats.wilcoxon(ta, tb, zero_method="wilcox")
    return dict(n_pairs=int(len(ta)), p=float(res.pvalue),
                median_diff=float(np.median(ta - tb)), ci=bootstrap_median_diff(ta, tb))


def mcnemar_exact(x_a, x_b):
    """x_a, x_b — булевы массивы исходов (например, столкновение) по одним и тем же сидам."""
    b = int(np.sum(x_a & ~x_b))
    c = int(np.sum(~x_a & x_b))
    if b + c == 0:
        return dict(b=b, c=c, p=1.0)
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return dict(b=b, c=c, p=float(p))


def paired_binary_test(df_a, df_b, col):
    m = df_a.merge(df_b, on="seed", suffixes=("_a", "_b"))
    return mcnemar_exact(m[f"{col}_a"].to_numpy(bool), m[f"{col}_b"].to_numpy(bool))


def bootstrap_median_diff(ta, tb, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = ta - tb
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def holm(pvals):
    """Поправка Холма. На входе список p-значений, на выходе скорректированные (в том же порядке)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
