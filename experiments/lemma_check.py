"""Численная проверка условия на дальность срабатывания разделения.

Два аппарата летят навстречу, каждый со скоростью w/2. Разделение «видит» пару только при d < R (как любое правило,
равное нулю при d >= R) и тогда тормозит каждый аппарат с максимальным ускорением a вдоль линии; вне R аппараты
продолжают сближение. Ищем минимальное R, при котором min d >= d_min при всех фазах начального положения
(сдвиг старта внутри одного шага), и сравниваем с формулой R*(w).

Запуск:  python -m experiments.lemma_check
"""
import numpy as np


def r_star(w, a, h, d_min, order):
    m0 = 0 if order == "explicit" else 1
    s = sum(max(w - 2 * a * h * m, 0.0) for m in range(m0, 1000)) * h
    return d_min + h * w + s


def min_dist(R, w, a, h, d_min, order, d0):
    v = w / 2.0                     # скорость каждого навстречу (сближение w)
    d, vi = d0, v
    dmin = d
    for _ in range(400):
        brake = d < R
        u = -a if brake else 0.0    # ускорение каждого вдоль своего направления движения (торможение)
        if order == "explicit":
            d -= 2 * h * vi
            vi = max(vi + h * u, 0.0)
        else:
            vi = max(vi + h * u, 0.0)
            d -= 2 * h * vi
        dmin = min(dmin, d)
        if vi == 0.0 and brake:
            break
    return dmin


def r_numeric(w, a, h, d_min, order):
    lo, hi = d_min, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = all(min_dist(mid, w, a, h, d_min, order, d0) >= d_min - 1e-9
                 for d0 in np.linspace(mid + 2 * w * h, mid + 4 * w * h, 41))   # все фазы входа в зону
        hi, lo = (mid, lo) if ok else (hi, mid)
    return hi


def main():
    a, h, d_min = 2.0, 1.0, 5.0
    print(" w, м/с | порядок        | R* по формуле | R минимальное численно | непрерывное d_min + w²/(4a)")
    for w in (2.0, 5.0, 10.0, 15.0, 20.0):
        for order in ("explicit", "semi_implicit"):
            rf, rn = r_star(w, a, h, d_min, order), r_numeric(w, a, h, d_min, order)
            print(f" {w:5.0f}  | {order:14s} | {rf:13.1f} | {rn:22.1f} | {d_min + w * w / (4 * a):6.1f}")
    print("\nПравило с R = d_min = 5 м: минимальная дистанция при сближении w и худшей фазе входа")
    for w in (2.0, 5.0, 10.0, 20.0):
        worst = min(min_dist(5.0, w, a, h, d_min, "explicit", d0) for d0 in np.linspace(5 + 2 * w, 5 + 4 * w, 41))
        print(f"  w = {w:4.0f} м/с: min d = {worst:6.1f} м")


if __name__ == "__main__":
    main()
