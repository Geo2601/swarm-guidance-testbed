"""Параметры модели.

Базовый сценарий построен по модели работы [1] (см. README): Акимов А. А., Гнатенко Ю. А., Болбаков Р. Г.
Информационно-управляющие системы, 2025, № 6, с. 2–14, doi 10.31799/1684-8853-2025-6-2-14.

Источник каждого параметра:
  [1, с. N]     — значение из работы [1], N — номер страницы журнала;
  [1, рис. N]   — оценено по рисунку работы [1] (приблизительно);
  [ДОПУЩЕНИЕ]   — в [1] не указано, выбрано для стенда; проверяется анализом чувствительности.
"""
from dataclasses import dataclass, asdict, replace
import math
import os


@dataclass(frozen=True)
class Config:
    # --- Кинематика ---
    dim: int = 2                 # [1, с. 8] плоскость, d = 2
    dt: float = 1.0              # [1, с. 8] Δt = 1 с
    v_max: float = 10.0          # [1, с. 8] V_max = 10 м/с
    a_max: float = 2.0           # [1, с. 8] a_max = 2 м/с²
    t_max: int = 30              # [1, с. 8, шаг 7.2] останов при k = k_max = 30
    integrator: str = os.environ.get("SWARMSIM_INTEGRATOR", "explicit")
    # Переменная окружения SWARMSIM_INTEGRATOR задаёт схему для всех расчётов процесса (по умолчанию explicit).
    # 'explicit'      — p ← p + vΔt, затем v ← v + uΔt (положение по скорости прошлого шага);
    # 'semi_implicit' — v ← v + uΔt, затем p ← p + vΔt [1, с. 6, 8, шаг 6.3].

    # --- Группа и цель ---
    n_agents: int = 4            # [1, с. 8] N = 4
    r_capture: float = 5.0       # [1, с. 8] R_cap = 5 м
    d_min: float = 5.0           # [1, с. 8] d_min = 5 м — порог безопасности
    target_speed: float = 0.0    # [1, с. 9] сц. 1: 0; сц. 2: 10 м/с

    # --- Начальные условия ---
    start_mode: str = "surround"
    # 'surround' — аппараты вокруг цели со случайных направлений [1, рис. 3, 6; с. 8:
    #              «сократить дистанцию с разных направлений»];
    # 'cluster'  — аппараты кучкой с одной стороны.
    start_dist_range: tuple = (85.0, 110.0)  # [1, рис. 1] стартовая дистанция до цели
    start_min_sep: float = 10.0  # [ДОПУЩЕНИЕ] минимальный стартовый интервал между аппаратами
    start_spread: float = 50.0   # [ДОПУЩЕНИЕ] только для 'cluster'
    start_distance: tuple = (100.0, 120.0)  # [ДОПУЩЕНИЕ] только для 'cluster'
    start_at_rest: bool = True   # [ДОПУЩЕНИЕ] нулевая начальная скорость
    target_heading: str = "random"  # [ДОПУЩЕНИЕ] цель движется прямолинейно, курс случайный
    crossing_jitter_deg: float = 30.0  # [ДОПУЩЕНИЕ] для target_heading='crossing'

    # --- GWO (закон наведения) ---
    gwo_k_max: int = 30          # [1, с. 8] k_max = 30; a_m(k) = 2(1 − k/k_max) [с. 5]
    gwo_a_clamp: bool = True     # несущественно: испытание останавливается при k = k_max
    gwo_leaders: str = "centroids"
    # 'centroids'    — [1, с. 5, шаг 4; с. 7, п. 4.3]: X_j^(i) = c_i − A ⊙ |C ⊙ c_i − p_j|,
    #                  c_i — центроиды целей; при M < 3 точки целей дублируются, т. е. при одной цели
    #                  c_1 = c_2 = c_3 = q. Роли α, β, δ в формулу не входят.
    # 'agents'       — классический GWO с лидерами-аппаратами (для сравнения; захват не обеспечивает).
    gwo_weight_scale: float = 1.0 / (2.0 * math.sqrt(2))  # [1, с. 6–7] w_g = a(k) / (2√d), d = 2

    # --- Boids ---
    k_sep: float = 0.1           # [1, с. 8]
    k_ali: float = 0.1           # [1, с. 8]
    k_coh: float = 0.1           # [1, с. 8]
    sep_law: str = "paper"
    # 'paper'   — базовый закон [1, с. 7, шаг 5]: F_S = −k_s Σ_{j: r_ij < d_min} (p_j − p_i) / r_ij²,
    #             т. е. отталкивание k_s/r действует только ближе d_min;
    # 'inverse' — k_s/r по всем соседям в радиусе обзора [1, с. 6].
    post_separate: float = None  # опция: после шага пары разводятся до этой дистанции (только с Boids; None — выкл.)
    sep_radius: float = None    # радиус действия разделения для 'inverse' (None — neighbor_radius)
    neighbor_radius: float = 30.0  # [ДОПУЩЕНИЕ] радиус обзора для выравнивания и сплочения

    # --- Пропорциональная навигация ---
    pn_gain: float = 3.0         # [ДОПУЩЕНИЕ] навигационная постоянная N (типично 3–5)

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


SCENARIO_1 = Config(target_speed=0.0)
SCENARIO_2 = Config(target_speed=10.0)
