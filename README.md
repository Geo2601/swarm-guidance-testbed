# swarm-guidance-testbed

Стенд для имитационного моделирования законов группового наведения на плоскости. Группа аппаратов сближается с целью, неподвижной или движущейся прямолинейно. Скорость и ускорение ограничены, к закону наведения можно добавить модуль Boids. Все варианты сравниваются на одинаковых случайных числах (парный дизайн), поэтому различия между законами не тонут в разбросе начальных условий.

[English summary below](#english-summary)

## Что есть в стенде

**Модель** (`swarmsim/`)
- кинематика: шаг Δt, |v| ≤ V_max, |u| ≤ a_max, две схемы интегрирования (`explicit`, `semi_implicit`);
- модуль Boids: разделение, выравнивание, сплочение; два закона разделения (`paper` — только ближе d_min, `inverse` — по всем соседям в радиусе обзора);
- сценарии: аппараты стартуют вокруг цели со случайных направлений, цель неподвижна или идёт прямолинейно;
- несколько целей: K-means-центроиды, назначение целей (жадное, венгерский метод, минимизация максимального времени, распределение ролей по кольцу);
- статистика: точный критерий Мак-Немара, парные тесты, поправка Холма, бутстреп.

**Законы наведения** (`swarmsim/guidance.py`, `swarmsim/metaguide.py`)

| Группа | Законы |
|---|---|
| Классические | `pursuit` — погоня; `parallel` — параллельное сближение (курс на точку встречи); `pn` — пропорциональная навигация; `attract` — притяжение к цели без шума |
| На основе популяционных метаэвристик | `gwo` [2]; `woa`, `woa_spiral`, `woa_encircle` [3]; `hho`, `hho_hard`, `hho_soft`, `hho_soft_j1`, `hho_noclip` [4]; `pso` [5]; `pso_hist`; `spso2011` [6] |
| Контрольные варианты GWO | `gwo_rel` — член через разность положений; `gwo_c1` — C = 1; `gwo_iso` — скалярные случайные множители |
| Регулятор по скорости | `gwo_v`, `gwo_rel_v`, `woa_v`, `pso_v`, `attract_v` — та же предложенная точка, но аппарат отслеживает её по скорости |

У законов на основе метаэвристик общий регулятор u = w(k)·(X − p): оператор метаэвристики предлагает точку X, регулятор переводит её в ускорение. Меняется только оператор.

**Опыты** (`experiments/`)

| Скрипт | Что считает |
|---|---|
| `ablation` | закон наведения × модуль Boids, парные тесты |
| `safety_front`, `lemma_check` | дальность срабатывания разделения и компромисс «быстродействие — безопасность» |
| `multi_target`, `figures_multi`, `robustness` | несколько целей: методы распределения и устойчивость выводов |
| `meta_redundancy` | группа против независимых одиночных попыток: 1 − (1 − p₁)^N |
| `meta_shift`, `meta_scale` | зависимость от сдвига начала системы координат и её масштаб |
| `meta_rotation` | зависимость от направления осей (угла подхода к цели) |
| `meta_invariant_controls` | контроли с изотропным шумом (`spso2011`, `gwo_iso`) |
| `meta_sens`, `meta_hho` | чувствительность к Δt, радиусу захвата, горизонту, профилю Boids; HHO |
| `meta_passes` | разбор сближений одиночного аппарата |
| `meta_model`, `meta_model_check`, `meta_semi_moment` | полуаналитическая модель вероятности захвата через момент относительного движения и её проверка |
| `meta_multi_model`, `meta_multi_incidental` | модель нескольких целей с переназначением после захвата |
| `meta_analyze`, `meta_rotation_analyze`, `meta_sens_analyze` | сводки по результатам опытов |
| `export_viewer` | данные для просмотрщика |

**Просмотрщик** (`viewer/`). Страница проигрывает испытания: слева и справа один и тот же старт с разными законами, график минимальной дистанции между аппаратами, сводная таблица.

**Независимые реализации** (`independent_check_meta/`, `independent_check_multi/`). Код написан без импорта `swarmsim` и нужен для сверки ключевых чисел.

## Установка

Нужен Python 3.10 или новее.

```bash
pip install -r requirements.txt
```

Скрипты запускаются из корня репозитория, корень должен быть в `PYTHONPATH`:

```bash
# Linux / macOS
export PYTHONPATH=.
# Windows, PowerShell
$env:PYTHONPATH="."
# Windows, cmd
set PYTHONPATH=.
```

## Запуск

```bash
python tests/test_model.py                                  # проверки модели, все должны вывести ok

python -m experiments.ablation --scenario 1 --seeds 100 --profile paper
python -m experiments.ablation --scenario 2 --seeds 100 --profile paper_safe --target-speed 5

python -m experiments.meta_shift --seeds 1000 --jobs 8      # сдвиг начала координат
python -m experiments.meta_rotation --seeds 1000 --jobs 8   # поворот осей
python -m experiments.meta_redundancy --seeds 1000 --jobs 8 # группа против независимых попыток
python -m experiments.meta_analyze                          # сводки по двум первым опытам

python -m experiments.multi_target --seeds 100 --jobs 8     # несколько целей

python -m experiments.export_viewer --seeds 30 --stat-seeds 100
```

Результаты пишутся в папку `results/`. Просмотрщик собирается в `viewer/viewer.html` и открывается двойным щелчком, без сервера.

Схему интегрирования для всех расчётов процесса задаёт переменная окружения `SWARMSIM_INTEGRATOR` (`explicit` по умолчанию или `semi_implicit`). Параметры модели — в `swarmsim/config.py`.

## Базовая модель

Базовый сценарий и модуль Boids построены по модели работы [1]. В `swarmsim/config.py` у каждого параметра указан источник: значение из [1] с номером страницы или пометка `[ДОПУЩЕНИЕ]`, если в [1] параметр не задан и выбран для стенда.

## Литература

1. Акимов А. А., Гнатенко Ю. А., Болбаков Р. Г. Гибридный алгоритм глобального планирования и локального взаимодействия для перехвата целей роем БПЛА // Информационно-управляющие системы. 2025. № 6. С. 2–14. doi:10.31799/1684-8853-2025-6-2-14
2. Mirjalili S., Mirjalili S. M., Lewis A. Grey Wolf Optimizer // Advances in Engineering Software. 2014. Vol. 69. P. 46–61. doi:10.1016/j.advengsoft.2013.12.007
3. Mirjalili S., Lewis A. The Whale Optimization Algorithm // Advances in Engineering Software. 2016. Vol. 95. P. 51–67. doi:10.1016/j.advengsoft.2016.01.008
4. Heidari A. A., Mirjalili S., Faris H., Aljarah I., Mafarja M., Chen H. Harris hawks optimization: Algorithm and applications // Future Generation Computer Systems. 2019. Vol. 97. P. 849–872. doi:10.1016/j.future.2019.02.028
5. Clerc M., Kennedy J. The particle swarm — explosion, stability, and convergence in a multidimensional complex space // IEEE Transactions on Evolutionary Computation. 2002. Vol. 6, no. 1. P. 58–73. doi:10.1109/4235.985692
6. Zambrano-Bigiarini M., Clerc M., Rojas R. Standard particle swarm optimisation 2011 at CEC-2013: A baseline for future PSO improvements // 2013 IEEE Congress on Evolutionary Computation. 2013. P. 2337–2344. doi:10.1109/CEC.2013.6557848
7. Reynolds C. W. Flocks, herds and schools: A distributed behavioral model // ACM SIGGRAPH Computer Graphics. 1987. Vol. 21, no. 4. P. 25–34. doi:10.1145/37402.37406

## Лицензия

MIT, см. [LICENSE](LICENSE).

---

## English summary

A simulation testbed for multi-vehicle guidance laws in the plane. A group of vehicles approaches a stationary or constant-velocity target under speed and acceleration limits, optionally with a Boids interaction module. All variants share the same random numbers trial by trial (paired design).

**Included**
- classical laws: pursuit, parallel navigation (lead pursuit to the meeting point), proportional navigation, noise-free attraction;
- guidance laws built from population metaheuristics: GWO [2], WOA [3], HHO [4], PSO [5], SPSO-2011 [6], plus control variants (difference-form GWO, C = 1, scalar random factors) and velocity-tracking regulators;
- Boids module [7] with two separation laws, multi-target assignment methods;
- experiments: law × Boids ablation, separation range and safety trade-off, multi-target allocation, shift and rotation of the coordinate frame, group vs. independent attempts, sensitivity to step, capture radius and horizon, a semi-analytical capture-probability model;
- an offline trajectory viewer and two independent re-implementations for cross-checking.

**Quick start**

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python tests/test_model.py
python -m experiments.ablation --scenario 1 --seeds 100 --profile paper
python -m experiments.export_viewer --seeds 30 --stat-seeds 100   # then open viewer/viewer.html
```

The baseline scenario and Boids module follow the model of [1]; every parameter in `swarmsim/config.py` is tagged with its source or marked as an assumption. Code comments are in Russian. License: MIT.
