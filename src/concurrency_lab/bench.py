"""Утилиты для честных замеров.

Главная идея: ОДИН замер ничего не значит. Время выполнения шумит из-за
планировщика ОС, других процессов, кэшей CPU, турбобуста частоты и т.д.
Поэтому каждый вариант мы запускаем несколько раз и сообщаем mean ± std.
Если два варианта отличаются меньше, чем на величину std, — честно говорим,
что разница не значима.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Папка results/ лежит в корне репозитория. Считаем путь от этого файла,
# а не от текущей директории (os.getcwd()), чтобы скрипты работали,
# откуда бы их ни запустили.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Measurement:
    """Результат замера одного варианта (например, "threads, 16 воркеров")."""

    name: str
    params: dict
    wall_times: list[float] = field(default_factory=list)  # "настенное" время, сек
    cpu_times: list[float] = field(default_factory=list)   # CPU-время процесса, сек

    @property
    def mean(self) -> float:
        return statistics.mean(self.wall_times)

    @property
    def std(self) -> float:
        # stdev требует минимум 2 точки; при одном запуске честно пишем 0
        return statistics.stdev(self.wall_times) if len(self.wall_times) > 1 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean"] = self.mean
        d["std"] = self.std
        return d


def measure(name: str, fn: Callable[[], object], repeats: int = 3, **params) -> Measurement:
    """Запускает fn() `repeats` раз и собирает время.

    Почему time.perf_counter(), а не time.time():
      - time.time() — "часы на стене": их может подвести NTP-синхронизация,
        перевод часов; разрешение на некоторых ОС ~15 мс.
      - perf_counter() — монотонный счётчик с максимальным разрешением,
        создан именно для замера интервалов.

    Почему отдельно time.process_time():
      - это CPU-время ТЕКУЩЕГО процесса (все его потоки вместе).
      - Сравнение wall vs cpu очень показательно:
          * I/O-bound: wall >> cpu (процесс в основном ЖДЁТ сеть)
          * CPU-bound в потоках под GIL: cpu ≈ wall (работает по сути одно ядро)
      - Ограничение: время дочерних процессов (multiprocessing) сюда НЕ входит.
        Для процессов смотрим только на wall time.
    """
    m = Measurement(name=name, params=params)
    for _ in range(repeats):
        t0_wall, t0_cpu = time.perf_counter(), time.process_time()
        fn()
        m.wall_times.append(time.perf_counter() - t0_wall)
        m.cpu_times.append(time.process_time() - t0_cpu)
    print(f"  {name:<34} {m.mean:8.3f} ± {m.std:.3f} s   (cpu {statistics.mean(m.cpu_times):.3f} s)")
    return m


def gil_enabled() -> bool:
    """True, если GIL включён.

    В Python 3.13+ есть экспериментальная free-threaded сборка (python3.13t),
    в 3.14 она стала официально поддерживаемой (PEP 779). В ней GIL можно
    выключить. sys._is_gil_enabled() есть только в 3.13+, поэтому getattr
    с запасным вариантом: в старых версиях GIL включён всегда.
    """
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def system_info() -> dict:
    """Окружение, в котором сделан замер. Без этого результат невоспроизводим:
    на 2 ядрах и на 16 ядрах multiprocessing даст совсем разный выигрыш."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "gil_enabled": gil_enabled(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def save_results(experiment: str, measurements: list[Measurement], extra: dict | None = None) -> Path:
    """Сохраняем сырые данные в JSON.

    Почему JSON, а не сразу картинку: графики можно перестроить в любой момент
    (scripts/plot_results.py), а данные остаются первичным артефактом
    исследования. Это принцип "отделяй вычисления от визуализации".
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{experiment}.json"
    payload = {
        "experiment": experiment,
        "system": system_info(),
        "measurements": [m.to_dict() for m in measurements],
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n  -> сохранено в {path.relative_to(RESULTS_DIR.parent)}")
    return path
