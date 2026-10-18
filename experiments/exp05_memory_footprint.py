"""Эксперимент 05. Сколько стоит один поток и одна корутина?

ВОПРОС: почему говорят, что asyncio масштабируется на десятки тысяч
соединений, а потоки — нет?

МЕТОД: создаём N "ожидающих" единиц (потоков или корутин), которые
ничего не делают, только ждут события. Меряем:
  - прирост RSS (Resident Set Size — реально занятая физическая память);
  - время создания.

Нюанс про память потоков: ОС резервирует каждому потоку стек (на Linux
по умолчанию 8 МБ ВИРТУАЛЬНОЙ памяти), но физически страницы выделяются
по мере использования. Поэтому RSS на поток — десятки КБ, а не 8 МБ.
Но кроме памяти поток — это объект ядра ОС: планировщик, лимиты на
число потоков (ulimit -u), стоимость переключения контекста.

Запуск:  python experiments/exp05_memory_footprint.py
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time

import psutil

from concurrency_lab.bench import RESULTS_DIR, system_info

THREAD_COUNTS = [100, 1_000, 2_000]  # больше — рискованно упереться в лимиты ОС
CORO_COUNTS = [100, 1_000, 10_000, 100_000]  # корутины можно в сотни тысяч


def rss_mb() -> float:
    gc.collect()  # убираем мусор, чтобы не мерить "хвосты" прошлых прогонов
    return psutil.Process().memory_info().rss / 2**20


def measure_threads(n: int) -> dict:
    stop = threading.Event()
    before = rss_mb()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=stop.wait, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    created = time.perf_counter() - t0
    after = rss_mb()
    stop.set()
    for t in threads:
        t.join()
    return {
        "kind": "thread",
        "n": n,
        "create_s": created,
        "rss_delta_mb": after - before,
        "kb_per_unit": (after - before) * 1024 / n,
    }


def measure_coroutines(n: int) -> dict:
    async def run() -> dict:
        stop = asyncio.Event()
        before = rss_mb()
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(stop.wait()) for _ in range(n)]
        await asyncio.sleep(0)  # даём всем задачам стартовать и встать на ожидание
        created = time.perf_counter() - t0
        after = rss_mb()
        stop.set()
        await asyncio.gather(*tasks)
        return {
            "kind": "coroutine",
            "n": n,
            "create_s": created,
            "rss_delta_mb": after - before,
            "kb_per_unit": (after - before) * 1024 / n,
        }

    return asyncio.run(run())


def main() -> None:
    results = []
    for n in THREAD_COUNTS:
        r = measure_threads(n)
        results.append(r)
        print(
            f"  threads    n={n:>7}: +{r['rss_delta_mb']:7.1f} MB  "
            f"({r['kb_per_unit']:5.1f} KB/шт)  создание {r['create_s']:.3f} s"
        )
    for n in CORO_COUNTS:
        r = measure_coroutines(n)
        results.append(r)
        print(
            f"  coroutines n={n:>7}: +{r['rss_delta_mb']:7.1f} MB  "
            f"({r['kb_per_unit']:5.1f} KB/шт)  создание {r['create_s']:.3f} s"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "exp05_memory_footprint.json").write_text(
        json.dumps({"system": system_info(), "results": results}, indent=2, ensure_ascii=False)
    )
    print("\n  -> сохранено в results/exp05_memory_footprint.json")


if __name__ == "__main__":
    main()
