"""Эксперимент 03. Блокирующий код внутри asyncio — самая частая ошибка.

ВОПРОС: что происходит с event loop, если внутри корутины вызвать
блокирующую функцию (time.sleep, requests.get, тяжёлый расчёт)?
И как это правильно исправлять?

МЕТОД: "пульсометр" (heartbeat). Фоновая корутина каждые 10 мс просыпается
и записывает, насколько она опоздала. Опоздание = event loop был занят
и не мог её разбудить. Это ровно то, что почувствуют ВСЕ остальные
клиенты вашего async-сервиса: пока loop занят, никто не обслуживается.

СЦЕНАРИИ (в каждом — 3 "задачи" по 0.3 с):
  1. asyncio.sleep           — правильный неблокирующий ввод-вывод
  2. time.sleep              — блокирующий вызов прямо в корутине (ОШИБКА)
  3. to_thread(time.sleep)   — блокирующий I/O вынесен в поток (исправление)
  4. CPU в корутине          — тяжёлый расчёт прямо в loop (ОШИБКА)
  5. to_thread(CPU)          — расчёт в потоке: loop жив, но GIL общий
  6. ProcessPool(CPU)        — расчёт в процессе: loop полностью свободен

Запуск:  python experiments/exp03_blocking_in_async.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from concurrency_lab.bench import RESULTS_DIR, system_info
from concurrency_lab.workloads import count_primes

TICK = 0.010          # пульс 10 мс
WORK = 0.3            # длительность одной "задачи" для sleep-сценариев
PRIMES_N = 60_000     # подобрано примерно на ~0.1-0.3 с CPU


async def heartbeat(lags: list[float], stop: asyncio.Event) -> None:
    """Каждые TICK секунд проверяем, вовремя ли нас разбудили."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + TICK
        await asyncio.sleep(TICK)
        lags.append(max(0.0, loop.time() - expected))


async def scenario(name: str, job) -> dict:
    lags: list[float] = []
    stop = asyncio.Event()
    hb = asyncio.create_task(heartbeat(lags, stop))
    await asyncio.sleep(0.05)  # даём пульсометру стартовать

    t0 = time.perf_counter()
    await asyncio.gather(*(job() for _ in range(3)))
    total = time.perf_counter() - t0

    stop.set()
    await hb
    res = {
        "scenario": name,
        "total_s": total,
        "max_lag_ms": max(lags) * 1000 if lags else total * 1000,
        "p50_lag_ms": statistics.median(lags) * 1000 if lags else 0.0,
        "ticks": len(lags),
    }
    print(f"  {name:<22} total={total:5.2f}s  max_lag={res['max_lag_ms']:7.1f} ms  "
          f"median_lag={res['p50_lag_ms']:5.1f} ms  ticks={len(lags)}")
    return res


async def main_async() -> list[dict]:
    loop = asyncio.get_running_loop()
    # Пул процессов создаём заранее и переиспользуем: запуск процессов дорогой,
    # в реальном сервисе пул живёт всё время работы приложения.
    with ProcessPoolExecutor(max_workers=3) as proc_pool:
        await loop.run_in_executor(proc_pool, count_primes, 0, 10)  # прогрев пула

        async def good_sleep():
            await asyncio.sleep(WORK)

        async def bad_sleep():
            time.sleep(WORK)  # <- блокирует ВЕСЬ поток, а значит и event loop

        async def thread_sleep():
            # to_thread (3.9+) = run_in_executor(default ThreadPoolExecutor, ...)
            # Блокирующий вызов уходит в отдельный поток, loop продолжает работу.
            await asyncio.to_thread(time.sleep, WORK)

        async def bad_cpu():
            count_primes(0, PRIMES_N)  # расчёт прямо в корутине: loop стоит

        async def thread_cpu():
            # Loop не заблокирован полностью, но расчёт держит GIL, и потоку
            # с loop приходится его отбирать. Интерпретатор переключает потоки
            # примерно раз в sys.getswitchinterval() = 5 мс -> пульс опаздывает.
            await asyncio.to_thread(count_primes, 0, PRIMES_N)

        async def process_cpu():
            # Отдельный процесс = отдельный GIL. Loop свободен полностью.
            # Цена: аргументы и результат проходят через pickle (здесь копейки).
            await loop.run_in_executor(proc_pool, count_primes, 0, PRIMES_N)

        results = []
        print("I/O-подобная нагрузка (ожидание 0.3 с x3):")
        results.append(await scenario("asyncio.sleep (ok)", good_sleep))
        results.append(await scenario("time.sleep (BAD)", bad_sleep))
        results.append(await scenario("to_thread(sleep) (ok)", thread_sleep))
        print("CPU-нагрузка (count_primes x3):")
        results.append(await scenario("cpu in loop (BAD)", bad_cpu))
        results.append(await scenario("to_thread(cpu)", thread_cpu))
        results.append(await scenario("process pool(cpu) (ok)", process_cpu))
        return results


def main() -> None:
    results = asyncio.run(main_async())
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "exp03_blocking_in_async.json").write_text(
        json.dumps({"system": system_info(), "results": results}, indent=2, ensure_ascii=False))
    print("\n  -> сохранено в results/exp03_blocking_in_async.json")


if __name__ == "__main__":
    main()
