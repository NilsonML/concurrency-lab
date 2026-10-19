"""Эксперимент 01. I/O-bound: N HTTP-запросов к медленному серверу.

ВОПРОС: как быстрее всего сделать много сетевых запросов —
последовательно, потоками, процессами или asyncio? И как время зависит
от уровня конкурентности (сколько запросов "в полёте" одновременно)?

ГИПОТЕЗА:
  - Последовательно: время ≈ N * delay (ждём каждый ответ по очереди).
  - Потоки и asyncio дадут почти одинаковый выигрыш: при ожидании сети
    GIL отпускается, так что потоки не мешают друг другу.
  - asyncio будет экономнее по CPU и памяти и лучше масштабироваться
    на большую конкурентность.
  - Процессы тоже ускорят, но заплатят за создание процессов (дорого).

ТЕОРЕТИЧЕСКИЙ МИНИМУМ при конкурентности c:
  ceil(N / c) * delay — "волны" по c запросов, каждая длится delay.
  Сравнение с ним показывает накладные расходы подхода.

Запуск:  python experiments/exp01_io_bound.py
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import httpx

from concurrency_lab.bench import measure, save_results
from concurrency_lab.server import running_server

N_REQUESTS = 100  # сколько запросов делаем за один прогон
DELAY = 0.05  # задержка ответа сервера, сек (как типичный ответ БД/сервиса)
REPEATS = 3  # повторов на вариант -> mean ± std
CONCURRENCY = [1, 4, 16, 64, 100]  # одинаковые уровни для потоков и asyncio = честное сравнение


# ---------------------------------------------------------------------------
# 1. Последовательно — бейзлайн. Без бейзлайна любые "ускорения" бессмысленны.
# ---------------------------------------------------------------------------
def run_sequential(url: str) -> None:
    # Один Client на все запросы: он держит пул соединений (keep-alive).
    # Если вызывать httpx.get() каждый раз, на КАЖДЫЙ запрос будет новое
    # TCP-соединение — это частая ошибка, которая сильно замедляет код.
    with httpx.Client() as client:
        for _ in range(N_REQUESTS):
            client.get(url).raise_for_status()


# ---------------------------------------------------------------------------
# 2. Потоки (threading через ThreadPoolExecutor)
# ---------------------------------------------------------------------------
# Почему ThreadPoolExecutor, а не "голые" threading.Thread:
#   - пул ограничивает число потоков (1000 задач != 1000 потоков);
#   - потоки переиспользуются (создание потока — системный вызов, не бесплатно);
#   - удобные Future: результат и ИСКЛЮЧЕНИЯ возвращаются в главный поток.
#     В голом Thread исключение просто напечатается и потеряется.
#
# ДВА ВАРИАНТА — и это главная находка эксперимента (см. README, "Что пошло не так"):
#
# 2a) Свой httpx.Client в каждом потоке (threading.local).
#     Первая версия эксперимента была написана так: "у каждого потока свой
#     клиент — никакой конкуренции за общий объект". Оказалось, что создание
#     httpx.Client стоит ~50 мс CPU (загрузка CA-сертификатов в SSL-контекст,
#     даже для http://). 64 потока = 64 клиента = ~3 c CPU, и под GIL эта
#     работа выполняется по очереди. Потоки при c=64 стали МЕДЛЕННЕЕ, чем при c=16.
#
# 2b) Один общий Client на все потоки.
#     Пул соединений httpx/httpcore защищён блокировками, поэтому общий клиент
#     можно использовать из нескольких потоков. Дорогая инициализация
#     делается один раз. Так и нужно делать в реальном коде.
_thread_local = threading.local()


def _get_thread_local_client(url: str) -> int:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = _thread_local.client = httpx.Client()  # <- дорогая операция в КАЖДОМ потоке
    # Во время ожидания ответа (socket.recv) поток ОТПУСКАЕТ GIL — другие
    # потоки работают. Поэтому потоки и помогают в I/O-задачах.
    return client.get(url).status_code


def run_threads_local_client(url: str, workers: int) -> None:
    # Почему ThreadPoolExecutor, а не "голые" threading.Thread:
    #   - пул ограничивает число потоков (1000 задач != 1000 потоков);
    #   - потоки переиспользуются (создание потока — системный вызов);
    #   - Future возвращают результат и ИСКЛЮЧЕНИЯ в главный поток.
    #     В голом Thread исключение просто напечатается в stderr и потеряется.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        codes = list(pool.map(_get_thread_local_client, [url] * N_REQUESTS))
    assert all(c == 200 for c in codes)


def run_threads_shared_client(url: str, workers: int) -> None:
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    with httpx.Client(limits=limits) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        # list(...) дожидается всех результатов и пробрасывает исключение,
        # если какой-то запрос упал.
        codes = list(pool.map(lambda u: client.get(u).status_code, [url] * N_REQUESTS))
    assert all(c == 200 for c in codes)


# ---------------------------------------------------------------------------
# 3. asyncio
# ---------------------------------------------------------------------------
async def _run_async(url: str, concurrency: int) -> None:
    # Semaphore ограничивает число одновременных запросов.
    # БЕЗ ограничения на 100 000 задач мы бы открыли 100 000 соединений
    # и уронили бы сервер / упёрлись в лимит файловых дескрипторов ОС.
    # Ограничение конкурентности — обязательная практика в проде.
    sem = asyncio.Semaphore(concurrency)

    # Лимиты пула соединений httpx (по умолчанию max_connections=100).
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(limits=limits) as client:

        async def fetch() -> None:
            async with sem:
                # await = точка переключения: пока ждём ответ, event loop
                # отправляет другие запросы. Один поток, нет GIL-конкуренции.
                r = await client.get(url)
                r.raise_for_status()

        # TaskGroup (Python 3.11+) — "структурная конкурентность":
        # если одна задача упадёт, остальные будут отменены, а исключение
        # проброшено. Старый asyncio.gather по умолчанию так НЕ делает —
        # остальные задачи продолжают висеть "сиротами".
        async with asyncio.TaskGroup() as tg:
            for _ in range(N_REQUESTS):
                tg.create_task(fetch())


def run_asyncio(url: str, concurrency: int) -> None:
    # asyncio.run создаёт новый event loop, выполняет корутину и закрывает loop.
    # В реальном приложении run вызывается ОДИН раз в точке входа.
    asyncio.run(_run_async(url, concurrency))


# ---------------------------------------------------------------------------
# 4. Процессы — для полноты картины (для I/O так делать НЕ стоит)
# ---------------------------------------------------------------------------
_process_client: httpx.Client | None = None


def _get_in_process(url: str) -> int:
    # У каждого процесса своя память, поэтому глобальная переменная здесь —
    # "своя" для каждого процесса. Клиент создаём лениво, один раз на процесс.
    # (Первая версия делала httpx.get() на каждый запрос = новый клиент и новое
    # TCP-соединение на каждый запрос — это было в разы медленнее.)
    global _process_client
    if _process_client is None:
        _process_client = httpx.Client()
    return _process_client.get(url).status_code


def run_processes(url: str, workers: int) -> None:
    # Пул создаётся ВНУТРИ замера специально: стоимость запуска процессов —
    # часть честной цены этого подхода для разовой задачи.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        codes = list(pool.map(_get_in_process, [url] * N_REQUESTS))
    assert all(c == 200 for c in codes)


def main() -> None:
    measurements = []
    with running_server() as base:
        url = f"{base}/?delay={DELAY}"
        httpx.get(url)  # "прогрев": первый запрос включает импорт/инициализацию

        print(f"N={N_REQUESTS}, delay={DELAY}s, repeats={REPEATS}")
        print("Бейзлайн:")
        measurements.append(
            measure(
                "sequential",
                lambda: run_sequential(url),
                REPEATS,
                approach="sequential",
                concurrency=1,
            )
        )

        print("Потоки, свой клиент в каждом потоке (первая версия):")
        for c in CONCURRENCY:
            measurements.append(
                measure(
                    f"threads-local c={c}",
                    lambda c=c: run_threads_local_client(url, c),
                    REPEATS,
                    approach="threads_local_client",
                    concurrency=c,
                )
            )
            # lambda c=c — фиксируем текущее значение c. Без этого все lambda
            # захватили бы ПЕРЕМЕННУЮ c (late binding) и увидели бы последнее значение.

        print("Потоки, общий клиент (исправленная версия):")
        for c in CONCURRENCY:
            measurements.append(
                measure(
                    f"threads-shared c={c}",
                    lambda c=c: run_threads_shared_client(url, c),
                    REPEATS,
                    approach="threads",
                    concurrency=c,
                )
            )

        print("asyncio:")
        for c in CONCURRENCY:
            measurements.append(
                measure(
                    f"asyncio c={c}",
                    lambda c=c: run_asyncio(url, c),
                    REPEATS,
                    approach="asyncio",
                    concurrency=c,
                )
            )

        print("Процессы:")
        for w in sorted({2, os.cpu_count() or 2, 16}):
            # (у пула процессов есть ещё накладные расходы: pickle аргументов и
            #  результатов, запуск интерпретаторов — порядка десятков мс на процесс)
            measurements.append(
                measure(
                    f"processes w={w}",
                    lambda w=w: run_processes(url, w),
                    REPEATS,
                    approach="processes",
                    concurrency=w,
                )
            )

    theory = {c: math.ceil(N_REQUESTS / c) * DELAY for c in CONCURRENCY}
    save_results(
        "exp01_io_bound",
        measurements,
        {
            "config": {"n_requests": N_REQUESTS, "delay": DELAY, "repeats": REPEATS},
            "theoretical_min": theory,
        },
    )


# Охрана __main__ ОБЯЗАТЕЛЬНА для multiprocessing на Windows и macOS:
# там дочерние процессы стартуют методом "spawn" — заново импортируют этот
# модуль. Без охраны каждый дочерний процесс снова запустил бы main() ->
# бесконечное порождение процессов.
if __name__ == "__main__":
    main()
