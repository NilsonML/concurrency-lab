"""Эксперимент 02. CPU-bound: где GIL мешает, а где нет.

ВОПРОС: ускоряют ли потоки вычислительную задачу? А процессы?
И важно ли, ЧТО именно считается — чистый Python или C-код?

ДВЕ НАГРУЗКИ:
  A) count_primes — чистый Python. Интерпретатор всё время держит GIL.
  B) hash_bytes   — sha256 в C-коде hashlib, который ОТПУСКАЕТ GIL.

ГИПОТЕЗА (обычный CPython с GIL):
  A) потоки: ускорения нет (может быть даже медленнее из-за переключений);
     процессы: ускорение ≈ min(число воркеров, число ядер).
  B) потоки: ускорение есть, т.к. GIL отпущен во время хеширования.
  На free-threaded сборке (python3.13t / 3.14t) потоки ускорят и A.

Почему считаем speedup = t_sequential / t_variant: это безразмерная
величина, её можно сравнивать между машинами. Абсолютные секунды —
нельзя (у всех разные процессоры).

Запуск:  python experiments/exp02_cpu_bound.py
Сравнить без GIL:  python3.14t experiments/exp02_cpu_bound.py
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from concurrency_lab.bench import gil_enabled, measure, save_results
from concurrency_lab.workloads import count_primes, hash_bytes, split_range

PRIMES_UP_TO = 600_000   # подобрано так, чтобы последовательный прогон шёл ~1-3 c
CHUNKS = 16              # кусков работы больше, чем воркеров -> балансировка нагрузки
HASH_DATA = os.urandom(1_000_000)  # 1 МБ — заведомо больше порога, после которого hashlib отпускает GIL
HASH_TASKS = 16
HASH_ROUNDS = 60
REPEATS = 3


def _primes_task(bounds: tuple[int, int]) -> int:
    return count_primes(*bounds)


def _hash_task(_: int) -> str:
    return hash_bytes(HASH_DATA, HASH_ROUNDS)


def primes_sequential() -> int:
    return sum(_primes_task(b) for b in split_range(PRIMES_UP_TO, CHUNKS))


def primes_pool(executor_cls, workers: int) -> int:
    # Один и тот же код для потоков и процессов: у ThreadPoolExecutor и
    # ProcessPoolExecutor общий интерфейс (concurrent.futures.Executor).
    # Это удобно: сравниваем ТОЛЬКО модель исполнения, остальное одинаково.
    with executor_cls(max_workers=workers) as pool:
        return sum(pool.map(_primes_task, split_range(PRIMES_UP_TO, CHUNKS)))


def hash_sequential() -> None:
    for i in range(HASH_TASKS):
        _hash_task(i)


def hash_threads(workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_hash_task, range(HASH_TASKS)))


def hash_processes(workers: int) -> None:
    # Важная деталь: HASH_DATA (1 МБ) — глобальная переменная модуля.
    # При spawn (Windows/macOS) каждый процесс заново импортирует модуль и
    # сгенерирует СВОИ случайные данные; при fork (Linux) получит копию.
    # Для хеширования это не важно, но запомни: глобальное состояние в
    # multiprocessing ведёт себя по-разному на разных ОС.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_hash_task, range(HASH_TASKS)))


def main() -> None:
    cores = os.cpu_count() or 2
    workers_list = sorted({2, cores, cores * 2})
    print(f"Ядер: {cores}, GIL включён: {gil_enabled()}")

    # Проверка корректности ДО замеров: параллельная версия должна давать
    # тот же ответ. Быстрый, но неправильный код никому не нужен.
    expected = primes_sequential()
    assert primes_pool(ThreadPoolExecutor, 2) == expected
    assert primes_pool(ProcessPoolExecutor, 2) == expected

    m = []
    print("\nA) Чистый Python (count_primes):")
    m.append(measure("primes sequential", primes_sequential, REPEATS,
                     workload="primes", approach="sequential", workers=1))
    for w in workers_list:
        m.append(measure(f"primes threads w={w}", lambda w=w: primes_pool(ThreadPoolExecutor, w),
                         REPEATS, workload="primes", approach="threads", workers=w))
    for w in workers_list:
        m.append(measure(f"primes processes w={w}", lambda w=w: primes_pool(ProcessPoolExecutor, w),
                         REPEATS, workload="primes", approach="processes", workers=w))

    print("\nB) C-код, отпускающий GIL (hashlib.sha256):")
    m.append(measure("hash sequential", hash_sequential, REPEATS,
                     workload="hash", approach="sequential", workers=1))
    for w in workers_list:
        m.append(measure(f"hash threads w={w}", lambda w=w: hash_threads(w),
                         REPEATS, workload="hash", approach="threads", workers=w))
    for w in workers_list:
        m.append(measure(f"hash processes w={w}", lambda w=w: hash_processes(w),
                         REPEATS, workload="hash", approach="processes", workers=w))

    tag = "" if gil_enabled() else "_nogil"
    save_results(f"exp02_cpu_bound{tag}", m,
                 {"config": {"primes_up_to": PRIMES_UP_TO, "chunks": CHUNKS,
                             "hash_tasks": HASH_TASKS, "hash_rounds": HASH_ROUNDS}})


if __name__ == "__main__":
    main()
