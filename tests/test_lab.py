"""Тесты. Зачем они в исследовательском проекте:
бенчмарк, который считает неправильно, хуже, чем никакого. Тесты проверяют,
что все варианты (последовательный, потоки, процессы, asyncio) дают
ОДИНАКОВЫЙ результат — значит, мы сравниваем скорость одной и той же работы.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from concurrency_lab.server import running_server
from concurrency_lab.workloads import count_primes, split_range

# Эксперименты лежат вне пакета — добавляем папку в путь, чтобы их импортировать.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import exp04_race_conditions as races  # noqa: E402


def test_count_primes_known_values():
    # Известные значения функции π(n): простых чисел меньше 10 — четыре (2, 3, 5, 7)
    assert count_primes(0, 10) == 4
    assert count_primes(0, 100) == 25
    assert count_primes(0, 1000) == 168


def test_split_range_covers_everything():
    bounds = split_range(1003, 7)
    assert bounds[0][0] == 0 and bounds[-1][1] == 1003
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))  # без дыр и пересечений


@pytest.mark.parametrize("executor", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_parallel_primes_equal_sequential(executor):
    chunks = split_range(20_000, 8)
    expected = sum(count_primes(a, b) for a, b in chunks)
    with executor(max_workers=2) as pool:
        assert sum(pool.map(count_primes, *zip(*chunks))) == expected


def test_server_responds_and_respects_delay():
    with running_server() as url:
        r = httpx.get(f"{url}/?delay=0.2")
        assert r.status_code == 200 and r.text == "ok"
        assert r.elapsed.total_seconds() >= 0.2


def test_locked_counter_is_exact():
    assert races.run_counter(use_lock=True) == races.THREADS * races.INCREMENTS


def test_bank_with_lock_never_overdraws():
    assert races.bank_threads(use_lock=True) == 0
    assert asyncio.run(races.bank_async(await_between=True, use_lock=True)) == 0


def test_async_without_await_is_atomic():
    # Ключевое свойство кооперативной многозадачности: без await переключения нет
    assert asyncio.run(races.bank_async(await_between=False, use_lock=False)) == 0
