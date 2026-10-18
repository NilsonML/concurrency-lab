"""Эксперимент 04. Гонки данных: потоки (вытесняющая многозадачность)
против asyncio (кооперативная многозадачность).

ВОПРОС: защищает ли GIL от гонок? Бывают ли гонки в asyncio, где всего
один поток? Как их чинить?

ГЛАВНАЯ ИДЕЯ:
  - Потоки переключает ОС/интерпретатор В ЛЮБОЙ момент между байткод-
    инструкциями. Ты не контролируешь, где тебя прервут -> гонка возможна
    в любом месте, где читается и затем пишется общее состояние.
  - asyncio переключается ТОЛЬКО на await. Между двумя await код атомарен
    относительно других корутин. Гонка возможна только если между
    "проверить" и "изменить" стоит await.

Запуск:  python experiments/exp04_race_conditions.py
"""

from __future__ import annotations

import asyncio
import dis
import sys
import threading
import time

THREADS = 8
INCREMENTS = 100_000


# ---------------------------------------------------------------------------
# 1. counter += 1 — НЕ атомарная операция
# ---------------------------------------------------------------------------
class Counter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def increment_split(self) -> None:
        """Та же логика, но между чтением и записью есть вызов функции."""
        current = self.value
        current = _plus_one(
            current
        )  # вызов функции = точка, где CPython проверяет "не пора ли отдать GIL"
        self.value = current


def _plus_one(x: int) -> int:
    return x + 1


def show_bytecode() -> None:
    """Смотрим, во что компилируется self.value += 1.

    Это несколько инструкций: прочитать атрибут, прибавить, записать обратно.
    Если поток прервут после чтения, но до записи — другой поток успеет
    записать своё значение, а мы затем перезапишем его своим устаревшим.
    Это классический "lost update".
    """
    print("Байткод `self.value += 1`:")
    for ins in dis.get_instructions(Counter.increment):
        if ins.opname not in ("RESUME", "RETURN_CONST", "RETURN_VALUE", "LOAD_CONST"):
            print(f"    {ins.opname:<20} {ins.argrepr}")


def run_counter(use_lock: bool, split: bool = False) -> int:
    counter = Counter()
    lock = threading.Lock()
    inc = counter.increment_split if split else counter.increment

    def worker() -> None:
        for _ in range(INCREMENTS):
            if use_lock:
                # with lock: гарантирует release даже при исключении.
                # Ручные lock.acquire()/lock.release() без try/finally — путь к deadlock.
                with lock:
                    inc()
            else:
                inc()

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # ждём завершения всех потоков
    return counter.value


# ---------------------------------------------------------------------------
# 2. check-then-act: "проверил баланс -> списал"
# ---------------------------------------------------------------------------
def bank_threads(use_lock: bool) -> int:
    """10 потоков пытаются снять по 100 со счёта, где 100. Должен успеть один."""
    balance = {"value": 100}
    lock = threading.Lock()

    def withdraw() -> None:
        def critical() -> None:
            if balance["value"] >= 100:  # check
                time.sleep(0.001)  # тут поток почти наверняка переключится
                balance["value"] -= 100  # act

        if use_lock:
            with lock:
                critical()
        else:
            critical()

    threads = [threading.Thread(target=withdraw) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return balance["value"]


async def bank_async(await_between: bool, use_lock: bool) -> int:
    """То же самое на asyncio."""
    balance = {"value": 100}
    lock = asyncio.Lock()

    async def critical() -> None:
        if balance["value"] >= 100:
            if await_between:
                await asyncio.sleep(0.001)  # например, запрос в БД между проверкой и списанием
            balance["value"] -= 100

    async def withdraw() -> None:
        if use_lock:
            async with (
                lock
            ):  # asyncio.Lock, НЕ threading.Lock! threading.Lock заблокировал бы весь loop
                await critical()
        else:
            await critical()

    await asyncio.gather(*(withdraw() for _ in range(10)))
    return balance["value"]


# ---------------------------------------------------------------------------
# 3. Deadlock: два потока берут два лока в разном порядке
# ---------------------------------------------------------------------------
def deadlock_demo(ordered: bool) -> str:
    a, b = threading.Lock(), threading.Lock()
    status: list[str] = []
    barrier = threading.Barrier(2)  # оба потока возьмут первый лок одновременно

    def worker(first: threading.Lock, second: threading.Lock, name: str) -> None:
        with first:
            barrier.wait()
            # timeout вместо вечного ожидания — чтобы демо не повесило программу.
            # В проде acquire(timeout=...) тоже полезная защита.
            if second.acquire(timeout=1):
                second.release()
                status.append(f"{name}: ok")
            else:
                status.append(f"{name}: DEADLOCK (не дождался второго лока)")

    if ordered:
        # ИСПРАВЛЕНИЕ: глобальный порядок захвата — все берут сначала a, потом b.
        # Тогда цикл ожидания невозможен. Барьер тут не нужен (он бы сам
        # создал взаимоблокировку), поэтому разрешим проходить сразу.
        barrier = threading.Barrier(1)
        t1 = threading.Thread(target=worker, args=(a, b, "T1"))
        t2 = threading.Thread(target=worker, args=(a, b, "T2"))
    else:
        t1 = threading.Thread(target=worker, args=(a, b, "T1"))
        t2 = threading.Thread(target=worker, args=(b, a, "T2"))  # обратный порядок!
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return "; ".join(sorted(status))


def main() -> None:
    show_bytecode()

    expected = THREADS * INCREMENTS
    print(f"\n1) counter += 1, {THREADS} потоков x {INCREMENTS}, ожидаем {expected}")
    default_interval = sys.getswitchinterval()
    # switchinterval — как часто интерпретатор просит поток отдать GIL
    # (по умолчанию 5 мс). Уменьшая его, мы делаем переключения частыми,
    # и скрытая гонка проявляется чаще. Баг был всегда — мы лишь чаще его ловим.
    #
    # НЮАНС, найденный в ходе эксперимента: в CPython 3.10+ поток отдаёт GIL
    # не между ЛЮБЫМИ инструкциями, а только в определённых точках (вызовы
    # функций, переходы назад в циклах). Внутри `self.value += 1` таких точек
    # нет, поэтому на обычном CPython с GIL этот счётчик "случайно" не теряет
    # обновления. Это ДЕТАЛЬ РЕАЛИЗАЦИИ, а не гарантия: на free-threaded
    # сборке (3.13t/3.14t), в PyPy или в будущей версии гонка проявится.
    # Стоит добавить между чтением и записью вызов функции — и она видна сразу.
    for split in (False, True):
        label = "a.value += 1          " if not split else "чтение; f(); запись    "
        for interval in (default_interval, 1e-6):
            sys.setswitchinterval(interval)
            got = run_counter(use_lock=False, split=split)
            print(f"   {label} без lock, switch={interval:g}s: {got:>8}  потеряно {expected - got}")
    got = run_counter(use_lock=True, split=True)
    print(
        f"   чтение; f(); запись     с lock,   switch=1e-06s: {got:>8}  потеряно {expected - got}"
    )
    sys.setswitchinterval(default_interval)

    print("\n2) check-then-act, 10 снятий по 100 при балансе 100 (верный итог: 0)")
    print(f"   threads без lock:           баланс = {bank_threads(False)}")
    print(f"   threads с lock:             баланс = {bank_threads(True)}")
    print(f"   asyncio без await внутри:   баланс = {asyncio.run(bank_async(False, False))}")
    print(f"   asyncio с await, без lock:  баланс = {asyncio.run(bank_async(True, False))}")
    print(f"   asyncio с await, с lock:    баланс = {asyncio.run(bank_async(True, True))}")

    print("\n3) Deadlock")
    print(f"   разный порядок захвата: {deadlock_demo(ordered=False)}")
    print(f"   единый порядок захвата: {deadlock_demo(ordered=True)}")


if __name__ == "__main__":
    main()
