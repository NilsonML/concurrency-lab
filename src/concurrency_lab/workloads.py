"""Типовые нагрузки для экспериментов.

ВАЖНО для multiprocessing: функции, которые выполняются в других процессах,
должны быть определены на верхнем уровне модуля (не lambda, не вложенные).
Причина: аргументы и сама функция передаются в дочерний процесс через
pickle, а pickle сохраняет функцию как ссылку "модуль.имя". Lambda и
вложенные функции так сериализовать нельзя -> PicklingError.
"""

from __future__ import annotations

import hashlib


def count_primes(start: int, stop: int) -> int:
    """Чистая CPU-нагрузка на чистом Python: считаем простые числа перебором.

    Специально "медленный" наивный алгоритм: нам нужна ДОЛГАЯ работа
    интерпретатора Python, которая всё время держит GIL.
    (Для реальной задачи взяли бы решето Эратосфена.)
    """
    count = 0
    for n in range(max(start, 2), stop):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            count += 1
    return count


def hash_bytes(data: bytes, rounds: int) -> str:
    """CPU-нагрузка внутри C-кода, который ОТПУСКАЕТ GIL.

    hashlib для данных больше ~2 КБ освобождает GIL на время хеширования.
    То же делают NumPy (большинство операций), zlib, чтение файлов, сеть,
    PyTorch и др. Отсюда важный вывод: "потоки бесполезны для CPU-задач" —
    это правда только для ЧИСТОГО Python-кода.
    """
    digest = b""
    for _ in range(rounds):
        digest = hashlib.sha256(data + digest).digest()
    return digest.hex()


def split_range(stop: int, chunks: int) -> list[tuple[int, int]]:
    """Делим [0, stop) на `chunks` кусков для раздачи воркерам.

    Нюанс: проверка больших чисел на простоту дороже, чем маленьких,
    поэтому равные по длине куски НЕ равны по работе. С числом кусков,
    кратным числу воркеров и большим его (например, 16 кусков на 2-4
    воркера), пул сам выравнивает нагрузку: освободившийся воркер берёт
    следующий кусок. Это называется load balancing / work stealing.
    """
    step = stop // chunks
    bounds = [(i * step, (i + 1) * step) for i in range(chunks)]
    bounds[-1] = (bounds[-1][0], stop)
    return bounds
