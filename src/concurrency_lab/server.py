"""Локальный тестовый HTTP-сервер с управляемой задержкой.

ПОЧЕМУ СВОЙ ЛОКАЛЬНЫЙ СЕРВЕР, А НЕ НАСТОЯЩИЕ САЙТЫ:
  1. Воспроизводимость. У реального сайта задержка скачет (сеть, нагрузка,
     CDN), результаты будут шуметь сильнее, чем различия между подходами.
  2. Контроль. Мы задаём задержку точно (?delay=0.05) и знаем теоретический
     минимум времени, с которым можно сравнить результат.
  3. Этика. 1000 параллельных запросов к чужому сайту — это мини-DDoS,
     за который можно получить бан (и это просто некрасиво).

ПОЧЕМУ СЕРВЕР НА "ГОЛОМ" asyncio, БЕЗ ФРЕЙМВОРКА:
  - Нет внешних зависимостей.
  - Это сам по себе учебный пример: один поток, один event loop обслуживает
    сотни одновременных соединений. Пока одно соединение "спит"
    (await asyncio.sleep), loop переключается на другие.
  - Мы реализуем минимум HTTP/1.1: GET, Content-Length и keep-alive
    (повторное использование TCP-соединения — так работают пулы соединений
    в httpx/requests, и без keep-alive мы бы мерили в основном
    установку TCP-соединений).

ПОЧЕМУ СЕРВЕР ЗАПУСКАЕТСЯ В ОТДЕЛЬНОМ ПРОЦЕССЕ:
  Если запустить его потоком внутри процесса-клиента, сервер и клиент
  будут делить один GIL и мешать друг другу — мы бы мерили их
  взаимное влияние, а не подход клиента.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
import time
from contextlib import contextmanager
from urllib.parse import parse_qs, urlsplit


async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Обслуживает одно TCP-соединение (возможно, много запросов подряд — keep-alive).

    Каждое соединение — это отдельная КОРУТИНА. asyncio.start_server создаёт её
    сам на каждое новое подключение. Корутины дешёвые (~1-2 КБ памяти),
    поэтому тысячи соединений — норма. С потоками на соединение (модель
    "thread per connection") мы бы упёрлись в память и переключения ОС.
    """
    try:
        while True:
            # Читаем заголовки запроса до пустой строки. await = "отдай управление
            # event loop'у, пока данных нет". Поток при этом НЕ блокируется.
            raw = await reader.readuntil(b"\r\n\r\n")
            request_line = raw.split(b"\r\n", 1)[0].decode()
            _method, target, _version = request_line.split(" ", 2)

            query = parse_qs(urlsplit(target).query)
            delay = float(query.get("delay", ["0"])[0])

            # Имитация "медленной" работы: запроса в БД, другого сервиса и т.д.
            # ВАЖНО: asyncio.sleep, а НЕ time.sleep. time.sleep заблокировал бы
            # ВЕСЬ сервер на это время (см. эксперимент 03).
            await asyncio.sleep(delay)

            body = b"ok"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: keep-alive\r\n\r\n" + body
            )
            await writer.drain()  # ждём, пока буфер отправки освободится (backpressure)
    except (asyncio.IncompleteReadError, ConnectionResetError, ValueError):
        pass  # клиент закрыл соединение — нормальная ситуация
    finally:
        writer.close()


async def _serve(port: int, ready: mp.Event) -> None:
    # backlog — длина очереди ещё не принятых соединений. По умолчанию 100;
    # при сотнях одновременных клиентов его стоит увеличить, иначе часть
    # подключений получит отказ / будет ждать.
    server = await asyncio.start_server(_handle_connection, "127.0.0.1", port, backlog=2048)
    ready.set()
    async with server:
        await server.serve_forever()


def _run(port: int, ready: mp.Event) -> None:
    asyncio.run(_serve(port, ready))


def _free_port() -> int:
    """Просим ОС дать свободный порт (bind на порт 0) — так тесты не конфликтуют
    между собой и с другими программами на фиксированном порту вроде 8000."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def running_server():
    """Контекстный менеджер: поднимает сервер в отдельном процессе и гарантированно
    гасит его в finally — даже если эксперимент упал с ошибкой.
    Это паттерн RAII / "with" — ресурсы не утекают.

    Использование:
        with running_server() as base_url:
            httpx.get(f"{base_url}/?delay=0.1")
    """
    port = _free_port()
    ready = mp.Event()
    proc = mp.Process(target=_run, args=(port, ready), daemon=True)
    proc.start()
    if not ready.wait(timeout=10):
        proc.terminate()
        raise RuntimeError("тестовый сервер не стартовал за 10 секунд")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.join(timeout=5)


if __name__ == "__main__":
    # Ручной запуск для проверки: python -m concurrency_lab.server
    # затем в другом терминале: curl "http://127.0.0.1:<port>/?delay=1"
    with running_server() as url:
        print(f"Сервер слушает {url}  (Ctrl+C для остановки)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
