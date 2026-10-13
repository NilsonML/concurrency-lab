"""Строит графики из results/*.json.

Почему отдельный скрипт, а не графики внутри экспериментов:
  - эксперимент может идти минутами, а стиль графика хочется менять часто;
  - графики можно перестроить без перезапуска замеров;
  - данные (JSON) — первичны, картинки — производные.

Запуск:  python scripts/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI: работает на сервере и в CI
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Фиксированный порядок цветов: одна и та же сущность всегда одного цвета
# на всех графиках (asyncio всегда синий, потоки всегда оранжевые и т.д.).
COLORS = {
    "asyncio": "#2a78d6",
    "threads": "#eb6834",
    "processes": "#1baf7a",
    "threads_local_client": "#e87ba4",
    "sequential": "#52514e",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def plot_io_bound() -> None:
    data = load("exp01_io_bound")
    ms = data["measurements"]
    fig, ax = plt.subplots(figsize=(8, 4.8))

    labels = {
        "threads_local_client": "потоки, клиент в каждом потоке (ошибка)",
        "threads": "потоки, общий клиент",
        "asyncio": "asyncio",
        "processes": "процессы",
    }
    for approach, label in labels.items():
        pts = sorted((m["params"]["concurrency"], m["mean"], m["std"])
                     for m in ms if m["params"]["approach"] == approach)
        if not pts:
            continue
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, marker="o", markersize=5, linewidth=2, capsize=3,
                    color=COLORS[approach], label=label)

    theory = sorted((int(k), v) for k, v in data["theoretical_min"].items())
    ax.plot(*zip(*theory), linestyle="--", color=MUTED, linewidth=1.2,
            label="теоретический минимум ⌈N/c⌉·delay")
    seq = next(m["mean"] for m in ms if m["params"]["approach"] == "sequential")
    ax.axhline(seq, color=MUTED, linewidth=1, linestyle=":")
    ax.text(1.05, seq * 1.08, f"последовательно: {seq:.1f} c", color=MUTED, fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("конкурентность (одновременных запросов / воркеров)")
    ax.set_ylabel("время на 100 запросов, c (log)")
    cfg = data["config"]
    ax.set_title(f"I/O-bound: {cfg['n_requests']} запросов, задержка сервера "
                 f"{int(cfg['delay'] * 1000)} мс", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp01_io_bound.png", dpi=150)
    plt.close(fig)


def plot_cpu_bound() -> None:
    data = load("exp02_cpu_bound")
    ms = data["measurements"]
    cores = data["system"]["cpu_count"]
    fig, ax = plt.subplots(figsize=(8, 4.4))

    workloads = {"primes": "чистый Python\n(count_primes)", "hash": "C-код, отпускает GIL\n(hashlib.sha256)"}
    approaches = ["threads", "processes"]
    width = 0.34
    for i, (wl, wl_label) in enumerate(workloads.items()):
        seq = next(m["mean"] for m in ms if m["params"]["workload"] == wl
                   and m["params"]["approach"] == "sequential")
        for j, ap in enumerate(approaches):
            m = next(m for m in ms if m["params"]["workload"] == wl
                     and m["params"]["approach"] == ap and m["params"]["workers"] == cores)
            speedup = seq / m["mean"]
            x = i + (j - 0.5) * (width + 0.02)
            ax.bar(x, speedup, width, color=COLORS[ap],
                   label={"threads": "потоки", "processes": "процессы"}[ap] if i == 0 else None)
            ax.text(x, speedup + 0.04, f"{speedup:.2f}×", ha="center", color=INK, fontsize=10)

    ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.text(-0.42, 1.03, "без ускорения (1×)", color=MUTED, fontsize=9, ha="left")
    ax.set_xticks(range(len(workloads)), list(workloads.values()))
    ax.set_ylabel("ускорение относительно последовательного")
    ax.set_ylim(0, max(2.3, cores + 0.3))
    gil = "GIL включён" if data["system"]["gil_enabled"] else "GIL выключен"
    ax.set_title(f"CPU-bound: {cores} воркера на {cores} ядрах, Python "
                 f"{data['system']['python']} ({gil})", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp02_cpu_bound.png", dpi=150)
    plt.close(fig)


def plot_blocking() -> None:
    data = load("exp03_blocking_in_async")["results"]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["scenario"] for r in data][::-1]
    lags = [r["max_lag_ms"] for r in data][::-1]
    bad = ["BAD" in n for n in names]
    colors = ["#e34948" if b else COLORS["asyncio"] for b in bad]
    ax.barh(names, lags, color=colors, height=0.6)
    for y, v in enumerate(lags):
        ax.text(v * 1.08, y, f"{v:.0f} мс", va="center", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("максимальное опоздание пульса event loop, мс (log)")
    ax.set_title("Блокирующий код в asyncio: насколько «замерзает» event loop", loc="left")
    ax.axvline(10, color=MUTED, linestyle="--", linewidth=1)
    ax.text(10.5, len(names) - 0.45, "период пульса 10 мс", color=MUTED, fontsize=8)
    ax.set_xlim(1, max(lags) * 3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp03_blocking_in_async.png", dpi=150)
    plt.close(fig)


def plot_memory() -> None:
    data = load("exp05_memory_footprint")["results"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for kind, label, color in (("thread", "потоки", COLORS["threads"]),
                               ("coroutine", "корутины (asyncio)", COLORS["asyncio"])):
        pts = [(r["n"], max(r["rss_delta_mb"], 0.05)) for r in data if r["kind"] == kind]
        x, y = zip(*pts)
        ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=label)
        ax.annotate(f"{y[-1]:.0f} МБ на {x[-1]:,}".replace(",", " "), (x[-1], y[-1]),
                    textcoords="offset points", xytext=(-10, 8), ha="right", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("сколько одновременно ожидающих единиц")
    ax.set_ylabel("прирост RSS, МБ (log)")
    ax.set_title("Память: ~15 КБ на поток против <1 КБ на корутину", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp05_memory_footprint.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_io_bound()
    plot_cpu_bound()
    plot_blocking()
    plot_memory()
    print("Графики сохранены в results/*.png")
