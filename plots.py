"""Tables and figures for the report, plus the offline threshold sweep.

    python plots.py sweep  --out results/sweep.csv
    python plots.py report --results results/full --out results/figures --sweep results/sweep.csv

The sweep replays every incoming task against history.jsonl at each threshold and scores
each reuse with the dataset's verify.py. It makes no model calls, and it is analysis only:
nothing it finds feeds back into policy.py.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import policy  # noqa: E402
import run  # noqa: E402

CONFIGS = list(run.CONFIGS)
COLORS = {"baseline": "#2a78d6", "task": "#eb6834", "content": "#1baf7a", "both": "#eda100"}
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 11)]
METRICS = [("accuracy", "Accuracy (%)"), ("tool_calls", "Tool calls / task"),
           ("model_requests", "Model requests / task"), ("input_tokens", "Input tokens / task"),
           ("wall_s", "Wall time / task (s)")]
FIGURES = ["efficiency_vs_accuracy.png", "threshold_sweep.png",
           "per_dataset_metrics.png", "overhead_vs_savings.png"]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 9, "legend.frameon": False})


# ---------------------------------------------------------------------------
# threshold sweep (no model calls)
# ---------------------------------------------------------------------------
def sweep(datasets, thresholds=THRESHOLDS):
    """Reuse decisions and their verified correctness at each threshold, gated or not."""
    rows = []
    for dataset in datasets:
        tasks, verifier = run.load_tasks(dataset, None), run.load_verifier(dataset)
        for mode in ("gated", "similarity_only"):
            for threshold in thresholds:
                task_policy = policy.TaskPolicy(run.env_dir(dataset), threshold, gated=(mode == "gated"))
                hits = [(t, task_policy.lookup(t)) for t in tasks]
                verdicts = [not verifier.shortfall(t["tid"], hit[0]) for t, hit in hits if hit]
                rows.append({"dataset": dataset, "threshold": threshold, "mode": mode,
                             "tasks": len(tasks), "reused": len(verdicts),
                             "correct_reuses": sum(verdicts), "false_reuses": verdicts.count(False),
                             "reuse_rate": len(verdicts) / len(tasks),
                             "precision": sum(verdicts) / len(verdicts) if verdicts else 1.0})
    return rows


# ---------------------------------------------------------------------------
# summary of run logs
# ---------------------------------------------------------------------------
def load_results(results):
    """{config: {dataset: rows}}, keeping only tids that every config finished."""
    logs = {c: {} for c in CONFIGS}
    for config in CONFIGS:
        folder = os.path.join(results, config)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            logs[config][name[:-len(".jsonl")]] = run.read_jsonl(os.path.join(folder, name))
    datasets = sorted(set.intersection(*(set(v) for v in logs.values())))
    for dataset in datasets:
        common = set.intersection(*({r["tid"] for r in logs[c][dataset]} for c in CONFIGS))
        for config in CONFIGS:
            logs[config][dataset] = [r for r in logs[config][dataset] if r["tid"] in common]
    return {c: {d: logs[c][d] for d in datasets} for c in CONFIGS}


def summarise(config, dataset, rows):
    n = max(len(rows), 1)
    mean = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    return {"config": config, "dataset": dataset, "tasks": len(rows),
            "accuracy": 100 * mean("passed"), "reuse_rate": 100 * mean("reused"),
            "tool_calls": mean("tool_calls"), "model_requests": mean("model_requests"),
            "input_tokens": mean("input_tokens"), "wall_s": mean("wall_s"),
            "overhead_s": sum(r["policy_s"] + r["content_s"] for r in rows) / n,
            "content_hits": mean("content_hits"), "est_cost": sum(r["est_cost"] for r in rows)}


def report(results, out, sweep_csv=None):
    """Write summary.csv and the four figures."""
    os.makedirs(out, exist_ok=True)
    logs = load_results(results)
    datasets = list(logs["baseline"])
    table = []
    for config in CONFIGS:
        table += [summarise(config, d, logs[config][d]) for d in datasets]
        table.append(summarise(config, "ALL", [r for d in datasets for r in logs[config][d]]))
    write_csv(table, os.path.join(out, "summary.csv"))
    cell = {(r["config"], r["dataset"]): r for r in table}

    _efficiency_vs_accuracy(cell, os.path.join(out, FIGURES[0]))
    _threshold_sweep(read_csv(sweep_csv) if sweep_csv else [], os.path.join(out, FIGURES[1]))
    _per_dataset(cell, datasets, os.path.join(out, FIGURES[2]))
    _overhead_vs_savings(cell, datasets, os.path.join(out, FIGURES[3]))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _efficiency_vs_accuracy(cell, path):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, config in enumerate(CONFIGS):  # staggered labels stay readable when points coincide
        r = cell[(config, "ALL")]
        ax.scatter(r["input_tokens"], r["accuracy"], s=70, color=COLORS[config],
                   edgecolor="white", linewidth=2, zorder=3, label=config)
        ax.annotate(config, (r["input_tokens"], r["accuracy"]), xytext=(8, 10 - 9 * i),
                    textcoords="offset points", color=INK)
    ax.set_ylim(80, 101)   # a full-scale-ish axis: one task is 1.6 points, not the whole plot
    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
    ax.set_xlabel("Mean input tokens per task")
    ax.set_ylabel("Verified accuracy (%)")
    ax.set_title("Efficiency vs correctness", color=INK, loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)


def _threshold_sweep(rows, path):
    datasets = sorted({r["dataset"] for r in rows})
    cols = min(4, max(1, len(datasets)))
    fig, axes = plt.subplots((len(datasets) + cols - 1) // cols or 1, cols, squeeze=False,
                             figsize=(3 * cols, 2.4 * max(1, (len(datasets) + cols - 1) // cols)))
    lines = [("gated", "reused", "#2a78d6", "-", "reused (gated)"),
             ("gated", "false_reuses", "#eb6834", "-", "false reuses (gated)"),
             ("similarity_only", "reused", "#2a78d6", "--", "reused (similarity only)"),
             ("similarity_only", "false_reuses", "#eb6834", "--", "false reuses (similarity only)")]
    for ax, dataset in zip(axes.flat, datasets):
        for mode, key, color, style, label in lines:
            points = sorted((r["threshold"], r[key]) for r in rows if r["dataset"] == dataset and r["mode"] == mode)
            ax.plot(*zip(*points), color=color, linestyle=style, linewidth=2, marker="o", markersize=3, label=label)
        ax.set_title(dataset, color=INK, loc="left")
        ax.set_xlabel("Similarity threshold")
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("Tasks")
    if datasets:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=4)
    fig.suptitle("Impact of the similarity threshold on task reuse", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, path)


def _grouped_bars(ax, datasets, configs, value):
    width = 0.8 / len(configs)
    for i, config in enumerate(configs):
        xs = [d + (i - (len(configs) - 1) / 2) * width for d in range(len(datasets))]
        ax.bar(xs, [value(config, d) for d in datasets], width * 0.9, color=COLORS[config], label=config)
    ax.set_xticks(range(len(datasets)), datasets)


def _per_dataset(cell, datasets, path):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(8, 2.2 * len(METRICS)), sharex=True)
    for ax, (key, label) in zip(axes, METRICS):
        _grouped_bars(ax, datasets, CONFIGS, lambda c, d: cell[(c, d)][key])
        ax.set_ylabel(label)
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.suptitle("Performance across datasets", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def _overhead_vs_savings(cell, datasets, path):
    configs = [c for c in CONFIGS if c != "baseline"]
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 3.2), sharey=True)
    for ax, config in zip(axes, configs):
        saved = [cell[("baseline", d)]["wall_s"] - cell[(config, d)]["wall_s"] for d in datasets]
        overhead = [cell[(config, d)]["overhead_s"] for d in datasets]
        xs = range(len(datasets))
        ax.bar([x - 0.2 for x in xs], saved, 0.36, color="#2a78d6", label="time saved vs baseline")
        ax.bar([x + 0.2 for x in xs], overhead, 0.36, color="#eb6834", label="MinHash overhead")
        ax.set_xticks(list(xs), datasets, rotation=45, ha="right")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_title(config, color=INK, loc="left")
    axes[0].set_ylabel("Seconds per task (symlog)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.suptitle("Computational overhead vs savings", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)


def _save(fig, path):
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------------------
# csv and command line
# ---------------------------------------------------------------------------
def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    def value(text):
        try:
            return float(text)
        except ValueError:
            return text
    with open(path, newline="") as handle:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--datasets", default=",".join(run.DEFAULT_DATASETS))
    s.add_argument("--out", default=os.path.join(run.HERE, "results", "sweep.csv"))
    r = sub.add_parser("report")
    r.add_argument("--results", default=os.path.join(run.HERE, "results", "full"))
    r.add_argument("--out", default=os.path.join(run.HERE, "results", "figures"))
    r.add_argument("--sweep", default=os.path.join(run.HERE, "results", "sweep.csv"))
    opts = parser.parse_args(argv)
    if opts.command == "sweep":
        write_csv(sweep(opts.datasets.split(",")), opts.out)
        print("wrote", opts.out)
    else:
        report(opts.results, opts.out, opts.sweep if os.path.exists(opts.sweep) else None)
        print("wrote", opts.out)


if __name__ == "__main__":
    main()
