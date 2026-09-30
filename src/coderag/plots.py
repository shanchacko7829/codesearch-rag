"""Charts from results/eval_*.json (DejaVu Sans so it renders without system fonts)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
LABELS = {"bm25": "BM25", "dense": "LSA dense", "hybrid": "Hybrid (RRF)", "rerank": "Hybrid + reranker"}
COLORS = {"bm25": "#8fa3b8", "dense": "#b8a48f", "hybrid": "#5b8def", "rerank": "#1f4fbf"}


def plot_methods(results: dict, out) -> None:
    repos = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / len(LABELS)
    for j, m in enumerate(LABELS):
        xs, vals, lo, hi = [], [], [], []
        for i, r in enumerate(repos):
            s = results[r]["modes"][m]["mrr@10"]
            xs.append(i + j * w)
            vals.append(s["value"])
            lo.append(s["value"] - s["ci95"][0])
            hi.append(s["ci95"][1] - s["value"])
        ax.bar(xs, vals, w, yerr=[lo, hi], capsize=3, label=LABELS[m], color=COLORS[m])
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(repos))])
    ax.set_xticklabels(repos)
    ax.set_ylabel("MRR@10 (95% CI, bootstrap over files)")
    ax.set_title("Natural-language code search, docstrings hidden")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_chunking(results: dict, out) -> None:
    repos = list(results)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    for j, (key, label, color) in enumerate((("ast", "AST chunks (function/class)", "#1f4fbf"),
                                             ("windows", "Fixed 40-line windows", "#b0b7c3"))):
        vals = [results[r]["chunking"][key]["hybrid"]["mrr@10"]["value"] for r in repos]
        ax.bar([i + j * w for i in range(len(repos))], vals, w, label=label, color=color)
    ax.set_xticks([i + w / 2 for i in range(len(repos))])
    ax.set_xticklabels(repos)
    ax.set_ylabel("MRR@10 (hybrid)")
    ax.set_title("Effect of chunking strategy")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def make_plots(results_dir="results", out_dir="docs") -> list[str]:
    results = {}
    for p in sorted(Path(results_dir).glob("eval_*.json")):
        results[p.stem[5:]] = json.loads(p.read_text())
    if not results:
        raise FileNotFoundError(f"no eval_*.json in {results_dir}; run `python -m coderag eval` first")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_methods(results, Path(out_dir) / "methods.png")
    plot_chunking(results, Path(out_dir) / "chunking.png")
    return ["methods.png", "chunking.png"]
