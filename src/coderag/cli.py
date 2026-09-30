"""Command line:  python -m coderag {fetch,index,search,ask,eval,plots,serve}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PRESETS


def _resolve(args):
    """(repo path, subdir) from either --preset or --repo/--subdir."""
    if getattr(args, "preset", None):
        from .fetch import fetch_preset
        return fetch_preset(args.preset, args.repos_dir), PRESETS[args.preset]["subdir"]
    if not args.repo:
        raise SystemExit("give --preset NAME or --repo PATH")
    return Path(args.repo), args.subdir or ""


def cmd_fetch(a):
    from .fetch import fetch_preset
    for name in (a.names or sorted(PRESETS)):
        print(name, "->", fetch_preset(name, a.repos_dir))


def cmd_index(a):
    from .chunker import chunk_repo
    from .dense import BACKENDS
    from .index import CodeIndex
    repo, sub = _resolve(a)
    chunks = chunk_repo(repo, sub, strategy=a.strategy, hide_docstrings=False, include_docs=a.docs)
    backend = BACKENDS[a.backend]() if a.backend != "lsa" else BACKENDS["lsa"](a.dims)
    idx = CodeIndex.build(chunks, backend, verbose=True)
    if a.reranker:
        from .rerank import Reranker
        idx.reranker = Reranker.load(a.reranker)
    out = a.out or f"indexes/{Path(repo).name}"
    idx.save(out)
    print(f"saved {len(chunks):,} chunks to {out}")


def cmd_search(a):
    from .index import CodeIndex
    idx = CodeIndex.load(a.index)
    for h in idx.search(a.query, k=a.k, mode=a.mode):
        print(f"{h.rank:>2}. {h.chunk.location:<48} {h.chunk.kind:<8} {h.chunk.name}  ({h.score:.3f})")


def _generator(a):
    from .qa import ExtractiveGenerator, OllamaGenerator
    return OllamaGenerator(a.model, a.ollama_host) if a.generator == "ollama" else ExtractiveGenerator()


def cmd_ask(a):
    from .index import CodeIndex
    from .qa import GeneratorError, answer_question
    idx = CodeIndex.load(a.index)
    try:
        r = answer_question(idx, a.question, _generator(a), k=a.k)
    except GeneratorError as exc:
        raise SystemExit(f"error: {exc}")
    print(r["answer"])
    print("\nSources:")
    for s in r["sources"]:
        print(f"  [{s['n']}] {s['location']}  {s['name']}")
    c = r["citations"]
    if c["invalid"]:
        print(f"\nWARNING: answer cites sources that do not exist: {c['invalid']}")


def cmd_eval(a):
    from .experiments import run_experiment
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    names = a.presets or sorted(PRESETS)
    for name in names:
        from .fetch import fetch_preset
        repo = fetch_preset(name, a.repos_dir)
        rr = Path(a.out_dir) / f"reranker_{name}.json"
        transfer = Path(a.out_dir) / f"reranker_{a.transfer_from}.json" if a.transfer_from and a.transfer_from != name else None
        res = run_experiment(repo, PRESETS[name]["subdir"], name, a.queries, a.train_queries, a.dims,
                             transfer_reranker=transfer if transfer and transfer.exists() else None, save_reranker=rr)
        (Path(a.out_dir) / f"eval_{name}.json").write_text(json.dumps(res, indent=2))
        print(f"wrote {a.out_dir}/eval_{name}.json")


def cmd_plots(a):
    from .plots import make_plots
    print("wrote", make_plots(a.results_dir, a.out_dir))


def cmd_serve(a):
    import uvicorn
    from .index import CodeIndex
    from .service import create_app
    uvicorn.run(create_app(CodeIndex.load(a.index), _generator(a)), host=a.host, port=a.port)


def build_parser():
    p = argparse.ArgumentParser(prog="coderag", description="Code-aware retrieval and Q&A with an evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def repo_args(s):
        s.add_argument("--preset", choices=sorted(PRESETS))
        s.add_argument("--repo")
        s.add_argument("--subdir", default="")
        s.add_argument("--repos-dir", default="repos")

    s = sub.add_parser("fetch", help="download the pinned benchmark repositories")
    s.add_argument("names", nargs="*", help=f"any of {sorted(PRESETS)} (default: all)")
    s.add_argument("--repos-dir", default="repos")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("index", help="build and save a search index")
    repo_args(s)
    s.add_argument("--out")
    s.add_argument("--strategy", choices=["ast", "windows"], default="ast")
    s.add_argument("--backend", choices=["lsa", "sbert"], default="lsa")
    s.add_argument("--dims", type=int, default=256)
    s.add_argument("--docs", action="store_true", help="also index .md/.rst documentation")
    s.add_argument("--reranker", help="path to a trained reranker json")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("search", help="search a saved index")
    s.add_argument("query")
    s.add_argument("--index", required=True)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--mode", default="auto", choices=["auto", "bm25", "dense", "hybrid", "rerank"])
    s.set_defaults(fn=cmd_search)

    def gen_args(s):
        s.add_argument("--generator", choices=["extractive", "ollama"], default="extractive")
        s.add_argument("--model", default="qwen2.5-coder:7b")
        s.add_argument("--ollama-host", default="http://localhost:11434")

    s = sub.add_parser("ask", help="answer a question with cited sources")
    s.add_argument("question")
    s.add_argument("--index", required=True)
    s.add_argument("-k", type=int, default=5)
    gen_args(s)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="run the benchmark (writes results/eval_*.json)")
    s.add_argument("presets", nargs="*", default=None)
    s.add_argument("--repos-dir", default="repos")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--queries", type=int, default=800)
    s.add_argument("--train-queries", type=int, default=1500)
    s.add_argument("--dims", type=int, default=256)
    s.add_argument("--transfer-from", default="django", help="also test a reranker trained on this repo")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("plots", help="draw charts from results/")
    s.add_argument("--results-dir", default="results")
    s.add_argument("--out-dir", default="docs")
    s.set_defaults(fn=cmd_plots)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--index", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    gen_args(s)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0
