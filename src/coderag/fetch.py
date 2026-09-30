"""Download the pinned repositories used by the benchmark.

Uses ``git`` when it is installed (and verifies the commit hash), otherwise downloads the
tag's zip archive with the standard library so nothing beyond Python is required.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .config import PRESETS


def extract_zip_archive(data: bytes, target: Path) -> None:
    """Extract a GitHub zip (single top-level folder) into ``target``, refusing path traversal."""
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts[1:]          # drop the "<repo>-<tag>/" prefix
            if not parts:
                continue
            dest = (target.joinpath(*parts)).resolve()
            if target not in dest.parents and dest != target:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))


def fetch_preset(name: str, dest="repos", method: str = "auto") -> Path:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' (choose from {sorted(PRESETS)})")
    p = PRESETS[name]
    target = Path(dest) / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if method in ("auto", "git") and shutil.which("git"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", p["tag"], p["url"], str(target)], check=True)
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        if head != p["commit"]:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"{name}: tag {p['tag']} resolved to {head}, expected {p['commit']}")
    else:
        url = f"{p['url']}/archive/refs/tags/{p['tag']}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "coderag"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            extract_zip_archive(resp.read(), target)
    return target
