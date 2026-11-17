"""Resumable downloads, shared by the three families.

Raw inputs here run to 157 GB for STHELAR alone, over links that are not fast, so a
download must survive being interrupted and must never silently half-write a file. This
shells out to ``curl -C -`` (or ``wget -c``) rather than reimplementing resume: both get
range requests, redirects and flaky connections right, and both are present everywhere
this will run.

Measured from the cluster this was written on, 2026-11-05, against EBI's FTP: **4.8 MB/s**
on one stream, and 1.6-1.9 MB/s each on four concurrent streams -- about 6.9 MB/s in
aggregate. Parallelism barely helps because the cap is upstream, which is why
:func:`fetch_all` is sequential by default. That figure is this link on that day, not a
property of the data, so a different site should measure its own.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    """One file to fetch, or one archive to fetch and unpack."""

    url: str
    dest: Path
    #: Expected size in bytes, when it is known. A finished file of the wrong size is
    #: re-fetched rather than trusted, which is the failure mode resume introduces.
    size: int | None = None
    #: ``"tar"`` / ``"zip"`` when ``dest`` is a directory to extract the download into,
    #: rather than the downloaded file itself.
    unpack: str | None = None
    #: The archive's single top-level directory, dropped so that its *contents* land
    #: directly in ``dest``; ``None`` when the members are already at the top level.
    #: Both shapes occur among these datasets, and the difference is not cosmetic:
    #: ``binned_outputs.tar.gz`` wraps everything in ``binned_outputs/`` and the breast
    #: ``outs.zip`` in ``outs/``, while the lung ``outs.zip`` has ``analysis.zarr.zip``
    #: and its siblings at the top. Declared per archive in ``download/urls.json`` rather
    #: than sniffed, so a changed layout is an error instead of a silent re-rooting.
    strip: str | None = None

    @property
    def name(self) -> str:
        return self.dest.name

    @property
    def archive(self) -> Path:
        """Where the archive itself is downloaded, beside the directory it fills."""
        suffix = ".tar.gz" if self.unpack == "tar" else ".zip"
        return self.dest.parent / f".{self.dest.name}{suffix}.part"


def _members(infos, strip: str | None):
    """Archive members, re-rooted by ``strip``, with anything unsafe refused."""
    for info in infos:
        name = getattr(info, "filename", None) or info.name
        rel = name
        if strip:
            prefix = strip.rstrip("/") + "/"
            if rel.rstrip("/") == strip.rstrip("/"):
                continue                    # the wrapper directory entry itself
            if not rel.startswith(prefix):
                raise DownloadError(
                    f"archive member {name!r} is not under the declared top-level "
                    f"directory {strip!r}; this archive's layout has changed")
            rel = rel[len(prefix):]
        if not rel or rel == "/":
            continue
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise DownloadError(f"archive member {name!r} escapes the destination")
        yield info, rel


def _extract(item: Item) -> int:
    """Extract ``item.archive`` into ``item.dest``; returns the number of files written."""
    import tarfile
    import zipfile

    item.dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if item.unpack == "tar":
        with tarfile.open(item.archive, "r:*") as tf:
            for info, rel in _members(tf, item.strip):
                out = item.dest / rel
                if info.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                if not info.isfile():
                    continue                # no links or devices in these archives
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(info)
                if src is None:
                    continue
                with src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    elif item.unpack == "zip":
        with zipfile.ZipFile(item.archive) as zf:
            for info, rel in _members(zf.infolist(), item.strip):
                out = item.dest / rel
                if info.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    else:
        raise DownloadError(f"unknown archive format {item.unpack!r}")
    return n


def _tool() -> list[str]:
    if shutil.which("curl"):
        return ["curl", "-fL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o"]
    if shutil.which("wget"):
        return ["wget", "-c", "-t", "5", "-O"]
    raise DownloadError("neither curl nor wget is available")


def complete(item: Item) -> bool:
    """Is this download already fully there?"""
    if item.unpack:
        # the archive is deleted once unpacked, so a non-empty destination is the only
        # evidence left; the size check belongs to the archive, not to what came out of it
        return item.dest.is_dir() and any(item.dest.iterdir())
    if not item.dest.exists():
        return False
    if item.size is None:
        return True
    return item.dest.stat().st_size == item.size


def _curl(target: Path, url: str, size: int | None, label: str) -> tuple[int, int, float]:
    """Fetch ``url`` to ``target``, resuming. Returns (had, got, seconds)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    had = target.stat().st_size if target.exists() else 0
    t0 = time.time()
    cmd = _tool() + [str(target), url]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise DownloadError(
            f"{label}: {cmd[0]} exited {proc.returncode}. The partial file is kept, "
            f"so re-running the download stage resumes it.")
    got = target.stat().st_size
    if size is not None and got != size:
        raise DownloadError(
            f"{label}: got {got} bytes, expected {size}. Delete {target} and re-run; "
            "a resumed download cannot recover from a truncated source.")
    return had, got, max(time.time() - t0, 1e-6)


def fetch(item: Item, dry_run: bool = False) -> dict:
    """Downloads one file (or archive), resuming a partial one, and reports what it did."""
    if complete(item):
        return {"file": item.name, "status": "present",
                "bytes": (None if item.unpack else item.dest.stat().st_size)}
    if dry_run:
        return {"file": item.name, "status": "would-download", "url": item.url}

    if item.unpack:
        # Downloaded to a hidden `.part` sibling and only then extracted, so an
        # interrupted run resumes the transfer instead of leaving a half-unpacked
        # directory that `complete()` would accept as finished.
        had, got, dt = _curl(item.archive, item.url, item.size, item.name)
        n = _extract(item)
        item.archive.unlink(missing_ok=True)
        return {"file": item.name, "status": "downloaded", "bytes": got,
                "files_unpacked": n, "resumed_from": had, "seconds": round(dt, 1),
                "mb_per_s": round((got - had) / dt / 2**20, 2)}

    had, got, dt = _curl(item.dest, item.url, item.size, item.name)
    return {"file": item.name, "status": "downloaded", "bytes": got,
            "resumed_from": had, "seconds": round(dt, 1),
            "mb_per_s": round((got - had) / dt / 2**20, 2)}


def fetch_all(items: list[Item], dry_run: bool = False) -> list[dict]:
    """Fetches a list of files one at a time, reporting each.

    Sequential on purpose: see the module docstring. Anything already complete is skipped,
    so this is also the cheap way to check a download before a long pipeline run.
    """
    out = []
    for item in items:
        res = fetch(item, dry_run=dry_run)
        out.append(res)
        rate = f" at {res['mb_per_s']} MB/s" if "mb_per_s" in res else ""
        print(f"  {res['status']:>16}  {res['file']}{rate}", flush=True)
    return out
