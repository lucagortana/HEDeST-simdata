"""Reads ``download/urls.json`` and turns it into things to fetch or to ask for.

Two kinds of source, and they need different handling:

* **direct URLs** -- STHELAR on EBI's FTP, and the 10x files on ``cf.10xgenomics.com``.
  These are downloaded by the ``download`` stage, archives unpacked in place.
* **entries with no URL** (one file: the breast cell-type spreadsheet) make the stage stop
  with the dataset page and the exact path to put the file at, which is something a person
  can act on. Guessing a URL instead does not fail cleanly -- it saves an HTML error page
  under the data file's name, and the first thing to notice is a confusing crash three
  stages later.

Every 10x URL here was verified rather than inferred from a pattern: a HEAD request has to
answer 200, and its ``Content-Length`` is recorded in ``bytes`` and checked after the
transfer. Where the file also existed locally the two byte counts agreed exactly, which is
what makes these the published files and not merely URLs that resolve.

Fill a URL in ``download/urls.json`` once you have it and the stage downloads it like any
other; or put the file where ``dest`` says and the stage sees it is already there.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..common import download as dl

#: ``download/urls.json``, relative to the repository root.
SOURCES_FILE = Path(__file__).resolve().parents[3] / "download" / "urls.json"


class MissingSource(RuntimeError):
    """A needed file has no URL and is not on disk. The message says what to do."""


@dataclass(frozen=True)
class Source:
    role: str
    dest: Path
    url: str | None
    what: str
    size: int | None = None
    optional: bool = False
    unpack: str | None = None
    #: the archive's top-level directory, stripped on extraction; see download.Item.strip
    strip: str | None = None

    @property
    def present(self) -> bool:
        if not self.dest.exists():
            return False
        if self.size is not None and self.dest.is_file():
            return self.dest.stat().st_size == self.size
        if self.dest.is_dir():
            return any(self.dest.iterdir())
        return True


def load() -> dict:
    if not SOURCES_FILE.exists():
        raise MissingSource(f"{SOURCES_FILE} is missing from the checkout")
    return json.loads(SOURCES_FILE.read_text())


def _fill(template: str, **fmt) -> str:
    out = template
    for k, v in fmt.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def sthelar(data_root: Path, slide: str) -> list[Source]:
    spec = load()["sthelar"]
    out = []
    for item in spec["per_slide"]:
        url = _fill(item["url"], base=spec["base"], slide=slide)
        out.append(Source(
            role=item["role"],
            dest=data_root / _fill(item["dest"], slide=slide),
            url=url, what=item["what"],
            size=spec.get("bytes", {}).get(item["role"], {}).get(slide)))
    return out


def visiumhd(data_root: Path, sample: str, patient: str) -> list[Source]:
    """``sample`` is the folder name (``Visium_HD_Human_Colon_Cancer_P1``) and so the URL
    stem; ``patient`` is the short key (``P1``) the recorded sizes are listed under."""
    spec = load()["visiumhd"]
    return [Source(role=i["role"], dest=data_root / _fill(i["dest"], sample=sample),
                   url=i["url"] and _fill(i["url"], base=spec["base"], sample=sample),
                   what=i["what"],
                   size=spec.get("bytes", {}).get(i["role"], {}).get(patient),
                   optional=i.get("optional", False), unpack=i.get("unpack"),
                   strip=i.get("strip"))
            for i in spec["per_sample"]]


def xenium(data_root: Path, key: str, slide: str) -> list[Source]:
    spec = load()["xenium"]
    # per_slide_extra adds this slide's own files and may also REPLACE a shared entry, by
    # repeating its role: the two slides' output bundles have different internal layouts
    # (`strip`), so breast overrides `outputs` rather than being listed twice.
    items: list[dict] = list(spec["per_slide"])
    by_role = {i["role"]: n for n, i in enumerate(items)}
    for extra in spec.get("per_slide_extra", {}).get(key, []):
        if extra["role"] in by_role:
            items[by_role[extra["role"]]] = extra
        else:
            by_role[extra["role"]] = len(items)
            items.append(extra)
    # the two slides sit under different release directories on the CDN (lung 2.0.0,
    # breast 1.0.1), so the base is per slide rather than per family
    base = spec["base"][key]
    return [Source(role=i["role"], dest=data_root / _fill(i["dest"], slide=slide),
                   url=i["url"] and _fill(i["url"], base=base, slide=slide),
                   what=i["what"],
                   size=spec.get("bytes", {}).get(i["role"], {}).get(key),
                   optional=i.get("optional", False), unpack=i.get("unpack"),
                   strip=i.get("strip"))
            for i in items]


def page_for(family: str, key: str | None = None) -> str:
    page = load()[family]["source_page"]
    return page[key] if isinstance(page, dict) and key else str(page)


def fetch_or_explain(sources: list[Source], family: str, label: str,
                     page: str, dry_run: bool = False) -> list[dict]:
    """Downloads what it can; raises with instructions for what it cannot.

    Anything already on disk is reported and skipped, so this doubles as the check that a
    manually placed download is complete before hours of pipeline run on it.
    """
    results, blocked = [], []
    for src in sources:
        if src.present:
            results.append({"role": src.role, "status": "present", "path": str(src.dest)})
            print(f"  {'present':>16}  {src.role}", flush=True)
            continue
        if src.url:
            results.append(dl.fetch(
                dl.Item(src.url, src.dest, src.size, src.unpack, src.strip),
                dry_run=dry_run))
            continue
        if not src.optional:
            blocked.append(src)

    if blocked:
        lines = [f"{label}: {len(blocked)} input(s) must be downloaded by hand.",
                 f"  dataset page: {page}",
                 "  These files are served behind a form, so this repo cannot fetch them "
                 "for you.", ""]
        for src in blocked:
            lines.append(f"  - {src.role}: {src.what}")
            lines.append(f"      put it at: {src.dest}")
        lines += ["",
                  "Then re-run the same command. Alternatively, paste the direct URLs into "
                  f"download/urls.json ({SOURCES_FILE.name}) and the download stage will "
                  "fetch them."]
        raise MissingSource("\n".join(lines))
    return results
