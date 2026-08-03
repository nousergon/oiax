"""Get the embedding model onto the machine before a prompt needs it.

`README.md` used to state it plainly: *"On first use, a ~90 MB ONNX embedding
model downloads and caches locally."* **First use is a prompt.** So the first
routed turn on any new machine paid a multi-second network download inside the
path whose whole premise is that it makes no network call — and it did so with
degrade-to-lexical-only as the only fallback, which means a machine with no
egress silently became a lexical-only router forever with nothing reading as an
error.

Two distinct problems, and this module addresses both:

1. **Latency and availability.** The design promises local, fast and offline.
   A first-use download contradicts all three, precisely on the turn where a new
   install is being evaluated.
2. **Supply chain.** An artifact fetched at first use, from the network, with
   nothing recording what arrived, is an unverified dependency entering the
   process *later* than every other one — after any install-time scanning has
   finished.

**The three states matter.** ``ready()`` on the embedder collapses everything
into one boolean, which is correct for the router and useless to an operator:

- ``PRESENT`` — loads with the network unavailable. The only state that means
  the promise holds.
- ``FETCHABLE`` — not cached, but the provider publishes it and it can be
  downloaded. A *first prompt will pay for it.*
- ``UNAVAILABLE`` — not cached and cannot be fetched. This machine will route
  lexical-only until something changes.

``PRESENT`` is established by **actually loading with the provider's offline
switch set**, not by looking for files and hoping. A cache-shaped directory that
does not load is the failure this check exists to catch.

Run it::

    python -m oiax.provision                 # fetch, verify, write a manifest
    python -m oiax.provision --check         # report the state, change nothing
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from oiax.embedding import DEFAULT_MODEL_ID, FastEmbedEmbedder, provider_publishes

__all__ = [
    "MANIFEST_NAME",
    "ProvisionStatus",
    "State",
    "check",
    "provision",
    "verify",
]

State = Literal["present", "fetchable", "unavailable"]

#: Written next to the cache. Not inside a provider-managed directory: the
#: provider owns its own layout and may rewrite it, and a manifest the provider
#: can clobber is a manifest that silently stops meaning anything.
MANIFEST_NAME = "oiax-model-manifest.json"

#: Environment switches the provider's downloader honours. Setting them is how
#: "present" is tested for real rather than inferred from a directory listing.
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


@dataclass
class ProvisionStatus:
    """What is on this machine, and what a first prompt would therefore pay."""

    state: State
    model_id: str
    cache_dir: str | None = None
    detail: str = ""
    files: int = 0
    bytes: int = 0
    manifest_ok: bool | None = None  # None = no manifest to check against

    @property
    def routes_semantically_offline(self) -> bool:
        return self.state == "present"

    def describe(self) -> str:
        lines = [f"model:  {self.model_id}", f"cache:  {self.cache_dir or '(provider default)'}"]
        if self.state == "present":
            size = f"{self.files} files, {self.bytes / 1e6:.0f} MB"
            lines.append(f"state:  PRESENT — loads offline, {size}")
        elif self.state == "fetchable":
            lines.append(
                "state:  FETCHABLE — not on this machine. THE FIRST PROMPT WILL PAY "
                "for the download; run `python -m oiax.provision` first."
            )
        else:
            lines.append(
                "state:  UNAVAILABLE — not cached and cannot be fetched. This machine "
                "will route LEXICAL-ONLY until that changes."
            )
        if self.detail:
            lines.append(f"detail: {self.detail}")
        if self.manifest_ok is True:
            lines.append("digest: verified against the manifest")
        elif self.manifest_ok is False:
            lines.append(
                "digest: MISMATCH — the cached files are not the ones that were "
                "provisioned. Delete the cache and re-provision."
            )
        return "\n".join(lines)


@contextlib.contextmanager
def _offline() -> Iterator[None]:
    """Force the provider's downloader offline for the duration."""
    saved = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cache_files(cache_dir: str | None) -> list[Path]:
    if not cache_dir:
        return []
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _loads_offline(model_id: str, cache_dir: str | None) -> tuple[bool, str]:
    """Can the model be loaded with the network switched off?

    The honest test, and the one the acceptance criterion names. A directory
    that *looks* like a populated cache but does not load is exactly what this
    is here to catch.
    """
    with _offline():
        embedder = FastEmbedEmbedder(model_id=model_id, cache_dir=cache_dir)
        try:
            vectors = embedder.embed(["provisioning check"])
        except Exception as exc:  # an unpublished id raises; let the caller see it
            return False, str(exc)
        if not len(vectors):
            return False, "the provider could not load the model offline"
        return True, ""


def check(model_id: str = DEFAULT_MODEL_ID, cache_dir: str | None = None) -> ProvisionStatus:
    """Report which of the three states this machine is in. Changes nothing."""
    cache_dir = cache_dir or os.environ.get("OIAX_MODEL_CACHE") or None

    present, why = _loads_offline(model_id, cache_dir)
    files = _cache_files(cache_dir)
    total = sum(p.stat().st_size for p in files)

    if present:
        return ProvisionStatus(
            state="present",
            model_id=model_id,
            cache_dir=cache_dir,
            files=len(files),
            bytes=total,
            manifest_ok=verify(cache_dir) if cache_dir else None,
        )

    # Not present. Distinguish "could be fetched" from "cannot" WITHOUT
    # fetching — the caller asked for a report, not a download.
    fetchable, detail = _is_fetchable(model_id)
    return ProvisionStatus(
        state="fetchable" if fetchable else "unavailable",
        model_id=model_id,
        cache_dir=cache_dir,
        detail=detail or why,
        files=len(files),
        bytes=total,
    )


def _is_fetchable(model_id: str) -> tuple[bool, str]:
    published = provider_publishes(model_id)
    if published is None:
        return False, "the provider package is not installed"
    if not published:
        # Configuration, not connectivity. Fetching will never help.
        return False, f"{model_id!r} is not a model this provider publishes"

    # A HEAD against the provider's own host would be a second, weaker copy of
    # its download logic. The honest answer is that fetchability is decided by
    # attempting it, so this reports the reachable case and lets `provision()`
    # find out for real.
    return True, "published by the provider; not yet on this machine"


def provision(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str | None = None,
    *,
    write_manifest: bool = True,
) -> ProvisionStatus:
    """Fetch the model if needed, then record what arrived.

    Raises on failure. A half-provisioned machine that reported success would
    send the cost back to the first prompt, which is the whole thing this
    module exists to prevent.
    """
    cache_dir = cache_dir or os.environ.get("OIAX_MODEL_CACHE") or None

    embedder = FastEmbedEmbedder(model_id=model_id, cache_dir=cache_dir)
    vectors = embedder.embed(["provisioning"])  # triggers the download
    if not len(vectors):
        raise RuntimeError(
            f"could not provision {model_id!r} into {cache_dir or 'the provider default'}. "
            "Nothing was cached, so a first prompt would still pay for the download."
        )

    if write_manifest and cache_dir:
        _write_manifest(model_id, cache_dir, embedder.dimension())

    status = check(model_id, cache_dir)
    if status.state != "present":
        raise RuntimeError(
            f"provisioning completed but the model still does not load offline "
            f"({status.detail}). Reporting success here would hand the cost back "
            f"to the first prompt."
        )
    return status


def _manifest_path(cache_dir: str) -> Path:
    return Path(cache_dir).parent / f"{Path(cache_dir).name}-{MANIFEST_NAME}"


def _write_manifest(model_id: str, cache_dir: str, dimension: int) -> Path:
    files = _cache_files(cache_dir)
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "dimension": dimension,
        "files": {str(p.relative_to(cache_dir)): _digest(p) for p in files},
        "bytes": sum(p.stat().st_size for p in files),
    }
    path = _manifest_path(cache_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify(cache_dir: str) -> bool | None:
    """Recompute digests against the manifest. ``None`` when there is none.

    ``None`` and ``True`` are deliberately different: an unverified cache is not
    a verified one, and a check that returned True for "nothing to compare
    against" would report the strongest possible answer for the weakest possible
    evidence.
    """
    path = _manifest_path(cache_dir)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        recorded: dict[str, str] = manifest["files"]
    except Exception:
        return False

    root = Path(cache_dir)
    for rel, digest in recorded.items():
        target = root / rel
        if not target.exists() or _digest(target) != digest:
            return False
    return True


@dataclass
class _Args:
    check_only: bool = False
    model_id: str = DEFAULT_MODEL_ID
    cache_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m oiax.provision",
        description="Put the embedding model on this machine before a prompt needs it.",
    )
    parser.add_argument("--check", action="store_true", help="report the state, change nothing")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where the model lives. Defaults to $OIAX_MODEL_CACHE, then the "
        "provider's own default. A manifest is only written when this is set, "
        "because the provider's default location is not ours to write beside.",
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if parsed.check:
        status = check(parsed.model_id, parsed.cache_dir)
        print(status.describe())
        # Non-zero on anything but PRESENT, so this is usable as a gate in an
        # image build or a bootstrap script.
        return 0 if status.state == "present" else 1

    try:
        status = provision(parsed.model_id, parsed.cache_dir)
    except Exception as exc:
        print(f"provisioning failed: {exc}", file=sys.stderr)
        return 1
    print(status.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
