"""Tests for model provisioning.

The acceptance criterion the issue names is the offline one: **a machine with
the model present and no network routes semantically.** It is marked
``network`` because a bare CI runner has no cached model and cannot satisfy it
without downloading ~90 MB, and a test suite that fetched a third-party artifact
would fail on someone else's laptop for reasons unrelated to the code.

Everything that can be tested without the artifact is tested without it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oiax.embedding import DEFAULT_MODEL_ID
from oiax.provision import (
    MANIFEST_NAME,
    ProvisionStatus,
    _manifest_path,
    _offline,
    check,
    provision,
    verify,
)


def _model_is_cached() -> bool:
    return check().state == "present"


needs_model = pytest.mark.skipif(
    not _model_is_cached(), reason="embedding model not cached on this machine"
)


# ── the three states are distinct ───────────────────────────────────────────


def test_status_reports_the_three_states_distinctly():
    # ready() collapses everything into one boolean, which is right for the
    # router and useless to an operator deciding whether a first prompt pays.
    present = ProvisionStatus(state="present", model_id="m", files=3, bytes=10**8)
    fetchable = ProvisionStatus(state="fetchable", model_id="m")
    unavailable = ProvisionStatus(state="unavailable", model_id="m")

    assert present.routes_semantically_offline
    assert not fetchable.routes_semantically_offline
    assert not unavailable.routes_semantically_offline
    assert "PRESENT" in present.describe()
    assert "FIRST PROMPT WILL PAY" in fetchable.describe()
    assert "LEXICAL-ONLY" in unavailable.describe()


def test_an_empty_cache_directory_is_fetchable_not_present(tmp_path):
    status = check(cache_dir=str(tmp_path))
    assert status.state == "fetchable"
    assert "not yet on this machine" in status.detail


def test_an_unpublished_model_is_unavailable_not_fetchable(tmp_path):
    # Configuration, not connectivity. Fetching will never help, and reporting
    # it as fetchable would send an operator to check their network.
    status = check(model_id="definitely/not-published", cache_dir=str(tmp_path))
    assert status.state == "unavailable"
    assert "not a model this provider publishes" in status.detail


def test_a_cache_shaped_directory_that_does_not_load_is_not_present(tmp_path):
    # PRESENT is established by loading offline, never by a directory listing.
    (tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx").mkdir()
    (tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx" / "model.onnx").write_bytes(b"not a model")
    assert check(cache_dir=str(tmp_path)).state != "present"


# ── the offline switch ──────────────────────────────────────────────────────


def test_offline_context_sets_and_restores_the_provider_switches():
    before = os.environ.get("HF_HUB_OFFLINE")
    with _offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == before


def test_offline_context_restores_a_pre_existing_value(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with _offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# ── the manifest ────────────────────────────────────────────────────────────


def test_verify_returns_none_when_there_is_no_manifest(tmp_path):
    # None and True are deliberately different. Returning True for "nothing to
    # compare against" would report the strongest answer for the weakest
    # evidence.
    assert verify(str(tmp_path)) is None


def test_verify_detects_a_changed_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "model.bin"
    target.write_bytes(b"original")
    import hashlib

    manifest = {
        "model_id": DEFAULT_MODEL_ID,
        "dimension": 384,
        "files": {"model.bin": hashlib.sha256(b"original").hexdigest()},
        "bytes": 8,
    }
    _manifest_path(str(cache)).write_text(json.dumps(manifest), encoding="utf-8")

    assert verify(str(cache)) is True
    target.write_bytes(b"tampered")
    assert verify(str(cache)) is False


def test_verify_detects_a_deleted_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.bin").write_bytes(b"x")
    import hashlib

    _manifest_path(str(cache)).write_text(
        json.dumps({"files": {"model.bin": hashlib.sha256(b"x").hexdigest()}}), encoding="utf-8"
    )
    (cache / "model.bin").unlink()
    assert verify(str(cache)) is False


def test_a_corrupt_manifest_is_a_failure_not_an_absence(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _manifest_path(str(cache)).write_text("{not json", encoding="utf-8")
    assert verify(str(cache)) is False


def test_the_manifest_lives_beside_the_cache_not_inside_it(tmp_path):
    # The provider owns its own directory layout and may rewrite it. A manifest
    # the provider can clobber silently stops meaning anything.
    path = _manifest_path(str(tmp_path / "cache"))
    assert MANIFEST_NAME in path.name
    assert path.parent == tmp_path


# ── the CLI is usable as a gate ─────────────────────────────────────────────


def test_check_exits_non_zero_when_the_model_is_absent(tmp_path, capsys):
    from oiax.provision import main

    rc = main(["--check", "--cache-dir", str(tmp_path)])
    capsys.readouterr()
    # Non-zero so an image build or bootstrap script can gate on it.
    assert rc == 1


def test_provision_raises_rather_than_reporting_a_half_done_machine(tmp_path):
    # A half-provisioned machine reporting success hands the cost straight back
    # to the first prompt, which is the entire thing this module prevents.
    with pytest.raises(Exception):
        provision(model_id="definitely/not-published", cache_dir=str(tmp_path))


# ── the acceptance criterion ────────────────────────────────────────────────


@needs_model
def test_a_provisioned_machine_routes_semantically_with_the_network_off():
    """The criterion #20 names, run against the real cached model."""
    from oiax import build_index, route, semantic_ready
    from oiax.corpus import PolicyDirCorpus

    corpus_dir = Path(__file__).resolve().parent.parent / "src/oiax/eval/corpora/reference-policies"
    with _offline():
        index = build_index(PolicyDirCorpus(corpus_dir))
        assert semantic_ready() is True
        assert index.corpus_separability is not None
        assert [h.name for h in route("we need to roll back a bad deploy", index)]


@needs_model
def test_check_reports_present_on_a_machine_that_has_it():
    status = check()
    assert status.state == "present"
    assert status.routes_semantically_offline
