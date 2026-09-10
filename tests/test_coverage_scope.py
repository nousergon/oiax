"""The coverage gate's *scope* is asserted here, not only its number.

repository-baseline-policy.md §4.2 C5: the way a coverage gate stops being
honest is by narrowing what it measures rather than by lowering the number —
which reads as an improvement in every report. Measured on symposion,
removing one flag moved the reported figure from 34.76% to 92.36% with no
new test code.

oiax's coverage configuration lives INLINE in `.github/workflows/ci.yml`
(one `--cov=oiax --cov-fail-under=75` pytest invocation), not in
`pyproject.toml` — there is no `[tool.coverage.run]` / `[tool.coverage.report]`
section, no `.coveragerc`, no `setup.cfg`. That is a deliberate difference
from repos (mnemon, morning-signal, scannerctl) that configure coverage in
`pyproject.toml`; migrating oiax's config there is a separate, larger change
and out of scope here. These tests read `.github/workflows/ci.yml` directly
instead, so they assert what a passing suite cannot otherwise notice:

* the measured source is the WHOLE ``oiax`` package (C1), via exactly one
  ``--cov=oiax`` flag — never a path or submodule narrower than that, and
  never a second, differently-scoped flag hiding in another job;
* the floor is enforced by a non-zero exit (C2) and is a ratchet that may be
  raised and never lowered (C3), via exactly one ``--cov-fail-under=`` flag;
* no ``omit`` list has appeared anywhere (pyproject.toml, .coveragerc,
  setup.cfg) beyond the pinned, currently-empty, justified set — a future PR
  that introduces one silently shrinks the denominator without this test
  noticing the change in words, so this test forces the diff to be reviewed
  instead of waved through as "coverage went up";
* every source module under ``src/`` is inside the measured ``oiax``
  package — nothing is invisible to the gate by living outside it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "src" / "oiax"

#: The floor may be RAISED here as coverage improves. Lowering it is a policy
#: amendment (repository-baseline-policy.md §4.2 C3), not a code change.
MINIMUM_FLOOR = 75

#: Nothing is omitted today. Widening this set is a scope decision, not a
#: drive-by coverage bump — it needs a reviewed justification alongside it,
#: same as the entries mnemon and morning-signal pin in pyproject.toml.
EXPECTED_OMIT: frozenset[str] = frozenset()


def _ci_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def test_coverage_source_is_the_whole_package() -> None:
    """C1 — exactly one ``--cov=`` flag, naming the whole oiax package."""
    matches = re.findall(r"--cov=(\S+)", _ci_text())
    assert len(matches) == 1, (
        f"expected exactly one --cov= flag in ci.yml, found {matches!r}. "
        "A second occurrence (e.g. added to another job) must be reviewed, "
        "not silently picked between."
    )
    assert matches[0] == "oiax", (
        f"coverage source must be exactly the oiax package, got {matches[0]!r}. "
        "Narrowing it to a submodule (e.g. oiax.router) measures the tested "
        "subset and reports it as the repository."
    )


def test_coverage_floor_is_enforced_and_never_lowered() -> None:
    """C2 + C3 — the gate exits non-zero below a floor that only ratchets up."""
    matches = re.findall(r"--cov-fail-under=(\d+)", _ci_text())
    assert len(matches) == 1, (
        f"expected exactly one --cov-fail-under= flag in ci.yml, found {matches!r}. "
        "A second occurrence must be reviewed, not silently picked between."
    )
    fail_under = int(matches[0])
    assert fail_under >= MINIMUM_FLOOR, (
        f"coverage floor {fail_under} is below the ratchet {MINIMUM_FLOOR}. "
        "A floor is raised as coverage improves and never lowered to make a "
        "change pass (repository-baseline-policy.md §4.2 C3)."
    )


def test_no_coverage_config_introduces_an_unreviewed_omit() -> None:
    """A shrunk denominator is a narrowing this test forces into review."""
    coveragerc = REPO_ROOT / ".coveragerc"
    setup_cfg = REPO_ROOT / "setup.cfg"
    assert not coveragerc.exists(), (
        ".coveragerc has appeared — oiax's coverage config lives inline in "
        "ci.yml today; a new config file is a scope change that needs review, "
        "not a silent addition this test should pass through."
    )
    assert not setup_cfg.exists(), (
        "setup.cfg has appeared — oiax's coverage config lives inline in "
        "ci.yml today; a new config file is a scope change that needs review, "
        "not a silent addition this test should pass through."
    )

    if not PYPROJECT.exists():
        return
    tool = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    coverage = tool.get("coverage", {})
    omit = set(coverage.get("run", {}).get("omit", []))
    added = omit - EXPECTED_OMIT
    assert not added, (
        f"pyproject.toml's [tool.coverage.run] omit gained unreviewed entries: "
        f"{sorted(added)}. Each omitted path removes files from the "
        "denominator, raising the reported figure without adding a test — "
        "update EXPECTED_OMIT here alongside a reviewed justification if this "
        "is deliberate."
    )


def test_every_source_module_is_inside_the_measured_package() -> None:
    """No source file is invisible to the gate by living outside src/oiax."""
    src = REPO_ROOT / "src"
    stray = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in src.rglob("*.py")
        if ".egg-info" not in p.parts
        and PACKAGE_ROOT not in p.parents
        and p != PACKAGE_ROOT
    )
    assert not stray, (
        f"source modules outside the measured package are invisible to the "
        f"coverage gate: {stray}"
    )
