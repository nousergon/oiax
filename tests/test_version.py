"""`oiax.__version__` must be the installed distribution's version."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import oiax


def test_version_matches_installed_metadata():
    assert oiax.__version__ == version("oiax")


def test_installed_metadata_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert version("oiax") == pyproject["project"]["version"]
