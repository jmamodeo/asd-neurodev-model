import importlib.metadata
import pathlib
import re
import tomllib

import asd_model

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    citation = (ROOT / "CITATION.cff").read_text()
    cff_version = re.search(r"^version: (\S+)", citation, re.MULTILINE).group(1)
    assert asd_model.__version__ == cff_version
    assert importlib.metadata.version("asd-neurodev-model") == cff_version


def _requirement_pins(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_runtime_pins_come_from_requirements_txt():
    pins = _requirement_pins((ROOT / "requirements.txt").read_text())
    assert pins

    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    assert "dependencies" in pyproject["project"]["dynamic"]
    assert "dependencies" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["dependencies"]["file"] == ["requirements.txt"]

    runtime_requires = [req for req in importlib.metadata.requires("asd-neurodev-model") if "extra ==" not in req]
    assert sorted(runtime_requires) == sorted(pins)
