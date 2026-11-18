import importlib.metadata
import pathlib
import re

import asd_model

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    citation = (ROOT / "CITATION.cff").read_text()
    cff_version = re.search(r"^version: (\S+)", citation, re.MULTILINE).group(1)
    assert asd_model.__version__ == cff_version
    assert importlib.metadata.version("asd-neurodev-model") == cff_version
