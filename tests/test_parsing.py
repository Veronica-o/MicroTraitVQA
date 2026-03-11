"""
Tests for archive parsing.
Integration tests (marked with @pytest.mark.integration) are skipped
automatically if no test archive is found — put one in tests/data/ to run them.
"""

import io
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, ".")
from bio_vqa.vqa import _get_text, parse_archive
import xml.etree.ElementTree as ET

# look for a test archive
_ARCHIVE = next(
    (p for p in ["tests/data/PMC11047695_tar.gz", "tests/data/PMC11050112_tar.gz",
                 "PMC11047695_tar.gz", "PMC11050112_tar.gz"] if Path(p).exists()),
    None
)


# --- _get_text ---

def test_get_text_plain():
    el = ET.fromstring("<p>Hello world</p>")
    assert _get_text(el) == "Hello world"

def test_get_text_nested():
    el = ET.fromstring("<caption><title>Fig 1</title><p>A chromatogram.</p></caption>")
    r = _get_text(el)
    assert "Fig 1" in r and "chromatogram" in r

def test_get_text_mixed_content():
    el = ET.fromstring("<p>Start <b>bold</b> end.</p>")
    r = _get_text(el)
    assert "Start" in r and "bold" in r and "end" in r

def test_get_text_empty():
    assert _get_text(ET.fromstring("<p></p>")) == ""


# --- parse_archive error cases (no real archive needed) ---

def test_parse_raises_on_no_nxml(tmp_path):
    fake = str(tmp_path / "empty.tar.gz")
    with tarfile.open(fake, "w:gz") as tar:
        content = b"dummy"
        info = tarfile.TarInfo(name="dummy.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    with pytest.raises(FileNotFoundError):
        with tempfile.TemporaryDirectory() as work:
            parse_archive(fake, work)


# --- integration tests (need a real archive) ---

@pytest.mark.skipif(_ARCHIVE is None, reason="No archive in tests/data/")
def test_parse_returns_figures():
    with tempfile.TemporaryDirectory() as work:
        figs = parse_archive(_ARCHIVE, work)
    assert len(figs) > 0

@pytest.mark.skipif(_ARCHIVE is None, reason="No archive in tests/data/")
def test_parse_figure_fields():
    with tempfile.TemporaryDirectory() as work:
        figs = parse_archive(_ARCHIVE, work)
        f = figs[0]
    assert f.figure_id and f.label and f.caption and f.filename
    assert Path(f.image_path).exists()

@pytest.mark.skipif(_ARCHIVE is None, reason="No archive in tests/data/")
def test_parse_unique_ids():
    with tempfile.TemporaryDirectory() as work:
        figs = parse_archive(_ARCHIVE, work)
    ids = [f.figure_id for f in figs]
    assert len(ids) == len(set(ids))
