"""Tests for the Paderborn downloader (network fully mocked)."""

import sys
import types
from pathlib import Path
from urllib.error import HTTPError

import pytest

import aether_pdm.ingest.download_paderborn as mod


class _FakeResponse:
    """httpx-style stream response used by the mocked downloader."""

    def __init__(self, content: bytes, status_code: int = 200):
        self._content = content
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError("http://fake", self.status_code, "error", None, None)
        return None

    def iter_bytes(self, chunk_size=1024 * 1024):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


class _FakeClient:
    """httpx.Client stand-in: stream() returns a fake response."""

    def __init__(self, response: _FakeResponse, *args, **kwargs):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method, url):
        return self._response


def _patch_client(monkeypatch, response: _FakeResponse):
    monkeypatch.setattr(
        mod.httpx,
        "Client",
        lambda *args, **kwargs: _FakeClient(response, *args, **kwargs),
    )


def test_download_subset_calls_download_and_extract(monkeypatch, tmp_path):
    """Subset download should call download_rar then extract_rar per file."""
    downloaded: list[str] = []
    extracted: list[str] = []

    def fake_download(file_id, out_dir, base_url=mod.BASE_URL, timeout=600):
        downloaded.append(file_id)
        rar = Path(out_dir) / f"{file_id}.rar"
        rar.touch()
        return rar

    def fake_extract(rar_path, out_dir):
        extracted.append(rar_path.name)
        return [Path(out_dir) / f"{rar_path.stem}.mat"]

    monkeypatch.setattr(mod, "download_rar", fake_download)
    monkeypatch.setattr(mod, "extract_rar", fake_extract)

    paths = mod.download_subset(["K001", "KI04", "KO04"], out_dir=tmp_path)

    assert downloaded == ["K001", "KI04", "KO04"]
    assert extracted == ["K001.rar", "KI04.rar", "KO04.rar"]
    assert len(paths) == 3
    assert paths[0] == tmp_path / "K001.mat"


def test_download_rar_writes_file(tmp_path, monkeypatch):
    """download_rar should stream the archive to disk (progress included)."""
    content = b"R" * (15 * 1024 * 1024)  # > 10 MB so progress prints once
    _patch_client(monkeypatch, _FakeResponse(content))

    dest = mod.download_rar("K001", tmp_path)

    assert dest == tmp_path / "K001.rar"
    assert dest.read_bytes() == content


def test_download_rar_raises_http_error_on_404(tmp_path, monkeypatch):
    """A 404 response should raise urllib HTTPError for fallback handling."""
    _patch_client(monkeypatch, _FakeResponse(b"", status_code=404))

    with pytest.raises(HTTPError):
        mod.download_rar("KO04", tmp_path)


def test_download_subset_falls_back_ko_to_ka(monkeypatch, tmp_path):
    """KO04 404 -> retry KA04 (artificial outer race), no crash."""
    calls: list[str] = []

    def fake_download(file_id, out_dir, base_url=mod.BASE_URL, timeout=600):
        calls.append(file_id)
        if file_id == "KO04":
            raise HTTPError("http://fake/KO04.rar", 404, "Not Found", None, None)
        rar = Path(out_dir) / f"{file_id}.rar"
        rar.touch()
        return rar

    monkeypatch.setattr(mod, "download_rar", fake_download)
    monkeypatch.setattr(
        mod, "extract_rar", lambda rar_path, out_dir: [Path(out_dir) / f"{rar_path.stem}.mat"]
    )

    paths = mod.download_subset(["KO04"], out_dir=tmp_path)

    assert calls == ["KO04", "KA04"]
    assert paths == [tmp_path / "KA04.mat"]


def test_download_subset_no_fallback_raises(monkeypatch, tmp_path):
    """A non-KO 404 with no fallback should propagate the HTTPError."""
    def fake_download(file_id, out_dir, base_url=mod.BASE_URL, timeout=600):
        raise HTTPError("http://fake/K001.rar", 404, "Not Found", None, None)

    monkeypatch.setattr(mod, "download_rar", fake_download)

    with pytest.raises(HTTPError):
        mod.download_subset(["K001"], out_dir=tmp_path)


def test_download_subset_skip_extract(monkeypatch, tmp_path):
    """extract=False should download but not call extract_rar."""
    def fake_download(file_id, out_dir, base_url=mod.BASE_URL, timeout=600):
        return Path(out_dir) / f"{file_id}.rar"

    monkeypatch.setattr(mod, "download_rar", fake_download)
    monkeypatch.setattr(
        mod, "extract_rar", lambda *a, **k: pytest.fail("extract_rar should not be called")
    )

    paths = mod.download_subset(["K001"], out_dir=tmp_path, extract=False)

    assert paths == []


def test_extract_rar_returns_mat_paths(tmp_path, monkeypatch):
    """rarfile backend: extractall called, .mat paths returned."""
    extracted_to: list[str] = []

    class FakeRarFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extractall(self, out_dir):
            extracted_to.append(str(out_dir))
            Path(out_dir, "K001.mat").touch()

    fake = types.ModuleType("rarfile")
    fake.RarFile = FakeRarFile
    monkeypatch.setitem(sys.modules, "rarfile", fake)

    rar_path = tmp_path / "K001.rar"
    rar_path.touch()
    out_dir = tmp_path / "extracted"
    out_dir.mkdir()

    paths = mod.extract_rar(rar_path, out_dir)

    assert extracted_to == [str(out_dir)]
    assert paths == [out_dir / "K001.mat"]


def test_extract_rar_patool_fallback(tmp_path, monkeypatch):
    """When rarfile is missing, patool should be used as fallback."""
    calls: list[tuple[str, str]] = []

    fake = types.ModuleType("patoolib")

    def extract_archive(path, outdir=None):
        calls.append((path, outdir))
        Path(outdir, "K001.mat").touch()

    fake.extract_archive = extract_archive
    monkeypatch.setitem(sys.modules, "rarfile", None)
    monkeypatch.setitem(sys.modules, "patoolib", fake)

    rar_path = tmp_path / "K001.rar"
    rar_path.touch()
    out_dir = tmp_path / "extracted"
    out_dir.mkdir()

    paths = mod.extract_rar(rar_path, out_dir)

    assert calls == [(str(rar_path), str(out_dir))]
    assert paths == [out_dir / "K001.mat"]


def test_extract_rar_no_rarfile_raises(tmp_path, monkeypatch):
    """Neither backend available -> RuntimeError with setup instructions."""
    monkeypatch.setitem(sys.modules, "rarfile", None)
    monkeypatch.setitem(sys.modules, "patoolib", None)

    rar_path = tmp_path / "K001.rar"
    rar_path.touch()

    with pytest.raises(RuntimeError, match="rarfile"):
        mod.extract_rar(rar_path, tmp_path)
