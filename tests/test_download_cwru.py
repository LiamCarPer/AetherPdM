"""Tests for the CWRU downloader (network fully mocked)."""

import httpx

import aether_pdm.ingest.download_cwru as mod
from aether_pdm.ingest.normalize_cwru import TEST_FILES, VAL_FILES


class _FakeResponse:
    """Minimal httpx.Response stand-in used by the mocked client."""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://fake")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


class _FakeClient:
    """httpx.Client stand-in: get() returns a fixed response and records URLs."""

    def __init__(self, response: _FakeResponse, *args, **kwargs):
        self._response = response
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url):
        self.urls.append(url)
        return self._response


def _patch_client(monkeypatch, response: _FakeResponse) -> _FakeClient:
    client = _FakeClient(response)
    monkeypatch.setattr(mod.httpx, "Client", lambda *args, **kwargs: client)
    return client


def test_download_file_preserves_numeric_name(tmp_path, monkeypatch):
    """The .mat file must be saved as {file_id}.mat, never a fault-type name."""
    content = b"%MAT-file fake payload 97"
    client = _patch_client(monkeypatch, _FakeResponse(content))

    ok = mod.download_file("97", tmp_path)

    assert ok is True
    assert client.urls == [f"{mod.CWRU_BASE_URL}97.mat"]
    assert (tmp_path / "97.mat").read_bytes() == content
    assert not (tmp_path / "normal.mat").exists()


def test_download_file_failure_returns_false(tmp_path, monkeypatch):
    """Hard 404s must fail immediately and be reported as False, not raised."""
    client = _patch_client(monkeypatch, _FakeResponse(b"", status_code=404))

    assert mod.download_file("97", tmp_path) is False
    assert not (tmp_path / "97.mat").exists()
    assert len(client.urls) == 1  # 404 is not retried


def test_download_file_retries_transient_errors(tmp_path, monkeypatch):
    """Transient 5xx must be retried with backoff until the file lands."""
    calls: list[str] = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            calls.append(url)
            if len(calls) == 1:
                return _FakeResponse(b"", status_code=503)
            return _FakeResponse(b"%MAT retried")

    monkeypatch.setattr(mod.httpx, "Client", Client)

    assert mod.download_file("97", tmp_path) is True
    assert (tmp_path / "97.mat").read_bytes() == b"%MAT retried"
    assert len(calls) == 2


def test_downloader_file_ids_cover_val_and_test():
    """CWRU_FILE_IDS (from the catalog) must cover VAL_FILES | TEST_FILES."""
    ids = set(mod.CWRU_FILE_IDS)

    assert VAL_FILES | TEST_FILES <= ids
    assert len(mod.CWRU_FILE_IDS) == len(ids)  # no duplicates


def test_download_single_mat_files_flat_layout(tmp_path, monkeypatch):
    """download_single_mat_files must write flat {file_id}.mat into dest/cwru/."""
    client = _patch_client(monkeypatch, _FakeResponse(b"%MAT fake"))

    ok = mod.download_single_mat_files(tmp_path, file_ids=["97", "98"])

    assert ok is True
    assert (tmp_path / "cwru" / "97.mat").read_bytes() == b"%MAT fake"
    assert (tmp_path / "cwru" / "98.mat").read_bytes() == b"%MAT fake"
    assert len(client.urls) == 2


def test_download_single_mat_files_reports_partial_failure(tmp_path, monkeypatch, capsys):
    """A missing file must not abort the run: others download, False is returned."""

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            if url.endswith("97.mat"):
                return _FakeResponse(b"%MAT ok")
            return _FakeResponse(b"", status_code=404)

    monkeypatch.setattr(mod.httpx, "Client", Client)

    ok = mod.download_single_mat_files(tmp_path, file_ids=["97", "98"])

    assert ok is False
    assert (tmp_path / "cwru" / "97.mat").read_bytes() == b"%MAT ok"
    assert not (tmp_path / "cwru" / "98.mat").exists()
    assert "FAILED (1/2 files): 98" in capsys.readouterr().out
