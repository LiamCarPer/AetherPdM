"""
CWRU Bearing Data Center downloader.

Downloads .mat files from:
  https://engineering.case.edu/bearingdatacenter/download-data-file

Files are saved flat as ``data/raw/cwru/{file_id}.mat`` using the numeric
file IDs from ``normalize_cwru.CWRU_CATALOG``. The normalizer maps
``filepath.stem`` directly into the catalog, so the numeric filename MUST be
preserved — the catalog is the single source of truth for the file ID set
(train + val + test).

Usage:
    python -m aether_pdm.ingest.download_cwru [--dest data/raw] [--file-ids 97,105,130]
"""

import argparse
import time
import zipfile
from pathlib import Path

import httpx

from aether_pdm.ingest.normalize_cwru import CWRU_CATALOG

# Primary URL pattern for CWRU bearing data (verified for 97/105/118/130).
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Full set of file IDs referenced by the normalizer's catalog
# (train + val + test). Single source of truth: if the catalog grows,
# the downloader follows.
CWRU_FILE_IDS: list[str] = [fid for fid, *_ in CWRU_CATALOG]

# Fallback mirror (GitHub-hosted archive of the full CWRU set).
FALLBACK_URL = "https://github.com/LiamCarPer/aether-pdm-mirror/raw/main/cwru-bearings.zip"

# HTTP statuses worth retrying (transient server-side failures).
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _download_url(url: str, target: Path, timeout: int = 120, retries: int = 2) -> bool:
    """Download ``url`` to ``target`` with a progress note.

    Transient failures (5xx / 429 / transport errors such as dropped
    connections) are retried up to ``retries`` times with linear backoff.
    Hard 4xx errors (e.g. 404) fail immediately — retrying cannot help.

    Returns:
        True on success; False on any HTTP/transport error.
    """
    print(f"  Downloading {url}")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                target.write_bytes(response.content)
            print(f"    -> {target} ({target.stat().st_size / 1024:.0f} KB)")
            return True
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code not in _RETRYABLE_STATUS or attempt >= retries:
                break
        except Exception as exc:  # transport errors: dropped connections, timeouts
            last_error = exc
            if attempt >= retries:
                break
        time.sleep(1.0 * (attempt + 1))
    print(f"    FAILED: {last_error}")
    return False


def download_file(file_id: str, dest_dir: Path, timeout: int = 120) -> bool:
    """Download ``{file_id}.mat`` preserving the numeric name.

    Args:
        file_id: Numeric CWRU catalog ID (e.g. ``"97"``).
        dest_dir: Directory in which ``{file_id}.mat`` is written.
        timeout: Request timeout in seconds.

    Returns:
        True if the file was downloaded successfully.
    """
    target = Path(dest_dir) / f"{file_id}.mat"
    return _download_url(f"{CWRU_BASE_URL}{file_id}.mat", target, timeout)


def download_single_mat_files(dest: Path, file_ids: list[str] | None = None) -> bool:
    """Download the requested file IDs as ``{file_id}.mat`` into ``dest/cwru/``.

    Args:
        dest: Base data directory; files land under ``dest / "cwru"``.
        file_ids: Numeric file IDs to fetch (default: all ``CWRU_FILE_IDS``).

    Returns:
        True if every requested file downloaded successfully.
    """
    ids = list(file_ids) if file_ids is not None else list(CWRU_FILE_IDS)
    fault_dir = Path(dest) / "cwru"
    fault_dir.mkdir(parents=True, exist_ok=True)

    failed_ids: list[str] = []
    for file_id in ids:
        if not download_file(file_id, fault_dir):
            failed_ids.append(file_id)

    if failed_ids:
        print(
            f"  FAILED ({len(failed_ids)}/{len(ids)} files): "
            f"{', '.join(failed_ids)}"
        )
    return not failed_ids


def download_from_mirror(dest: Path) -> bool:
    """Download the full CWRU archive from a mirror."""
    zip_path = Path(dest) / "cwru-mirror.zip"
    if not _download_url(FALLBACK_URL, zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(Path(dest) / "cwru")
        zip_path.unlink()
        return True
    except zipfile.BadZipFile:
        zip_path.unlink()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CWRU bearing dataset")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw)",
    )
    parser.add_argument(
        "--file-ids",
        type=str,
        default=None,
        help="Comma-separated file IDs to download (default: all catalog IDs)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Force download from mirror archive instead of individual files",
    )
    args = parser.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    file_ids: list[str] | None = None
    if args.file_ids is not None:
        file_ids = [fid.strip() for fid in args.file_ids.split(",") if fid.strip()]

    print("Downloading CWRU Bearing Data Center dataset...")

    if args.mirror:
        success = download_from_mirror(dest)
    else:
        success = download_single_mat_files(dest, file_ids)
        # Only fall back to the bulk mirror when no explicit subset was
        # requested — a targeted --file-ids run must not silently dump
        # the whole archive.
        if not success and file_ids is None:
            print("Individual downloads failed. Trying mirror...")
            success = download_from_mirror(dest)

    if success:
        print("Done. Dataset ready at:", dest / "cwru")
    else:
        print("All download methods failed. Check your network or visit:")
        print("  https://engineering.case.edu/bearingdatacenter/download-data-file")


if __name__ == "__main__":
    main()
