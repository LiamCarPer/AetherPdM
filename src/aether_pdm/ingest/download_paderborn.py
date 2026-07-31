"""
Download and extract a small subset of the Paderborn University bearing dataset.

Files are distributed as .rar archives containing .mat files. This module
downloads a configurable subset (default: K001 healthy, KI04 inner race,
KO04 outer race) and extracts the .mat files for normalization.

``rarfile`` is intentionally NOT a hard dependency. The extractor try-imports
it (with ``unrar``/``7-Zip`` as the system backend) and falls back to
``patool``; if neither is available it raises a clear, actionable error.

Usage:
    python -m aether_pdm.ingest.download_paderborn \
        [--out data/raw/paderborn] [--subset K001,KI04,KO04]
"""

import argparse
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import httpx

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
DEFAULT_SUBSET = ["K001", "KI04", "KO04"]  # healthy, inner_race, outer_race

# Real outer-race (KOxx) archives are not always present on the server;
# fall back to the artificial-damage outer-race archives (KAxx) when 404.
KO_FALLBACK = {
    "KO04": "KA04",
    "KO05": "KA05",
}

# 10 MiB progress reporting granularity.
_PROGRESS_STEP_BYTES = 10 * 1024 * 1024


def download_rar(
    file_id: str,
    out_dir: Path,
    base_url: str = BASE_URL,
    timeout: int = 600,
) -> Path:
    """Download ``file_id.rar`` from the Paderborn data center.

    Streams the archive with ``httpx`` and prints progress every 10 MB so a
    150-180 MB archive does not look stalled.

    Args:
        file_id: Archive stem, e.g. ``"K001"``.
        out_dir: Directory in which the ``.rar`` file is written.
        base_url: Base URL of the Paderborn data center.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded ``.rar`` file.

    Raises:
        urllib.error.HTTPError: On 404 (fallback logic handled by caller).
        httpx.HTTPError: On other transport/status errors.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url.rstrip('/')}/{file_id}.rar"
    dest = out_dir / f"{file_id}.rar"

    print(f"  Downloading {url}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            if response.status_code == 404:
                raise HTTPError(url, response.status_code, "Not Found", Message(), None)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            next_report = _PROGRESS_STEP_BYTES
            with open(dest, "wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024) if total else 0.0
                        print(f"    ... {mb:.0f} MB / {total_mb:.0f} MB")
                        next_report += _PROGRESS_STEP_BYTES

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"    -> {dest} ({size_mb:.0f} MB)")
    return dest


def extract_rar(rar_path: Path, out_dir: Path) -> list[Path]:
    """Extract a .rar archive to ``out_dir``.

    Extraction backend order:
    1. ``rarfile`` (requires a system ``unrar`` or ``7-Zip`` binary)
    2. ``patool`` (``patoolib``) as a fallback

    Args:
        rar_path: Path to the ``.rar`` archive.
        out_dir: Directory in which the archive is extracted.

    Returns:
        Sorted list of extracted ``.mat`` paths.

    Raises:
        RuntimeError: If neither backend is importable, with setup instructions.
    """
    rar_path = Path(rar_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rarfile
    except ImportError:
        rarfile = None  # type: ignore

    if rarfile is not None:
        with rarfile.RarFile(str(rar_path)) as archive:
            archive.extractall(str(out_dir))
        return sorted(out_dir.rglob("*.mat"))

    try:
        import patoolib
    except ImportError:
        raise RuntimeError(
            "Cannot extract .rar archives: neither 'rarfile' nor 'patool' is installed. "
            "Run 'uv add rarfile' and install a backend binary (unrar or 7-Zip), "
            "or 'uv add patool'. Then re-run the downloader."
        )

    patoolib.extract_archive(str(rar_path), outdir=str(out_dir))
    return sorted(out_dir.rglob("*.mat"))


def download_subset(
    subset: list[str] | None = None,
    out_dir: Path = Path("data/raw/paderborn"),
    extract: bool = True,
) -> list[Path]:
    """Download the configured subset of Paderborn archives.

    For each file ID:
    1. ``download_rar`` from the data center.
    2. On 404, if the ID starts with ``KO``, retry the artificial outer-race
       equivalent (``KA`` + suffix) so the default subset still succeeds.
    3. Optionally extract the archive.

    Args:
        subset: List of file IDs (default ``DEFAULT_SUBSET``).
        out_dir: Destination directory.
        extract: Whether to extract archives after download.

    Returns:
        List of extracted ``.mat`` paths (empty when ``extract=False``).

    Raises:
        urllib.error.HTTPError: If a download 404s with no fallback available.
    """
    subset = list(subset or DEFAULT_SUBSET)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_paths: list[Path] = []
    for file_id in subset:
        try:
            rar_path = download_rar(file_id, out_dir)
        except HTTPError as exc:
            fallback = KO_FALLBACK.get(file_id)
            if fallback is None and file_id.startswith("KO"):
                fallback = "KA" + file_id[2:]
            if fallback is None:
                raise
            print(
                f"  {file_id}.rar not found ({exc.code}); "
                f"trying {fallback}.rar (artificial damage)"
            )
            rar_path = download_rar(fallback, out_dir)

        if extract:
            mat_paths.extend(extract_rar(rar_path, out_dir))

    return mat_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Paderborn bearing dataset subset")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/paderborn"),
        help="Destination directory (default: data/raw/paderborn)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=",".join(DEFAULT_SUBSET),
        help="Comma-separated file IDs (default: K001,KI04,KO04)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives only, do not extract .mat files",
    )
    args = parser.parse_args()

    subset = [item.strip() for item in args.subset.split(",") if item.strip()]
    mat_paths = download_subset(
        subset=subset,
        out_dir=args.out,
        extract=not args.no_extract,
    )

    if args.no_extract:
        print(f"Downloaded {len(subset)} archives to {args.out}")
    else:
        print(f"Extracted {len(mat_paths)} .mat files:")
        for path in mat_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
