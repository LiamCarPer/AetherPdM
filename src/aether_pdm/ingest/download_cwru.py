"""
CWRU Bearing Data Center downloader.

Downloads .mat files from:
  https://engineering.case.edu/bearingdatacenter/download-data-file

Structured into:
  data/raw/cwru/{normal,inner_race,outer_race,ball}/*.mat

Usage:
    python -m aether_pdm.ingest.download_cwru [--dest data/raw/cwru]
"""

import argparse
import zipfile
from pathlib import Path

import httpx

# Primary URLs for CWRU bearing data
# The official CWRU data is distributed as multiple .mat files organized by fault type.
CWRU_URLS = {
    "normal": "https://engineering.case.edu/sites/default/files/97.mat",
    "inner_race": "https://engineering.case.edu/sites/default/files/105.mat",
    "outer_race": "https://engineering.case.edu/sites/default/files/130.mat",
    "ball": "https://engineering.case.edu/sites/default/files/118.mat",
}

# Fallback mirror (Kaggle-hosted archive)
FALLBACK_URL = "https://github.com/LiamCarPer/aether-pdm-mirror/raw/main/cwru-bearings.zip"

MIRROR_URL = "https://www.dropbox.com/scl/fo/somekey/cwru-bearings.zip"  # placeholder


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file with progress indicator. Returns True on success."""
    print(f"  Downloading {url}")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
        print(f"    -> {dest} ({len(dest.read_bytes()) / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def download_single_mat_files(dest: Path) -> bool:
    """Download individual .mat files by fault type."""
    fault_dir = dest / "cwru"
    all_ok = True
    for fault_type, url in CWRU_URLS.items():
        target = fault_dir / fault_type / f"{fault_type}.mat"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not download_file(url, target):
            all_ok = False
    return all_ok


def download_from_mirror(dest: Path) -> bool:
    """Download the full CWRU archive from a mirror."""
    zip_path = dest / "cwru-mirror.zip"
    if not download_file(FALLBACK_URL, zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest / "cwru")
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
        "--mirror",
        action="store_true",
        help="Force download from mirror archive instead of individual files",
    )
    args = parser.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    print("Downloading CWRU Bearing Data Center dataset...")

    if args.mirror:
        success = download_from_mirror(dest)
    else:
        success = download_single_mat_files(dest)
        if not success:
            print("Individual downloads failed. Trying mirror...")
            success = download_from_mirror(dest)

    if success:
        print("Done. Dataset ready at:", dest / "cwru")
    else:
        print("All download methods failed. Check your network or visit:")
        print("  https://engineering.case.edu/bearingdatacenter/download-data-file")


if __name__ == "__main__":
    main()
