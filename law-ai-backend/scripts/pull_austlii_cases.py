"""
AustLII Case Downloader & Converter — works in both dev (local) and prod (Azure).

Downloads RTF files from AustLII's direct URL path, converts to Markdown locally.
Increments case number from 1 and stops after N consecutive 404s.

URL pattern: https://www.austlii.edu.au/au/cases/cth/{court}/{year}/{number}.rtf

Supported courts:
  FamCA, FamCAFC, FMCAfam, FCCA, FedCFamC2F, FedCFamC1F, FedCFamC1A

Usage:
    python scripts/pull_austlii_cases.py --court FamCAFC --years 2021
    python scripts/pull_austlii_cases.py --court FamCA --years 2018-2024
    python scripts/pull_austlii_cases.py --court all --years 2020-2024
    python scripts/pull_austlii_cases.py --court FamCAFC --years 2021 --storage azure

Requirements:
    pip install pypandoc requests
    pip install azure-storage-blob   # only needed for --storage azure
    Pandoc must be installed: https://pandoc.org/installing.html
"""

import argparse
import os
import sys
import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

import requests

try:
    import pypandoc
    # Ensure pandoc binary is available
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("Pandoc binary not found. Downloading...")
        pypandoc.download_pandoc()
        print("Pandoc installed successfully.")
except ImportError:
    print("ERROR: pypandoc not installed. Run: pip install pypandoc")
    exit(1)

# Suppress InsecureRequestWarning (verify=False for corporate proxies like Zscaler)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("austlii_puller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_STORAGE = os.environ.get("AUSTLII_STORAGE", "local")
AZURE_CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
BLOB_CONTAINER_MD = os.environ.get("AUSTLII_BLOB_CONTAINER_MD", "austlii-cases-md")
REQUEST_DELAY_SECONDS = int(os.environ.get("AUSTLII_REQUEST_DELAY", "3"))
MAX_CONSECUTIVE_404 = int(os.environ.get("AUSTLII_MAX_CONSECUTIVE_404", "10"))

LOCAL_MD_DIR = str(
    (Path(__file__).resolve().parent.parent / "AustLII_cases_md_famca_tree").resolve()
)
FAILED_LOG_PATH = str(
    (Path(__file__).resolve().parent.parent / "logs" / "failed_pullAustLII.log").resolve()
)

SUPPORTED_COURTS = [
    "FamCA",
    "FamCAFC",
    "FMCAfam",
    "FCCA",
    "FedCFamC2F",
    "FedCFamC1F",
    "FedCFamC1A",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def log_pull_failure(case_id: str, url: str, error_message: str) -> None:
    os.makedirs(os.path.dirname(FAILED_LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] CASE: {case_id} | URL: {url} | ERROR: {error_message}\n")


# ─────────────────────────────────────────────
# Storage abstraction
# ─────────────────────────────────────────────

class StorageBackend(ABC):
    @abstractmethod
    def file_exists(self, path: str) -> bool: ...
    @abstractmethod
    def write_text(self, path: str, content: str) -> None: ...


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _full_path(self, path: str) -> str:
        full = os.path.join(self.base_dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def file_exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def write_text(self, path: str, content: str) -> None:
        with open(self._full_path(path), "w", encoding="utf-8") as f:
            f.write(content)


class AzureBlobStorage(StorageBackend):
    def __init__(self, connection_string: str, container_name: str):
        from azure.storage.blob import BlobServiceClient
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container = self.blob_service.get_container_client(container_name)
        try:
            self.container.create_container()
        except Exception:
            pass

    def file_exists(self, path: str) -> bool:
        return self.container.get_blob_client(path).exists()

    def write_text(self, path: str, content: str) -> None:
        self.container.get_blob_client(path).upload_blob(content.encode("utf-8"), overwrite=True)


def create_storage(mode: str) -> StorageBackend:
    if mode == "azure":
        if not AZURE_CONN_STR:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING env var is required for azure mode")
        return AzureBlobStorage(AZURE_CONN_STR, BLOB_CONTAINER_MD)
    return LocalStorage(LOCAL_MD_DIR)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def parse_year_range(years_str: str) -> list[int]:
    years = []
    for part in years_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def build_rtf_url(court: str, year: int, number: int) -> str:
    """Direct RTF URL — this is the pattern that works (same as the original script)."""
    return f"https://www.austlii.edu.au/au/cases/cth/{court}/{year}/{number}.rtf"


def build_case_id(court: str, year: int, number: int) -> str:
    return f"{court}_{year}_{number}"


def convert_rtf_to_md(rtf_content: bytes) -> str:
    """Convert RTF bytes to Markdown using Pandoc (via pypandoc). Cleans up temp file + memory."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False) as tmp:
            tmp.write(rtf_content)
            tmp_path = tmp.name
        md_text = pypandoc.convert_file(tmp_path, "md", format="rtf", extra_args=["--wrap=none"])
        return md_text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        del rtf_content  # Free RTF bytes from memory immediately


# ─────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────

def download_court_year(
    court: str,
    year: int,
    md_storage: StorageBackend,
    max_consecutive_404: int = MAX_CONSECUTIVE_404,
) -> dict:
    """
    Download all cases for a court/year by incrementing case number from 1.
    Downloads RTF from direct URL, converts to Markdown via Pandoc.
    Stops after max_consecutive_404 consecutive 404 responses.
    """
    stats = {"downloaded": 0, "converted": 0, "skipped": 0, "failed": 0}
    year_folder = f"{court}_{year}"
    consecutive_404 = 0
    case_number = 1

    logger.info("Starting %s/%d — incrementing from 1, stop after %d consecutive 404s",
                court, year, max_consecutive_404)

    while consecutive_404 < max_consecutive_404:
        case_id = build_case_id(court, year, case_number)
        md_path = f"{year_folder}/{case_id}.md"

        # Skip if already downloaded
        if md_storage.file_exists(md_path):
            stats["skipped"] += 1
            case_number += 1
            consecutive_404 = 0
            continue

        url = build_rtf_url(court, year, case_number)

        try:
            response = requests.get(url, headers=HEADERS, timeout=30, verify=False)

            if response.status_code == 200:
                # Check if we actually got RTF (not HTML from a proxy/redirect)
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    consecutive_404 = 0
                    logger.warning("  [⚠️] Got HTML instead of RTF for %s (proxy/redirect?)", case_id)
                    log_pull_failure(case_id, url, f"Got HTML instead of RTF (Content-Type: {content_type})")
                    stats["failed"] += 1
                    case_number += 1
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                consecutive_404 = 0
                stats["downloaded"] += 1

                # Convert RTF → Markdown, then free memory
                try:
                    md_text = convert_rtf_to_md(response.content)
                    if md_text and len(md_text.strip()) > 100:
                        md_storage.write_text(md_path, md_text)
                        stats["converted"] += 1
                        logger.info("  [OK] %s (%d chars)", case_id, len(md_text))
                    else:
                        char_count = len(md_text) if md_text else 0
                        logger.warning("  [⚠️] Empty/short: %s (%d chars)", case_id, char_count)
                        log_pull_failure(case_id, url, f"Empty conversion ({char_count} chars)")
                        stats["failed"] += 1
                except Exception as e:
                    logger.error("  [❌] Conversion failed: %s — %s", case_id, e)
                    log_pull_failure(case_id, url, f"Conversion error: {e}")
                    stats["failed"] += 1
                finally:
                    response.close()  # Free HTTP response + RTF bytes from memory

            elif response.status_code == 404:
                consecutive_404 += 1
                if consecutive_404 >= max_consecutive_404:
                    logger.info("  [STOP] %d consecutive 404s at case #%d. "
                                "Last valid case was ~#%d.",
                                max_consecutive_404, case_number,
                                case_number - max_consecutive_404)
            else:
                consecutive_404 = 0
                logger.error("  [❌] HTTP %d: %s", response.status_code, url)
                log_pull_failure(case_id, url, f"HTTP {response.status_code}")
                stats["failed"] += 1

        except requests.exceptions.Timeout:
            consecutive_404 = 0
            logger.error("  [❌] Timeout: %s", url)
            log_pull_failure(case_id, url, "Timeout")
            stats["failed"] += 1
        except Exception as e:
            consecutive_404 = 0
            logger.error("  [❌] Error: %s — %s", case_id, e)
            log_pull_failure(case_id, url, str(e))
            stats["failed"] += 1

        case_number += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("  %s/%d DONE: Downloaded=%d Converted=%d Skipped=%d Failed=%d (scanned 1-%d)",
                court, year, stats["downloaded"], stats["converted"],
                stats["skipped"], stats["failed"], case_number - 1)
    return stats


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

def test_single_case(court: str = "FamCAFC", year: int = 2021, number: int = 1) -> None:
    """Debug: download one RTF and try converting it."""
    case_id = build_case_id(court, year, number)
    url = build_rtf_url(court, year, number)
    print(f"Fetching: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30, verify=False)
    except Exception as e:
        print(f"[FAILED] {type(e).__name__}: {e}")
        return

    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content length: {len(response.content)} bytes")

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        print(f"\n[⚠️] Server returned HTML instead of RTF!")
        print(f"This likely means:")
        print(f"  1. Corporate proxy (Zscaler) is intercepting the request")
        print(f"  2. AustLII no longer serves .rtf for this case")
        print(f"\nHTML preview (first 500 chars):")
        print(response.text[:500])
        return

    if response.status_code != 200:
        print(f"Response: {response.text[:500]}")
        return

    print("\nConverting RTF → Markdown...")
    try:
        md_text = convert_rtf_to_md(response.content)
        print(f"Markdown length: {len(md_text)} chars")
        print(f"\nPreview (first 500 chars):\n{'='*60}")
        print(md_text[:500])
        print(f"{'='*60}")
    except Exception as e:
        print(f"[FAILED] Conversion error: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AustLII cases → Markdown (local or Azure)")
    parser.add_argument("--court", required=True,
                        choices=SUPPORTED_COURTS + ["all"],
                        help="Court to crawl, or 'all' for all 7 courts")
    parser.add_argument("--years", required=True,
                        help="Year or range: '2021', '2018-2021', '2018,2019,2021'")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, choices=["local", "azure"],
                        help=f"Storage backend (default: {DEFAULT_STORAGE})")
    parser.add_argument("--max-404", type=int, default=MAX_CONSECUTIVE_404,
                        help=f"Stop after N consecutive 404s (default: {MAX_CONSECUTIVE_404})")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: download and convert one case only")
    args = parser.parse_args()

    # Test mode
    if args.test:
        court = args.court if args.court != "all" else "FamCAFC"
        year = parse_year_range(args.years)[0]
        test_single_case(court, year, 1)
        exit(0)

    max_404 = args.max_404
    logger.info("Storage: %s | Max consecutive 404s: %d", args.storage, max_404)

    md_backend = create_storage(args.storage)
    courts = SUPPORTED_COURTS if args.court == "all" else [args.court]
    years = parse_year_range(args.years)

    total_stats = {"downloaded": 0, "converted": 0, "skipped": 0, "failed": 0}

    for court in courts:
        logger.info("\n" + "=" * 60)
        logger.info("  COURT: %s | Years: %s", court, years)
        logger.info("=" * 60)

        for year in years:
            stats = download_court_year(court, year, md_backend, max_consecutive_404=max_404)
            for key in total_stats:
                total_stats[key] += stats[key]

    logger.info("\n" + "=" * 60)
    logger.info("  ALL COMPLETE")
    logger.info("  Downloaded=%d Converted=%d Skipped=%d Failed=%d",
                total_stats["downloaded"], total_stats["converted"],
                total_stats["skipped"], total_stats["failed"])
    logger.info("  Failure log: %s", FAILED_LOG_PATH)
    logger.info("=" * 60)
