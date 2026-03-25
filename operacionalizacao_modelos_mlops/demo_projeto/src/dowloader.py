import os
import fnmatch
import zipfile
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from src.utils.logger import get_logger

def check_kaggle_credentials(secrets_path: Path) -> None:
    """
    Verify Kaggle credentials are available either via environment variable.
    """
    load_dotenv(dotenv_path=str(secrets_path))
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return has_env

def list_remote_files(
    dataset: str,
    file_pattern: str,
    logging_config: dict[str, Any]
) -> list[str]:
    """
    Query the Kaggle API to list files matching file_pattern in the dataset.

    Args:
        dataset:        Kaggle dataset slug, e.g. 'shibumohapatra/house-price'.
        file_pattern:   Glob pattern to filter filenames, e.g. '*.csv'.
        logging_config: Optional logging config dict.

    Returns:
        Sorted list of matching remote filenames (names only, no paths).

    Raises:
        EnvironmentError: If credentials are missing.
        RuntimeError:     If the Kaggle API call fails.
    """
    logger = get_logger(__name__, logging_config=logging_config)

    try:
        
        from kaggle.api.kaggle_api_extended import KaggleApi

        # autenticação na API do Kaggle
        api = KaggleApi()
        api.authenticate()

        # lista de conjuntos de dados desejados
        files = api.dataset_list_files(dataset).files

        # lista de arquivos que desejamos
        matched = sorted(
            f.name for f in files if fnmatch.fnmatch(f.name, file_pattern)
        )
        logger.info(
            "Found %d remote file(s) matching '%s' in '%s'",
            len(matched),
            file_pattern,
            dataset,
        )
        return matched
    
    except Exception as exc:
        raise RuntimeError(
            f"Failed to list files in Kaggle dataset '{dataset}': {exc}"
        ) from exc
        
def download_dataset(
    dataset: str,
    expected_files: list[str],
    destination_dir: Path,
    skip_if_exists: bool = True,
    force: bool = False,
    logging_config: dict[str, Any] | None = None,
) -> list[Path]:
    """
    Download the specified files from a Kaggle dataset.

    Uses dataset_download_file() per file to avoid downloading the full
    dataset zip and to allow selective file targeting.

    Args:
        dataset:         Kaggle slug, e.g. 'shibumohapatra/house-price'.
        expected_files:  List of exact filenames to download.
        destination_dir: Local directory where files will be saved.
        skip_if_exists:  Skip download if all expected files are already present.
        force:           Override skip_if_exists — always re-download.
        logging_config:  Optional logging config dict.

    Returns:
        List of Path objects for each file in destination_dir after download.

    Raises:
        EnvironmentError: If Kaggle credentials are missing.
        RuntimeError:     If any file download fails.
    """
    log_cfg = logging_config
    logger = get_logger(__name__, log_cfg)

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if not force and skip_if_exists and _files_already_present(destination_dir, expected_files):
        logger.info(
            "All %d expected file(s) already present in '%s'. Skipping download. "
            "(Use --force-download to re-download.)",
            len(expected_files),
            destination_dir,
        )
        return [destination_dir / f for f in expected_files]

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(f"Kaggle API authentication failed: {exc}") from exc

    downloaded: list[Path] = []
    for filename in expected_files:
        dest_path = destination_dir / filename
        if not force and skip_if_exists and dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info("  [SKIP] '%s' already exists (%s)", filename, _format_size(dest_path.stat().st_size))
            downloaded.append(dest_path)
            continue

        logger.info("  [DOWN] Downloading '%s'...", filename)
        try:
            api.dataset_download_file(
                dataset=dataset,
                file_name=filename,
                path=str(destination_dir),
                force=True,
                quiet=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{filename}' from '{dataset}': {exc}"
            ) from exc

        # Handle generic zip extraction: Kaggle may return {filename}.zip
        # regardless of the original file extension (e.g., .csv.zip, .parquet.zip)
        zip_path = Path(str(dest_path) + ".zip")
        if zip_path.exists() and not dest_path.exists():
            _unzip_file(zip_path, destination_dir, logger)

        if dest_path.exists():
            size = dest_path.stat().st_size
            logger.info("  [OK]   '%s' saved (%s)", filename, _format_size(size))
            downloaded.append(dest_path)
        else:
            # Check if there's any file that looks like what we expected (case differences)
            candidates = list(destination_dir.glob(f"*{Path(filename).suffix}"))
            hint = f" Found: {[c.name for c in candidates]}" if candidates else ""
            raise RuntimeError(
                f"Download of '{filename}' appeared to succeed but file not found at '{dest_path}'.{hint}"
            )

    logger.info(
        "Download complete: %d/%d file(s) in '%s'",
        len(downloaded),
        len(expected_files),
        destination_dir,
    )
    return downloaded

def _files_already_present(destination_dir: Path, expected_files: list[str]) -> bool:
    """
    Return True only if every expected filename exists in destination_dir
    with a non-zero file size.
    """
    for filename in expected_files:
        path = destination_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def _unzip_file(zip_path: Path, destination_dir: Path, logger: Any) -> None:
    """Unzip a .zip file into destination_dir and remove the archive."""
    logger.info("  [UNZIP] Extracting '%s'...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir)
    zip_path.unlink()
    logger.info("  [UNZIP] Done. Archive removed.")