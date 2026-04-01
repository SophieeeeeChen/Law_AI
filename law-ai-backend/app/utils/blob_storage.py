"""
Azure Blob Storage utilities for ChromaDB and SQLite migration.

Usage:
1. Upload your local chroma_db folder and SQLite DB to Azure Blob Storage (one-time)
2. Set environment variables: AZURE_STORAGE_CONNECTION_STRING, AZURE_BLOB_CONTAINER_NAME
3. Call ensure_chroma_db_exists() and ensure_sqlite_db_exists() on app startup
"""

import os
import zipfile
import logging
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = BACKEND_DIR / "chroma_db"
CHROMA_ZIP_NAME = "chroma_db.zip"

# SQLite paths
DATA_DIR = Path(os.environ.get("DATA_DIR", BACKEND_DIR / "data"))
SQLITE_DB_NAME = os.environ.get("SQLITE_DB_NAME", "sophieai.db")
SQLITE_DB_PATH = DATA_DIR / SQLITE_DB_NAME
SQLITE_BLOB_NAME = "sophieai.db"


def download_chroma_from_blob() -> bool:
    """
    Download chroma_db.zip from Azure Blob Storage and extract it.
    Returns True if successful, False otherwise.
    """
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("azure-storage-blob not installed. Run: pip install azure-storage-blob")
        return False

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER_NAME", "sophieai-data")

    if not connection_string:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set. Skipping blob download.")
        return False

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=CHROMA_ZIP_NAME
        )

        # Check if blob exists
        if not blob_client.exists():
            logger.warning(f"Blob '{CHROMA_ZIP_NAME}' not found in container '{container_name}'")
            return False

        # Download to temp file
        zip_path = CHROMA_DB_DIR.parent / CHROMA_ZIP_NAME
        logger.info(f"Downloading {CHROMA_ZIP_NAME} from Azure Blob Storage...")

        with open(zip_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

        # Extract
        logger.info(f"Extracting to {CHROMA_DB_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(CHROMA_DB_DIR.parent)

        # Cleanup zip
        zip_path.unlink()
        logger.info("ChromaDB downloaded and extracted successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to download ChromaDB from blob: {e}")
        return False


def ensure_chroma_db_exists() -> bool:
    """
    Ensure chroma_db folder exists. If not, download from Azure Blob.
    Call this on app startup.
    
    Returns True if chroma_db is available, False otherwise.
    """
    # Check if already exists locally
    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        logger.info(f"ChromaDB found at {CHROMA_DB_DIR}")
        return True

    # Try to download from Azure
    logger.info("ChromaDB not found locally. Attempting to download from Azure Blob...")
    return download_chroma_from_blob()


def upload_chroma_to_blob() -> bool:
    """
    Upload local chroma_db folder to Azure Blob Storage as a zip file.
    Run this locally to upload your existing embeddings.
    
    Usage:
        python -c "from app.utils.blob_storage import upload_chroma_to_blob; upload_chroma_to_blob()"
    """
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("azure-storage-blob not installed. Run: pip install azure-storage-blob")
        return False

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER_NAME", "sophieai-data")

    if not connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING not set.")
        return False

    if not CHROMA_DB_DIR.exists():
        logger.error(f"ChromaDB directory not found: {CHROMA_DB_DIR}")
        return False

    try:
        # Create zip file
        zip_path = CHROMA_DB_DIR.parent / CHROMA_ZIP_NAME
        logger.info(f"Zipping {CHROMA_DB_DIR}...")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in CHROMA_DB_DIR.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(CHROMA_DB_DIR.parent)
                    zipf.write(file_path, arcname)

        # Upload to blob
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create container if doesn't exist
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            logger.info(f"Created container: {container_name}")

        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=CHROMA_ZIP_NAME
        )

        logger.info(f"Uploading {zip_path.name} to Azure Blob Storage...")
        with open(zip_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # Cleanup local zip
        zip_path.unlink()

        logger.info(f"Successfully uploaded ChromaDB to {container_name}/{CHROMA_ZIP_NAME}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload ChromaDB: {e}")
        return False


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload_chroma_to_blob()
    else:
        ensure_chroma_db_exists()


# =============================================================================
# SQLite Database - Azure Blob Storage Functions
# =============================================================================

def _get_blob_client(blob_name: str):
    """Helper to get a blob client for the given blob name."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("azure-storage-blob not installed. Run: pip install azure-storage-blob")
        return None

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.environ.get("AZURE_BLOB_CONTAINER_NAME", "sophieai-data")

    if not connection_string:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set.")
        return None

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client.get_blob_client(container=container_name, blob=blob_name)


def download_sqlite_from_blob() -> bool:
    """
    Download SQLite database from Azure Blob Storage.
    Returns True if successful, False otherwise.
    """
    blob_client = _get_blob_client(SQLITE_BLOB_NAME)
    if blob_client is None:
        return False

    try:
        if not blob_client.exists():
            logger.info(f"No existing SQLite DB in blob storage. Starting fresh.")
            return False

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Download
        logger.info(f"Downloading {SQLITE_BLOB_NAME} from Azure Blob Storage...")
        with open(SQLITE_DB_PATH, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

        logger.info(f"SQLite DB downloaded to {SQLITE_DB_PATH}")
        return True

    except Exception as e:
        logger.error(f"Failed to download SQLite from blob: {e}")
        return False


def upload_sqlite_to_blob() -> bool:
    """
    Upload local SQLite database to Azure Blob Storage.
    Call this after important database changes or on a schedule.
    
    Usage:
        python -c "from app.utils.blob_storage import upload_sqlite_to_blob; upload_sqlite_to_blob()"
    """
    blob_client = _get_blob_client(SQLITE_BLOB_NAME)
    if blob_client is None:
        return False

    if not SQLITE_DB_PATH.exists():
        logger.warning(f"SQLite DB not found at {SQLITE_DB_PATH}. Nothing to upload.")
        return False

    try:
        # Create a backup copy first (SQLite best practice - avoid uploading while in use)
        backup_path = SQLITE_DB_PATH.parent / f"{SQLITE_DB_NAME}.backup"
        shutil.copy2(SQLITE_DB_PATH, backup_path)

        logger.info(f"Uploading {SQLITE_BLOB_NAME} to Azure Blob Storage...")
        with open(backup_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # Cleanup backup
        backup_path.unlink()

        logger.info(f"SQLite DB uploaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to upload SQLite to blob: {e}")
        return False


def ensure_sqlite_db_exists() -> bool:
    """
    Ensure SQLite database exists. If not, download from Azure Blob.
    Call this on app startup.
    
    Returns True if database is available (local or downloaded), False otherwise.
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already exists locally
    if SQLITE_DB_PATH.exists():
        logger.info(f"SQLite DB found at {SQLITE_DB_PATH}")
        return True

    # Try to download from Azure
    logger.info("SQLite DB not found locally. Attempting to download from Azure Blob...")
    downloaded = download_sqlite_from_blob()
    
    if not downloaded:
        logger.info("No remote DB found. A new database will be created on first use.")
    
    return True  # App can still start - SQLAlchemy will create tables


def backup_sqlite_to_blob() -> bool:
    """
    Create a timestamped backup of SQLite database in Azure Blob Storage.
    Useful for periodic backups without overwriting the main copy.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_blob_name = f"backups/sophieai_{timestamp}.db"
    
    blob_client = _get_blob_client(backup_blob_name)
    if blob_client is None:
        return False

    if not SQLITE_DB_PATH.exists():
        logger.warning(f"SQLite DB not found at {SQLITE_DB_PATH}. Nothing to backup.")
        return False

    try:
        # Create local backup copy
        backup_path = SQLITE_DB_PATH.parent / f"{SQLITE_DB_NAME}.backup"
        shutil.copy2(SQLITE_DB_PATH, backup_path)

        logger.info(f"Uploading backup as {backup_blob_name}...")
        with open(backup_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # Cleanup local backup
        backup_path.unlink()

        logger.info(f"Backup created: {backup_blob_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to backup SQLite to blob: {e}")
        return False


def get_sqlite_db_path() -> Path:
    """
    Get the SQLite database path. Use this in your database config.
    """
    return SQLITE_DB_PATH
