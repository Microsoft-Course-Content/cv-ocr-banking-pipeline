"""
Azure Blob Storage Connector for CV & OCR Pipeline.
Stores uploaded document images and processing results.
Falls back to local filesystem when Azure is not configured.
"""

import os
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class BlobStorageConnector:
    """Manages image/document storage for the CV & OCR pipeline."""

    def __init__(self):
        self.conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.container = os.getenv("BLOB_CONTAINER_CV", "cv-ocr-documents")
        self.use_azure = bool(self.conn_str) and AZURE_AVAILABLE
        self.blob_service = None

        if self.use_azure:
            try:
                self.blob_service = BlobServiceClient.from_connection_string(self.conn_str)
                try:
                    self.blob_service.create_container(self.container)
                except Exception:
                    pass
                logger.info("Azure Blob Storage connected for CV & OCR pipeline")
            except Exception as e:
                logger.warning(f"Blob init failed: {e}. Using local.")
                self.use_azure = False

        if not self.use_azure:
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)

    async def store_image(self, file_bytes: bytes, filename: str, content_type: str = "image/png") -> dict:
        """Store uploaded document image."""
        blob_name = f"images/{datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}_{filename}"

        if self.use_azure:
            client = self.blob_service.get_blob_client(self.container, blob_name)
            client.upload_blob(file_bytes, overwrite=True,
                               content_settings=ContentSettings(content_type=content_type))
            path = f"https://{self.blob_service.account_name}.blob.core.windows.net/{self.container}/{blob_name}"
        else:
            local_name = blob_name.replace("/", "_")
            path = os.path.join("uploads", local_name)
            with open(path, "wb") as f:
                f.write(file_bytes)

        return {"storage_path": path, "storage_type": "azure_blob" if self.use_azure else "local", "blob_name": blob_name}

    async def store_result(self, doc_id: str, result: dict) -> str:
        """Store processing result as JSON."""
        blob_name = f"results/{doc_id}.json"
        data = json.dumps(result, default=str, indent=2)

        if self.use_azure:
            client = self.blob_service.get_blob_client(self.container, blob_name)
            client.upload_blob(data, overwrite=True,
                               content_settings=ContentSettings(content_type="application/json"))
            return f"azure://{self.container}/{blob_name}"
        else:
            path = os.path.join("outputs", f"{doc_id}.json")
            with open(path, "w") as f:
                f.write(data)
            return path
