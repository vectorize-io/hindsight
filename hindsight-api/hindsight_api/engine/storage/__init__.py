"""File storage backends for uploaded files."""

from collections.abc import Callable

from .base import FileStorage
from .postgresql import PostgreSQLFileStorage

__all__ = ["FileStorage", "PostgreSQLFileStorage", "create_file_storage"]


def create_file_storage(
    storage_type: str,
    pool_getter: Callable | None = None,
    schema: str | None = None,
    **kwargs,
) -> FileStorage:
    """
    Create file storage backend based on configuration.

    Args:
        storage_type: "native" (PostgreSQL BYTEA) or "s3" (S3-compatible object storage)
        pool_getter: Database pool getter (required for native)
        schema: Database schema (for native multi-tenant)
        **kwargs: Additional args passed to storage backend

    Returns:
        FileStorage instance

    Raises:
        ValueError: If storage_type is unknown or required args are missing
    """
    if storage_type == "native":
        if not pool_getter:
            raise ValueError("pool_getter required for native (PostgreSQL) storage")
        return PostgreSQLFileStorage(pool_getter=pool_getter, schema=schema)
    elif storage_type == "s3":
        from ...config import get_config
        from .s3 import S3FileStorage

        config = get_config()
        bucket = config.file_storage_s3_bucket
        if not bucket:
            raise ValueError("HINDSIGHT_API_FILE_STORAGE_S3_BUCKET is required for S3 storage")
        return S3FileStorage(
            bucket=bucket,
            region=config.file_storage_s3_region,
            endpoint=config.file_storage_s3_endpoint,
            access_key_id=config.file_storage_s3_access_key_id,
            secret_access_key=config.file_storage_s3_secret_access_key,
        )
    elif storage_type == "gcs":
        from ...config import get_config
        from .gcs import GCSFileStorage

        config = get_config()
        bucket = config.file_storage_gcs_bucket
        if not bucket:
            raise ValueError("HINDSIGHT_API_FILE_STORAGE_GCS_BUCKET is required for GCS storage")
        return GCSFileStorage(
            bucket=bucket,
            service_account_key=config.file_storage_gcs_service_account_key,
        )
    elif storage_type == "azure":
        from ...config import get_config
        from .azure import AzureFileStorage

        config = get_config()
        container = config.file_storage_azure_container
        if not container:
            raise ValueError("HINDSIGHT_API_FILE_STORAGE_AZURE_CONTAINER is required for Azure storage")
        return AzureFileStorage(
            container_name=container,
            account_name=config.file_storage_azure_account_name,
            account_key=config.file_storage_azure_account_key,
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}. Supported: 'native', 's3', 'gcs', 'azure'.")
