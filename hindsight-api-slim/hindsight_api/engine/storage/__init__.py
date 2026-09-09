"""File storage backends for uploaded files."""

from collections.abc import Callable

from .base import FileStorage
from .postgresql import PostgreSQLFileStorage

__all__ = ["FileStorage", "PostgreSQLFileStorage", "create_file_storage"]


def create_file_storage(
    storage_type: str,
    pool_getter: Callable | None = None,
    schema: str | None = None,
    schema_getter: Callable | None = None,
    **kwargs,
) -> FileStorage:
    """
    Create file storage backend based on configuration.

    Args:
        storage_type: "native" (PostgreSQL BYTEA) or "s3" (S3-compatible object storage)
        pool_getter: Database pool getter (required for native)
        schema: Static database schema (for native single-tenant)
        schema_getter: Callable returning current schema at query time (for native multi-tenant)
        **kwargs: Additional args passed to storage backend

    Returns:
        FileStorage instance

    Raises:
        ValueError: If storage_type is unknown or required args are missing
    """
    # A deployment can register its own file backend through the extension
    # mechanism, exactly as it registers a memories store. Checked first so an
    # external store can supply one without this factory naming any provider.
    extension_storage = _load_file_storage_extension()
    if extension_storage is not None:
        return extension_storage

    if storage_type == "native":
        if not pool_getter:
            raise ValueError("pool_getter required for native (PostgreSQL) storage")
        return PostgreSQLFileStorage(pool_getter=pool_getter, schema=schema, schema_getter=schema_getter)
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
        raise ValueError(
            f"Unknown storage type: {storage_type}. Supported: 'native', 's3', 'gcs', 'azure'. "
            "External backends load through HINDSIGHT_API_FILE_STORAGE_EXTENSION."
        )


def _load_file_storage_extension() -> FileStorage | None:
    """Load a file-storage backend registered through the extension mechanism.

    ``HINDSIGHT_API_FILE_STORAGE_EXTENSION=module.path:ClassName`` names a
    :class:`FileStorage` implementation; every other ``HINDSIGHT_API_FILE_STORAGE_*``
    variable is handed to it as a lowercased config dict. This keeps
    provider-specific backends out of this factory — the same pattern the memories
    store uses via ``HINDSIGHT_API_MEMORIES_EXTENSION``. Returns ``None`` when no
    extension is configured, so the built-in backends below stay the default.
    """
    import importlib
    import os

    from ...config import get_config

    ext_path = get_config().file_storage_extension
    if not ext_path:
        return None
    if ":" not in ext_path:
        raise ValueError(
            f"Invalid HINDSIGHT_API_FILE_STORAGE_EXTENSION '{ext_path}'. Expected format 'module.path:ClassName'."
        )
    module_path, class_name = ext_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, FileStorage)):
        raise ValueError(f"File storage extension '{class_name}' must inherit from FileStorage.")

    prefix = "HINDSIGHT_API_FILE_STORAGE_"
    config = {
        key[len(prefix) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(prefix) and key != "HINDSIGHT_API_FILE_STORAGE_EXTENSION"
    }
    return cls(config)
