"""Abstract base class for file converters."""

from abc import ABC, abstractmethod


class FileConverter(ABC):
    """Abstract base for file to markdown converters."""

    @abstractmethod
    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Convert file to markdown.

        Args:
            file_data: Raw file bytes
            filename: Original filename (used for format detection)

        Returns:
            Markdown content as string

        Raises:
            ValueError: If file format is not supported
            RuntimeError: If conversion fails
        """
        pass

    @abstractmethod
    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """
        Check if converter supports this file type.

        Args:
            filename: File name (used for extension check)
            content_type: MIME type (optional)

        Returns:
            True if this converter can handle the file
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Get converter name.

        Returns:
            Converter name (e.g., "markitdown")
        """
        pass
