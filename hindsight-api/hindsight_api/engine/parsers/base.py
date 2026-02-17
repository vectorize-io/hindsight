"""Abstract base class for file parsers."""

from abc import ABC, abstractmethod


class FileParser(ABC):
    """Abstract base for file to markdown parsers."""

    @abstractmethod
    async def convert(self, file_data: bytes, filename: str) -> str:
        """
        Parse file to markdown.

        Args:
            file_data: Raw file bytes
            filename: Original filename (used for format detection)

        Returns:
            Markdown content as string

        Raises:
            ValueError: If file format is not supported
            RuntimeError: If parsing fails
        """
        pass

    @abstractmethod
    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """
        Check if parser supports this file type.

        Args:
            filename: File name (used for extension check)
            content_type: MIME type (optional)

        Returns:
            True if this parser can handle the file
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Get parser name.

        Returns:
            Parser name (e.g., "markitdown")
        """
        pass
