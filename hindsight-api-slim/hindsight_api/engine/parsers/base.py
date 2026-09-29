"""Abstract base class for file parsers."""

from abc import ABC, abstractmethod


class UnsupportedFileTypeError(Exception):
    """Raised by a parser when it does not support the given file type."""

    pass


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
            UnsupportedFileTypeError: If the file type is not supported by this parser
            RuntimeError: If parsing fails for another reason
        """
        pass

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """
        Check if parser supports this file type.

        Override this for local/static extension-based filtering.
        Parsers that delegate to a remote service should leave this as True
        and raise UnsupportedFileTypeError from convert() instead.

        Args:
            filename: File name (used for extension check)
            content_type: MIME type (optional)

        Returns:
            True if this parser can handle the file (default: True)
        """
        return True

    @abstractmethod
    def name(self) -> str:
        """
        Get parser name.

        Returns:
            Parser name (e.g., "markitdown")
        """
        pass
