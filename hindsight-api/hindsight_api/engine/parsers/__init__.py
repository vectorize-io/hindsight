"""File parser implementations."""

from .base import FileParser, UnsupportedFileTypeError
from .iris import IrisParser
from .markitdown import MarkitdownParser

__all__ = ["FileParser", "UnsupportedFileTypeError", "IrisParser", "MarkitdownParser", "FileParserRegistry"]


class FileParserRegistry:
    """Registry for file parsers with auto-detection."""

    def __init__(self):
        """Initialize empty parser registry."""
        self._parsers: dict[str, FileParser] = {}

    def register(self, parser: FileParser):
        """
        Register a parser.

        Args:
            parser: FileParser instance
        """
        self._parsers[parser.name()] = parser

    def get_parser(
        self,
        name: str | None,
        filename: str,
        content_type: str | None = None,
    ) -> FileParser:
        """
        Get parser by name or auto-detect.

        Args:
            name: Parser name (e.g., "markitdown") or None for auto-detect
            filename: File name for auto-detection
            content_type: MIME type (optional)

        Returns:
            FileParser instance

        Raises:
            ValueError: If no suitable parser found
        """
        if name:
            # Explicit parser requested — return it directly, let the parser
            # raise UnsupportedFileTypeError from convert() if needed
            if name not in self._parsers:
                raise ValueError(f"Parser '{name}' not found. Available: {list(self._parsers.keys())}")
            return self._parsers[name]

        # Auto-detect parser
        for parser in self._parsers.values():
            if parser.supports(filename, content_type):
                return parser

        raise ValueError(f"No parser found for {filename}. Available parsers: {list(self._parsers.keys())}")

    def list_parsers(self) -> list[str]:
        """Get list of registered parser names."""
        return list(self._parsers.keys())
