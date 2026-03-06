"""File parser implementations."""

import logging
from dataclasses import dataclass

from .base import FileParser, UnsupportedFileTypeError
from .iris import IrisParser
from .markitdown import MarkitdownParser

__all__ = [
    "FileParser",
    "UnsupportedFileTypeError",
    "IrisParser",
    "MarkitdownParser",
    "FileParserRegistry",
    "ConvertResult",
]


@dataclass
class ConvertResult:
    """Result of a successful file conversion."""

    content: str
    parser_name: str


logger = logging.getLogger(__name__)


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

    async def convert_with_fallback(
        self,
        parsers: list[str],
        file_data: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> ConvertResult:
        """
        Try each parser in order, falling back on failure or empty content.

        Moves to the next parser if the current one raises UnsupportedFileTypeError
        or returns empty content. Any other exception (RuntimeError, network error,
        etc.) also triggers a fallback so the chain is exhausted before failing.

        Args:
            parsers: Ordered list of parser names to try
            file_data: Raw file bytes
            filename: Original filename
            content_type: MIME type (optional)

        Returns:
            ConvertResult with the parsed content and the name of the parser that succeeded

        Raises:
            ValueError: If a parser name is not registered
            RuntimeError: If all parsers fail or return empty content
        """
        last_error: Exception | None = None
        for name in parsers:
            parser = self.get_parser(name, filename, content_type)
            try:
                content = await parser.convert(file_data, filename)
                if content and content.strip():
                    return ConvertResult(content=content, parser_name=name)
                logger.warning(f"Parser '{name}' returned empty content for '{filename}', trying next")
                last_error = RuntimeError(f"Parser '{name}' returned no content for '{filename}'")
            except UnsupportedFileTypeError as e:
                logger.warning(f"Parser '{name}' does not support '{filename}', trying next: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"Parser '{name}' failed for '{filename}', trying next: {e}")
                last_error = e

        raise last_error or RuntimeError(f"No parsers available for '{filename}'")

    def list_parsers(self) -> list[str]:
        """Get list of registered parser names."""
        return list(self._parsers.keys())
