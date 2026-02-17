"""File converter implementations."""

from .base import FileConverter
from .markitdown import MarkitdownConverter

__all__ = ["FileConverter", "MarkitdownConverter", "ConverterRegistry"]


class ConverterRegistry:
    """Registry for file converters with auto-detection."""

    def __init__(self):
        """Initialize empty converter registry."""
        self._converters: dict[str, FileConverter] = {}

    def register(self, converter: FileConverter):
        """
        Register a converter.

        Args:
            converter: FileConverter instance
        """
        self._converters[converter.name()] = converter

    def get_converter(
        self,
        name: str | None,
        filename: str,
        content_type: str | None = None,
    ) -> FileConverter:
        """
        Get converter by name or auto-detect.

        Args:
            name: Converter name (e.g., "markitdown") or None for auto-detect
            filename: File name for auto-detection
            content_type: MIME type (optional)

        Returns:
            FileConverter instance

        Raises:
            ValueError: If no suitable converter found
        """
        if name:
            # Explicit converter requested
            if name not in self._converters:
                raise ValueError(f"Converter '{name}' not found. Available: {list(self._converters.keys())}")
            return self._converters[name]

        # Auto-detect converter
        for converter in self._converters.values():
            if converter.supports(filename, content_type):
                return converter

        raise ValueError(f"No converter found for {filename}. Available converters: {list(self._converters.keys())}")

    def list_converters(self) -> list[str]:
        """Get list of registered converter names."""
        return list(self._converters.keys())
