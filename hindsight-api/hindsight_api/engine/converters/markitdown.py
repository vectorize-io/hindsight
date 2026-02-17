"""Markitdown converter implementation."""

import asyncio
import logging
import tempfile
from pathlib import Path

from .base import FileConverter

logger = logging.getLogger(__name__)


class MarkitdownConverter(FileConverter):
    """
    Markitdown file converter.

    Uses Microsoft's markitdown library to convert various file formats
    to markdown including PDF, Office docs, images (via OCR), audio, HTML.

    Supported formats:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - PowerPoint (.pptx, .ppt)
    - Excel (.xlsx, .xls)
    - Images (.jpg, .jpeg, .png) - with OCR
    - HTML (.html, .htm)
    - Text (.txt, .md)
    - Audio (.mp3, .wav) - with transcription
    """

    def __init__(self):
        """Initialize markitdown converter."""
        # Lazy import to avoid requiring markitdown for all users
        try:
            from markitdown import MarkItDown

            self._markitdown = MarkItDown()
        except ImportError as e:
            raise ImportError(
                "markitdown package is required for file conversion. Install with: pip install markitdown"
            ) from e

    async def convert(self, file_data: bytes, filename: str) -> str:
        """Convert file to markdown using markitdown."""
        # markitdown is synchronous, so we run it in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._convert_sync, file_data, filename)

    def _convert_sync(self, file_data: bytes, filename: str) -> str:
        """Synchronous conversion (runs in thread pool)."""
        # Write to temp file (markitdown requires file path)
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            # Convert using markitdown
            result = self._markitdown.convert(tmp_path)

            if not result or not result.text_content:
                raise RuntimeError(f"No content extracted from '{filename}'")

            return result.text_content

        except Exception as e:
            logger.error(f"Markitdown conversion failed for {filename}: {e}")
            raise RuntimeError(f"Failed to convert '{filename}': {e}") from e

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def supports(self, filename: str, content_type: str | None = None) -> bool:
        """Check if markitdown supports this file type."""
        # Supported extensions (from markitdown docs)
        supported_extensions = {
            # Documents
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            # Images (with OCR)
            ".jpg",
            ".jpeg",
            ".png",
            # Web
            ".html",
            ".htm",
            # Text
            ".txt",
            ".md",
            ".csv",
            # Audio (with transcription)
            ".mp3",
            ".wav",
        }

        ext = Path(filename).suffix.lower()
        return ext in supported_extensions

    def name(self) -> str:
        """Get converter name."""
        return "markitdown"
