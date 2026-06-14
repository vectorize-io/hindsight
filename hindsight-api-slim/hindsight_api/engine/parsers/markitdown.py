"""Markitdown parser implementation."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from hindsight_api.config import DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT

from .base import FileParser

logger = logging.getLogger(__name__)


class MarkitdownParser(FileParser):
    """
    Markitdown file parser.

    Uses Microsoft's markitdown library to convert various file formats
    to markdown including PDF, Office docs, images with optional OCR,
    audio, HTML.

    Supported formats:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - PowerPoint (.pptx, .ppt)
    - Excel (.xlsx, .xls)
    - Images (.jpg, .jpeg, .png) - optional OCR
    - HTML (.html, .htm)
    - Text (.txt, .md)
    - Audio (.mp3, .wav) - with transcription
    """

    def __init__(
        self,
        *,
        ocr_enabled: bool = False,
        ocr_api_key: str | None = None,
        ocr_base_url: str | None = None,
        ocr_model: str | None = None,
        ocr_prompt: str | None = None,
        ocr_default_headers: dict | None = None,
    ):
        """Initialize markitdown parser."""
        # Lazy import to avoid requiring markitdown for all users
        try:
            from markitdown import MarkItDown
        except ImportError as e:
            raise ImportError(
                "markitdown package is required for file parsing. Install with: pip install markitdown"
            ) from e

        self._ocr_enabled = ocr_enabled
        kwargs = {}
        if ocr_enabled:
            kwargs = self._build_ocr_kwargs(
                api_key=ocr_api_key,
                base_url=ocr_base_url,
                model=ocr_model,
                prompt=ocr_prompt,
                default_headers=ocr_default_headers,
            )

        self._markitdown = MarkItDown(**kwargs)

    def _build_ocr_kwargs(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        prompt: str | None,
        default_headers: dict | None,
    ) -> dict:
        """Build MarkItDown kwargs for OpenAI-compatible image OCR."""
        if not model or not model.strip():
            raise ValueError(
                "Markitdown OCR is enabled but no model is configured. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_MODEL or HINDSIGHT_API_LLM_MODEL."
            )
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Markitdown OCR is enabled but no API key is configured. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_API_KEY, HINDSIGHT_API_LLM_API_KEY, or OPENAI_API_KEY."
            )

        from openai import OpenAI

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        return {
            "llm_client": OpenAI(**client_kwargs),
            "llm_model": model,
            "llm_prompt": prompt or DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT,
        }

    async def convert(self, file_data: bytes, filename: str) -> str:
        """Parse file to markdown using markitdown."""
        # markitdown is synchronous, so we run it in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._convert_sync, file_data, filename)

    def _convert_sync(self, file_data: bytes, filename: str) -> str:
        """Synchronous parsing (runs in thread pool)."""
        if self._is_image_file(filename) and not self._ocr_enabled:
            raise RuntimeError(
                "Image OCR is not enabled for the markitdown parser. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_ENABLED=true and configure a vision-capable "
                "OpenAI-compatible model, or choose an OCR-capable parser."
            )

        # Write to temp file (markitdown requires file path)
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            # Parse using markitdown
            result = self._markitdown.convert(tmp_path)

            if not result or not result.text_content:
                raise RuntimeError(f"No content extracted from '{filename}'")

            return result.text_content

        except Exception as e:
            logger.error(f"Markitdown parsing failed for {filename}: {e}")
            raise RuntimeError(f"Failed to parse '{filename}': {e}") from e

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Return whether the file type needs OCR to extract useful text."""
        return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}

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
            # Images (optional OCR)
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
        """Get parser name."""
        return "markitdown"
