"""Markitdown parser implementation."""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hindsight_api.config import DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT

from .base import FileParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkitdownOcrOptions:
    """OpenAI-compatible OCR options passed through to MarkItDown."""

    # Keep this typed as object so the OpenAI SDK import stays lazy for non-OCR users.
    llm_client: object
    llm_model: str
    llm_prompt: str


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
        if ocr_enabled:
            ocr_options = self._build_ocr_options(
                api_key=ocr_api_key,
                base_url=ocr_base_url,
                model=ocr_model,
                prompt=ocr_prompt,
            )
            self._markitdown = MarkItDown(
                llm_client=ocr_options.llm_client,
                llm_model=ocr_options.llm_model,
                llm_prompt=ocr_options.llm_prompt,
            )
        else:
            self._markitdown = MarkItDown()

    def _build_ocr_options(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        prompt: str | None,
    ) -> MarkitdownOcrOptions:
        """Build MarkItDown options for OpenAI-compatible image OCR."""
        if not model or not model.strip():
            raise ValueError(
                "Markitdown OCR is enabled but no model is configured. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_MODEL to an OpenAI-compatible OCR/vision model "
                "with image-input support."
            )
        if not api_key:
            raise ValueError(
                "Markitdown OCR is enabled but no API key is configured. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_API_KEY."
            )
        if not base_url or not base_url.strip():
            raise ValueError(
                "Markitdown OCR is enabled but no base URL is configured. "
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_BASE_URL to an OpenAI-compatible OCR/vision endpoint."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package is required when Markitdown OCR is enabled.") from e

        return MarkitdownOcrOptions(
            llm_client=OpenAI(api_key=api_key, base_url=base_url.strip()),
            llm_model=model.strip(),
            llm_prompt=prompt or DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT,
        )

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
                "Set HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_ENABLED=true and configure an OpenAI-compatible "
                "OCR/vision endpoint with image-input support, or choose an OCR-capable parser."
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
