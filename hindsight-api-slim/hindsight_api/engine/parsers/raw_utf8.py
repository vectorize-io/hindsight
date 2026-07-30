"""Source-faithful UTF-8 file parser."""

from .base import FileParser


class RawUtf8Parser(FileParser):
    """Decode UTF-8 text without normalizing or sanitizing its contents."""

    async def convert(self, file_data: bytes, filename: str) -> str:
        """Strictly decode UTF-8 bytes and reject text PostgreSQL cannot preserve."""
        try:
            content = file_data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"'{filename}' is not valid UTF-8") from error

        if "\x00" in content:
            raise ValueError(f"'{filename}' contains a NUL byte")

        return content

    def name(self) -> str:
        """Return the registered parser name."""
        return "raw_utf8"

    def contract_version(self) -> str:
        """Return the stable source-fidelity contract version."""
        return "1"

    def preserves_source_text(self) -> bool:
        """Raw UTF-8 output is already the approved source text."""
        return True
