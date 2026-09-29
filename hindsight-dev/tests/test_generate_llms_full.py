from pathlib import Path

from hindsight_dev.generate_llms_full import clean_markdown, extract_section, get_docs_dir


def test_extract_section_joins_parts_and_dedents():
    code = "\n".join(
        [
            "// [docs:a] // [docs:b]",
            "import { X } from 'x';",
            "// [/docs:a] // [/docs:b]",
            "func main() {",
            "\t// [docs:a]",
            "\tx := 1",
            "\t// [/docs:a]",
            "}",
        ]
    )
    assert extract_section(code, "a") == "import { X } from 'x';\n\tx := 1"
    assert extract_section(code, "b") == "import { X } from 'x';"


def test_every_code_snippet_renders_its_example_code():
    # clean_markdown raises when a section is missing, so this also catches a renamed section
    for page in get_docs_dir().rglob("*.md*"):
        cleaned = clean_markdown(page.read_text())
        assert "<CodeSnippet" not in cleaned, page
        assert "raw-loader" not in cleaned, page


def test_code_import_lines_survive():
    cleaned = clean_markdown(Path(get_docs_dir() / "sdks" / "nodejs.mdx").read_text())
    assert "import { HindsightClient } from '@vectorize-io/hindsight-client';" in cleaned
    assert 'from "npm:@vectorize-io/hindsight-client"' in cleaned
