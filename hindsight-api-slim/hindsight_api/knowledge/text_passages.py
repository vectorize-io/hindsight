"""Text passages for knowledge banks: structure first, small, no overlap. No LLM.

Measured on LegalBench-RAG (6,889 queries) and Chroma's chunking set, EmbeddingGemma 2
(hindsight-dev/benchmarks/knowledge_banks_mm/CHUNKING.md):

* small passages win at an equal amount of returned text, down to ~128 tokens;
* splitting on paragraphs and headings adds 1-2 points over plain recursive splitting;
* overlap, semantic (embedding-breakpoint) chunking, late chunking and putting the heading
  path in the embedded text do not help (late chunking hurts by 10-16 points).

The previous splitter (recursive 512 tokens, 64 overlap) was 7 points worse at 4k returned
characters. Search returns each hit with its neighbouring passages (see ``store``), which is
the query-time step that paid; trimming passages to "only the relevant sentences" did not.

Every passage keeps exact character offsets into the text, so neighbours can be stitched
back without guessing where a passage came from (``str.find`` lands on the wrong copy in
repetitive documents, which is what the recursive splitter used to do).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..engine.token_encoding import count_tokens

#: Biggest boundary first. The empty string is the last resort: split anywhere.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
MIN_SENTENCE_TOKENS = 6


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


def _strip(text: str, start: int, end: int) -> _Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(start, end)


# --- recursive (last resort for an overlong sentence) -------------------------------------------


def _split_on(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    # Keep the separator on the left piece so joining pieces reproduces the text.
    parts = text.split(separator)
    out = [p + separator for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _split_recursive(text: str, max_tokens: int, separators: tuple[str, ...]) -> list[str]:
    """Pieces that each fit in ``max_tokens``, splitting on the first separator that helps."""
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    for i, separator in enumerate(separators):
        pieces = _split_on(text, separator)
        if len(pieces) < 2:
            continue
        out: list[str] = []
        for piece in pieces:
            if count_tokens(piece) <= max_tokens:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, max_tokens, separators[i + 1 :]))
        return out
    return [text]


def _recursive_spans(text: str, start: int, end: int, max_tokens: int) -> list[_Span]:
    """Spans of ``text[start:end]`` cut to fit ``max_tokens``, merged greedily, no overlap."""
    pieces = _split_recursive(text[start:end], max_tokens, SEPARATORS)
    out: list[_Span] = []
    run_start = cursor = start
    for piece in pieces:
        if cursor > run_start and count_tokens(text[run_start : cursor + len(piece)]) > max_tokens:
            out.append(_strip(text, run_start, cursor))
            run_start = cursor
        cursor += len(piece)
    if cursor > run_start:
        out.append(_strip(text, run_start, cursor))
    return [s for s in out if s.end > s.start]


# --- sentences ----------------------------------------------------------------------------------

# A sentence ends at . ? ! ; or : followed by whitespace, or at a newline. Not after common
# abbreviations or a lone enumerator ("1.", "(a)."), which would cut "Section 2.1. The" apart.
_END = re.compile(r"(?<=[.?!;:])[\"')\]]*\s+|\n+")
_NO_BREAK = re.compile(
    r"(?:\b(?:No|Nos|Sec|Secs|Art|Inc|Ltd|Co|Corp|Mr|Mrs|Ms|Dr|U\.S|e\.g|i\.e|etc|vs|v|St|Jr|Sr)\.|"
    r"(?:^|\s)\(?[0-9ivxlcA-Za-z]{1,4}[.)]|\d\.)$"
)


def _sentences(text: str, start: int, end: int, max_tokens: int) -> list[_Span]:
    """Sentence spans of ``text[start:end]``; tiny fragments merge forward, overlong ones are cut."""
    raw: list[_Span] = []
    cursor = start
    for m in _END.finditer(text, start, end):
        if "\n" not in m.group(0) and _NO_BREAK.search(text[max(cursor, m.start() - 12) : m.start()]):
            continue
        raw.append(_Span(cursor, m.end()))
        cursor = m.end()
    if cursor < end:
        raw.append(_Span(cursor, end))
    out: list[_Span] = []
    pending: _Span | None = None
    for span in raw:
        span = _strip(text, pending.start if pending else span.start, span.end)
        if span.end <= span.start:
            continue
        tokens = count_tokens(text[span.start : span.end])
        if tokens < MIN_SENTENCE_TOKENS:
            pending = span
            continue
        pending = None
        if tokens > max_tokens:
            out.extend(_recursive_spans(text, span.start, span.end, max_tokens))
        else:
            out.append(span)
    if pending is not None:
        if out:
            out[-1] = _Span(out[-1].start, pending.end)
        else:
            out.append(pending)
    return out


# --- structure ----------------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
# Contract and report headers: "ARTICLE IV", "Section 5.02 Covenants.", "12. TERMINATION", all caps.
_ARTICLE = re.compile(r"^(?:ARTICLE|Article|PART|Part|SCHEDULE|Schedule|EXHIBIT|Exhibit)\s+[0-9IVXLC]+\b")
_SECTION = re.compile(r"^(?:Section|SECTION|§)\s*(\d+(?:\.\d+)*)\.?\s*([^.\n]{0,80})")
# "5. NONDISCRIMINATION. Grantee shall ..." / "6.2 Term of Agreement." -- a numbered, title-like
# lead-in (<= 8 words, each capitalized or a short joiner), alone or followed by the body text.
_NUMBERED = re.compile(r"^(\(?\d+(?:\.\d+)*\)?)\.?\s+([A-Z][^.\n:]{0,80})(?:[.:]|$)")
_JOINERS = {"of", "and", "the", "to", "in", "for", "on", "or", "a", "an", "by", "with", "&"}


def _title_like(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 8 and all(w[0].isupper() or w.lower() in _JOINERS or not w[0].isalpha() for w in words)


def _opens_section(line: str) -> bool:
    """Whether a line is a heading: a markdown heading, an article, a section or a numbered title, or a
    short all-caps line."""
    s = line.strip()
    if not s:
        return False
    if _MD_HEADING.match(s) or _ARTICLE.match(s) or _SECTION.match(s):
        return True
    m = _NUMBERED.match(s)
    if m and _title_like(m.group(2)):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(s) <= 120 and len(letters) >= 4 and all(c.isupper() for c in letters) and len(s.split()) <= 12


@dataclass(frozen=True)
class _Block:
    span: _Span
    opens_section: bool


def _blocks(text: str) -> list[_Block]:
    """Paragraph blocks. Blank lines separate paragraphs; when a text has (almost) none, newlines do."""
    sep = re.compile(r"\n[ \t]*\n+") if text.count("\n\n") * 20 >= text.count("\n") else re.compile(r"\n+")
    out: list[_Block] = []
    start = 0
    for m in [*sep.finditer(text), None]:
        end = m.start() if m else len(text)
        span = _strip(text, start, end)
        start = m.end() if m else len(text)
        if span.end <= span.start:
            continue
        out.append(_Block(span, _opens_section(text[span.start : span.end].split("\n", 1)[0])))
    return out


def split_into_passages(text: str, *, passage_size: int) -> list[str]:
    """Split a document into passages of at most ``passage_size`` tokens, structure first.

    A heading starts a new passage; paragraphs are merged up to ``passage_size``; a paragraph
    bigger than that is cut by sentences. No overlap: it did not help (CHUNKING.md).

    A bare heading (a heading line with nothing under it yet) waits for its content: the next
    heading or the first sentences of a cut paragraph join it. A heading used to become a passage
    of its own there ("## Observations", "Core concepts"),
    and a few words nearly equal to a short query outranked every real answer.
    """
    if passage_size <= 0:
        raise ValueError("passage_size must be positive")
    spans: list[_Span] = []
    run: _Span | None = None
    run_tokens = 0
    bare = False  # the run holds only bare headings so far

    def flush() -> None:
        nonlocal run, run_tokens, bare
        if run is not None:
            spans.append(run)
        run, run_tokens, bare = None, 0, False

    for block in _blocks(text):
        chunk = text[block.span.start : block.span.end]
        tokens = count_tokens(chunk)
        heading = block.opens_section and "\n" not in chunk
        if block.opens_section and not bare:
            flush()
        if tokens > passage_size or (bare and not heading and run_tokens + tokens > passage_size):
            # Cut by sentences. Waiting headings lead the first piece: the sentences come in groups
            # small enough to fit beside them, so the piece stays within passage_size.
            lead, lead_tokens = (run, run_tokens) if bare else (None, 0)
            if not bare:
                flush()
            run, run_tokens, bare = None, 0, False
            piece, piece_tokens = lead, lead_tokens
            room = max(1, passage_size - lead_tokens - 1)  # -1: the blank line after the heading
            for sentence in _sentences(text, block.span.start, block.span.end, room):
                n = count_tokens(text[sentence.start : sentence.end]) + (1 if piece is lead and lead else 0)
                if piece is not None and piece is not lead and piece_tokens + n > passage_size:
                    spans.append(piece)
                    piece, piece_tokens = None, 0
                piece = _Span(piece.start if piece else sentence.start, sentence.end)
                piece_tokens += n
            if piece is not None:
                spans.append(piece)
            continue
        if run is not None and not bare and run_tokens + tokens > passage_size:
            flush()
        if run is None:
            run = block.span
            bare = heading
        else:
            run = _Span(run.start, block.span.end)
            bare = bare and heading
        run_tokens += tokens
    flush()
    # Each passage is an exact slice of the input, in order.
    return [text[span.start : span.end] for span in spans]
