"""The pre-#3756 chunking oracle: ``RecursiveCharacterTextSplitter``, vendored.

Retain's plain-text chunking used to be langchain's ``RecursiveCharacterTextSplitter``.
#3756 replaced it with a lazy re-implementation, and the tests that guard that swap are
differential — they only mean anything if the thing they diff against is genuinely the old
splitter and not a paraphrase of the new one.

Keeping ``langchain-text-splitters`` installed just to provide that oracle dragged in
``langchain-core`` -> ``langsmith`` -> ``orjson``, so the algorithm lives here instead,
transcribed from ``langchain_text_splitters`` 1.1.2 (MIT) — ``character.py``'s
``_split_text_with_regex`` / ``RecursiveCharacterTextSplitter._split_text`` and
``base.py``'s ``TextSplitter._merge_splits`` / ``_join_docs``.

It is specialised to the one configuration retain ever used::

    RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=0, length_function=len,
        is_separator_regex=False, separators=_RECURSIVE_TEXT_SEPARATORS,
    )

so ``keep_separator`` is its default ``True`` (the "start" sense: a separator leads the
NEXT piece), ``strip_whitespace`` is its default ``True``, and the merge separator is
therefore always ``""``. The branches that configuration cannot reach — regex separators,
``keep_separator="end"``, non-zero overlap, token length functions — are left out rather
than transcribed untested.

``test_chunking_reference_matches_langchain.py`` is what makes this trustworthy: while
langchain was still installed, this module was diffed against the real splitter over every
corpus the chunking tests use plus 20,000 fuzzed documents, and matched byte-for-byte.

**Do not refactor this to call — or share anything with — the live implementation in
``fact_extraction``.** The moment the two share a code path, every test that diffs them
passes vacuously. That is the same rule the file it replaced carried.
"""

import re

# The pop loop and the merge accounting below both branch on the overlap; retain always
# ran with none. Named rather than folded away so the transcription still lines up with
# langchain's source.
_CHUNK_OVERLAP = 0


def _split_text_with_regex_keep_start(text: str, separator_pattern: str) -> list[str]:
    """``_split_text_with_regex(..., keep_separator=True)`` — separators lead the next piece."""
    if separator_pattern:
        # The parentheses in the pattern keep the delimiters in the result.
        splits_ = re.split(f"({separator_pattern})", text)
        splits = [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]
        if len(splits_) % 2 == 0:
            splits += splits_[-1:]
        splits = [splits_[0], *splits]
    else:
        splits = list(text)
    return [s for s in splits if s]


def _join_docs(docs: list[str], separator: str) -> str | None:
    text = separator.join(docs).strip()
    return text or None


def _merge_splits(splits: list[str], separator: str, chunk_size: int) -> list[str]:
    """``TextSplitter._merge_splits`` — pack pieces up to ``chunk_size``."""
    separator_len = len(separator)

    docs: list[str] = []
    current_doc: list[str] = []
    total = 0
    for d in splits:
        len_ = len(d)
        if total + len_ + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            # langchain logs a warning when total > chunk_size here; nothing observable.
            if len(current_doc) > 0:
                doc = _join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                # With no overlap to preserve this drains current_doc completely.
                while total > _CHUNK_OVERLAP or (
                    total + len_ + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0
                ):
                    total -= len(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += len_ + (separator_len if len(current_doc) > 1 else 0)
    doc = _join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """``RecursiveCharacterTextSplitter._split_text`` — descend the separator tiers."""
    final_chunks: list[str] = []

    # Pick the first separator that occurs in this text; the last one is the fallback.
    separator = separators[-1]
    new_separators: list[str] = []
    for i, s_ in enumerate(separators):
        separator_ = re.escape(s_)
        if not s_:
            separator = s_
            break
        if re.search(separator_, text):
            separator = s_
            new_separators = separators[i + 1 :]
            break

    splits = _split_text_with_regex_keep_start(text, re.escape(separator))

    # Merge what fits; recurse into anything that does not.
    good_splits: list[str] = []
    # keep_separator is on, so pieces already carry their separator and are joined bare.
    merge_separator = ""
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, merge_separator, chunk_size))
                good_splits = []
            if not new_separators:
                final_chunks.append(s)
            else:
                final_chunks.extend(_split_text(s, new_separators, chunk_size))
    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, merge_separator, chunk_size))
    return final_chunks


def recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    """The exact call retain's plain-text chunking made before #3756."""
    return _split_text(text, separators, max_chars)
