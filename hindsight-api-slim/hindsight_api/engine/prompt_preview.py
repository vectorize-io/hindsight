"""Render the prompts an operation would send, without calling an LLM.

Missions (``retain_mission``, ``observations_mission``, ``reflect_mission``) are
edited per bank in the control plane, but a mission only means something once you
see the prompt it lands in — and for retain and observations it lands in the *user*
message, never the system prompt, because both keep their system prefix
bank-agnostic so one provider-side cache serves every bank. A preview that showed
only the system prompt would therefore show a configured mission as absent, which
is exactly backwards. Every operation here renders both messages.

A message is reported as **blocks**: the active ones concatenate back to the exact
text sent, and each names the setting that produced it and carries that setting's
current value, so the same screen can show what the prompt says and let you change
it. Inactive blocks are settings that are switched off, listed in the position they
*would* occupy — the mission you have not written yet is the one you came here to
write, and it has no text to sit beside.

Each renderer calls the same builder the real operation calls, so a preview cannot
drift from what is sent.
"""

import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from ..config import HindsightConfig

PreviewOperation = Literal["retain", "consolidation", "reflect"]

PREVIEW_OPERATIONS: tuple[str, ...] = get_args(PreviewOperation)

BlockSource = Literal["config", "builtin"]
InputKind = Literal["text", "boolean", "choice", "complex"]

# Stand-ins for the runtime data a real call carries (the chunk being retained, the
# facts being consolidated, the question being reflected on). Bracketed rather than
# lorem-ipsum so nobody mistakes filler for part of the prompt.
#
# These are fixed, not caller-supplied. A preview answers "what does this bank send",
# and letting a caller pass its own sample only moved the question — the prompt would
# then depend on something the bank does not hold.
PLACEHOLDER_RETAIN_CONTENT = "«the text being retained»"
PLACEHOLDER_RETAIN_CONTEXT = "«context supplied with the document»"
PLACEHOLDER_CONSOLIDATION_FACTS = "«the batch of new facts being consolidated»"
PLACEHOLDER_CONSOLIDATION_OBSERVATIONS = "«the observations this bank already holds»"
PLACEHOLDER_REFLECT_QUERY = "«the question being asked»"

# Reflect shapes its system prompt around per-request options rather than bank config.
# The preview renders the fullest form — every tool available — because a prompt with
# a tool section missing invites the reader to conclude the bank cannot use it.
PREVIEW_REFLECT_INCLUDE_OBSERVATIONS = True
PREVIEW_REFLECT_HAS_MENTAL_MODELS = True

# `chunks` is not an extraction style — it is the absence of extraction. The retain
# path returns before any LLM queue or lock is touched (`_extract_facts_chunks`), so
# there is no prompt to show. Falling through to the concise template, which is what
# the prompt builder does for any unrecognised mode, would invent one.
CHUNKS_MODE_EXPLANATION = (
    "Chunks mode stores each chunk verbatim as its own memory and never calls an LLM, "
    "so retain sends no prompt at all. Entity labels, the mission and the custom "
    "instructions have no effect in this mode. Switch to concise, verbose, verbatim or "
    "custom to see a prompt."
)


@dataclass(frozen=True)
class PromptBlock:
    """One block of a message: its text, and the setting that decides it.

    The **active** blocks of a message concatenate back to the exact text sent —
    nothing dropped, reordered or duplicated — so a client can render them
    separately without showing the reader something the model never receives.

    An **inactive** block has no text. It marks a setting that is switched off, at
    the point in the message where it would land if it were on. What that would do
    is for the client to say: everything identifying a block here is a machine
    value, never display copy, so the wording stays with the UI that localises it.

    A block is identified by whichever of these applies, in that order: ``field`` is
    the config field behind it; ``section`` is a slug for a part the preview names
    itself and no single field owns (``bank_identity``, ``disposition``,
    ``directives``); ``heading`` is the section heading the prompt text carries at
    that point, extracted from the prompt rather than authored here. A block with
    none of the three is unremarkable built-in wording.
    """

    text: str
    source: BlockSource
    field: str = ""
    section: str = ""
    heading: str = ""
    active: bool = True
    value: str | None = None
    kind: InputKind = "text"
    choices: list[str] | None = None
    # Whether the bank may override the field at all. Decided centrally from the
    # config layer's allowlist — see `render_prompt_preview`.
    editable: bool = False


@dataclass(frozen=True)
class PromptMessage:
    """One message of the request, as the blocks it is built from."""

    role: Literal["system", "user"]
    blocks: list[PromptBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The message as sent — the active blocks partition it exactly."""
        return "".join(block.text for block in self.blocks if block.active)


@dataclass(frozen=True)
class RunSetting:
    """A setting that shapes the operation without appearing in its prompt.

    Chunk sizes decide how the input is cut before extraction ever runs, so they
    change what comes back while contributing no prompt text. They cannot be blocks —
    blocks partition the message, and these are in none of it — but leaving them out
    entirely made the lab look as though the prompt were the whole story.
    """

    field: str
    value: str | None
    kind: InputKind = "text"
    editable: bool = False


@dataclass(frozen=True)
class PromptPreview:
    """What one call of the requested operation would send.

    ``skipped_reason`` is set when the configuration means no prompt is sent at all
    (chunks mode); ``messages`` is then empty and the reason is the whole answer.
    """

    messages: list[PromptMessage]
    response_schema: dict[str, Any] | None = None
    skipped_reason: str | None = None
    # The retain strategy these prompts were rendered under, and the names a client
    # can offer instead of asking for the bank config to find them.
    strategy: str | None = None
    strategies: list[str] = field(default_factory=list)
    run_settings: list[RunSetting] = field(default_factory=list)


# A section heading is either a line fenced by rules of box-drawing characters (how
# the retain and consolidation prompts write them) or a markdown `##` heading (how
# the reflect prompt does). Naming a built-in block after its own heading beats
# numbering the leftovers "(1/2)", "(2/2)" — those said only that the block had been
# split, which is an artefact of where the settings land, not something the reader
# needs to know.
_HEADING = re.compile(
    r"^(?:═+\s*\n(?P<fenced>[^\n═][^\n]*)\n═+\s*|#{2,3} +(?P<markdown>[^\n]+))$",
    re.MULTILINE,
)


def _heading(text: str) -> str:
    """The section heading this block of prompt text carries, or "" if it has none.

    Extracted from the prompt rather than authored here, so it is content and not
    display copy. Trimmed at the first dash or bracket — "SELECTIVITY - CRITICAL
    (Reduces 90% of unnecessary output)" is written to shout at a model, not to
    label a list — and title-cased, since the fenced form is shouted.
    """
    match = _HEADING.search(text)
    if not match:
        return ""
    raw = match.group("fenced") or match.group("markdown") or ""
    heading = re.split(r"\s[-–(]", raw.strip(), maxsplit=1)[0].strip()
    if not heading:
        return ""
    return f"{heading[0].upper()}{heading[1:].lower()}"


def _partition(text: str, pieces: list[PromptBlock]) -> list[PromptBlock]:
    """Cut ``text`` into blocks, one per piece, with the gaps kept as built-ins.

    Each piece's ``text`` is a fragment one of the prompt builders substituted or
    appended, so it is present verbatim — but matching is best-effort by design: an
    empty fragment (an unset setting) or one a builder reworded is skipped rather
    than cut at a wrong offset, and that text simply stays in the surrounding
    built-in block. Everything not claimed by a piece becomes a built-in block, so
    the blocks always concatenate back to ``text``.
    """
    matches: list[tuple[int, int, PromptBlock]] = []
    # An inactive piece has no text to find, so it cannot be placed by matching. It is
    # anchored to the last piece before it that *did* match, which keeps it where the
    # author listed it — "this is where that setting would land" is the whole point of
    # showing it, and a switched-off setting floated to the end says nothing.
    pending: list[tuple[int, PromptBlock]] = []
    cursor = 0
    for piece in pieces:
        if not piece.text:
            # A piece that contributes nothing still earns a place when it is
            # identifiable — that is the whole point of an off block, showing where a
            # switched-off setting would land. An anonymous empty piece is absent.
            if piece.field or piece.section:
                pending.append((len(matches), piece))
            continue
        index = text.find(piece.text, cursor)
        if index == -1:
            # Fall back to searching from the start: pieces are listed in emission
            # order, but a reordering upstream should degrade to an unordered match
            # rather than dropping the block.
            index = text.find(piece.text)
            if index == -1 or any(index < end and start < index + len(piece.text) for start, end, _ in matches):
                continue
        matches.append((index, index + len(piece.text), piece))
        cursor = index + len(piece.text)

    blocks: list[PromptBlock] = []
    cursor = 0

    def add_gap(gap: str) -> None:
        """Emit the unclaimed text as a built-in block — unless it is only the blank
        lines between two claimed pieces.

        An appended section brings its own surrounding newlines, so the text between
        it and its neighbour is often one or two characters of whitespace. As a block
        of its own that is a row saying "Extraction rules · 1 chars", which is noise;
        it belongs to the block before it, and the parts must still concatenate back
        to the message exactly.
        """
        if not gap:
            return
        if not gap.strip() and blocks:
            blocks[-1] = replace(blocks[-1], text=blocks[-1].text + gap)
            return
        blocks.append(PromptBlock(text=gap, source="builtin", heading=_heading(gap)))

    for matched, (start, end, piece) in enumerate(sorted(matches, key=lambda m: m[0])):
        add_gap(text[cursor:start])
        blocks.append(piece)
        cursor = end
        blocks.extend(inactive for anchor, inactive in pending if anchor == matched + 1)
    add_gap(text[cursor:])
    # Anything anchored before the first match (or when nothing matched) leads.
    # Two gaps are never adjacent — a matched piece always separates them — so there
    # is nothing to merge here. That was not true while runtime placeholders were
    # blocks: they cut the scaffolding into repeated one-line fragments.
    return [inactive for anchor, inactive in pending if anchor == 0] + blocks


def _setting(
    field_name: str,
    text: str,
    *,
    value: str | None,
    kind: InputKind = "text",
    choices: list[str] | None = None,
) -> PromptBlock:
    """A block produced by a setting. Inactive when it contributes no text."""
    return PromptBlock(
        text=text,
        source="config",
        field=field_name,
        active=bool(text),
        value=value,
        kind=kind,
        choices=choices,
    )


def _named(section: str, text: str, *, active: bool = True) -> PromptBlock:
    """A built-in block the preview names itself, because no single field owns it."""
    return PromptBlock(text=text, source="builtin", section=section, active=active)


def _language_blocks(config: "HindsightConfig", default_rule: str) -> list[PromptBlock]:
    """Whichever half of ``llm_output_language`` is actually present.

    Exactly one ever is — an explicit language drops the keep-the-source-language
    rule outright rather than arguing with it (see default_language_section) — so
    offering both would show one setting as two blocks, one permanently switched
    off. There is no switch: setting a language *replaces* the default rule.
    """
    from .prompt_utils import output_language_directive

    value = config.llm_output_language
    if not value and not default_rule:
        return []
    if value:
        return [_setting("llm_output_language", output_language_directive(value).strip(), value=value)]
    return [_setting("llm_output_language", default_rule, value=None)]


def _labels_pieces(
    section: str,
    *,
    entity_labels_value: str | None,
    free_form: bool,
    free_form_line: str,
) -> list[PromptBlock]:
    """Split the entity-labels section around the line the free-form flag decides.

    Cut rather than overlapped: the partition matches by locating text, and a piece
    contained inside another piece can only be dropped. Head and tail stay attributed
    to ``entity_labels``; a client showing the control once per field renders them as
    one setting split by another.
    """
    labels = _setting("entity_labels", "", value=entity_labels_value, kind="complex")
    if not section:
        # No labels configured: the section is absent, so the flag decides nothing and
        # only the (switched-off) labels block is worth reporting.
        return [labels]

    head, _, tail = section.partition(free_form_line)
    if not _:
        return [replace(labels, text=section, active=True)]
    return [
        replace(labels, text=head, active=bool(head)),
        _setting("entities_allow_free_form", free_form_line, value=str(free_form).lower(), kind="boolean"),
        replace(labels, text=tail, active=bool(tail)),
    ]


def _retain_run_settings(config: "HindsightConfig") -> list[RunSetting]:
    """Retain's chunking: what the extractor is handed, decided before any prompt."""
    return [
        RunSetting(field="retain_chunk_size", value=_as_text(config.retain_chunk_size)),
        RunSetting(field="retain_structured_chunk_size", value=_as_text(config.retain_structured_chunk_size)),
    ]


def _as_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _render_retain(config: "HindsightConfig") -> PromptPreview:
    from ..config import RETAIN_EXTRACTION_MODES
    from .retain.entity_labels import parse_entity_labels
    from .retain.fact_extraction import (
        _DEFAULT_LANGUAGE_RULE,
        CAUSAL_RELATIONSHIPS_SECTION,
        _build_labels_prompt_section,
        _retain_mission_preamble,
        build_chunk_prompt_parts,
        build_free_form_entities_instruction,
    )

    mode = config.retain_extraction_mode or "concise"
    mode_block = _setting(
        "retain_extraction_mode",
        "",
        value=mode,
        kind="choice",
        choices=list(RETAIN_EXTRACTION_MODES),
    )
    if mode == "chunks":
        # The mode still comes back, switched off, so the control that got the reader
        # here is the one thing they can still reach. Returning nothing left them at a
        # dead end: no prompt, and no way to pick a mode that would produce one. The
        # chunk sizes matter more here than anywhere: in this mode they are the only
        # thing deciding what gets stored.
        return PromptPreview(
            messages=[PromptMessage(role="system", blocks=[mode_block])],
            skipped_reason=CHUNKS_MODE_EXPLANATION,
            run_settings=_retain_run_settings(config),
        )

    # Retain stamps the current time on an item that carries no timestamp — only an
    # explicit null leaves it unset (see the orchestrator's event_date_value).
    # Passing None here rendered "Event Date: Unknown", which is not the line the
    # model gets for an ordinary retain.
    from .retain.orchestrator import utcnow

    parts = build_chunk_prompt_parts(
        config,
        chunk=PLACEHOLDER_RETAIN_CONTENT,
        event_date=utcnow(),
        context=PLACEHOLDER_RETAIN_CONTEXT,
    )

    labels_section = _build_labels_prompt_section(
        parse_entity_labels(config.entity_labels), config.entities_allow_free_form
    )
    system_pieces = [
        *_language_blocks(config, _DEFAULT_LANGUAGE_RULE),
        # Only offered in custom mode, where it is the extraction rules. In any other
        # mode the field is inert — the builder never reads it — so an off block for
        # it would point at a slot that does not exist in this prompt.
        *(
            [
                _setting(
                    "retain_custom_instructions",
                    config.retain_custom_instructions or "",
                    value=config.retain_custom_instructions,
                )
            ]
            if mode == "custom"
            else []
        ),
        _setting(
            "retain_extract_causal_links",
            CAUSAL_RELATIONSHIPS_SECTION.strip() if config.retain_extract_causal_links else "",
            value=str(config.retain_extract_causal_links).lower(),
            kind="boolean",
        ),
        # The labels section is decided by two settings, so it is reported as two:
        # `entities_allow_free_form` owns one sentence in the middle of it, and
        # attributing the whole section to `entity_labels` hid a flag that really does
        # change what the model is told. The section only exists when labels are
        # configured, which is why the flag has no block of its own without them.
        *_labels_pieces(
            labels_section.strip(),
            entity_labels_value=json.dumps(config.entity_labels) if config.entity_labels else None,
            free_form=config.entities_allow_free_form,
            free_form_line=build_free_form_entities_instruction(config.entities_allow_free_form),
        ),
    ]

    mission = (config.retain_mission or "").strip()
    user_pieces = [
        _setting(
            "retain_mission",
            _retain_mission_preamble(config).strip() if mission else "",
            value=config.retain_mission,
        ),
    ]

    # The mode picks the whole system prompt, so it belongs on the built-in blocks
    # rather than floating in a list of its own — those blocks *are* the mode.
    system_blocks = _partition(parts.system_prompt, system_pieces)
    for i, block in enumerate(system_blocks):
        if block.source == "builtin":
            system_blocks[i] = replace(
                block,
                field=mode_block.field,
                value=mode_block.value,
                kind=mode_block.kind,
                choices=mode_block.choices,
            )

    schema = parts.response_schema.model_json_schema() if hasattr(parts.response_schema, "model_json_schema") else None
    return PromptPreview(
        messages=[
            PromptMessage(role="system", blocks=system_blocks),
            PromptMessage(role="user", blocks=_partition(parts.user_message, user_pieces)),
        ],
        response_schema=schema,
        run_settings=_retain_run_settings(config),
    )


def _render_consolidation(config: "HindsightConfig") -> PromptPreview:
    from .consolidation.prompts import (
        _DEFAULT_LANGUAGE_RULE,
        build_consolidation_input,
        build_consolidation_system_prompt,
        build_mission_section,
    )

    system_prompt = build_consolidation_system_prompt(llm_output_language=config.llm_output_language)
    user_prompt = build_consolidation_input(
        facts_text=PLACEHOLDER_CONSOLIDATION_FACTS,
        observations_text=PLACEHOLDER_CONSOLIDATION_OBSERVATIONS,
        observations_mission=config.observations_mission,
    )

    # Unlike retain's, this mission always occupies its slot — unset just means the
    # built-in default fills it — so it is never an off block. Built by the same
    # helper the user message uses, heading included: the heading would otherwise be
    # a built-in block of its own that `## MISSION` names "Mission" too.
    user_pieces = [
        _setting(
            "observations_mission",
            build_mission_section(config.observations_mission),
            value=config.observations_mission,
        ),
    ]
    return PromptPreview(
        messages=[
            PromptMessage(
                role="system",
                blocks=_partition(system_prompt, _language_blocks(config, _DEFAULT_LANGUAGE_RULE)),
            ),
            PromptMessage(role="user", blocks=_partition(user_prompt, user_pieces)),
        ],
    )


def _render_reflect(
    config: "HindsightConfig",
    bank_profile: dict[str, Any],
    directives: list[dict[str, Any]] | None,
) -> PromptPreview:
    from .reflect.prompts import (
        _TOOLS_LANGUAGE_RULE,
        bank_disposition_line,
        bank_name_line,
        build_agent_user_prompt,
        build_directives_section,
        build_system_prompt_for_tools,
    )

    # reflect_mission out-ranks the legacy banks.mission column (see
    # _overlay_bank_config_disposition_mission), and the profile is what the prompt
    # builder reads — so apply it here rather than trusting the stored column.
    profile = dict(bank_profile)
    if config.reflect_mission:
        profile["mission"] = config.reflect_mission

    system_prompt = build_system_prompt_for_tools(
        profile,
        None,
        directives=directives,
        has_mental_models=PREVIEW_REFLECT_HAS_MENTAL_MODELS,
        include_observations=PREVIEW_REFLECT_INCLUDE_OBSERVATIONS,
        budget=None,
        llm_output_language=config.llm_output_language,
    )
    user_prompt = build_agent_user_prompt(PLACEHOLDER_REFLECT_QUERY, config.llm_output_language)

    # The identity head is the bank's own name and disposition traits, so it reads as
    # hardcoded prompt text unless it is called out. Both lines come from the prompt
    # module's own helpers, so there is one copy of the wording, not two.
    disposition = bank_disposition_line(profile)
    system_pieces = [
        _named("bank_identity", bank_name_line(profile)),
        _named("disposition", disposition, active=bool(disposition)),
        # Directives are the bank's hard rules, injected near the top of the agent's
        # prompt. They have their own page rather than a config field, so the block
        # is named for them and carries no editor.
        _named(
            "directives",
            build_directives_section(directives).strip() if directives else "",
            active=bool(directives),
        ),
        _setting("reflect_mission", (profile.get("mission") or "").strip(), value=config.reflect_mission),
        *_language_blocks(config, _TOOLS_LANGUAGE_RULE),
    ]
    return PromptPreview(
        messages=[
            PromptMessage(role="system", blocks=_partition(system_prompt, system_pieces)),
            PromptMessage(role="user", blocks=_partition(user_prompt, _language_blocks(config, ""))),
        ],
    )


def render_prompt_preview(
    operation: str,
    config: "HindsightConfig",
    bank_profile: dict[str, Any],
    directives: list[dict[str, Any]] | None = None,
) -> PromptPreview:
    """Render ``operation``'s prompts for this bank — no LLM call, no writes.

    Everything that shapes the prompt comes from the bank: its resolved config, its
    profile, its directives. The runtime data an operation would be given is a fixed
    bracketed placeholder, so what comes back is this bank's prompt and nothing else.

    Raises ``ValueError`` for an unknown operation.
    """
    if operation == "retain":
        preview = _render_retain(config)
    elif operation == "consolidation":
        preview = _render_consolidation(config)
    elif operation == "reflect":
        preview = _render_reflect(config, bank_profile, directives)
    else:
        raise ValueError(f"Unknown prompt preview operation '{operation}'. Allowed: {sorted(PREVIEW_OPERATIONS)}")

    # Which fields a bank may actually override is decided in one place — the config
    # layer's own allowlist — so mark them from it rather than by hand per renderer.
    # `llm_output_language` and `retain_extract_causal_links` shape the prompt but are
    # server-level, and a UI that offered to edit them would only collect a 400.
    from ..config import HindsightConfig

    configurable = HindsightConfig._CONFIGURABLE_FIELDS
    return PromptPreview(
        messages=[
            PromptMessage(
                role=message.role,
                blocks=[replace(block, editable=block.field in configurable) for block in message.blocks],
            )
            for message in preview.messages
        ],
        run_settings=[replace(s, editable=s.field in configurable) for s in preview.run_settings],
        response_schema=preview.response_schema,
        skipped_reason=preview.skipped_reason,
    )
