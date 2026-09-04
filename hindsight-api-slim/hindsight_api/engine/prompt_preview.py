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

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from ..config import HindsightConfig

PreviewOperation = Literal["retain", "observations", "reflect"]

PREVIEW_OPERATIONS: tuple[str, ...] = get_args(PreviewOperation)

BlockSource = Literal["config", "builtin"]
InputKind = Literal["text", "boolean", "choice", "complex"]

# Stand-ins for the runtime data a real call would carry (the chunk being retained,
# the facts being consolidated, the question being reflected on). Bracketed rather
# than lorem-ipsum so nobody mistakes filler for part of the prompt.
SAMPLE_RETAIN_CONTENT = "«the text being retained»"
SAMPLE_RETAIN_CONTEXT = "«context supplied with the document»"
SAMPLE_OBSERVATIONS_FACTS = "«the batch of new facts being consolidated»"
SAMPLE_OBSERVATIONS_EXISTING = "«the observations this bank already holds»"
SAMPLE_REFLECT_QUERY = "«the question being asked»"

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

EXTRACTION_MODES = ["concise", "verbose", "verbatim", "chunks", "custom"]


@dataclass(frozen=True)
class PromptBlock:
    """One block of a message: its text, and the setting that decides it.

    The **active** blocks of a message concatenate back to the exact text sent —
    nothing dropped, reordered or duplicated — so a client can render them
    separately without showing the reader something the model never receives.

    An **inactive** block has no text. It marks a setting that is switched off, at
    the point in the message where it would land if it were on, with ``note``
    explaining what turning it on would do. Without these, every unset setting would
    be invisible on the one screen built for changing them.

    ``source`` says what kind of thing produced the block, which is what tells a
    reader whether they can act on it: ``config`` is a setting (``field`` names it,
    and ``value``/``kind``/``choices``/``editable`` describe how to change it), and
    ``builtin`` is Hindsight's own fixed wording.

    The runtime data an operation is given — the chunk, the facts, the question — is
    NOT a block. It is a hole in the built-in scaffolding, marked inline with «…»,
    and cutting the scaffolding around it only produced a run of near-identical
    "Extraction request" blocks of eleven characters each.
    """

    label: str
    text: str
    source: BlockSource
    field: str = ""
    active: bool = True
    note: str | None = None
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
class PromptPreview:
    """What one call of ``operation`` would send.

    ``skipped_reason`` is set when the configuration means no prompt is sent at all
    (chunks mode); ``messages`` is then empty and the reason is the whole answer.
    """

    operation: str
    messages: list[PromptMessage]
    response_schema: dict[str, Any] | None = None
    skipped_reason: str | None = None


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


def _section_name(text: str, default: str) -> str:
    """Name a built-in block after its first section heading, else ``default``.

    The heading is trimmed at the first dash or bracket — "SELECTIVITY - CRITICAL
    (Reduces 90% of unnecessary output)" is a heading written to shout at a model,
    not a label to put in a list.
    """
    match = _HEADING.search(text)
    if not match:
        return default
    fenced = match.group("fenced")
    raw = fenced or match.group("markdown") or ""
    heading = re.split(r"\s[-–(]", raw.strip(), maxsplit=1)[0].strip()
    if not heading:
        return default
    name = f"{heading[0].upper()}{heading[1:].lower()}"
    # A fenced heading is a single shouted noun ("SELECTIVITY", "ENTITIES") and needs
    # the suffix to read as a label; a markdown one is already a phrase ("## CRITICAL
    # RULES"), and appending to it produced "Critical rules rules".
    return f"{name} rules" if fenced else name


def _partition(text: str, default_label: str, pieces: list[PromptBlock]) -> list[PromptBlock]:
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
            # Only a piece with a note earns a place when it contributes nothing:
            # that note is what makes an off block worth showing. A piece with neither
            # text nor note is simply absent.
            if piece.note:
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
    for matched, (start, end, piece) in enumerate(sorted(matches, key=lambda m: m[0])):
        if start > cursor:
            gap = text[cursor:start]
            blocks.append(PromptBlock(label=_section_name(gap, default_label), text=gap, source="builtin"))
        blocks.append(piece)
        cursor = end
        blocks.extend(inactive for anchor, inactive in pending if anchor == matched + 1)
    if cursor < len(text):
        gap = text[cursor:]
        blocks.append(PromptBlock(label=_section_name(gap, default_label), text=gap, source="builtin"))
    # Anything anchored before the first match (or when nothing matched) leads.
    # Two gaps are never adjacent — a matched piece always separates them — so there
    # is nothing to merge here. That was not true while runtime placeholders were
    # blocks: they cut the scaffolding into repeated one-line fragments.
    return [inactive for anchor, inactive in pending if anchor == 0] + blocks


def _setting(
    field_name: str,
    label: str,
    text: str,
    *,
    value: str | None,
    kind: InputKind = "text",
    choices: list[str] | None = None,
    note: str | None = None,
) -> PromptBlock:
    """A block produced by a setting. Inactive when it contributes no text."""
    return PromptBlock(
        label=label,
        text=text,
        source="config",
        field=field_name,
        active=bool(text),
        note=note,
        value=value,
        kind=kind,
        choices=choices,
    )


def _language_blocks(config: "HindsightConfig", default_rule: str) -> list[PromptBlock]:
    """The two mutually exclusive halves of ``llm_output_language``.

    Exactly one is ever present — an explicit language drops the keep-the-source-
    language rule outright rather than arguing with it (see default_language_section)
    — so both are offered and the absent one simply does not match.
    """
    from .prompt_utils import output_language_directive

    value = config.llm_output_language
    # Only the half that is actually present is returned. Offering both and letting
    # the absent one fall out as "off" would show one setting as two blocks, one of
    # them permanently switched off — but there is no switch: setting a language
    # *replaces* the keep-the-source-language rule, it does not enable a second one.
    if not value and not default_rule:
        return []
    if value:
        return [
            _setting("llm_output_language", "Output language", output_language_directive(value).strip(), value=value)
        ]
    return [_setting("llm_output_language", "Language rule", default_rule, value="match input")]


def _render_retain(
    config: "HindsightConfig", *, content: str, context: str, event_date: datetime | None
) -> PromptPreview:
    from .retain.entity_labels import parse_entity_labels
    from .retain.fact_extraction import (
        _DEFAULT_LANGUAGE_RULE,
        CAUSAL_RELATIONSHIPS_SECTION,
        _build_labels_prompt_section,
        _retain_mission_preamble,
        build_chunk_prompt_parts,
    )

    mode = config.retain_extraction_mode or "concise"
    if mode == "chunks":
        return PromptPreview(operation="retain", messages=[], skipped_reason=CHUNKS_MODE_EXPLANATION)

    # Retain stamps the current time when a caller sends no timestamp — only an
    # explicit null leaves it unset (see the orchestrator's event_date_value). A
    # preview that left it None showed "Event Date: Unknown", which is not the line
    # the model gets for an ordinary retain.
    from .retain.orchestrator import utcnow

    parts = build_chunk_prompt_parts(config, chunk=content, event_date=event_date or utcnow(), context=context)

    labels_section = _build_labels_prompt_section(
        parse_entity_labels(config.entity_labels), config.entities_allow_free_form
    )
    system_pieces = [
        *_language_blocks(config, _DEFAULT_LANGUAGE_RULE),
        # Only offered in custom mode, where it is the extraction rules. In any other
        # mode the field is inert — the builder never reads it — so an off block for
        # it would be pointing at a slot that does not exist in this prompt.
        *(
            [
                _setting(
                    "retain_custom_instructions",
                    "Custom extraction instructions",
                    config.retain_custom_instructions or "",
                    value=config.retain_custom_instructions,
                    note=(
                        "Unset, so custom mode falls back to the built-in concise rules above. "
                        "Write instructions to replace them."
                    ),
                )
            ]
            if mode == "custom"
            else []
        ),
        _setting(
            "retain_extract_causal_links",
            "Causal relationships",
            CAUSAL_RELATIONSHIPS_SECTION.strip() if config.retain_extract_causal_links else "",
            value=str(config.retain_extract_causal_links).lower(),
            kind="boolean",
            note="Would ask the model to link facts that caused one another, and widen the response schema.",
        ),
        _setting(
            "entity_labels",
            "Entity labels",
            labels_section.strip(),
            value=str(config.entity_labels) if config.entity_labels else None,
            kind="complex",
            note="Would restrict entities to a controlled vocabulary of key:value labels.",
        ),
    ]

    mission = (config.retain_mission or "").strip()
    user_pieces = [
        _setting(
            "retain_mission",
            "Mission",
            _retain_mission_preamble(config).strip() if mission else "",
            value=config.retain_mission,
            note="Would be prepended here, taking priority over the general extraction rules.",
        ),
    ]

    # The mode picks the whole system prompt, so it belongs on the first built-in
    # block rather than floating in a separate list — that block *is* the mode.
    system_blocks = _partition(parts.system_prompt, "Extraction rules", system_pieces)
    for i, block in enumerate(system_blocks):
        if block.source == "builtin":
            system_blocks[i] = replace(
                block, field="retain_extraction_mode", value=mode, kind="choice", choices=EXTRACTION_MODES
            )

    schema = parts.response_schema.model_json_schema() if hasattr(parts.response_schema, "model_json_schema") else None
    return PromptPreview(
        operation="retain",
        messages=[
            PromptMessage(role="system", blocks=system_blocks),
            PromptMessage(role="user", blocks=_partition(parts.user_message, "Extraction request", user_pieces)),
        ],
        response_schema=schema,
    )


def _render_observations(config: "HindsightConfig", *, facts_text: str, observations_text: str) -> PromptPreview:
    from .consolidation.prompts import (
        _DEFAULT_LANGUAGE_RULE,
        build_consolidation_input,
        build_consolidation_system_prompt,
    )

    system_prompt = build_consolidation_system_prompt(llm_output_language=config.llm_output_language)
    user_prompt = build_consolidation_input(
        facts_text=facts_text,
        observations_text=observations_text,
        observations_mission=config.observations_mission,
    )

    mission = (config.observations_mission or "").strip()
    user_pieces = [
        _setting(
            "observations_mission",
            "Mission",
            mission,
            value=config.observations_mission,
            note="Unset, so the built-in default mission below is used instead.",
        ),
    ]
    return PromptPreview(
        operation="observations",
        messages=[
            PromptMessage(
                role="system",
                blocks=_partition(
                    system_prompt, "Consolidation rules", _language_blocks(config, _DEFAULT_LANGUAGE_RULE)
                ),
            ),
            PromptMessage(role="user", blocks=_partition(user_prompt, "Consolidation input", user_pieces)),
        ],
    )


def _render_reflect(
    config: "HindsightConfig",
    bank_profile: dict[str, Any],
    *,
    query: str,
    context: str | None,
    include_observations: bool,
    has_mental_models: bool,
    budget: str | None,
    directives: list[dict[str, Any]] | None,
) -> PromptPreview:
    from .reflect.prompts import (
        _TOOLS_LANGUAGE_RULE,
        build_agent_user_prompt,
        build_directives_section,
        build_system_prompt_for_tools,
    )

    # reflect_mission out-ranks the legacy banks.mission column (see
    # _overlay_bank_config_disposition_mission), and the caller may be previewing an
    # unsaved edit to it — so apply it here rather than trusting the stored profile.
    profile = dict(bank_profile)
    if config.reflect_mission:
        profile["mission"] = config.reflect_mission

    system_prompt = build_system_prompt_for_tools(
        profile,
        context,
        directives=directives,
        has_mental_models=has_mental_models,
        include_observations=include_observations,
        budget=budget,
        llm_output_language=config.llm_output_language,
    )
    user_prompt = build_agent_user_prompt(query, config.llm_output_language)

    # The identity head is built from the bank's own name and disposition sliders, so
    # it reads as hardcoded prompt text unless it is named and its origin stated.
    # Rebuilt here exactly as build_system_prompt_for_tools writes it; if that
    # wording changes these simply stop matching and fold back into the surrounding
    # built-in block rather than mislabelling anything.
    disposition = profile.get("disposition") or {}
    traits = ", ".join(f"{t}={disposition[t]}" for t in ("skepticism", "literalism", "empathy") if t in disposition)
    system_pieces = [
        PromptBlock(
            label="Bank identity",
            text=f"## Memory Bank: {profile.get('name', 'Assistant')}",
            source="builtin",
            note="This bank's name — rename it on the General tab.",
        ),
        PromptBlock(
            label="Disposition",
            text=f"Disposition: {traits}" if traits else "",
            source="builtin",
            active=bool(traits),
            note="The bank's skepticism, literalism and empathy traits — set them in the Reflect section.",
        ),
        # Directives are the bank's hard rules, injected near the top of the agent's
        # prompt. They have their own page rather than a config field, so the block
        # names them and says where they live instead of offering an editor.
        PromptBlock(
            label="Directives",
            text=build_directives_section(directives).strip() if directives else "",
            source="builtin",
            active=bool(directives),
            note="Hard rules the agent must follow — add them on the bank's Directives list.",
        ),
        _setting(
            "reflect_mission",
            "Mission",
            (profile.get("mission") or "").strip(),
            value=config.reflect_mission,
            note="Would become the agent's role, replacing the built-in default one above.",
        ),
        *_language_blocks(config, _TOOLS_LANGUAGE_RULE),
    ]
    user_pieces = _language_blocks(config, "")
    return PromptPreview(
        operation="reflect",
        messages=[
            PromptMessage(role="system", blocks=_partition(system_prompt, "Reflect agent rules", system_pieces)),
            PromptMessage(role="user", blocks=_partition(user_prompt, "Question", user_pieces)),
        ],
    )


def render_prompt_preview(
    operation: str,
    config: "HindsightConfig",
    bank_profile: dict[str, Any],
    *,
    content: str | None = None,
    context: str | None = None,
    event_date: datetime | None = None,
    existing_observations: str | None = None,
    include_observations: bool = True,
    has_mental_models: bool = True,
    budget: str | None = None,
    directives: list[dict[str, Any]] | None = None,
) -> PromptPreview:
    """Render ``operation``'s prompts against ``config`` — no LLM call, no writes.

    ``content`` stands in for whatever the operation would be given at runtime: the
    text being retained, the facts being consolidated, or the question being
    reflected on. Omit it and a bracketed placeholder is used instead, so the
    surrounding instructions stay readable.

    Raises ``ValueError`` for an unknown operation.
    """
    preview = _render(
        operation,
        config,
        bank_profile,
        content=content,
        context=context,
        event_date=event_date,
        existing_observations=existing_observations,
        include_observations=include_observations,
        has_mental_models=has_mental_models,
        budget=budget,
        directives=directives,
    )

    # Which fields a bank may actually override is decided in one place — the config
    # layer's own allowlist — so mark them from it rather than by hand per renderer.
    # `llm_output_language` and `retain_extract_causal_links` shape the prompt but are
    # server-level, and a UI that offered to edit them would only collect a 400.
    from ..config import HindsightConfig

    configurable = HindsightConfig._CONFIGURABLE_FIELDS
    return PromptPreview(
        operation=preview.operation,
        messages=[
            PromptMessage(
                role=message.role,
                blocks=[
                    replace(block, editable=block.field in configurable and block.kind != "complex")
                    for block in message.blocks
                ],
            )
            for message in preview.messages
        ],
        response_schema=preview.response_schema,
        skipped_reason=preview.skipped_reason,
    )


def _render(
    operation: str,
    config: "HindsightConfig",
    bank_profile: dict[str, Any],
    *,
    content: str | None,
    context: str | None,
    event_date: datetime | None,
    existing_observations: str | None,
    include_observations: bool,
    has_mental_models: bool,
    budget: str | None,
    directives: list[dict[str, Any]] | None,
) -> PromptPreview:
    """Dispatch to the per-operation renderer. See :func:`render_prompt_preview`."""
    if operation == "retain":
        return _render_retain(
            config,
            content=content or SAMPLE_RETAIN_CONTENT,
            context=context if context is not None else SAMPLE_RETAIN_CONTEXT,
            event_date=event_date,
        )
    if operation == "observations":
        return _render_observations(
            config,
            facts_text=content or SAMPLE_OBSERVATIONS_FACTS,
            observations_text=existing_observations or SAMPLE_OBSERVATIONS_EXISTING,
        )
    if operation == "reflect":
        return _render_reflect(
            config,
            bank_profile,
            query=content or SAMPLE_REFLECT_QUERY,
            context=context,
            include_observations=include_observations,
            has_mental_models=has_mental_models,
            budget=budget,
            directives=directives,
        )
    raise ValueError(f"Unknown prompt preview operation '{operation}'. Allowed: {sorted(PREVIEW_OPERATIONS)}")
