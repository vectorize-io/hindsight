"""Every prompt that asks for operations must describe the operations we accept.

The op vocabulary is written down twice: once as Pydantic models the applier
validates against, and once in prose inside each system prompt. Those two drifted
in exactly the way this kind of duplication always drifts — the delta prompt was
rewritten for the id-addressed schema while the retraction prompt, added on main
in parallel, kept telling the model to say ``"index": N``. Nothing failed loudly:
the ops validated to nothing, every retraction was dropped, and unsaying a
retracted fact silently stopped working.

A test for the prompt that drifted does not exist by construction — it is the one
nobody wrote. So this asserts over *every* prompt that carries an operations
vocabulary, and a new one is caught by being added to a list rather than by
someone remembering.
"""

from __future__ import annotations

import re

import pytest

from hindsight_api.engine.reflect import prompts
from hindsight_api.engine.reflect.delta_ops import Operation

# Prompts that instruct a model to emit operations. Add a prompt here when you
# add one; the point of the list is that forgetting is what this test catches.
_OPERATION_PROMPTS = {
    "STRUCTURED_DELTA_SYSTEM_PROMPT": prompts.STRUCTURED_DELTA_SYSTEM_PROMPT,
    "STRUCTURED_RETRACTION_SYSTEM_PROMPT": prompts.STRUCTURED_RETRACTION_SYSTEM_PROMPT,
}

_OP_SHAPE_RX = re.compile(r'\{"op":\s*"(\w+)"([^`]*)')


def _operation_models() -> dict[str, type]:
    import typing

    return {m.model_fields["op"].default: m for m in typing.get_args(typing.get_args(Operation)[0])}


def _prompt_names() -> list[str]:
    return sorted(_OPERATION_PROMPTS)


@pytest.mark.parametrize("prompt_name", _prompt_names())
class TestPromptsMatchTheSchema:
    def test_every_named_op_exists(self, prompt_name: str):
        """A prompt offering an op we do not implement wastes a whole refresh."""
        known = set(_operation_models())
        named = {m.group(1) for m in _OP_SHAPE_RX.finditer(_OPERATION_PROMPTS[prompt_name])}
        assert named, f"{prompt_name} documents no operations"
        assert named <= known, f"{prompt_name} offers unknown ops: {sorted(named - known)}"

    def test_no_op_shape_uses_a_field_the_schema_rejects(self, prompt_name: str):
        """Every field named in a shape must exist on that op's model.

        This is what caught the retraction prompt: ``index`` is not a field on any
        v2 operation, so a model following that prompt produced ops that failed
        validation and were dropped.
        """
        models = _operation_models()
        prompt = _OPERATION_PROMPTS[prompt_name]
        for match in _OP_SHAPE_RX.finditer(prompt):
            op_name, rest = match.group(1), match.group(2)
            shape = rest.split("``")[0]
            fields = set(re.findall(r'"(\w+)":', shape))
            allowed = set(models[op_name].model_fields)
            assert fields <= allowed, (
                f"{prompt_name} tells the model to send {sorted(fields - allowed)} on "
                f"{op_name!r}; the schema accepts {sorted(allowed)}"
            )

    def test_blocks_are_never_described_as_typed_objects(self, prompt_name: str):
        """v1's typed block union is gone; a prompt still describing it teaches
        the model a payload that cannot validate."""
        prompt = _OPERATION_PROMPTS[prompt_name]
        for legacy in ('"type": "paragraph"', '"type": "bullet_list"', '"type": "ordered_list"'):
            assert legacy not in prompt, f"{prompt_name} still documents the v1 block shape {legacy}"

    def test_block_addressing_is_by_id(self, prompt_name: str):
        """Positional addressing is the defect in #3273 and is not in the schema."""
        prompt = _OPERATION_PROMPTS[prompt_name]
        if "remove_block" in prompt or "replace_block" in prompt:
            assert "block_id" in prompt, f"{prompt_name} addresses blocks without naming block_id"


def test_the_list_covers_every_operations_prompt():
    """A prompt asking for ``{"operations": [...]}`` that is not in the list above
    is exactly the prompt this test cannot check — so find it here instead."""
    unlisted = [
        name
        for name in dir(prompts)
        if name.endswith("_SYSTEM_PROMPT")
        and isinstance(getattr(prompts, name), str)
        and '"operations"' in getattr(prompts, name)
        and name not in _OPERATION_PROMPTS
    ]
    assert not unlisted, f"these prompts ask for operations but are not checked: {unlisted}"
