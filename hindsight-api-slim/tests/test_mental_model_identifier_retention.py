"""Tests for the write-time identifier-retention gate.

The gate refuses a mental-model refresh that drops too many anchored
identifiers over existing real content. The logic lives in
``engine/identifier_retention.py`` as pure functions precisely so it can be
tested without a database or an LLM; the engine call site is three lines.

The cases that matter are the two false-signal directions:
  * a moved identifier counts as KEPT (set semantics), or the gate would
    refuse every reordering and be switched off within a day
    (test_moved_identifier_counts_as_kept);
  * THIS GATE returns no-refuse whenever ``has_delta_baseline`` is false, so it
    cannot block a bootstrap write over empty/PENDING content
    (test_bootstrap_write_is_never_refused). Scoped deliberately: other guards
    on the same path -- the emptiness check and the #3112 delta-window guard,
    both ABOVE the call site so their more precise refusals win -- have their
    own conditions, and this file does not speak for them. The call-site
    ordering itself is pinned by the engine-level tests in
    ``test_mental_model_dry_run_refresh.py``.
"""

from __future__ import annotations

import unittest

from hindsight_api.engine.identifier_retention import (
    DEFAULT_IDENTIFIER_LOSS_REFUSE,
    ENV_IDENTIFIER_LOSS_REFUSE,
    evaluate,
    extract_identifiers,
    format_warning,
    lost_identifiers,
    refuse_threshold,
)

BEFORE = (
    "Decision of 2026-07-27 recorded in TRIAL-CLOSE-RUNBOOK.md; "
    "overlay at commit 5ef5a70; see D:\\HQ_runtime\\patches\\build.py "
    "and HINDSIGHT_API_REFLECT_WALL_TIMEOUT, tracked as HQ-123 on 127.0.0.1:18888."
)


class ExtractionTests(unittest.TestCase):
    def test_taxonomy_covers_the_audit_classes(self):
        ids = extract_identifiers(BEFORE)
        for expected in (
            "2026-07-27",
            "TRIAL-CLOSE-RUNBOOK.md",
            "5ef5a70",
            "HINDSIGHT_API_REFLECT_WALL_TIMEOUT",
            "HQ-123",
            ":18888",
        ):
            self.assertIn(expected, ids, f"{expected} should be an identifier")

    def test_empty_and_none_are_safe(self):
        self.assertEqual(extract_identifiers(None), set())
        self.assertEqual(extract_identifiers(""), set())

    def test_port_class_needs_a_host_prefix(self):
        """Pinning a real limit of the shared taxonomy, measured 2026-08-10.

        `\\b:\\d{4,5}\\b` requires a word character before the colon, so
        `127.0.0.1:18888` and `localhost:6280` match but a bare ` :18888`
        does not. That is inherited from the offline probe's pattern and must
        NOT be "fixed" here: the gate and the probe agreeing on what counts as
        an identifier is the point, and a divergence would make the two
        instruments contradict each other on the same event.
        """
        self.assertIn(":18888", extract_identifiers("at 127.0.0.1:18888 now"))
        self.assertIn(":6280", extract_identifiers("localhost:6280"))
        self.assertEqual(extract_identifiers("on :18888."), set())


class GradingTests(unittest.TestCase):
    def test_no_loss_is_silent(self):
        refuse, warning = evaluate(BEFORE, BEFORE + " Plus a new sentence.", True)
        self.assertFalse(refuse)
        self.assertIsNone(warning)

    def test_moved_identifier_counts_as_kept(self):
        """The main false-positive risk: reordering is not loss."""
        reordered = " ".join(reversed(BEFORE.split()))
        self.assertEqual(lost_identifiers(BEFORE, reordered), set())
        refuse, warning = evaluate(BEFORE, reordered, True)
        self.assertFalse(refuse)
        self.assertIsNone(warning)

    def test_single_loss_warns_but_proceeds(self):
        after = BEFORE.replace("2026-07-27", "recently")
        refuse, warning = evaluate(BEFORE, after, True, threshold=3)
        self.assertFalse(refuse, "one lost identifier is often legitimate churn")
        self.assertIsNotNone(warning)
        self.assertIn("2026-07-27", warning, "the warning must name it verbatim")

    def test_three_losses_refuse(self):
        after = "Everything was rewritten and the anchors are gone."
        lost = lost_identifiers(BEFORE, after)
        self.assertGreaterEqual(len(lost), 3)
        refuse, warning = evaluate(BEFORE, after, True, threshold=3)
        self.assertTrue(refuse)
        self.assertIn("dropped", warning)

    def test_threshold_zero_never_refuses_but_still_warns(self):
        after = "Everything was rewritten and the anchors are gone."
        refuse, warning = evaluate(BEFORE, after, True, threshold=0)
        self.assertFalse(refuse)
        self.assertIsNotNone(warning, "the signal must stay visible when disabled")

    def test_bootstrap_write_is_never_refused(self):
        """With has_delta_baseline false THIS gate always returns no-refuse.

        Asserted at threshold=1, the most aggressive setting, so the pass is
        not an artifact of a lenient default. Scope: this gate only — other
        guards on the write path may still reject a bootstrap candidate.
        """
        after = "brand new document with no identifiers"
        refuse, warning = evaluate(BEFORE, after, False, threshold=1)
        self.assertFalse(refuse)
        self.assertIsNone(warning)

    def test_warning_caps_the_named_list(self):
        lost = {f"HQ-{i}" for i in range(25)}
        text = format_warning(lost)
        self.assertIn("+15 more", text)
        self.assertIn("25 identifier(s)", text)


class ThresholdEnvTests(unittest.TestCase):
    def setUp(self):
        import os

        self._os = os
        self._prev = os.environ.get(ENV_IDENTIFIER_LOSS_REFUSE)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._prev is None:
            self._os.environ.pop(ENV_IDENTIFIER_LOSS_REFUSE, None)
        else:
            self._os.environ[ENV_IDENTIFIER_LOSS_REFUSE] = self._prev

    def test_default_when_unset(self):
        self._os.environ.pop(ENV_IDENTIFIER_LOSS_REFUSE, None)
        self.assertEqual(refuse_threshold(), DEFAULT_IDENTIFIER_LOSS_REFUSE)

    def test_env_override(self):
        self._os.environ[ENV_IDENTIFIER_LOSS_REFUSE] = "5"
        self.assertEqual(refuse_threshold(), 5)

    def test_garbage_env_falls_back_rather_than_raising(self):
        """This sits on a write path; a typo'd env var must not break refreshes."""
        self._os.environ[ENV_IDENTIFIER_LOSS_REFUSE] = "three"
        self.assertEqual(refuse_threshold(), DEFAULT_IDENTIFIER_LOSS_REFUSE)

    def test_negative_is_clamped(self):
        self._os.environ[ENV_IDENTIFIER_LOSS_REFUSE] = "-2"
        self.assertEqual(refuse_threshold(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
