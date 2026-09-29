"""Hindsight-Superagent error types."""


class HindsightError(Exception):
    """Exception raised when a Hindsight memory operation fails."""


class GuardBlockedError(HindsightError):
    """Exception raised when Superagent Guard blocks an input.

    Attributes:
        classification: The guard classification ("block").
        reasoning: Why the input was blocked.
        violation_types: List of violation type strings.
        cwe_codes: List of CWE codes matched.
    """

    def __init__(
        self,
        reasoning: str,
        violation_types: list[str],
        cwe_codes: list[str],
    ) -> None:
        self.classification = "block"
        self.reasoning = reasoning
        self.violation_types = violation_types
        self.cwe_codes = cwe_codes
        super().__init__(f"Input blocked by Superagent Guard: {reasoning} (violations: {violation_types})")
