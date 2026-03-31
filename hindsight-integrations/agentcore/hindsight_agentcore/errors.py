class HindsightAgentCoreError(Exception):
    """Exception raised when a Hindsight memory operation fails inside AgentCore Runtime."""
    pass


class BankResolutionError(HindsightAgentCoreError):
    """Raised when bank ID resolution fails — fails closed to prevent memory leakage."""
    pass
