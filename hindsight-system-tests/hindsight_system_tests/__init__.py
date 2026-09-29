"""Support code for the Hindsight blackbox system tests.

Nothing here is imported by the product. The package exists so the tests
themselves stay short: they declare what the LLM says, then talk to a real
server through the public client.
"""

from .reflect import reflect_loop
from .rulebook import ChatRequest, LLMStub, Stubs
from .server import HindsightServer, StubServer, start_hindsight_server, start_stub_server
from .waiting import wait_until_settled

__all__ = [
    "ChatRequest",
    "HindsightServer",
    "LLMStub",
    "StubServer",
    "Stubs",
    "start_hindsight_server",
    "reflect_loop",
    "start_stub_server",
    "wait_until_settled",
]
