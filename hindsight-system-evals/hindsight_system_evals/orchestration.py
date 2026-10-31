"""A scripted decision model, and a server that uses it.

A decision-orchestrated reflect searches once, asks the decision model whether that
evidence answers the question, and on "no" forces the LLM to write one more search.
Whether the LLM follows that forced turn is a property of the LLM and of how the call
is made, not of the bank: on a server that parses tool calls but does not enforce
``tool_choice``, a model can answer the forced search with ``done``, and the answer is
written from the first round alone. On a small bank the first round usually holds the
answer anyway, so a graded answer cannot show it. This decision model always says the
evidence is incomplete, and counts how often it is asked: a second verdict is only asked
after a second search round ran.

Every other question gets the answer that leaves retrieval as it was: candidates ranked
in the order they came, all of them kept, every memory relevant to the page.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from hindsight_system_evals.server import EvalServer, start_eval_server

#: The evidence-sufficiency question's id and the level it is answered with: the middle
#: of "does not answer / answers part / fully answers", so a follow-up is due.
SUFFICIENCY_QUESTION = "sufficient"
_PARTLY = 1


@dataclass
class ScriptedDecisionModel:
    url: str = ""
    #: The ``state`` of every sufficiency question since the last ``reset``.
    verdicts: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self._lock:
            self.verdicts.clear()

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        answers = {}
        for key, question in body.get("questions", {}).items():
            criteria = question.get("criteria")
            if question.get("type") == "score":
                levels = len(criteria) if isinstance(criteria, list) else 2
                level = _PARTLY if key == SUFFICIENCY_QUESTION else levels - 1
                if key == SUFFICIENCY_QUESTION:
                    with self._lock:
                        self.verdicts.append(str(body.get("state", "")))
                answers[key] = {
                    "score": float(level),
                    "probabilities": {str(i): 1.0 if i == level else 0.0 for i in range(levels)},
                }
            elif question.get("type") == "choice":
                options = list(criteria or {})
                # Falling probabilities by position: a ranking keeps the order retrieval gave.
                weights = [1.0 / (i + 1) for i in range(len(options))]
                total = sum(weights) or 1.0
                choice = "yes" if "yes" in options else (options[0] if options else None)
                answers[key] = {"choice": choice, "probabilities": {o: w / total for o, w in zip(options, weights)}}
            else:
                answers[key] = {"noul": 1.0}
        return {"answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


def _handler(model: ScriptedDecisionModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            reply = json.dumps(model.answer(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


@dataclass
class OrchestratedServer:
    server: EvalServer
    decision_model: ScriptedDecisionModel


@contextlib.contextmanager
def server_with_scripted_decision_model(log_path: Path) -> Iterator[OrchestratedServer]:
    """A server of this run's model whose decision model is ``ScriptedDecisionModel``."""
    model = ScriptedDecisionModel()
    http = ThreadingHTTPServer(("127.0.0.1", 0), _handler(model))
    model.url = f"http://127.0.0.1:{http.server_address[1]}"
    threading.Thread(target=http.serve_forever, daemon=True).start()
    overrides = {
        "HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER": "typesafe",
        "HINDSIGHT_EVAL_SET_DECISION_MODEL_BASE_URL": model.url,
        "HINDSIGHT_EVAL_SET_DECISION_MODEL_MODEL": "scripted",
        "HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY": "unused",
    }
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        server = start_eval_server(log_path=log_path)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    try:
        yield OrchestratedServer(server=server, decision_model=model)
    finally:
        server.stop()
        http.shutdown()


@dataclass(frozen=True)
class FollowUpCase:
    id: str
    category = "orchestration"
    #: Asked of reflect, and the topic of the page.
    question: str
    facts: tuple[str, ...]


#: Two, for the category's two-case floor. Small on purpose: the first round finds
#: everything, which is the point — only the decision model's "no" calls for more.
CASES = (
    FollowUpCase(
        id="fu-escalation",
        question="Who handles urgent issues for the Brenner Logistik account?",
        facts=(
            "Brenner Logistik signed a three-year contract in January 2026.",
            "Inside support, the Brenner Logistik account is tracked under the code BLX.",
            "Urgent BLX tickets go to Rainer Moser.",
        ),
    ),
    FollowUpCase(
        id="fu-release",
        question="When is the next release of the billing service, and who approves it?",
        facts=(
            "The billing service ships on the first Tuesday of every month.",
            "Billing releases need sign-off from the payments lead.",
            "Mara Ilic has been the payments lead since March 2026.",
        ),
    ),
)
