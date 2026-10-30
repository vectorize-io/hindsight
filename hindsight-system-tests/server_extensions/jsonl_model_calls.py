"""A test-only model call extension, loaded by the server under test (not by the tests).

It keeps every call it is handed in memory and writes them all, one JSON object per line,
to ``HINDSIGHT_API_MODEL_CALL_PATH`` only in ``on_shutdown`` — so a call in the file proves
both that it was recorded and that the server flushed its buffer as it stopped. It lives
outside the ``hindsight_system_tests`` package for the reason ``tag_scope_validator`` does.
"""

import threading

from hindsight_api.extensions import ModelCall, ModelCallExtension


class JsonlModelCalls(ModelCallExtension):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self.path = config["path"]
        self.lines: list[str] = []
        # Calls arrive from whichever thread and event loop made them.
        self.lock = threading.Lock()

    def record(self, call: ModelCall) -> None:
        line = call.model_dump_json()
        with self.lock:
            self.lines.append(line)

    async def on_shutdown(self) -> None:
        with self.lock:
            lines, self.lines = self.lines, []
        with open(self.path, "a") as f:
            f.writelines(line + "\n" for line in lines)
