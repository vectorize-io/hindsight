## Long-Term Memory (Hindsight)

You have access to persistent memory via the `recall` and `retain` tools provided by the **hindsight** MCP server.

**At the start of every new task:**
Call `recall` (hindsight server) with a query summarizing the task. Include relevant context from the recalled memories in your response before proceeding.

**During a task:**
Call `retain` (hindsight server) to store any significant decisions, discoveries, or user preferences as they emerge — don't wait until the end.

**At the end of a task:**
Call `retain` (hindsight server) with a concise summary of what was accomplished, decisions made, and any patterns worth remembering for future tasks.

Memory persists across sessions. Use it to avoid repeating work and to build on past context.
