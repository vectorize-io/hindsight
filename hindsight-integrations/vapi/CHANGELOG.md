# Changelog

## 0.1.0 (2026-04-24)

- Initial release: `HindsightVapiWebhook` for adding persistent memory to Vapi voice calls
- `assistant-request` handler recalls memories by caller phone number and returns `assistantOverrides` with a system message
- `end-of-call-report` handler retains the full transcript to Hindsight (fire-and-forget)
- `build_assistant_overrides()` helper for outbound calls where there is no `assistant-request` webhook
- Configurable recall budget (`low`, `mid`, `high`) and token limit
- Global `configure()` helper for shared connection settings
- Framework-agnostic: wires into any HTTP server (FastAPI, Flask, aiohttp, etc.) in two lines
