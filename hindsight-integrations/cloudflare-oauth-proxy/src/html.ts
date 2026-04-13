/**
 * HTML escaping and the login page template.
 */

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function loginPage(stateKey: string, error?: string): string {
  const errorBlock = error ? `<p class="error">${escapeHtml(error)}</p>` : "";
  return `<!DOCTYPE html>
<html>
<head>
  <title>Hindsight MCP</title>
  <style>
    body{font-family:system-ui;background:#0a0a0a;color:#e0e0e0;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}
    .card{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:2rem;width:320px}
    h2{margin-top:0}
    input{width:100%;padding:10px;margin:8px 0;border:1px solid #444;border-radius:6px;background:#0a0a0a;color:#e0e0e0;box-sizing:border-box;font-size:16px}
    button{width:100%;padding:10px;margin-top:12px;border:none;border-radius:6px;background:#3b82f6;color:white;font-size:16px;cursor:pointer}
    button:hover{background:#2563eb}
    .error{color:#ef4444;font-size:14px}
    .info{color:#888;font-size:13px;margin-top:12px}
  </style>
</head>
<body>
  <div class="card">
    <h2>Hindsight MCP</h2>
    <p>Authorize Claude to access your memory.</p>
    ${errorBlock}
    <form method="POST" action="/authorize">
      <input type="hidden" name="stateKey" value="${escapeHtml(stateKey)}" />
      <input type="password" name="password" placeholder="Password" autofocus required />
      <button type="submit">Authorize</button>
    </form>
    <p class="info">You only need to do this once per session.</p>
  </div>
</body>
</html>`;
}
