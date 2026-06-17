// One of the two header panels (API / Control plane): health dot, status pill,
// URL, optional detail, and state-aware Start/Restart/Stop.
export function ServicePanel({ title, running, statusText, healthOk, url, detail, busy, onStart, onRestart, onStop }) {
  return (
    <div class="svc">
      <div class="svc-top">
        <span class="svc-name">
          <span class={"hdot " + (healthOk ? "ok" : "bad")} />
          {title}
        </span>
        <span class={"status " + (running ? "on" : "off")}>{statusText}</span>
      </div>
      <a class="svc-url" href={url || "#"} target="_blank" rel="noopener">
        {url}
      </a>
      <div class="svc-detail">{detail}</div>
      <div class="svc-actions">
        <button class="ghost sm" disabled={busy || running} onClick={onStart}>
          Start
        </button>
        <button class="ghost sm" disabled={busy || !running} onClick={onRestart}>
          Restart
        </button>
        <button class="ghost sm" disabled={busy || !running} onClick={onStop}>
          Stop
        </button>
      </div>
    </div>
  );
}
