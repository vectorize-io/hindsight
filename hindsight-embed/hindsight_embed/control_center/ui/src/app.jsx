import { Fragment } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { api, forgetToken, pn, profileParam, syncProfileUrl, tabParam } from "./api.js";
import { ServicePanel } from "./service-panel.jsx";

const TABS = [
  { id: "quick", label: "Quick config" },
  { id: "config", label: "Configuration" },
  { id: "files", label: "Files" },
  { id: "logs", label: "Logs" },
];

const escHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Highlight active KEY=value lines; dim comments/blanks.
function envHighlight(text) {
  return text
    .split("\n")
    .map((line) => {
      const t = line.trim();
      if (t === "") return "<span></span>";
      if (t.startsWith("#")) return `<span class="l-off">${escHtml(line)}</span>`;
      const eq = line.indexOf("=");
      if (eq > 0)
        return (
          `<span class="l-on"><span class="k">${escHtml(line.slice(0, eq))}</span>` +
          `<span class="eq">=</span>${escHtml(line.slice(eq + 1))}</span>`
        );
      return `<span class="l-on">${escHtml(line)}</span>`;
    })
    .join("\n");
}

export function App() {
  const [version, setVersion] = useState("");
  const [noAuth, setNoAuth] = useState(false);
  const [providers, setProviders] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [current, setCurrent] = useState(null); // "" = default, null = none
  const [tab, setTab] = useState("quick");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState({ text: "", kind: "" });

  // Quick-config form
  const [cfg, setCfg] = useState(null);
  const [form, setForm] = useState({ provider: "", apiKey: "", model: "", apiPort: "", uiPort: "", apiVersion: "", cpVersion: "" });

  // status / health
  const [daemonRunning, setDaemonRunning] = useState(false);
  const [uiRunning, setUiRunning] = useState(false);
  const [daemonText, setDaemonText] = useState("—");
  const [uiText, setUiText] = useState("—");
  const [health, setHealth] = useState({ api_ok: false, api_detail: "", ui_ok: false });
  const [paths, setPaths] = useState(null);

  // env editor + logs
  const [envText, setEnvText] = useState("");
  const [envPath, setEnvPath] = useState("");
  const [envMsg, setEnvMsg] = useState({ text: "", kind: "" });
  const [envEffective, setEnvEffective] = useState(true); // show only active KEY=value lines
  const [logText, setLogText] = useState("");
  const [logPath, setLogPath] = useState("");
  const [logLines, setLogLines] = useState(200);
  const [logAuto, setLogAuto] = useState(true);
  const [logSource, setLogSource] = useState("daemon"); // "daemon" (API) or "ui" (control plane)

  const busyRef = useRef(false);
  busyRef.current = busy;
  const envPre = useRef(null);
  const envArea = useRef(null);

  const onUnauthorized = (e) => {
    if (e && e.unauthorized) {
      forgetToken();
      setNoAuth(true);
      return true;
    }
    return false;
  };

  // ----- data loads ----------------------------------------------------
  async function loadProfiles() {
    const { profiles } = await api("GET", "/api/profiles");
    setProfiles(profiles);
    return profiles;
  }

  async function selectProfile(name) {
    setCurrent(name);
    syncProfileUrl(name);
    setMsg({ text: "", kind: "" });
    try {
      const [cfgv, pathsv, env] = await Promise.all([
        api("GET", `/api/profiles/${pn(name)}/config`),
        api("GET", `/api/profiles/${pn(name)}/paths`),
        api("GET", `/api/profiles/${pn(name)}/env`),
      ]);
      setCfg(cfgv);
      setForm({
        provider: cfgv.provider || (providers[0] && providers[0].id) || "openai",
        apiKey: "",
        model: cfgv.model || "",
        apiPort: String(cfgv.api_port),
        uiPort: cfgv.ui_port_is_default ? "" : String(cfgv.ui_port),
        apiVersion: cfgv.api_version || "",
        cpVersion: cfgv.cp_version || "",
      });
      setPaths(pathsv);
      setEnvText(env.content);
      setEnvPath(env.path);
      setEnvMsg({ text: "", kind: "" });
    } catch (e) {
      onUnauthorized(e);
    }
  }

  async function loadLogs(name = current) {
    if (name === null) return;
    try {
      const r = await api("GET", `/api/profiles/${pn(name)}/logs?lines=${logLines}&source=${logSource}`);
      setLogPath(r.path);
      setLogText(r.exists ? r.content || "(empty)" : "(no log file yet)");
    } catch (e) {
      if (!onUnauthorized(e)) setLogText(e.message);
    }
  }

  async function refreshHealth(name = current) {
    if (name === null || busyRef.current) return;
    try {
      const h = await api("GET", `/api/profiles/${pn(name)}/health`);
      setHealth(h);
      setDaemonRunning(h.api_ok);
      setUiRunning(h.ui_ok);
      setDaemonText(h.api_ok ? "Running" : "Stopped");
      setUiText(h.ui_ok ? "Running" : "Stopped");
    } catch (e) {
      onUnauthorized(e);
    }
  }

  // ----- actions -------------------------------------------------------
  function keyPayload() {
    if (form.apiKey) return form.apiKey;
    return cfg && cfg.has_api_key ? "__unchanged__" : "";
  }

  async function save() {
    setMsg({ text: "Saving…", kind: "" });
    try {
      await api("POST", `/api/profiles/${pn(current)}/config`, {
        provider: form.provider,
        api_key: keyPayload(),
        model: form.model.trim(),
        api_port: form.apiPort.trim(),
        ui_port: form.uiPort.trim(),
        api_version: form.apiVersion.trim(),
        cp_version: form.cpVersion.trim(),
      });
      setMsg({ text: "Saved. Restart the daemon to apply.", kind: "ok" });
      await selectProfile(current);
      loadProfiles();
    } catch (e) {
      if (!onUnauthorized(e)) setMsg({ text: e.message, kind: "err" });
    }
  }

  async function saveEnv() {
    setEnvMsg({ text: "Saving…", kind: "" });
    try {
      await api("POST", `/api/profiles/${pn(current)}/env`, { content: envText });
      setEnvMsg({ text: "Saved. Restart the daemon to apply.", kind: "ok" });
      loadProfiles();
    } catch (e) {
      if (!onUnauthorized(e)) setEnvMsg({ text: e.message, kind: "err" });
    }
  }

  async function daemonAction(action) {
    setBusy(true);
    setLogSource("daemon");
    setTab("logs"); // jump to the daemon log so the user can watch it
    setDaemonText({ start: "Starting…", stop: "Stopping…", restart: "Restarting…" }[action] || "Working…");
    try {
      const r = await api("POST", `/api/profiles/${pn(current)}/daemon/${action}`);
      setDaemonRunning(r.running);
      setDaemonText(r.running ? "Running" : "Stopped");
    } catch (e) {
      if (!onUnauthorized(e)) setMsg({ text: e.message, kind: "err" });
    } finally {
      setBusy(false);
      loadProfiles();
      loadLogs();
    }
  }

  async function cpAction(action) {
    setBusy(true);
    setLogSource("ui");
    setTab("logs"); // jump to the control-plane log so the user can see why it starts/fails
    setUiText({ start: "Starting…", stop: "Stopping…", restart: "Restarting…" }[action] || "Working…");
    try {
      const u = await api("POST", `/api/profiles/${pn(current)}/ui/${action}`);
      setUiRunning(u.running);
      setUiText(u.running ? "Running" : "Stopped");
    } catch (e) {
      if (!onUnauthorized(e)) setMsg({ text: e.message, kind: "err" });
    } finally {
      setBusy(false);
      loadProfiles();
      refreshHealth();
    }
  }

  async function deleteProfile() {
    if (!current) return; // default profile isn't deletable
    if (!confirm(`Delete profile "${current}"? This stops its daemon and removes its config and logs. This cannot be undone.`))
      return;
    setBusy(true);
    try {
      const r = await api("POST", `/api/profiles/${pn(current)}/delete`);
      if (!r.ok) {
        setMsg({ text: r.message, kind: "err" });
        return;
      }
      setCurrent(null);
      history.replaceState(null, "", location.pathname);
      loadProfiles();
    } catch (e) {
      if (!onUnauthorized(e)) setMsg({ text: e.message, kind: "err" });
    } finally {
      setBusy(false);
    }
  }

  // ----- effects -------------------------------------------------------
  // Boot: health (version), providers, profiles, optional deep-link.
  useEffect(() => {
    (async () => {
      try {
        const h = await fetch("/api/health").then((r) => r.json());
        setVersion("v" + h.version);
        const { providers } = await api("GET", "/api/providers");
        setProviders(providers);
        const list = await loadProfiles();
        if (profileParam !== null) {
          const internal = profileParam === "default" ? "" : profileParam;
          if (list.some((p) => p.name === internal)) {
            await selectProfile(internal);
            if (tabParam && TABS.some((t) => t.id === tabParam)) setTab(tabParam);
          }
        }
      } catch (e) {
        onUnauthorized(e);
      }
    })();
  }, []);

  // Health poll for the selected profile (every 4s).
  useEffect(() => {
    if (current === null) return;
    refreshHealth(current);
    const id = setInterval(() => refreshHealth(current), 4000);
    return () => clearInterval(id);
  }, [current]);

  // Log tail poll while the Logs tab is open.
  useEffect(() => {
    if (current === null || tab !== "logs") return;
    loadLogs(current);
    if (!logAuto) return;
    const id = setInterval(() => loadLogs(current), 2000);
    return () => clearInterval(id);
  }, [current, tab, logAuto, logLines, logSource]);

  // Keep the highlight layer scroll-synced with the textarea.
  const syncEnvScroll = () => {
    if (envPre.current && envArea.current) {
      envPre.current.scrollTop = envArea.current.scrollTop;
      envPre.current.scrollLeft = envArea.current.scrollLeft;
    }
  };

  // ----- render --------------------------------------------------------
  const defaultVer = version.replace(/^v/, "") || "embed default";
  // "Effective only" shows just active KEY=value lines (read-only); the full
  // envText is preserved for editing/saving when the toggle is off.
  const envDisplay = envEffective
    ? envText.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).join("\n")
    : envText;

  if (noAuth) {
    return (
      <Fragment>
        <AppBar version={version} />
        <div class="banner">
          No access token yet. Open the control center once from the CLI — after that this URL is remembered and you can
          bookmark it: <code>hindsight-embed control start</code>
        </div>
      </Fragment>
    );
  }

  return (
    <Fragment>
      <AppBar version={version} />
      <div class="shell">
        <aside class="sidebar">
          <div class="sect">Profiles</div>
          <ul class="profiles">
            {profiles.map((p) => (
              <li
                key={p.name}
                class={p.name === current ? "sel" : ""}
                onClick={() => selectProfile(p.name)}
                ref={(el) => p.name === current && el && el.scrollIntoView({ block: "nearest" })}
              >
                <span class={"dot " + (p.daemon_running ? "on" : "off")} />
                <span class="nm">{p.display_name}</span>
                {p.is_active && <span class="badge">active</span>}
              </li>
            ))}
          </ul>
        </aside>

        <main class="content">
          {current === null ? (
            <div class="empty">Select a profile from the left to view and edit it.</div>
          ) : (
            <Fragment>
              <div class="chead">
                <div class="chead-row">
                  <span class="ptitle grad-text">{cfg ? cfg.display_name : ""}</span>
                  <span style="margin-left:auto" />
                  {current !== "" && (
                    <button class="ghost danger" disabled={busy} onClick={deleteProfile}>
                      Delete profile
                    </button>
                  )}
                </div>
                <div class="services">
                  <ServicePanel
                    title="API"
                    running={daemonRunning}
                    statusText={daemonText}
                    healthOk={health.api_ok}
                    url={paths && paths.daemon_url}
                    detail={health.api_ok ? health.api_detail : "unreachable"}
                    busy={busy}
                    onStart={() => daemonAction("start")}
                    onRestart={() => daemonAction("restart")}
                    onStop={() => daemonAction("stop")}
                  />
                  <ServicePanel
                    title="Control plane"
                    running={uiRunning}
                    statusText={uiText}
                    healthOk={health.ui_ok}
                    url={paths && paths.ui_url}
                    detail=""
                    busy={busy}
                    onStart={() => cpAction("start")}
                    onRestart={() => cpAction("restart")}
                    onStop={() => cpAction("stop")}
                  />
                </div>
              </div>

              <div class="tabs">
                {TABS.map((t) => (
                  <div key={t.id} class={"tab " + (tab === t.id ? "active " : "") + (tab === t.id ? "grad-text" : "")} onClick={() => setTab(t.id)}>
                    {t.label}
                  </div>
                ))}
              </div>

              {tab === "quick" && cfg && (
                <div class="panel narrow">
                  <label>
                    Provider
                    <select value={form.provider} onChange={(e) => setForm({ ...form, provider: e.target.value })}>
                      {providers.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    API key
                    <input
                      type="password"
                      autocomplete="off"
                      value={form.apiKey}
                      onInput={(e) => setForm({ ...form, apiKey: e.target.value })}
                    />
                    <span class="hint">
                      {cfg.has_api_key ? `A key is stored (${cfg.api_key_masked}). Leave blank to keep it.` : "No key stored yet."}
                    </span>
                  </label>
                  <label>
                    Model <span class="hint">(blank = provider default)</span>
                    <input type="text" autocomplete="off" placeholder="provider default" value={form.model} onInput={(e) => setForm({ ...form, model: e.target.value })} />
                  </label>
                  <div class="ports">
                    <label>
                      API port
                      <input type="number" value={form.apiPort} onInput={(e) => setForm({ ...form, apiPort: e.target.value })} />
                    </label>
                    <label>
                      UI port <span class="hint">(blank = API + 10000)</span>
                      <input type="number" placeholder={String(cfg.ui_port)} value={form.uiPort} onInput={(e) => setForm({ ...form, uiPort: e.target.value })} />
                    </label>
                  </div>
                  <div class="ports">
                    <label>
                      API version <span class="hint">(blank = {defaultVer})</span>
                      <input type="text" autocomplete="off" placeholder={defaultVer} value={form.apiVersion} onInput={(e) => setForm({ ...form, apiVersion: e.target.value })} />
                    </label>
                    <label>
                      Control plane version <span class="hint">(blank = {defaultVer})</span>
                      <input type="text" autocomplete="off" placeholder={defaultVer} value={form.cpVersion} onInput={(e) => setForm({ ...form, cpVersion: e.target.value })} />
                    </label>
                  </div>
                  <div class="row">
                    <button onClick={save} disabled={busy}>
                      Save
                    </button>
                  </div>
                  <div class={"msg " + msg.kind}>{msg.text}</div>
                </div>
              )}

              {tab === "config" && (
                <div class="panel">
                  <div class="note">
                    The profile's <code>{envPath}</code> — it contains your API key in plaintext. Active{" "}
                    <code>KEY=value</code> lines are highlighted; comments are dimmed. Save, then restart the daemon to apply.
                  </div>
                  <label class="row" style="margin-bottom:12px; color:var(--color-muted); font-size:12px">
                    <input type="checkbox" style="width:auto" checked={envEffective} onChange={(e) => setEnvEffective(e.target.checked)} /> Effective only (hide comments &amp; blank lines)
                  </label>
                  <div class="env-editor">
                    <pre class="env-hl" ref={envPre} aria-hidden="true" dangerouslySetInnerHTML={{ __html: envHighlight(envDisplay) }} />
                    <textarea
                      ref={envArea}
                      spellcheck={false}
                      readOnly={envEffective}
                      value={envDisplay}
                      onInput={(e) => setEnvText(e.target.value)}
                      onScroll={syncEnvScroll}
                    />
                  </div>
                  <div class="row" style="margin-top:12px">
                    <button class="ghost" onClick={() => selectProfile(current)}>
                      Reload
                    </button>
                    {!envEffective && (
                      <button onClick={saveEnv} disabled={busy}>
                        Save .env
                      </button>
                    )}
                  </div>
                  {envEffective && <div class="hint" style="margin-top:8px">Read-only view — uncheck “Effective only” to edit the raw file.</div>}
                  <div class={"msg " + envMsg.kind}>{envMsg.text}</div>
                </div>
              )}

              {tab === "files" && paths && (
                <div class="panel">
                  <div class="note">
                    On-disk locations and URLs for this profile — its config file, daemon log, lock file, local pg0 database
                    directory, and the daemon / control-plane addresses.
                  </div>
                  <dl class="files">
                    {[
                      ["Port", String(paths.port)],
                      ["Config (.env)", paths.config_path],
                      ["Daemon log", paths.log_path],
                      ["Lock file", paths.lock_path],
                      ["Database URL", paths.database_url],
                      ["Database dir", paths.database_path || "—"],
                      ["Daemon URL", paths.daemon_url],
                      ["Control plane", paths.ui_url],
                    ].map(([k, v]) => (
                      <Fragment key={k}>
                        <dt>{k}</dt>
                        <dd>{v}</dd>
                      </Fragment>
                    ))}
                  </dl>
                </div>
              )}

              {tab === "logs" && (
                <div class="panel">
                  <div class="row" style="margin-bottom:12px">
                    <select style="width:auto" value={logSource} onChange={(e) => setLogSource(e.target.value)}>
                      <option value="daemon">API (daemon)</option>
                      <option value="ui">Control plane</option>
                    </select>
                    <label class="row" style="margin:0; color:var(--color-muted); font-size:12px">
                      <input type="checkbox" style="width:auto" checked={logAuto} onChange={(e) => setLogAuto(e.target.checked)} /> auto-refresh
                    </label>
                    <select style="width:auto" value={logLines} onChange={(e) => setLogLines(Number(e.target.value))}>
                      <option value={100}>100 lines</option>
                      <option value={200}>200 lines</option>
                      <option value={500}>500 lines</option>
                    </select>
                    <button class="ghost" onClick={() => loadLogs()}>
                      Refresh
                    </button>
                    <span class="hint">{logPath}</span>
                  </div>
                  <pre class="logs">{logText || "—"}</pre>
                </div>
              )}
            </Fragment>
          )}
        </main>
      </div>
    </Fragment>
  );
}

function AppBar({ version }) {
  return (
    <div class="appbar">
      <img src="./logo.png" alt="Hindsight" />
      <span class="title grad-text">Embed Control Center</span>
      <span class="ver">{version}</span>
    </div>
  );
}
