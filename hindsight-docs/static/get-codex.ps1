[CmdletBinding()]
param (
    [switch]$Uninstall,
    [ValidateSet("local", "cloud")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"

$GITHUB_RAW = "https://raw.githubusercontent.com/vectorize-io/hindsight/main/hindsight-integrations/codex"
$INSTALL_DIR = Join-Path $env:USERPROFILE ".hindsight\codex"
$SCRIPTS_DIR = Join-Path $INSTALL_DIR "scripts"
$CODEX_DIR = Join-Path $env:USERPROFILE ".codex"
$HOOKS_FILE = Join-Path $CODEX_DIR "hooks.json"
$CONFIG_FILE = Join-Path $CODEX_DIR "config.toml"

$SCRIPT_FILES = @(
    "scripts/session_start.py",
    "scripts/recall.py",
    "scripts/retain.py",
    "scripts/lib/__init__.py",
    "scripts/lib/bank.py",
    "scripts/lib/client.py",
    "scripts/lib/config.py",
    "scripts/lib/content.py",
    "scripts/lib/daemon.py",
    "scripts/lib/llm.py",
    "scripts/lib/state.py"
)

function Write-Info($Msg) {
    Write-Host "i " -ForegroundColor Blue -NoNewline
    Write-Host $Msg
}

function Write-Success($Msg) {
    Write-Host "√ " -ForegroundColor Green -NoNewline
    Write-Host $Msg
}

function Write-Warn($Msg) {
    Write-Host "! " -ForegroundColor Yellow -NoNewline
    Write-Host $Msg
}

function Write-ErrorMsg($Msg) {
    Write-Host "x " -ForegroundColor Red -NoNewline
    Write-Host $Msg
    exit 1
}

function Write-Step($Msg) {
    Write-Host ""
    Write-Host "▸ $Msg" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Banner {
    Write-Host ""
    Write-Host "  HINDSIGHT FOR CODEX CLI" -ForegroundColor Cyan
    Write-Host "  Give your Codex agent persistent memory" -ForegroundColor DarkGray
    Write-Host ""
}

function Download-File($Url, $Dest) {
    $dir = Split-Path $Dest -Parent
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Invoke-WebRequest -Uri $Url -OutFile $Dest
}

function Get-PythonCommand {
    foreach ($cmd in @("py", "python", "python3")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            return $cmd
        }
    }
    return $null
}

# ─────────────────────────────────────────────────────────────
# Uninstall
# ─────────────────────────────────────────────────────────────
if ($Uninstall) {
    Write-Host "Uninstalling Hindsight Codex integration..."

    if (Test-Path $SCRIPTS_DIR) {
        Remove-Item -Path $SCRIPTS_DIR -Recurse -Force
        Write-Host "  Removed $SCRIPTS_DIR"
    }

    if (Test-Path $HOOKS_FILE) {
        Remove-Item -Path $HOOKS_FILE -Force
        Write-Host "  Removed $HOOKS_FILE"
    }

    # if (Test-Path $CONFIG_FILE) {
    #     $content = Get-Content $CONFIG_FILE -Raw
    #     if ($null -eq $content) { $content = "" }

    #     $content = $content -replace "(?m)^codex_hooks\s*=\s*true\s*\r?\n?", ""

    #     Set-Content -Path $CONFIG_FILE -Value $content -Encoding UTF8
    #     Write-Host "  Removed codex_hooks from $CONFIG_FILE"
    # }

    Write-Host ""
    Write-Success "Uninstall complete."
    exit 0
}

# ─────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────
Write-Banner

# Step 1: Check prerequisites
Write-Step "Checking prerequisites"

$pythonCmd = Get-PythonCommand
if (-not $pythonCmd) {
    Write-ErrorMsg "Python 3 is required. Install from https://python.org"
}
Write-Success "Python available ($pythonCmd)"

# Step 2: Select mode
if ([string]::IsNullOrWhiteSpace($Mode)) {
    Write-Host "Select deployment mode:"
    Write-Host ""
    Write-Host "  1) Local - Run Hindsight on your machine (default)"
    Write-Host "  2) Cloud - Connect to Hindsight Cloud"
    Write-Host ""

    $choice = Read-Host "Enter choice [1]"
    if ($choice -eq "2") {
        $Mode = "cloud"
    }
    else {
        $Mode = "local"
    }
    Write-Host ""
}

Write-Info "Mode: $Mode"

# Step 3: Download scripts
Write-Step "Downloading hook scripts"

foreach ($file in $SCRIPT_FILES) {
    $url = "$GITHUB_RAW/$file"
    $dest = Join-Path $INSTALL_DIR $file
    Download-File $url $dest
}

Write-Success "Scripts installed to $SCRIPTS_DIR"

# Step 4: Download default settings
$SETTINGS_DST = Join-Path $INSTALL_DIR "settings.json"

if (!(Test-Path $SETTINGS_DST)) {
    Download-File "$GITHUB_RAW/settings.json" $SETTINGS_DST
    Write-Success "Default settings written to $SETTINGS_DST"
}
else {
    $SETTINGS_TMP = Join-Path $INSTALL_DIR "settings.json.new"
    Download-File "$GITHUB_RAW/settings.json" $SETTINGS_TMP

    try {
        $existing = Get-Content $SETTINGS_DST -Raw | ConvertFrom-Json -AsHashtable
        $upstream = Get-Content $SETTINGS_TMP -Raw | ConvertFrom-Json -AsHashtable

        foreach ($key in $upstream.Keys) {
            if (-not $existing.ContainsKey($key)) {
                $existing[$key] = $upstream[$key]
            }
        }

        if ($upstream.ContainsKey("version")) {
            $existing["version"] = $upstream["version"]
        }

        $existing | ConvertTo-Json -Depth 20 | Set-Content -Path $SETTINGS_DST -Encoding UTF8
        Remove-Item $SETTINGS_TMP -Force
        Write-Success "Settings updated, user customizations preserved"
    }
    catch {
        if (Test-Path $SETTINGS_TMP) {
            Remove-Item $SETTINGS_TMP -Force
        }
        Write-Info "Keeping existing settings at $SETTINGS_DST"
    }
}

# Step 5: Configure connection
if ($Mode -eq "cloud") {
    Write-Step "Configuring Hindsight Cloud connection"

    $DEFAULT_CLOUD_URL = "https://api.hindsight.vectorize.io"

    Write-Host "Enter your Hindsight Cloud connection details."
    Write-Host "Get these from https://ui.hindsight.vectorize.io"
    Write-Host ""

    $CLOUD_URL = Read-Host "Cloud API URL [$DEFAULT_CLOUD_URL]"
    if ([string]::IsNullOrWhiteSpace($CLOUD_URL)) {
        $CLOUD_URL = $DEFAULT_CLOUD_URL
    }

    $CLOUD_TOKEN = Read-Host "API Token"
    if ([string]::IsNullOrWhiteSpace($CLOUD_TOKEN)) {
        Write-ErrorMsg "API Token is required for cloud mode"
    }

    $USER_CONFIG = Join-Path $env:USERPROFILE ".hindsight\codex.json"
    $userConfigDir = Split-Path $USER_CONFIG -Parent
    if (!(Test-Path $userConfigDir)) {
        New-Item -ItemType Directory -Path $userConfigDir -Force | Out-Null
    }

    $configObj = @{
        hindsightApiUrl   = $CLOUD_URL
        hindsightApiToken = $CLOUD_TOKEN
    }

    $configObj | ConvertTo-Json -Depth 10 | Set-Content -Path $USER_CONFIG -Encoding UTF8
    Write-Success "Cloud config saved to $USER_CONFIG"
}
else {
    Write-Step "Checking LLM configuration"

    if ($env:OPENAI_API_KEY) {
        Write-Success "OpenAI API key detected"
    }
    elseif ($env:ANTHROPIC_API_KEY) {
        Write-Success "Anthropic API key detected"
    }
    elseif ($env:GEMINI_API_KEY) {
        Write-Success "Gemini API key detected"
    }
    elseif ($env:GROQ_API_KEY) {
        Write-Success "Groq API key detected"
    }
    else {
        Write-Warn "No LLM API key detected. Set one before using Codex:"
        Write-Host '    $env:OPENAI_API_KEY="sk-your-key"' -ForegroundColor Cyan
        Write-Host '    or $env:ANTHROPIC_API_KEY="your-key"' -ForegroundColor Cyan
    }
}

# Step 6: Write hooks.json
Write-Step "Configuring Codex hooks"

if (!(Test-Path $CODEX_DIR)) {
    New-Item -ItemType Directory -Path $CODEX_DIR -Force | Out-Null
}

$SessionScript = (Join-Path $SCRIPTS_DIR "session_start.py").Replace("\", "/")
$RecallScript = (Join-Path $SCRIPTS_DIR "recall.py").Replace("\", "/")
$RetainScript = (Join-Path $SCRIPTS_DIR "retain.py").Replace("\", "/")

$hooksObj = @{
    hooks = @{
        SessionStart     = @(
            @{
                hooks = @(
                    @{
                        type    = "command"
                        command = "$pythonCmd `"$SessionScript`""
                        timeout = 5
                    }
                )
            }
        )
        UserPromptSubmit = @(
            @{
                hooks = @(
                    @{
                        type    = "command"
                        command = "$pythonCmd `"$RecallScript`""
                        timeout = 12
                    }
                )
            }
        )
        Stop             = @(
            @{
                hooks = @(
                    @{
                        type    = "command"
                        command = "$pythonCmd `"$RetainScript`""
                        timeout = 30
                    }
                )
            }
        )
    }
}

$hooksObj | ConvertTo-Json -Depth 20 | Set-Content -Path $HOOKS_FILE -Encoding UTF8
Write-Success "Hooks written to $HOOKS_FILE"

# Step 7: Enable codex_hooks in config.toml
if (!(Test-Path $CONFIG_FILE)) {
    New-Item -ItemType File -Path $CONFIG_FILE -Force | Out-Null
}

$tomlContent = Get-Content $CONFIG_FILE -Raw
if ($null -eq $tomlContent) { $tomlContent = "" }

if ($tomlContent -match "(?m)^codex_hooks\s*=\s*true\s*$") {
    Write-Info "codex_hooks already enabled"
}
else {
    if ($tomlContent -match "(?m)^\[features\]\s*$") {
        $tomlContent = $tomlContent -replace "(?m)^\[features\]\s*$", "[features]`r`ncodex_hooks = true"
    }
    else {
        if ($tomlContent.Length -gt 0 -and -not $tomlContent.EndsWith("`r`n")) {
            $tomlContent += "`r`n"
        }
        $tomlContent += "[features]`r`ncodex_hooks = true`r`n"
    }

    Set-Content -Path $CONFIG_FILE -Value $tomlContent -Encoding UTF8
    Write-Success "Added codex_hooks = true to $CONFIG_FILE"
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  √ Installation Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "  Hindsight memory is now active in Codex CLI."
Write-Host ""
Write-Host "  Configuration:"
Write-Host "    $INSTALL_DIR\settings.json   (plugin defaults)" -ForegroundColor Cyan
Write-Host "    $env:USERPROFILE\.hindsight\codex.json   (personal overrides)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start a new Codex session to activate."
Write-Host ""
Write-Host "  Uninstall:"
Write-Host "    .\install-hindsight-codex.ps1 -Uninstall" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Documentation: https://hindsight.vectorize.io/sdks/integrations/codex"
Write-Host ""
