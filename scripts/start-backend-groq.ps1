# Starts the web backend on Groq instead of the offline engine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start-backend-groq.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start-backend-groq.ps1 -Model qwen/qwen3.8-27b
#
# The key is read from GROQ_API_KEY in the repository-root .env and handed to
# the backend through this process's environment only; it is never printed and
# the .env file is not modified. The backend itself deliberately does not load
# .env (see server/app/config.py), which is why a plain uvicorn start is offline.

param(
    [int]$Port = 8000,
    [string]$Model = "openai/gpt-oss-120b"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"
$python = Join-Path $root "server\.venv\Scripts\python.exe"

if (-not (Test-Path $envFile)) { throw "No .env at $root - add GROQ_API_KEY=gsk_... there first." }
if (-not (Test-Path $python)) { throw "No server\.venv - create it first (see README, Quick start)." }

$match = Select-String -Path $envFile -Pattern '^\s*GROQ_API_KEY\s*=\s*["'']?([^"''\s#]+)' | Select-Object -First 1
if (-not $match) { throw "GROQ_API_KEY is not set in $envFile." }

# Gemini would take precedence over LLM_API_KEY, so make sure it is not in play.
Remove-Item Env:GEMINI_API_KEY, Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue
$env:LLM_API_KEY = $match.Matches[0].Groups[1].Value
$env:LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:LLM_MODEL = $Model

# Backup model: with NVIDIA_API_KEY in .env, NVIDIA's free gpt-oss-20b answers
# whenever Groq's free rate limit is reached. Without it, Groq alone.
$nvidia = Select-String -Path $envFile -Pattern '^\s*NVIDIA_API_KEY\s*=\s*["'']?([^"''\s#]+)' | Select-Object -First 1
$backup = "none"
if ($nvidia) {
    $env:LLM_FALLBACK_API_KEY = $nvidia.Matches[0].Groups[1].Value
    $env:LLM_FALLBACK_BASE_URL = "https://integrate.api.nvidia.com/v1"
    $env:LLM_FALLBACK_MODEL = "openai/gpt-oss-20b"
    $backup = "NVIDIA openai/gpt-oss-20b"
}

Write-Host "Backend on Groq ($Model, backup: $backup), port $Port. Check: http://localhost:$Port/api/health -> provider openai-compatible"
Set-Location (Join-Path $root "server")
& $python -m uvicorn app.main:app --reload --port $Port
