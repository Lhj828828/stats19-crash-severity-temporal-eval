$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project Python environment not found: $python"
}

& $python (Join-Path $project "run_d16_reproduction.py") --all @args
exit $LASTEXITCODE
