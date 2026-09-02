$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source

& $python (Join-Path $project "run_cas_public_reproduction.py") --all @args
exit $LASTEXITCODE
