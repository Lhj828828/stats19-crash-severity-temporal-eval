$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Entrypoint = Join-Path $ProjectRoot "code\d15_reproducibility.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Entrypoint)) {
    throw "D15 entrypoint was not found: $Entrypoint"
}

# Match the frozen D10-D11 execution cap and avoid uncontrolled CPU fan-out.
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"

& $Python $Entrypoint --run-tests
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
