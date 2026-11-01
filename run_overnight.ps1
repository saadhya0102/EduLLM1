param(
    [ValidateSet("all", "dry-run", "data", "train", "frontier", "evaluate", "status")]
    [string]$Stage = "all",
    [switch]$WithOpus,
    [switch]$Force,
    [int]$MaxLen = 768
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cmdArgs = @(
    "-m", "src.run_8k",
    "--profile", "overnight",
    "--stage", $Stage,
    "--max-len", $MaxLen
)
if ($WithOpus) { $cmdArgs += "--with-opus" }
if ($Force) { $cmdArgs += "--force" }

python @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Overnight experiment failed with exit code $LASTEXITCODE"
}
