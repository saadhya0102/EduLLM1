param(
    [ValidateSet(
        "all", "dry-run", "data", "train", "frontier", "evaluate", "status",
        "pause-next", "rollback"
    )]
    [string]$Stage = "all",
    [switch]$WithOpus,
    [switch]$Force,
    [int]$MaxLen = 1024,
    [double]$AfterMinutes = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cmdArgs = @(
    "-m", "src.run_8k",
    "--profile", "staged2k",
    "--stage", $Stage,
    "--max-len", $MaxLen
)
if ($WithOpus) { $cmdArgs += "--with-opus" }
if ($Force) { $cmdArgs += "--force" }
if ($AfterMinutes -gt 0) {
    $cmdArgs += @("--after-minutes", [string]$AfterMinutes)
}

python @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Staged 2K extension failed with exit code $LASTEXITCODE"
}
