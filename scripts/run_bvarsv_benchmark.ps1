param(
    [int]$NRep = 10000,
    [int]$NBurn = 2000,
    [int]$ThinFac = 10,
    [int]$Seed = 5813,
    [int]$ItPrint = 1000,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LocalRscript = Join-Path $RepoRoot "..\tools\R-4.6.0\bin\Rscript.exe"
$LocalLib = Join-Path $RepoRoot "..\R-library\4.6"

if (Test-Path -LiteralPath $LocalRscript) {
    $Rscript = (Resolve-Path -LiteralPath $LocalRscript).Path
    $env:R_LIBS_USER = (New-Item -ItemType Directory -Force -Path $LocalLib).FullName
} else {
    $RscriptCmd = Get-Command Rscript.exe -ErrorAction Stop
    $Rscript = $RscriptCmd.Source
}

$Args = @(
    "scripts/run_bvarsv_benchmark.R",
    "--nrep=$NRep",
    "--nburn=$NBurn",
    "--thinfac=$ThinFac",
    "--seed=$Seed",
    "--itprint=$ItPrint"
)

if ($Install) {
    $Args += "--install"
}

Push-Location $RepoRoot
try {
    & $Rscript @Args
} finally {
    Pop-Location
}

