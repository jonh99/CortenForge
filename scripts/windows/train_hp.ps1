# High-performance training helper for Windows/NVIDIA.
# Sets wgpu env vars to prefer the discrete GPU, then calls the cargo alias from repo root.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $repoRoot
Write-Host "Working dir:" (Get-Location)

$env:WGPU_POWER_PREF = "high-performance"
$env:WGPU_BACKEND = "dx12"
# Uncomment if you have multiple adapters and want to force NVIDIA:
# $env:WGPU_ADAPTER_NAME = "NVIDIA"

# Ensure logs directory exists for status writes.
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$inputRoot = "assets/datasets/captures_filtered"
if (-not (Test-Path $inputRoot)) {
    Write-Host "Input root not found at '$inputRoot'. Override via '-- --input-root <path>' if needed."
}

cargo train_hp -- @Args

Pop-Location
