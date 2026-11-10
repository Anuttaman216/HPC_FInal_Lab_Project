# demo.ps1 - Windows launcher for the live demo (runs inside the Docker image).
#
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1          # guided demo
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Shell   # Linux shell for your own commands
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -NoPause # demo without waiting for ENTER
param([switch]$Shell, [switch]$NoPause)

$root = Split-Path -Parent $PSScriptRoot
$docker = "docker"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
}

& $docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop is not running. Starting it - wait for the whale icon to stop animating, then run this again." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    exit 1
}

& $docker image inspect hpcg-analyzer *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Building the hpcg-analyzer image (first time only, needs internet, ~3 min)..." -ForegroundColor Yellow
    & $docker build -t hpcg-analyzer -f "$root\docker\Dockerfile" "$root\docker"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$mount = "${root}:/work"
if ($Shell) {
    Write-Host "Linux shell inside the container. Project is in /work. Type 'exit' to leave." -ForegroundColor Cyan
    & $docker run --rm -it -v $mount -w /work hpcg-analyzer bash
} elseif ($NoPause) {
    & $docker run --rm -e DEMO_NO_PAUSE=1 -v $mount -w /work hpcg-analyzer bash scripts/demo.sh
} else {
    & $docker run --rm -it -v $mount -w /work hpcg-analyzer bash scripts/demo.sh
}
