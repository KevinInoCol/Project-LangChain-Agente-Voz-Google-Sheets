# Instala FFmpeg en Windows usando el primer gestor disponible: winget, Chocolatey, Scoop o conda.
# Uso: powershell -ExecutionPolicy Bypass -File install_ffmpeg.ps1
$ErrorActionPreference = "Stop"

function Test-Command($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Update-SessionPath {
    # winget/choco actualizan el PATH del registro, no el de esta sesión: lo recargamos.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

if (Test-Command ffmpeg) {
    Write-Host "✅ FFmpeg ya está instalado: $((Get-Command ffmpeg).Source)"
    ffmpeg -version | Select-Object -First 1
    exit 0
}

if (Test-Command winget) {
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
} elseif (Test-Command choco) {
    choco install ffmpeg -y
} elseif (Test-Command scoop) {
    scoop install ffmpeg
} elseif (Test-Command conda) {
    Write-Host "⚠️  Sin winget/choco/scoop; instalando con conda en el entorno activo."
    conda install -y -c conda-forge ffmpeg
} else {
    Write-Host "❌ No se encontró winget, choco, scoop ni conda."
    Write-Host "   Descarga el build 'release essentials' de https://www.gyan.dev/ffmpeg/builds/,"
    Write-Host "   descomprímelo en C:\ffmpeg y añade C:\ffmpeg\bin al PATH del usuario."
    exit 1
}

Update-SessionPath
if (Test-Command ffmpeg) {
    Write-Host "✅ FFmpeg instalado: $((Get-Command ffmpeg).Source)"
    ffmpeg -version | Select-Object -First 1
    Write-Host "ℹ️  Cierra y abre otras terminales (y VS Code) para que vean el PATH nuevo."
} else {
    Write-Host "⚠️  La instalación terminó pero 'ffmpeg' aún no está en el PATH de esta sesión."
    Write-Host "   Abre una terminal nueva y prueba: ffmpeg -version"
    exit 1
}
