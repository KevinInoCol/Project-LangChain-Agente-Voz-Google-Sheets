#!/usr/bin/env bash
# Instala FFmpeg en macOS o Linux usando el primer gestor de paquetes disponible.
# Uso: bash install_ffmpeg.sh
set -euo pipefail

if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ FFmpeg ya está instalado: $(command -v ffmpeg)"
    ffmpeg -version | head -1
    exit 0
fi

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

OS="$(uname -s)"
echo "🔎 Sistema detectado: $OS"

if [ "$OS" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    elif command -v conda >/dev/null 2>&1; then
        echo "⚠️  Homebrew no encontrado; instalando con conda en el entorno activo."
        conda install -y -c conda-forge ffmpeg
    else
        echo "❌ Ni Homebrew ni conda. Instala Homebrew (https://brew.sh) y vuelve a ejecutar."
        exit 1
    fi
elif [ "$OS" = "Linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update && $SUDO apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y ffmpeg-free || $SUDO dnf install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        $SUDO pacman -S --noconfirm ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
        $SUDO zypper install -y ffmpeg
    elif command -v conda >/dev/null 2>&1; then
        conda install -y -c conda-forge ffmpeg
    else
        echo "❌ No se encontró apt, dnf, pacman, zypper ni conda."
        exit 1
    fi
else
    echo "❌ Sistema no soportado por este script: $OS (en Windows usa install_ffmpeg.ps1)."
    exit 1
fi

hash -r
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ FFmpeg instalado: $(command -v ffmpeg)"
    ffmpeg -version | head -1
else
    echo "⚠️  La instalación terminó pero 'ffmpeg' no está en el PATH. Abre una terminal nueva y prueba 'ffmpeg -version'."
    exit 1
fi
