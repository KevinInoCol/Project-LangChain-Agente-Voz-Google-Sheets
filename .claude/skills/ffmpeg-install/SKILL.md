---
name: ffmpeg-install
description: Instala y verifica FFmpeg en Windows, macOS o Linux — requisito de sistema de openai-whisper, que lo llama por subprocess para leer el audio. Detecta el sistema operativo y el gestor de paquetes disponible (winget/choco/scoop, Homebrew, apt/dnf/pacman/zypper o conda), instala, comprueba que `ffmpeg` quedó en el PATH y prueba una transcripción mínima. Use when the user asks to "instalar ffmpeg", "instalar fmpeg/ffmpg", "configurar el entorno del agente de voz", or when Whisper fails with FileNotFoundError [WinError 2] / "No such file or directory: 'ffmpeg'", "ffmpeg not found", or `whisper.transcribe()` errors on load_audio.
---

# Instalar FFmpeg (Windows / macOS / Linux)

`openai-whisper` **no trae FFmpeg**: `whisper.transcribe("test.wav")` ejecuta el binario `ffmpeg` del sistema para decodificar el audio. `pip install ffmpeg` **no sirve** (instala un wrapper de Python vacío, no el programa). FFmpeg se instala con el gestor de paquetes del sistema operativo y tiene que quedar en el `PATH`.

## Síntoma típico

```
FileNotFoundError: [WinError 2] El sistema no puede encontrar el archivo especificado     # Windows
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'                          # macOS / Linux
```
Sale en `whisper/audio.py → load_audio` al primer `transcribe()`.

## Workflow

### 1. ¿Ya está instalado?

```bash
ffmpeg -version
```
Si imprime `ffmpeg version ...`, no hay nada que instalar — salta al paso 4.

### 2. Instala con el script del sistema operativo

Los scripts detectan el gestor de paquetes disponible y usan el primero que encuentren.

**macOS / Linux**
```bash
bash .claude/skills/ffmpeg-install/scripts/install_ffmpeg.sh
```

**Windows** (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File .claude\skills\ffmpeg-install\scripts\install_ffmpeg.ps1
```

Los comandos piden `sudo` / UAC según el gestor: avisa al usuario antes de ejecutarlos. Si Claude no puede introducir la contraseña, sugiere al usuario correrlo él mismo con `! <comando>`.

### 3. Comandos manuales (si el script no aplica)

| Sistema | Gestor | Comando |
|---|---|---|
| **Windows** | winget (Windows 10/11, viene instalado) | `winget install --id Gyan.FFmpeg -e` |
| | Chocolatey | `choco install ffmpeg -y` (PowerShell como administrador) |
| | Scoop | `scoop install ffmpeg` |
| **macOS** | Homebrew | `brew install ffmpeg` |
| **Ubuntu / Debian** | apt | `sudo apt update && sudo apt install -y ffmpeg` |
| **Fedora** | dnf | `sudo dnf install -y ffmpeg-free` (o `ffmpeg` desde RPM Fusion) |
| **Arch** | pacman | `sudo pacman -S --noconfirm ffmpeg` |
| **openSUSE** | zypper | `sudo zypper install -y ffmpeg` |
| **Cualquiera** | conda | `conda install -y -c conda-forge ffmpeg` (solo dentro de ese entorno) |

**Sin Homebrew en macOS:** instálalo primero desde https://brew.sh o usa la vía conda.
**Sin gestor en Windows:** descarga el build "release essentials" de https://www.gyan.dev/ffmpeg/builds/, descomprime en `C:\ffmpeg` y añade `C:\ffmpeg\bin` al PATH del usuario.

### 4. Verifica

```bash
ffmpeg -version
python -c "import shutil; print(shutil.which('ffmpeg'))"
```
El segundo comando **debe** imprimir una ruta, ejecutado con el mismo Python/entorno que corre el agente. Si imprime `None`, ese Python no ve el `PATH` donde quedó FFmpeg.

Prueba de punta a punta con Whisper (genera 1 s de silencio y lo transcribe):
```bash
ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 /tmp/ffmpeg_check.wav \
  && python -c "import whisper; m=whisper.load_model('tiny'); print('Whisper OK:', repr(m.transcribe('/tmp/ffmpeg_check.wav', fp16=False)['text']))"
```
En Windows, cambia `/tmp/ffmpeg_check.wav` por `$env:TEMP\ffmpeg_check.wav` y el `\` final por un backtick o pon todo en una línea.

## Problemas frecuentes

| Problema | Causa | Solución |
|---|---|---|
| Windows: instaló bien pero `ffmpeg` "no se reconoce" | El PATH nuevo solo lo ven terminales nuevas | Cerrar y abrir la terminal **y VS Code** (su terminal integrada hereda el PATH viejo) |
| Funciona en la terminal pero no en el script | El agente corre en otro entorno (conda/venv) o desde el IDE | Comprobar `shutil.which('ffmpeg')` con ese Python; o `conda install -c conda-forge ffmpeg` en ese entorno |
| Se instaló `pip install ffmpeg` / `ffmpeg-python` | Son wrappers de Python, no el binario | `pip uninstall ffmpeg` y seguir esta skill |
| Fedora: `ffmpeg` no existe | No está en los repos base | `ffmpeg-free` o habilitar RPM Fusion |
| macOS: `brew: command not found` | Falta Homebrew | Instalar Homebrew o usar conda |
| `winget` no existe | Windows antiguo o App Installer desactualizado | Actualizar "App Installer" en Microsoft Store, o usar choco/scoop |

## Reglas

- **Nunca** "arregles" el error con `pip install ffmpeg`.
- Verifica siempre con el **mismo intérprete** que ejecuta el agente, no solo en la terminal.
- No metas FFmpeg en `requirements.txt`: no es un paquete de Python. Documéntalo como requisito del sistema en el README.
