# Agente de Voz con LangChain y Whisper

Agente conversacional por voz que permite hacer consultas en lenguaje natural sobre un dataset de propiedades en alquiler de São Paulo, Brasil. El usuario habla, Whisper transcribe, un agente LangChain v1 analiza el dataframe pandas y responde en voz alta usando OpenAI TTS.

## Arquitectura

```
Voz del usuario
      ↓
  Whisper (STT)
      ↓
  Agente LangChain v1  ←→  query_dataframe (tool pandas)
      ↓
  OpenAI TTS
      ↓
  Respuesta en audio
```

## Módulos

| Archivo | Descripción |
|---|---|
| `talking_llm_part_1_save_audio.py` | Grabación de audio con hotkey y guardado en WAV |
| `talking_llm_part_2_llm_y_TTS.py` | Transcripción con Whisper + respuesta del LLM + TTS |
| `talking_llm_part_3_completo_con_agente.py` | Versión completa con agente LangChain v1 + tool pandas |

## Requisitos

- Python 3.9 / 3.10 / 3.11
- Clave de API de OpenAI

## Instalación

**1. Crear y activar el entorno virtual:**

```bash
conda create -n LangChain-Agente-Voz-Sheets python=3.11
conda activate LangChain-Agente-Voz-Sheets
```

**2. Instalar dependencias:**

```bash
pip install -r requirements.txt
```

**3. Configurar variables de entorno:**

Crea un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=sk-...
```

**4. Añadir el dataset:**

Coloca el archivo `df_rent.csv` en la carpeta `Datos/`.

## Uso

Ejecuta el agente completo:

```bash
python talking_llm_part_3_completo_con_agente.py
```

- Presiona `Cmd` para **comenzar** a grabar tu pregunta.
- Presiona `Cmd` de nuevo para **detener** la grabación.
- El agente procesará tu pregunta y responderá en voz alta.

## Dataset

El dataset `df_rent.csv` contiene propiedades en alquiler de São Paulo con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `Price` | Precio mensual del alquiler (R$) |
| `Condo` | Valor del condominio mensual (R$) |
| `Size` | Tamaño en metros cuadrados |
| `Rooms` | Número de habitaciones |
| `Toilets` | Número de baños |
| `Suites` | Número de suites |
| `Parking` | Espacios de estacionamiento |
| `Elevator` | Tiene ascensor (1/0) |
| `Furnished` | Está amueblado (1/0) |
| `Swimming Pool` | Tiene piscina (1/0) |
| `District` | Barrio de São Paulo |
| `Latitude` | Coordenada geográfica |
| `Longitude` | Coordenada geográfica |

## Stack tecnológico

- **LangChain v1** — framework del agente (`create_agent`, `@tool`, `init_chat_model`)
- **LangGraph v1** — runtime del agente (usado internamente por LangChain v1)
- **OpenAI Whisper** — transcripción de voz a texto (STT)
- **OpenAI TTS** — síntesis de voz (modelo `tts-1`, voz `alloy`)
- **OpenAI GPT-4.1 mini** — modelo de lenguaje del agente
- **sounddevice / soundfile** — captura y reproducción de audio
- **pandas** — análisis del dataset
