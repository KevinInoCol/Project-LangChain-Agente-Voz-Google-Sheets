# Plantillas del scaffold

Esqueletos mínimos. Adáptalos al dominio, pero **no cambies la ubicación de los archivos ni los
nombres de las variables de entorno**. Verifica las APIs de LangChain con el MCP `docs-langchain`
antes de copiar (skills `langchain-mcp-docs` y `langchain-mcp-reference`).

---

## `.env.example`

```bash
# Copia este archivo como .env y rellena los valores reales:
#   cp .env.example .env
# El .env está en .gitignore: nunca lo subas al repositorio.

# ============================================================
# OpenAI  (obligatorio)
# ============================================================
# LLM del agente + TTS. https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# ============================================================
# PostgreSQL  (histórico de conversación)
# ============================================================
# Guarda los checkpoints de LangGraph (la memoria del agente) y la tabla
# <prefijo>_chat_log (auditoría legible con SQL).
# Si estas variables están vacías el agente NO se cae: usa RAM y avisa.
#
# Supabase: Project Settings > Database > Connection string > Session pooler
DB_USER=postgres.xxxxxxxxxxxxxxxx
DB_PASSWORD=
DB_HOST=aws-0-<region>.pooler.supabase.com

# IMPORTANTE: usa 5432 (pooler en modo "session"). El 6543 es "transaction",
# no soporta prepared statements y rompe el checkpointer de LangGraph con
# DuplicatePreparedStatement.
DB_PORT=5432
DB_NAME=postgres

# ============================================================
# Sesión  (opcional)
# ============================================================
# Al arrancar se imprime un Session ID. Ponlo aquí para retomar esa
# conversación en vez de empezar una nueva.
# VOICE_SESSION_ID=
```

---

## `conversation_history/chat_history.py`

Estructura obligatoria en 5 secciones (ver el proyecto de referencia para la implementación completa):

```python
"""Histórico de conversación del agente (PostgreSQL).

1. CHECKPOINTER — get_checkpointer(): lo que hace que el agente RECUERDE.
2. LOG PLANO   — registrar_turno() / listar_conversacion(): auditoría con SQL.

Variables de entorno (.env): DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
Si faltan o la conexión falla, cae a InMemorySaver con aviso visible.
"""

import atexit, os, uuid
from contextlib import ExitStack
from urllib.parse import quote_plus
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# 1. CONFIGURACIÓN
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "postgres")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    if all([DB_USER, DB_PASSWORD, DB_HOST]) else None
)
TABLA_LOG = "<prefijo>_chat_log"

_stack = ExitStack(); _checkpointer = None; _persistente = False

# 2. SESIONES
def nueva_sesion() -> str: ...          # str(uuid.uuid4())
def validar_sesion(session_id) -> str: ...  # UUID válido o uno nuevo

# 3. CHECKPOINTER
def get_checkpointer(): ...             # PostgresSaver singleton + .setup() + _crear_tabla_log()
def _avisar_sin_persistencia(motivo): ...   # InMemorySaver + banner de aviso
def historial_persistente() -> bool: ...
def cerrar() -> None: ...               # _stack.close()
atexit.register(cerrar)

# 4. LOG PLANO
def _crear_tabla_log() -> None: ...     # id, session_id UUID, rol, contenido, creado_en + índice
def registrar_turno(session_id, pregunta, respuesta) -> None: ...   # no-op si no persistente
def listar_conversacion(session_id) -> list[tuple]: ...

# 5. INSPECCIÓN MANUAL
if __name__ == "__main__":
    ...  # python conversation_history/chat_history.py <session_id>
```

`conversation_history/__init__.py`:

```python
from .chat_history import (
    cerrar, get_checkpointer, historial_persistente,
    listar_conversacion, nueva_sesion, registrar_turno, validar_sesion,
)

__all__ = [
    "cerrar", "get_checkpointer", "historial_persistente",
    "listar_conversacion", "nueva_sesion", "registrar_turno", "validar_sesion",
]
```

---

## `tools/<nombre_tool>.py` — dataset batch local

```python
"""Tool que le permite al agente consultar el DataFrame con pandas."""

import pandas as pd
from langchain.tools import tool


def ejecutar_consulta(df: pd.DataFrame, code: str) -> str:
    """Capa imperativa: testeable sin LLM ni micrófono."""
    ...


def get_query_dataframe_tool(df: pd.DataFrame):
    """Factory: cierra el DataFrame en clausura, el LLM solo ve `code`."""

    @tool
    def query_dataframe(code: str) -> str:
        """<docstring que LEE EL LLM: qué hace, qué formato espera, un ejemplo>"""
        return ejecutar_consulta(df, code)

    return query_dataframe
```

## `tools/<nombre_tool>.py` — un endpoint de API

Un módulo por endpoint. El cliente HTTP compartido va en `tools/_http.py`, que **no** define tools.

```python
"""Tool: buscar un cliente por documento.  GET /clientes/search"""

from langchain.tools import tool
from ._http import api_get


def buscar_cliente_por_documento(documento: str) -> dict:
    """Capa imperativa: se puede testear con requests-mock, sin LLM."""
    return api_get("/clientes/search", params={"doc": documento})


@tool
def buscar_cliente(documento: str) -> str:
    """Busca un cliente por su número de documento y devuelve nombre, email y
    estado de su cuenta. Usa solo dígitos, sin puntos ni guiones.
    Ejemplo: buscar_cliente("40123456")"""
    cliente = buscar_cliente_por_documento(documento)
    return f"{cliente['nombre']} — {cliente['email']} — {cliente['estado']}"
```

Cuando la tool no necesita dependencias en clausura, se exporta la `@tool` directamente; si necesita
un cliente configurado o credenciales por sesión, usa la factory `get_<tool>_tool(...)`.

`tools/__init__.py`:

```python
from .buscar_cliente import buscar_cliente, buscar_cliente_por_documento
from .crear_reserva import crear_reserva, crear_reserva_api

__all__ = ["buscar_cliente", "buscar_cliente_por_documento", "crear_reserva", "crear_reserva_api"]
```

---

## `prompt/prompt.yaml`

```yaml
name: <slug>-system-prompt
version: 1.0.0
description: >
  System prompt de <Nombre>, <rol> por voz especializado en <dominio>.
  Opera sobre <fuente de datos> mediante las tools <lista>, dentro de un agente
  LangChain con STT (Whisper) y TTS (OpenAI tts-1).
language: es
author: <email>
created_at: <YYYY-MM-DD>
tags: [voz, tool-calling, <dominio>]

# Placeholders inyectados en runtime por agent.py
variables:
  - <placeholder_1>

system_prompt: |
  <Identidad>
  Eres <nombre>, ... Atiendes por voz: el usuario te habla y tu respuesta se
  lee en voz alta. Siempre respondes en <idioma>.
  </Identidad>

  <Personalidad>
  ...
  </Personalidad>

  <Objetivo_Principal>
  ...
  </Objetivo_Principal>

  <Contexto_De_Los_Datos>
  ... {placeholder_1} ...
  </Contexto_De_Los_Datos>

  <Herramientas_Disponibles>
  - <tool>(<args>) -> str
    Qué hace y cuándo usarla.
  </Herramientas_Disponibles>

  <Reglas_De_Uso_De_Tools>
  1. Antes de dar cualquier dato, llama a la tool. Nunca inventes.
  ...
  </Reglas_De_Uso_De_Tools>

  <Instrucciones_Generales>
  ...
  </Instrucciones_Generales>

  <Formato_De_Respuesta>
  Tu respuesta será leída en voz alta por un sintetizador, así que:
  - No uses tablas, asteriscos, guiones decorativos ni Markdown.
  - Frases completas y naturales, como en una conversación hablada.
  - Nada de símbolos: di "metros cuadrados", no "m²".
  - Resume en 2 o 3 puntos clave; no listes todo.
  - Nunca muestres el código ni la salida cruda de la tool.
  </Formato_De_Respuesta>

  <Seguridad>
  - No respondas fuera del dominio.
  - Rechaza código destructivo, acceso a archivos o llamadas de red.
  </Seguridad>

  <IMPORTANTE>
  - Todo dato debe venir de una llamada a una tool.
  - Nada de Markdown: el texto se escucha, no se lee.
  - Si el dato no existe, dilo; no especules.
  </IMPORTANTE>
```

---

## `model_config/model_config.yaml`

```yaml
name: <slug>-model-config
version: 1.0.0
description: >
  Configuración de los modelos del agente de voz: LLM (init_chat_model),
  ventana de memoria (SummarizationMiddleware), límites del grafo, STT y TTS.
  Las instrucciones del agente viven en prompt/prompt.yaml.
author: <email>
created_at: <YYYY-MM-DD>
tags: [voz, configuracion, llm, whisper, tts]

# LLM del agente  ->  init_chat_model(model, **params)
llm:
  model: openai:gpt-4.1-mini   # siempre con prefijo de proveedor
  params:
    temperature: 0
    max_tokens: 500            # la respuesta se escucha; corta es mejor
    timeout: 60
    max_retries: 2

# Memoria  ->  SummarizationMiddleware(model, trigger=tuple(...), keep=tuple(...))
# [tipo, valor] con tipo en messages | tokens | fraction. Son MENSAJES, no turnos.
summarization:
  model: null                  # null = reutiliza el LLM del agente
  trigger: [messages, 20]
  keep: [messages, 10]

# Ejecución  ->  config de agent.invoke(...)
agent:
  recursion_limit: 50

# Speech-to-Text  ->  whisper.load_model(model_size) / transcribe(...)
stt:
  provider: whisper-local
  model_size: small            # tiny | base | small | medium | large
  fp16: false
  language: null               # null = autodetección

# Text-to-Speech  ->  client.audio.speech.create(...)
tts:
  provider: openai
  model: tts-1
  voice: alloy
  response_format: opus

# Audio de entrada  ->  sounddevice.InputStream(...)
audio:
  samplerate: 44100
  channels: 1
  dtype: int16
```

---

## `agent.py`

```python
"""Orquestador del agente: ensambla datos + system prompt + LLM + tools + memoria.

No sabe nada de audio, así que se puede probar sin micrófono:

    from agent import build_agent
    from conversation_history import nueva_sesion

    agent = build_agent()
    config = {"configurable": {"thread_id": nueva_sesion()}}
    r = agent.invoke({"messages": [{"role": "user", "content": "..."}]}, config)
    print(r["messages"][-1].content)
"""

from pathlib import Path

import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model

from conversation_history import get_checkpointer
from tools import get_<tool>_tool

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompt" / "prompt.yaml"
MODEL_CONFIG_PATH = BASE_DIR / "model_config" / "model_config.yaml"


def load_model_config() -> dict:
    """Carga model_config/model_config.yaml (LLM, memoria, STT, TTS, audio)."""
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(**contexto) -> str:
    """Carga prompt/prompt.yaml e inyecta los placeholders de `variables`."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    system_prompt = cfg["system_prompt"]
    for clave, valor in contexto.items():
        system_prompt = system_prompt.replace(f"{{{clave}}}", str(valor))
    return system_prompt


def build_agent(model: str | None = None, con_memoria: bool = True):
    """`model` es un override puntual; si es None manda model_config.yaml."""
    cfg = load_model_config()
    llm = init_chat_model(model or cfg["llm"]["model"], **cfg["llm"].get("params", {}))

    resumen = cfg["summarization"]
    return create_agent(
        llm,
        tools=[get_<tool>_tool(...)],
        system_prompt=load_system_prompt(...),
        #YAML no tiene tuplas: trigger/keep llegan como lista
        middleware=[
            SummarizationMiddleware(
                resumen.get("model") or llm,
                trigger=tuple(resumen["trigger"]),
                keep=tuple(resumen["keep"]),
            )
        ],
        checkpointer=get_checkpointer() if con_memoria else None,
    )
```

---

## Capa de voz — `talking_llm.py`

```python
class TalkingLLM:
    """Voz → Whisper → agent.invoke → TTS → audio. Sin lógica de negocio."""

    def __init__(self, session_id: str | None = None):
        self.cfg = load_model_config()                  # stt / tts / audio / agent
        self.whisper = whisper.load_model(self.cfg["stt"]["model_size"])
        self.agent = build_agent()
        self.session_id = validar_sesion(session_id) if session_id else nueva_sesion()
        self.config = {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": self.cfg["agent"]["recursion_limit"],
        }

    def convert_and_play(self):
        texto = self.transcribe()                       # Whisper
        r = self.agent.invoke({"messages": [{"role": "user", "content": texto}]}, self.config)
        respuesta = r["messages"][-1].content
        registrar_turno(self.session_id, texto, respuesta)
        self.speak(respuesta)                           # TTS
```

---

## `requirements.txt` (base)

```
langchain>=1.0.0
langchain-openai
langgraph>=1.0.0
langgraph-checkpoint-postgres
psycopg[binary]
openai
openai-whisper
python-dotenv
pyyaml
sounddevice
soundfile
pynput
numpy
```

`pandas` y `tabulate` solo si el proyecto usa `data/`.
