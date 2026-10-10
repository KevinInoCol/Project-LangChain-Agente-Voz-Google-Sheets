# CLAUDE.md

## Código LangChain: siempre con las dos skills de MCP

Antes de escribir, modificar, revisar o depurar cualquier código de LangChain / LangGraph / LangSmith
(cualquier archivo que importe `langchain*` o `langgraph`), invoca **las dos** skills del proyecto, en este orden:

1. **`langchain-mcp-docs`** — MCP `docs-langchain`: qué patrón/API es el vigente y recomendado.
2. **`langchain-mcp-reference`** — MCP `reference-langchain`: import, paquete y firma exacta de cada símbolo.

Sin excepciones, aunque el cambio parezca trivial o "ya se sepa" la API: el objetivo es código actual (LangChain 1.x),
con buenas prácticas y que funcione. Si alguno de los dos MCPs no está conectado, detente y avísalo antes de codear.
