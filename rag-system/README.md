# Pre-entrega 3: Sistema de Recuperación Semántica Local (RAG)

Sistema RAG (Retrieval-Augmented Generation) end-to-end con ingesta de datos, chunking por tokens, persistencia en ChromaDB y generación grounded con OpenAI.

## Estructura del Proyecto

```
rag-system/
├── data/                  # Carpeta para archivos .txt o .md
├── vectorstore/           # ChromaDB persistente (se crea automáticamente)
├── schemas.py             # Modelo Pydantic de respuesta
├── ingest.py              # Script de ingesta y chunking
├── rag.py                 # Cadena LCEL asíncrona con retriever
├── requirements.txt       # Dependencias
├── .env                   # Variables de entorno (no incluido en git)
└── .env.example           # Plantilla de variables de entorno
```

## Configuración del Entorno

1. Clonar el repositorio y navegar al directorio:
   ```bash
   cd rag-system
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurar variables de entorno:
   ```bash
   cp .env.example .env
   ```
   Editar `.env` y agregar tu API key de OpenAI:
   ```
   OPENAI_API_KEY=sk-tu-api-key-aqui
   ```

5. Agregar documentos:
   Colocar archivos `.txt` o `.md` en la carpeta `/data`.

## Ejecución

### Paso 1: Ingesta de datos

```bash
python ingest.py
```

Este script:
- Lee todos los archivos `.txt` y `.md` de `/data`
- Verifica si la colección ya existe en ChromaDB (evita re-indexación innecesaria)
- Fragmenta los documentos usando `RecursiveCharacterTextSplitter` con medición por **tokens** (tiktoken)
- Parámetros: `chunk_size=500`, `chunk_overlap=50`
- Persiste los embeddings en `./vectorstore` con ChromaDB

### Paso 2: Ejecutar el sistema RAG

```bash
python rag.py
```

Este script ejecuta dos pruebas automáticas:
1. **Pregunta con respuesta en el contexto**: Verifica que el modelo responda correctamente usando los documentos indexados.
2. **Pregunta trampa**: Verifica que el modelo no alucine y declare no tener acceso a información fuera del contexto.

## Arquitectura

### Ingesta (ingest.py)
- **Carga**: `TextLoader` para archivos `.txt` y `.md`
- **Chunking**: `RecursiveCharacterTextSplitter` con `tiktoken` para medición precisa de tokens
- **Persistencia**: `ChromaDB` con `persist_directory` para almacenamiento local
- **Idempotencia**: Verifica si la colección ya tiene documentos antes de indexar

### Recuperación (rag.py)
- **Retriever**: Configurado con `top_k=4` (dentro del rango 3-5 recomendado)
- **Consistencia**: Mismo modelo de embeddings (`text-embedding-3-small`) para indexación y búsqueda
- **Prompt**: Filtro estricto de veracidad mediante `ChatPromptTemplate`
- **Cadena LCEL**: Ensamblada con operador pipe (`|`)
- **Ejecución asíncrona**: `async def get_rag_response()` con `.ainvoke()`
- **Validación estructurada**: `Pydantic` + `.with_structured_output()`
- **Resiliencia**: `.with_retry()` para manejar salidas mal formadas

## Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| `tiktoken` para chunking | Evita perder contexto global por caracteres vs tokens |
| `chunk_size=500` | Balance entre granularidad y contexto suficiente |
| `chunk_overlap=50` | Mantiene continuidad semántica entre fragmentos |
| `top_k=4` | Suficiente para contexto relevante sin degradar atención |
| `with_structured_output()` | Garantiza formato de respuesta consistente |
| `with_retry()` | Maneja errores de formato del LLM automáticamente |
| `.ainvoke()` | No bloquea el event loop en aplicaciones async |
