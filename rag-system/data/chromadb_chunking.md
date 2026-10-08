# ChromaDB y Chunking Estratégico

ChromaDB es una base de datos vectorial de código abierto con modo de
persistencia local. Guarda los datos en disco en una carpeta (por ejemplo
./vectorstore) para que el conocimiento sobreviva al reinicio del servidor.

Operaciones CRUD en el espacio vectorial:
- Upsert: crea el vector o lo actualiza si el ID ya existe.
- get(): busca por IDs exactos o filtros de metadatos.
- query(): busca por similitud semántica, el corazón de RAG.
- Delete: elimina vectores que ya no son relevantes.

Los metadatos (autor, fecha, fuente) permiten filtrar antes de la búsqueda
vectorial y rastrear el origen de la información para citar fuentes.

Estrategia de chunking: RecursiveCharacterTextSplitter prueba separadores en
jerarquía (párrafos \n\n, saltos \n, espacios, caracteres). Se configura con
chunk_size=500 tokens y chunk_overlap=50 tokens. El overlap repite el final
del Chunk 1 al inicio del Chunk 2 para no perder contexto en el corte.

La medición debe hacerse por tokens con tiktoken, no por caracteres.
La métrica estándar para embeddings de OpenAI (text-embedding-3-small,
1536 dimensiones) es la similitud coseno. Nunca mezcles modelos de embeddings
entre indexación y consulta: los resultados serían aleatorios.

El retriever debe mantener un top_k de entre 3 y 5 fragmentos para evitar el
problema de "Contexto Infinito" y la degradación de atención (Lost in the Middle).
