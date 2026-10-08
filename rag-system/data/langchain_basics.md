# LangChain: Conceptos Fundamentales

LangChain es un framework para desarrollar aplicaciones con modelos de lenguaje (LLM).
Su pieza central es LCEL (LangChain Expression Language), un lenguaje declarativo
para componer cadenas usando el operador pipe (|).

Una cadena típica sigue el patrón: Prompt | Modelo | Parser.

El PromptTemplate toma un diccionario de variables y genera un mensaje formateado.
El ChatModel envía ese mensaje al LLM (por ejemplo gpt-4o-mini) y devuelve un AIMessage.
El OutputParser extrae el contenido útil, por ejemplo texto limpio con StrOutputParser
o un objeto validado con with_structured_output.

LCEL expone gratis async (ainvoke, abatch, astream), ejecución en paralelo con
RunnableParallel y observabilidad con LangSmith. El método batch procesa listas
de entradas de forma eficiente.

Para respuestas confiables se usa Pydantic: se define un esquema formal y el modelo
debe encajar su salida en ese molde. Si el JSON sale mal formado, se reintenta
con with_retry antes de devolver un error al usuario.
