import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from schemas import RAGResponse

load_dotenv()

VECTORSTORE_DIR = Path("./vectorstore")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
TOP_K = 4

SYSTEM_PROMPT = """Eres un asistente técnico. Responde solo basándote en el CONTEXTO proporcionado.
Si la respuesta no está allí, di que no tienes acceso a esa información.
No inventes datos ni hagas suposiciones fuera del contexto dado."""

HUMAN_PROMPT = """CONTEXTO:
{context}

PREGUNTA: {question}"""


def format_docs(docs: list) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        formatted.append(f"[Fragmento {i} | Fuente: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    structured_llm = llm.with_structured_output(RAGResponse).with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm
    )

    return chain


async def get_rag_response(query: str) -> RAGResponse:
    chain = build_chain()
    response = await chain.ainvoke(query)
    return response


async def main():
    print("=== PRUEBA 1: Pregunta con respuesta en el contexto ===")
    query1 = "¿Qué es LangChain y para qué se utiliza?"
    print(f"Pregunta: {query1}")
    response1 = await get_rag_response(query1)
    print(f"Respuesta: {response1.answer}")
    print(f"Referencias: {response1.references}")
    print(f"Grounded: {response1.grounded}")
    print()

    print("=== PRUEBA 2: Pregunta trampa (no debería estar en el contexto) ===")
    query2 = "¿Cuál es la capital de Marte?"
    print(f"Pregunta: {query2}")
    response2 = await get_rag_response(query2)
    print(f"Respuesta: {response2.answer}")
    print(f"Referencias: {response2.references}")
    print(f"Grounded: {response2.grounded}")


if __name__ == "__main__":
    asyncio.run(main())
