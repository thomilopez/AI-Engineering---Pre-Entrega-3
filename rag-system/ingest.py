import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tiktoken import get_encoding

load_dotenv()

DATA_DIR = Path("./data")
VECTORSTORE_DIR = Path("./vectorstore")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """Limpieza antes de fragmentar: elimina ruido que ensucia el embedding."""
    text = re.sub(r"\r\n?", "\n", text)          # normaliza saltos de línea
    text = re.sub(r"[ \t]+", " ", text)         # espacios/tabs duplicados
    text = re.sub(r"\n{3,}", "\n\n", text)       # más de 2 saltos -> párrafo
    text = re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)  # líneas con solo espacios
    return text.strip()


_TOKENIZER = get_encoding("cl100k_base")


def get_token_length(text: str) -> int:
    # Medición por TOKENS (no caracteres): error #1 que hace perder contexto.
    return len(_TOKENIZER.encode(text))


def load_documents(data_dir: Path) -> list:
    documents = []
    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() in (".txt", ".md"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=get_token_length,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def ingest():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    existing_count = vectorstore._collection.count()
    if existing_count > 0:
        print(f"La colección '{COLLECTION_NAME}' ya contiene {existing_count} fragmentos. Omitiendo indexación.")
        return

    if not DATA_DIR.exists():
        print(f"La carpeta {DATA_DIR} no existe. Creándola...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("Agrega archivos .txt o .md en la carpeta /data y vuelve a ejecutar.")
        return

    documents = load_documents(DATA_DIR)
    if not documents:
        print("No se encontraron archivos .txt o .md en la carpeta /data.")
        return

    chunks = split_documents(documents)
    print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")

    vectorstore.add_documents(chunks)
    print(f"Indexación completada. {len(chunks)} fragmentos guardados en {VECTORSTORE_DIR}.")


if __name__ == "__main__":
    ingest()
