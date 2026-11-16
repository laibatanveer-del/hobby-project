from pathlib import Path
from pypdf import PdfReader
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# document loading
def load_pdf(file_path):
    """
    Extract text from a PDF, page by page.

    Returns a list of dictionaries containing
    page content and metadata.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "id": f"{path.stem}_page_{page_number}",
            "text": text.strip(),
            "metadata": {
                "source": path.name,
                "page": page_number
            }
        })

    return documents


def split_documents(documents, chunk_size = 1000, chunk_overlap = 200):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size.")

    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "id": f'{document["id"]}_chunk_{chunk_index}',
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index
                    }
                })

            chunk_index += 1
            start += chunk_size - chunk_overlap

    return chunks


# Retreiver
def retrieve(query, top_k=3):

    results = search(query=query, n_results=top_k)

    retrieved_documents = []

    for i, document in enumerate(results["documents"][0]):
        retrieved_documents.append({
            "text": document,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents

# Vector DB 
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    embedding_function = DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="nutrition_documents",
        embedding_function=embedding_function
    )
    return collection


def add_documents(chunks):
    collection = get_collection()
    if not chunks:
        return

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )
    return collection.count()


def search(query, n_results= 3):
    collection = get_collection()
    return collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )


# LLM function
def generate_answer(question,retrieved_documents):
    context = ""

    for i, document in enumerate(retrieved_documents, start=1):
        metadata = document["metadata"]

        context += (
            f"\n[Source {i} | Page {metadata['page']}]\n"
            f"{document['text']}\n"
        )

    payload = {
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the "
                    "provided context. If the context does not "
                    "contain the answer, say you don't have "
                    "enough information. Mention the source "
                    "page numbers when answering."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n"
                    f"Question: {question}"
                )
            }
        ],
        "stream": False
    }

    request = Request(
        url="http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"].strip()

    except HTTPError as error:
        raise RuntimeError(
            f"Ollama API error: {error.code} - "
            f"{error.read().decode('utf-8')}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama: {error.reason}"
        ) from error
