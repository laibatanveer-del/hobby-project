from pathlib import Path
from pypdf import PdfReader
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import re

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


COLLECTION_NAME = "medical_documents"   # new collection; schema differs from nutrition_documents

def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )

def is_indexed(source):
    return bool(get_collection().get(where={"source": source}, limit=1)["ids"])

def delete_source(source):
    get_collection().delete(where={"source": source})


# --- chunking for typed elements ---
MD_SEP = re.compile(r"^\|[\s:\-|]*-{3,}[\s:\-|]*\|?\s*$")

def split_rows(text, max_chars=700):
    """Split tables by rows, repeating title+header in every part. Never cuts mid-row."""
    lines = text.splitlines()
    sep = next((i for i, l in enumerate(lines) if MD_SEP.match(l.strip())), None)
    header = lines[: sep + 1] if sep is not None else []
    body = lines[sep + 1 :] if sep is not None else lines
    header_len = len("\n".join(header))

    parts, current, size = [], [], header_len
    for line in body:
        if current and size + len(line) + 1 > max_chars:
            parts.append("\n".join(header + current))
            current, size = [], header_len
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(header + current))
    return parts


def split_elements(elements, chunk_size=1000, chunk_overlap=200, table_chars=700):
    text_docs = [e for e in elements if e["metadata"]["type"] == "text"]
    chunks = split_documents(text_docs, chunk_size, chunk_overlap)

    for e in elements:
        kind = e["metadata"]["type"]
        if kind == "text":
            continue
        label = f"[{kind.upper()} | page {e['metadata']['page']}]"
        for i, part in enumerate(split_rows(e["text"], table_chars)):
            chunks.append({
                "id": f"{e['id']}_chunk_{i}",
                "text": f"{label}\n{part}",
                "metadata": {**e["metadata"], "chunk_index": i},
            })
    return chunks

# Retreiver
def retrieve(query, top_k=5, where=None):
    results = search(query=query, n_results=top_k, where=where)
    return [
        {
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]

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


def search(query, n_results=3, where=None):
    collection = get_collection()
    return collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where=where,
    )


# LLM function
def generate_answer(question, retrieved_documents, model="qwen3:4b", think=False):
    context = ""
    for i, document in enumerate(retrieved_documents, start=1):
        m = document["metadata"]
        context += (
            f"\n[Source {i} | Page {m['page']} | {m.get('type', 'text')}]\n"
            f"{document['text']}\n"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Answer using only the provided context. Start directly with the answer: "
                "3-6 sentences or a short bullet list. Do not describe your reasoning and do not "
                "go through the sources one by one. Cite pages like (p. 7). "
                "If the context does not contain the answer, say you don't have enough information."
            )},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"},
        ],
        "stream": True,
        "think": think,              # False skips the reasoning trace
        "keep_alive": "10m",
        "options": {"num_ctx": 8192, "num_predict": 1024, "temperature": 0},
    }
    request = Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    parts = []
    try:
        with urlopen(request, timeout=120) as response:   # idle-gap timeout
            for line in response:
                msg = json.loads(line)
                piece = msg.get("message", {}).get("content", "")
                if piece:
                    print(piece, end="", flush=True)
                    parts.append(piece)
                if msg.get("done"):
                    break
        return "".join(parts).strip()
    except HTTPError as error:
        raise RuntimeError(f"Ollama API error: {error.code} - {error.read().decode()}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Ollama: {error.reason}") from error
