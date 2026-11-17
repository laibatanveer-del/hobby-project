# from src.base_func import *

# # Load PDF
# documents = load_pdf("documents/human-nutrition-text.pdf")

# # Split documents
# chunks = split_documents(documents)

# print(f"Chunks generated: {len(chunks)}")

# # Store chunks
# vector_db_count = add_documents(chunks)

# print(f"Documents stored: {vector_db_count}")

# # Get user question
# # question = "Which nutrients help maintain a healthy bones?"
# # question = "Tell me one beauty secret" (result shows it has no context and doc have nutrition data)
# question = "Tell me three eating principle"

# # Retrieve relevant documents
# retrieved_documents = retrieve(question, top_k=3)

# # Generate answer
# answer = generate_answer(question, retrieved_documents)

# # Display answer
# print("\nQuestion:", question)
# print("\nGenerated Answer:\n")
# print(answer)


from collections import Counter
from pathlib import Path

from src.base_func import *
from src.multimodal import load_pdf_multimodal

PDF_PATH = "documents/2024-ESH-MASTERplan-Blood-Pressure_publication_2025-1-1.pdf"
REINDEX = False   # True after changing chunking or prompts

source = Path(PDF_PATH).name
if REINDEX:
    delete_source(source)

if not is_indexed(source):
    elements = load_pdf_multimodal(PDF_PATH)
    chunks = split_elements(elements)
    print("Elements:", dict(Counter(e["metadata"]["type"] for e in elements)))
    print(f"Chunks generated: {len(chunks)}")
    print(f"Documents stored: {add_documents(chunks)}")
else:
    print(f"{source} already indexed, skipping ingestion")

# question = "What are the blood pressure thresholds for diagnosing hypertension?"
# question = "Any history of medical or genetics affects hypertension?"
# question = "Tell 3 symptoms of brain tumor?"
question = "How to select therapy for hypertension for patients over 80 years old?"


retrieved_documents = retrieve(question, top_k=8, where={"source": source})
for d in retrieved_documents:
    m = d["metadata"]
    print(f"  p{m['page']} [{m['type']}] dist={d['distance']:.3f}")
    
answer = generate_answer(question, retrieved_documents)
print("\nQuestion:", question)
print("\nGenerated Answer:\n")
print(answer)