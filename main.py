from src.base_func import *

# Load PDF
documents = load_pdf("documents/human-nutrition-text.pdf")

# Split documents
chunks = split_documents(documents)

print(f"Chunks generated: {len(chunks)}")

# Store chunks
vector_db_count = add_documents(chunks)

print(f"Documents stored: {vector_db_count}")

# Get user question
# question = "Which nutrients help maintain a healthy bones?"
# question = "Tell me one beauty secret" (result shows it has no context and doc have nutrition data)
question = "Tell me three eating principle"

# Retrieve relevant documents
retrieved_documents = retrieve(question, top_k=3)

# Generate answer
answer = generate_answer(question, retrieved_documents)

# Display answer
print("\nQuestion:", question)
print("\nGenerated Answer:\n")
print(answer)
