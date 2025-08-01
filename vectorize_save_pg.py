import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# === Set correct path to your PDF folder ===
pdf_dir = r"C:\Users\Lenovo\Desktop\interview.pro\docs"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# === Embedding model ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === PostgreSQL connection string ===
CONNECTION_STRING = "postgresql+psycopg2://postgres:baargavi123@localhost:5432/vector_db"

# === Text splitter settings ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# === Container for chunks ===
all_chunks = []

# === Process each PDF ===
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    if not chunks:
        print(f"⚠️ No content extracted from: {os.path.basename(pdf_path)}")
        continue

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf_path)

    print(f"\n✅ Processed: {os.path.basename(pdf_path)} | Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk.page_content)} chars):\n{chunk.page_content[:300]}...\n{'-'*60}")

    all_chunks.extend(chunks)

# === Store chunks to pgvector ===
try:
    PGVector.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name="interview_analytics",
        connection_string=CONNECTION_STRING,
    )
    print("\n✅ All documents embedded and saved to pgvector (PostgreSQL).")
except Exception as e:
    print("❌ Error saving to pgvector:", e)
