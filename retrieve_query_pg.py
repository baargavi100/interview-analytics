from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ modern import
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# === Embedding & LLM ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral")  # Or "tinyllama" etc.

# === PGVector connection ===
CONNECTION_STRING = "postgresql+psycopg2://postgres:baargavi123@localhost:5432/vector_db"
vectorstore = PGVector(
    collection_name="interview_analytics",
    connection_string=CONNECTION_STRING,
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever()

# === Retrieval QA Chain ===
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
)

# === Ask questions ===
print("\n📄 Ask questions about InterviewAnalytics.pdf")
print("Type 'exit' to quit.\n")

while True:
    query = input("🔎 Question: ")
    if query.lower() == "exit":
        break
    result = qa.invoke(query)  # ✅ Use invoke() instead of .run()
    print("\n📌 Answer:\n", result["result"])
