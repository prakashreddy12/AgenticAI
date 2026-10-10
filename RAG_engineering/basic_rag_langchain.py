from pathlib import Path

from langchain_community.document_loaders.text import TextLoader

# ── Chunking ───────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings & LLM (local, via Ollama) ───────────────────────────
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ── Vector Store (local, persisted to disk) ────────────────────────
from langchain_chroma import Chroma

print("✅ All imports successful")

print("Loading documents...\n")

base_dir = Path(__file__).resolve().parent
files_to_load = [
    base_dir / "sample_knowledge.txt",
    base_dir / "company_policy.txt",
]

# load both files and combine them into one list of documents
all_documents = []
for file_path in files_to_load:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing document: {file_path}")
    loader = TextLoader(str(file_path), encoding="utf-8")
    all_documents.extend(loader.load())

documents = all_documents

print(f"✅ Loaded {len(documents)} document(s)")
print(f"\n📄 Preview (first 300 chars):")
print(documents[0].page_content[:300])
print(f"\n🏷️  Metadata: {documents[0].metadata}")

print("Splitting document into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Total Chunks Created: {len(chunks)}")
print(f"\n--- Previewing first 3 chunks ---")

for index, chunk in enumerate(chunks[:3]):
    print(f"\n🔹 Chunk {index + 1} ({len(chunk.page_content)} chars)")
    print(chunk.page_content)
    print("-" * 50)

print("Loading Embedding Model...\n")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Embedding Model Ready")

test_vector = embeddings.embed_query("What is the leave policy?")
print(f"\n🔢 Vector dimension: {len(test_vector)}")
print(f"📊 First 5 values: {[round(v, 4) for v in test_vector[:5]]}")
print("Creating Chroma Vector Store...\n")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    #persist_directory="./chroma_db"
)

print("✅ Vector Store Created & Persisted to ./chroma_db")
print(f"📦 Total vectors stored: {vectorstore._collection.count()}")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

print("✅ Retriever Ready")

test_query = "What is the leave policy?"
test_results = retriever.invoke(test_query)

print(f"\n🔍 Test Query: '{test_query}'")
print(f"📄 Retrieved {len(test_results)} chunk(s):")
for i, doc in enumerate(test_results):
    print(f"\n  Chunk {i+1}: {doc.page_content[:150]}...")

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0
)

print("✅ LLM Ready")

question = "Who is the prime minister of India?"

# 1. Retrieve
retrieved_docs = retriever.invoke(question)

# 2. Join chunks into one context string
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 3. Build the grounded prompt
prompt = f"""You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, reply exactly with:
"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

# 4. Generate
response = llm.invoke(prompt)

print(f"❓ Question: {question}")
print(f"\n📚 Retrieved Context:\n{context}")
print(f"\n🤖 Answer: {response.content}")