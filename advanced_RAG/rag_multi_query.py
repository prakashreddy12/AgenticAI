from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

print("✅ All imports successful")
print("Loading document...\n")

loader = TextLoader("docs/company_policy.txt", encoding="utf-8")
documents = loader.load()

print(f"✅ Loaded {len(documents)} document(s)")
print(f"\n📄 Preview: {documents[0].page_content[:200]}")
print("Splitting document into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Total Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n🔹 Chunk {i+1}: {chunk.page_content}")
    print("-" * 50)
# ── Step 3: Embedding Model ──────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="nomic-embed-text")

test_vec = embeddings.embed_query("test")
print(f"✅ Embedding Model Ready — vector dim: {len(test_vec)}")
# ── Step 4: Create & Persist Vector Store ────────────────────────────────────
print("Creating Chroma Vector Store...\n")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Vector Store Ready — {vectorstore._collection.count()} vectors stored")
# ── Step 5: Retriever ────────────────────────────────────────────────────────
# k=2 per query — with 4 queries, we could get up to 8 chunks before dedup
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("✅ Retriever Ready (k=2 per query)")
print("Loading LLM...\n")

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0
)

print("✅ LLM Ready")
# ── Multi-Query Prompt ────────────────────────────────────────────────────────
# Instructs the LLM to generate 4 alternative phrasings of the user's question
# "Return ONLY the queries" avoids preamble text that breaks the line parser

multi_query_prompt = PromptTemplate.from_template(
    """
You are an expert search assistant.

Generate 4 different search queries for the user's question.
Each query should represent a different way of asking the same thing —
use synonyms, related terms, and different angles.

Return ONLY the queries. One query per line. No numbering. No explanation.

Question:
{question}
"""
)

# ── Multi-Query Chain (LCEL pipe) ───────────────────
multi_query_chain = multi_query_prompt | llm

print("✅ Multi-Query Chain Ready")

# ── Sanity Check: See what 4 queries look like for a sample question ──────────
sample_q = "What is the leave policy?"
sample_output = multi_query_chain.invoke({"question": sample_q})

print(f"\n🧪 Sample Multi-Query Generation:")
print(f"\n❓ Original Question: {sample_q}")
print(f"\n📋 Generated Queries (raw LLM output):")
print(sample_output.content.strip())

# ── Parse the raw output into a clean Python list ────────────────────────────
sample_list = [
    q.strip("- ").strip()
    for q in sample_output.content.strip().split("\n")
    if q.strip()   # skip empty lines
]

print(f"\n✅ Parsed into {len(sample_list)} queries:")
for i, q in enumerate(sample_list, 1):
    print(f"  {i}. {q}")
# ── Change this question and re-run ──────────────────────────────────────────
question = "What are my rights if I am terminated?"

# Generate 4 query variants
queries_raw = multi_query_chain.invoke({"question": question}).content.strip()
query_list = [q.strip("- ").strip() for q in queries_raw.split("\n") if q.strip()]

# Retrieve and deduplicate
all_docs = [doc for q in query_list for doc in retriever.invoke(q)]
seen = set()
unique_docs = []
for doc in all_docs:
    if doc.page_content not in seen:
        seen.add(doc.page_content)
        unique_docs.append(doc)

# Build context and answer
context = "\n\n".join(doc.page_content for doc in unique_docs)
prompt = f"""Answer ONLY using the context below.
If not found, say: "I could not find that information in the document."

Context:
{context}

Question: {question}
"""
answer = llm.invoke(prompt).content

print(f"❓ Question        : {question}")
print(f"\n📋 4 Queries used  :")
for i, q in enumerate(query_list, 1):
    print(f"   {i}. {q}")
print(f"\n📦 Raw chunks      : {len(all_docs)} → Unique: {len(unique_docs)}")
print(f"\n🤖 Answer          : {answer}")