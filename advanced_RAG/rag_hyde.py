from langchain_community.document_loaders.text import TextLoader
#from langchain_community.document_loaders.text import TextLoader
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
print("Loading Embedding Model...\n")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

test_vec = embeddings.embed_query("test")
print(f"✅ Embedding Model Ready — vector dim: {len(test_vec)}")

print("Creating Chroma Vector Store...\n")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Vector Store Ready — {vectorstore._collection.count()} vectors stored")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("✅ Retriever Ready")
print("Loading LLM...\n")

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0   # Deterministic — required for grounded RAG answers
)

print("✅ LLM Ready")

#── HyDE Prompt ───────────────────────────────────────────────────────────────
# Instructs the LLM to generate a hypothetical answer paragraph
# The output will be embedded and used as the search query — NOT as the answer

hyde_prompt = PromptTemplate.from_template(
    """
You are helping a retrieval system.

Generate a short hypothetical document paragraph that would
likely answer the user's question.

Write in the style of a formal policy or reference document.
It does NOT need to be factually correct — focus on style and vocabulary.

Question:
{question}

Hypothetical Document:
"""
)

# ── HyDE Chain (LCEL pipe) ───────────────────────────────────────────────────
# Fill {question} → send to LLM → return hypothetical paragraph
hyde_chain = hyde_prompt | llm

print("✅ HyDE Chain Ready")

# ── Sanity Check: See what a hypothetical document looks like ─────────────────
sample_q = "What is the leave policy?"
sample_hyde = hyde_chain.invoke({"question": sample_q})

print(f"\n🧪 Sample HyDE Test:")
print(f"\n❓ Question:")
print(f"   {sample_q}")
print(f"\n📝 Generated Hypothetical Document:")
print(sample_hyde.content.strip())
print("\n(This paragraph will be embedded and used as the search vector — not shown to the user)")

print("HyDE RAG System Ready. Type 'exit' to stop.\n")
print("=" * 60)

while True:

    question = input("\n❓ Ask a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    # ── STEP 8a: GENERATE HYPOTHETICAL DOCUMENT ───────────────────────────────
    # LLM writes a fake-but-stylistically-accurate answer paragraph
    # This becomes the search vector — it is NOT the final answer
    hypothetical_response = hyde_chain.invoke({"question": question})
    hypothetical_text = hypothetical_response.content.strip()

    print("\n===== ❓ USER QUESTION =====")
    print(question)

    print("\n===== 📝 HYPOTHETICAL DOCUMENT (search probe only) =====")
    print(hypothetical_text)
    print("(⚠️ This is NOT the answer — it's only used to find the right chunks)")

    # ── STEP 8b: RETRIEVE USING ORIGINAL QUESTION ────────────────────────────
    # For side-by-side comparison — shows what vanilla RAG would have retrieved
    original_docs = retriever.invoke(question)
    retrieved_docs = retriever.invoke(hypothetical_text)

    print("\n===== 📄 ORIGINAL RETRIEVAL (vanilla RAG — for comparison) =====")
    for index, doc in enumerate(retrieved_docs):
        print(f"\n  Chunk {index + 1}: {doc.page_content}")

    # ── STEP 8d: BUILD CONTEXT FROM REAL RETRIEVED CHUNKS ────────────────────
    # The hypothetical doc is discarded here — only real chunks go into the answer
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # ── STEP 8e: CRAFT FINAL ANSWER PROMPT ───────────────────────────────────
    # Original question + real context = grounded, trustworthy answer
    # The hypothetical document does NOT appear here
    prompt = f"""You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, reply exactly with:
"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    # ── STEP 8f: GENERATE FINAL ANSWER ───────────────────────────────────────
    response = llm.invoke(prompt)

    print("\n===== 🤖 FINAL ANSWER =====")
    print(response.content)