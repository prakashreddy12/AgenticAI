from langchain_community.document_loaders.text import TextLoader

# ── Chunking ───────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings (local, via Ollama) ──────────────────────────────────
from langchain_ollama import OllamaEmbeddings

# ── Vector Store (local, persisted to disk) ────────────────────────
from langchain_chroma import Chroma

# ── LLM — direct Ollama client, called from inside a graph node ──────────────
import ollama

# ── LangGraph ──────────────────────────────────────────────────────────────
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

print("✅ All imports successful")
print("Loading document...\n")

loader = TextLoader(
    "company_policy.txt",
    encoding="utf-8"
)

documents = loader.load()

print(f"✅ Loaded {len(documents)} document(s)")
print(f"\n📄 Preview (first 300 chars):")
print(documents[0].page_content[:300])
print(f"\n🏷️  Metadata: {documents[0].metadata}")

print("Splitting document into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
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
    persist_directory="./chroma_db_langgraph"
)

print("✅ Vector Store Created & Persisted to ./chroma_db_langgraph")
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
    # Print first 150 chars of each chunk
MODEL = "gpt-oss:120b-cloud"


class State(TypedDict):
    question: str
    context: str
    answer: str
    history: list[str]


print("✅ State schema defined")

def retrieve(state: State) -> dict:
    retrieved_docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    history = state.get("history", [])
    return {"context": context, "history": history + [f"Q: {state['question']}"]}


def generate(state: State) -> dict:
    history = state.get("history", [])
    history_text = "\n".join(history[-3:]) if history else "No prior questions yet."
    prompt = f"""You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, reply exactly with:
"I could not find that information in the document."

Previous conversation:
{history_text}

Context:
{state['context']}

Question:
{state['question']}
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return {"answer": response["message"]["content"]}


def no_context(state: State) -> dict:
    return {"answer": "I could not find that information in the document."}


def route_after_retrieve(state: State) -> str:
    return "generate" if state["context"].strip() else "no_context"


def human_review(state: State) -> dict:
    review = interrupt(
        {
            "message": "Review the retrieved context before answering.",
            "context": state["context"],
            "question": state["question"],
        }
    )

    if isinstance(review, str):
        state["context"] = review
    elif isinstance(review, dict):
        if "context" in review:
            state["context"] = review["context"]
        if "approve" in review and review["approve"] is False:
            return {"answer": "I stopped because the retrieved context was rejected by a human reviewer."}
    return {"context": state["context"]}


print("✅ Node functions defined")

graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("generate", generate)
graph_builder.add_node("no_context", no_context)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {"generate": "human_review", "no_context": "no_context"},
)
graph_builder.add_edge("human_review", "generate")
graph_builder.add_edge("generate", END)
graph_builder.add_edge("no_context", END)

graph = graph_builder.compile(checkpointer=MemorySaver())

print("✅ Graph compiled")
print("Nodes:", list(graph.get_graph().nodes.keys()))

# One approval per question, then resume only once.
thread_id = "rag-demo-thread"
question = "What is the work from home policy?"

result = graph.invoke(
    {"question": question, "history": [], "context": "", "answer": ""},
    config={"configurable": {"thread_id": thread_id}},
)

if "__interrupt__" in result:
    interrupt_data = result["__interrupt__"][0]
    payload = getattr(interrupt_data, "value", interrupt_data)
    print(f"\n⚠ Human review required for: {question}")
    print("Retrieved context:")
    print(payload.get("context", ""))

    edited = input("Press Enter to approve, or type revised context: ").strip()
    approved_context = edited if edited else payload.get("context", "")

    resumed = graph.invoke(
        Command(resume={"context": approved_context, "approved": True}),
        config={"configurable": {"thread_id": thread_id}},
    )
    print(f"\n❓ Question: {question}")
    print(f"📚 Retrieved Context:\n{resumed.get('context', '')}")
    print(f"🤖 Answer: {resumed.get('answer', 'No answer produced')}")
else:
    print(f"\n❓ Question: {question}")
    print(f"📚 Retrieved Context:\n{result.get('context', '')}")
    print(f"🤖 Answer: {result.get('answer', 'No answer produced')}")

print("✅ Human-in-the-loop approval complete")