from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

MODEL = "gpt-oss:120b-cloud"
OFFLOAD_KEY = "log-4471"

# Stand-in for external storage (a file, vector store, or database in a real system).
OFFLOAD_STORE = {}
class State(TypedDict):
    request: str
    raw_result: str
    summary: str
    detail_decision: str
    context: str
    answer: str
def offload_raw_result(state: State):
    OFFLOAD_STORE[OFFLOAD_KEY] = state["raw_result"]
    summary = state["raw_result"][:100] + "..."
    return {"summary": summary}

def build_context(state: State):
    lines = [
        f"TASK: {state['request']}",
        f"SUMMARY: {state['summary']}",
        f"NOTE: full text stored at '{OFFLOAD_KEY}' "
        f"({len(state['raw_result'])} chars, not sent unless needed).",
    ]
    return {"context": "\n".join(lines)}
def decide_detail_level(state: State):
    prompt = (
        "Does answering this request need the exact, full, verbatim text, or is a short "
        "summary enough?\n"
        "Return only FULL or SUMMARY.\n\n"
        f"Request: {state['request']}\n"
        f"Summary available: {state['summary']}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = response["message"]["content"].strip().upper()
    if decision not in ("FULL", "SUMMARY"):
        decision = "SUMMARY"
    return {"detail_decision": decision}

def route_after_decision(state: State):
    if state["detail_decision"] == "FULL":
        return "retrieve_full_text"
    return "ask_model"

def retrieve_full_text(state: State):
    full_text = OFFLOAD_STORE[OFFLOAD_KEY]
    context = state["context"] + f"\nFULL TEXT: {full_text}"
    return {"context": context}
def ask_model(state: State):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": state['context']}],
    )
    return {"answer": response["message"]["content"]}
graph_builder = StateGraph(State)
graph_builder.add_node("offload_raw_result", offload_raw_result)
graph_builder.add_node("build_context", build_context)
graph_builder.add_node("decide_detail_level", decide_detail_level)
graph_builder.add_node("retrieve_full_text", retrieve_full_text)
graph_builder.add_node("ask_model", ask_model)

graph_builder.add_edge(START, "offload_raw_result")
graph_builder.add_edge("offload_raw_result", "build_context")
graph_builder.add_edge("build_context", "decide_detail_level")
graph_builder.add_conditional_edges("decide_detail_level", route_after_decision)
graph_builder.add_edge("retrieve_full_text", "ask_model")
graph_builder.add_edge("ask_model", END)
graph = graph_builder.compile()
result = graph.invoke(
    {
        "request": "Give me the exact wording of the last error in the log",
        "raw_result": (
            "2026-09-24 09:14:02 INFO connection established\n"
            "2026-09-24 09:14:05 WARN retry attempt 1\n"
            "2026-09-24 09:14:09 ERROR connection reset by peer, session terminated"
        ),
    }
)
print("Context sent to the model:\n", result["context"])
print("\nAnswer:\n", result["answer"])