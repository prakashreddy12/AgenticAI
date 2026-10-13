from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

MODEL = "gpt-oss:120b-cloud"
BUDGET = 150
class State(TypedDict):
    sections: dict[str, str]
    budget: int
    cut_choice: str
    context: str
    dropped_section: str
    answer: str
def assemble_sections(state: State):
    sections = {
        "TASK": "What's the status of ticket KB-118?",
        "RAG": "KB-118: Open tickets owned by network-team are reviewed every weekday morning.",
        "MEMORY": "User asked about a  ticket (#4402) three turns ago and it was already closed.",
    }
    return {"sections": sections, "budget": BUDGET, "dropped_section": "NONE"}
def decide_cut(state: State):
    sections = state["sections"]
    
    total = sum(len(text) for text in sections.values())

    if total <= state["budget"]:
        return {"cut_choice": "NONE"}

    listing = "\n".join(f"{label}: {text}" for label, text in sections.items() if label != "TASK")
    prompt = (
        f"The assembled context is {total} characters, over the {state['budget']}-character budget. "
        "Choose ONE section to remove so it fits. Never choose TASK.\n"
        "Return only one of: RAG, MEMORY.\n\n"
        f"{listing}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response["message"]["content"].strip().upper()
    if choice not in ("RAG", "MEMORY"):
        choice = "MEMORY"  # safe default: fall back to the old fixed-priority answer
    return {"cut_choice": choice}

def route_after_cut(state: State):
    if state["cut_choice"] == "RAG":
        return "drop_rag_section"
    if state["cut_choice"] == "MEMORY":
        return "drop_memory_section"
    return "build_context"
def drop_rag_section(state: State):
    sections = dict(state["sections"])
    sections.pop("RAG", None)
    return {"sections": sections, "dropped_section": "RAG"}


def drop_memory_section(state: State):
    sections = dict(state["sections"])
    sections.pop("MEMORY", None)
    return {"sections": sections, "dropped_section": "MEMORY"}
PRIORITY_ORDER = ["TASK", "RAG", "MEMORY"]


def build_context(state: State):
    lines = [f"{label}: {state['sections'][label]}" for label in PRIORITY_ORDER if label in state["sections"]]
    return {"context": "\n".join(lines)}


def ask_model(state: State):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": state['context']}],
    )
    return {"answer": response["message"]["content"]}
graph_builder = StateGraph(State)
graph_builder.add_node("assemble_sections", assemble_sections)
graph_builder.add_node("decide_cut", decide_cut)
graph_builder.add_node("drop_rag_section", drop_rag_section)
graph_builder.add_node("drop_memory_section", drop_memory_section)
graph_builder.add_node("build_context", build_context)
graph_builder.add_node("ask_model", ask_model)

graph_builder.add_edge(START, "assemble_sections")
graph_builder.add_edge("assemble_sections", "decide_cut")
graph_builder.add_conditional_edges("decide_cut", route_after_cut)
graph_builder.add_edge("drop_rag_section", "build_context") #TASK,MEMORY
graph_builder.add_edge("drop_memory_section", "build_context") #TASK,RAG
graph_builder.add_edge("build_context", "ask_model")
graph_builder.add_edge("ask_model", END)
graph = graph_builder.compile()
result = graph.invoke({})
print(f"Budget: {BUDGET} characters")
print("Dropped section:", result["dropped_section"])
print("\nContext sent to the model:\n", result["context"])
print("\nAnswer:\n", result["answer"])