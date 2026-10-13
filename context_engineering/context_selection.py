from typing import TypedDict

from langgraph.graph import END, START, StateGraph
import ollama

MODEL = "gpt-oss:120b-cloud"

class State(TypedDict):
    request: str
    rag_candidates: list[tuple[str, str]]
    rag_choice: str
    selected_rag: list[tuple[str, str]]
    context: str
    answer: str

def classify_rag(state: State):
    labels = [label for label, _ in state["rag_candidates"]]
    listing = "\n".join(f"{label}: {text}" for label, text in state["rag_candidates"])
    prompt = (
        "Which of these knowledge base passages, if any, is relevant to the question?\n"
        f"Return only one label from this list, or NONE: {', '.join(labels)}\n\n"
        f"Question: {state['request']}\n\n{listing}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response["message"]["content"].strip().upper()
    valid = {label.upper() for label in labels} | {"NONE"}
    if choice not in valid:
        choice = "NONE"
    return {"rag_choice": choice}
def route_after_classify(state: State):
    return "include_rag" if state["rag_choice"] != "NONE" else "skip_rag"
