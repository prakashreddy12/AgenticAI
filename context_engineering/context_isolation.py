from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

MODEL = "gpt-oss:120b-cloud"
class State(TypedDict):
    ticket: str
    triage: str
    network_context: str
    network_result: str
    account_context: str
    account_result: str
    merged_context: str
    answer: str

def triage_ticket(state: State):
    prompt = (
        "A support ticket needs triage. Decide which specialist(s) should look at it.\n"
        "Return only one of: NETWORK, ACCOUNT, BOTH.\n\n"
        f"Ticket: {state['ticket']}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = response["message"]["content"].strip().upper()
    if decision not in ("NETWORK", "ACCOUNT", "BOTH"):
        decision = "BOTH"  # safe default: consulting an extra specialist beats missing one
    return {
        "triage": decision,
        "network_context": "(not consulted)",
        "network_result": "(not consulted)",
        "account_context": "(not consulted)",
        "account_result": "(not consulted)",
    }
def route_after_triage(state: State):
    if state["triage"] == "NETWORK":
        return ["build_network_context"]
    if state["triage"] == "ACCOUNT":
        return ["build_account_context"]
    return ["build_network_context", "build_account_context"]
def build_network_context(state: State):
    context = (
        f"TASK: {state['ticket']}\n"
        "FACT: The office VPN gateway had a scheduled restart at 09:00 today."
    )
    return {"network_context": context}


def ask_network_specialist(state: State):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": state['network_context'] + "\nAnswer in one short sentence."}],
    )
    return {"network_result": response["message"]["content"].strip()}
def build_account_context(state: State):
    context = (
        f"TASK: {state['ticket']}\n"
        "FACT: This user's account was not flagged for any lockouts or password resets today."
    )
    return {"account_context": context}


def ask_account_specialist(state: State):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": state['account_context'] + "\nAnswer in one short sentence."}],
    )
    return {"account_result": response["message"]["content"].strip()}
def merge_results(state: State):
    merged_context = f"NETWORK: {state['network_result']}\nACCOUNT: {state['account_result']}"
    answer = (
        f"Triage: {state['triage']}\n"
        f"Network specialist says: {state['network_result']}\n"
        f"Account specialist says: {state['account_result']}"
    )
    return {"merged_context": merged_context, "answer": answer}
graph_builder = StateGraph(State)
graph_builder.add_node("triage_ticket", triage_ticket)
graph_builder.add_node("build_network_context", build_network_context)
graph_builder.add_node("ask_network_specialist", ask_network_specialist)
graph_builder.add_node("build_account_context", build_account_context)
graph_builder.add_node("ask_account_specialist", ask_account_specialist)
graph_builder.add_node("merge_results", merge_results)

graph_builder.add_edge(START, "triage_ticket")
graph_builder.add_conditional_edges("triage_ticket", route_after_triage)
graph_builder.add_edge("build_network_context", "ask_network_specialist")
graph_builder.add_edge("ask_network_specialist", "merge_results")
graph_builder.add_edge("build_account_context", "ask_account_specialist")
graph_builder.add_edge("ask_account_specialist", "merge_results")
graph_builder.add_edge("merge_results", END)
graph = graph_builder.compile()
result = graph.invoke({"ticket": "User in Building 3 cannot reach internal systems."})
print("Network specialist saw only:\n", result["network_context"])
print("\nAccount specialist saw only:\n", result["account_context"])
print("\nMerged answer:\n", result["answer"])