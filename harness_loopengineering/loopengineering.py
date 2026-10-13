from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    attempt: int
    cost: float
    cost_per_attempt: float
    budget: float
    max_attempts: int
    threshold: int
    result: int
    passed: bool
def act(state: State):
    attempt = state["attempt"] + 1
    cost = state["cost"] + state["cost_per_attempt"]
    result = attempt * 5
    return {"attempt": attempt, "cost": cost, "result": result}
def verify(state: State):
    return {"passed": state["result"] >= state["threshold"]}
def route_after_verify(state: State):
    if state["passed"]:
        return "done"
    if state["cost"] >= state["budget"]:
        return "done"
    if state["attempt"] >= state["max_attempts"]:
        return "done"
    return "retry"
graph_builder = StateGraph(State)
graph_builder.add_node("act", act)
graph_builder.add_node("verify", verify)
graph_builder.add_edge(START, "act")
graph_builder.add_edge("act", "verify")
graph_builder.add_conditional_edges(
    "verify", route_after_verify, {"retry": "act", "done": END}
)
graph = graph_builder.compile();
def stop_reason(result):
    if result["passed"]:
        return "verification passed"
    if result["cost"] >= result["budget"]:
        return f"cost budget ({result['budget']}) reached"
    return f"iteration cap ({result['max_attempts']}) reached"


def run(label, initial_state):
    result = graph.invoke(initial_state)
    print(
        f"{label}: attempt={result['attempt']} cost={result['cost']} "
        f"result={result['result']} -> stopped because {stop_reason(result)}"
    )


run("Run A (reachable threshold)", {
    "attempt": 0, "cost": 0.0, "cost_per_attempt": 1.0,
    "budget": 100.0, "max_attempts": 5, "threshold": 10,
})
run("Run B (unreachable threshold, room in budget)", {
    "attempt": 0, "cost": 0.0, "cost_per_attempt": 1.0,
    "budget": 100.0, "max_attempts": 3, "threshold": 1000,
})
run("Run C (unreachable threshold, tight budget)", {
    "attempt": 0, "cost": 0.0, "cost_per_attempt": 10.0,
    "budget": 25.0, "max_attempts": 10, "threshold": 1000,
})