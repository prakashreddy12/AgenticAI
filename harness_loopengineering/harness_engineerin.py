from typing import TypedDict

from langgraph.graph import END, START, StateGraph

ALLOWED_ACTIONS = {"lookup_order", "check_shipping_status"}


class State(TypedDict):
    requested_action: str
    trace: list
    allowed: bool
    output: str
def check_permission(state: State):
    allowed = state["requested_action"] in ALLOWED_ACTIONS
    if allowed:
        note = f"check_permission: '{state['requested_action']}' is allowed"
    else:
        note = f"check_permission: '{state['requested_action']}' is blocked"
    trace = state["trace"] + [note]
    return {"trace": trace, "allowed": allowed}
def route_after_permission(state: State):
    return "execute" if state["allowed"] else "reject"
def execute_action(state: State):
    trace = state["trace"] + [f"execute_action: ran '{state['requested_action']}'"]
    return {"trace": trace, "output": f"'{state['requested_action']}' completed."}


def reject_action(state: State):
    trace = state["trace"] + [
        f"reject_action: refused '{state['requested_action']}' (not on the allowed list)"
    ]
    return {
        "trace": trace,
        "output": f"'{state['requested_action']}' was blocked by the harness.",
    }
graph_builder = StateGraph(State)
graph_builder.add_node("check_permission", check_permission)
graph_builder.add_node("execute_action", execute_action)
graph_builder.add_node("reject_action", reject_action)
graph_builder.add_edge(START, "check_permission")
graph_builder.add_conditional_edges(
    "check_permission",
    route_after_permission,
    {"execute": "execute_action", "reject": "reject_action"},
)
graph_builder.add_edge("execute_action", END)
graph_builder.add_edge("reject_action", END)
graph = graph_builder.compile()
def run(requested_action):
    result = graph.invoke({"requested_action": requested_action, "trace": []})
    print(f"Requested action: {requested_action}")
    for line in result["trace"]:
        print(f"  - {line}")
    print(f"Output: {result['output']}\n")


run("lookup_order")
run("issue_refund")