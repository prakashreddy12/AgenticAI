from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    route: str
    request: str
    answer: str

def choose_route(state: State):
    if "password" in state["request"].lower():
        return {"route":"account"}
    return {"route":"general"}

def next_node(state: State):
    return state["route"]

def account_help(state: State):
    return {"answer":"open a password page"}

def general_help(state: State):
    return {"answer": "Support will assist"}

graph_builder = StateGraph(State)
graph_builder.add_node("route", choose_route)
#graph_builder.add_node("next_node", next_node)
graph_builder.add_node("account", account_help)
graph_builder.add_node("general", general_help)

graph_builder.add_edge(START, "route")
graph_builder.add_conditional_edges("route", next_node)
#graph_builder.add_edge("next_node", "account")
#graph_builder.add_edge("next_node", "general")
graph_builder.add_edge("account", END)
graph_builder.add_edge("general", END)

graph = graph_builder.compile()

response = graph.invoke({"request":"How to reset VPN?"})

print(response["answer"])
print(response)

