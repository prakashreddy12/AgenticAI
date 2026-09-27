"""Call a plain Python function from a graph node."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    price: float
    quantity: int
    total: float
    answer: str


# This is a normal Python function — our "tool"
def calculate_total(price, quantity):
    return price * quantity


# Graph node
def use_tool(state: State):
    total = calculate_total(
        state["price"],
        state["quantity"]
    )

    return {"total": total}


# Graph node
def create_answer(state: State):
    return {
        "answer": f"The total price is {state['total']}."
    }


# StateGraph now uses our State definition
graph_builder = StateGraph(State)

graph_builder.add_node("use_tool", use_tool)
graph_builder.add_node("answer", create_answer)

graph_builder.add_edge(START, "use_tool")
graph_builder.add_edge("use_tool", "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile()


result = graph.invoke({
    "price": 25,
    "quantity": 3
})

print(result["answer"])


