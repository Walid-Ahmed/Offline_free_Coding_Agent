# LangGraph graph construction

Project-authored offline documentation notes. Reviewed 2026-11-01 for LangGraph 1.x.
Source: https://docs.langchain.com/oss/python/langgraph/graph-api

## State updates and reducers

StateGraph combines a schema, node functions, and routing edges. Nodes return
updates for selected fields. Without a reducer, an update replaces that field.
For accumulated history, define a reducer or return the entire updated history.
Compile the builder before invoking it. START identifies entry; END terminates.

## A minimal local graph

This original example uses no model or network service:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    task: str
    explanation: str

def explain(state: State):
    return {"explanation": "Inspect locally: " + state["task"]}

builder = StateGraph(State)
builder.add_node("explain", explain)
builder.add_edge(START, "explain")
builder.add_edge("explain", END)
graph = builder.compile()
result = graph.invoke({"task": "main.py", "explanation": ""})
assert result["explanation"] == "Inspect locally: main.py"
```

## Routing and termination

Use add_conditional_edges with a router and explicit destination mapping for
branching. Validate model actions before routing. Graph recursion limits count
super-steps; they are not a count of model replies. CodeHaven separately bounds
model actions. Graph orchestration itself does not isolate executable tools.
