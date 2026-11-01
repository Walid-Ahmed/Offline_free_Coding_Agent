# LangGraph checkpointing and persistence

Project-authored offline documentation notes. Reviewed 2026-11-01 for LangGraph 1.x.
Source: https://docs.langchain.com/oss/python/langgraph/persistence

## Checkpointer versus store

A checkpointer records execution state for a specific thread. A store holds
application data that can be shared across threads. Neither is automatically a
search index over repository files. CodeHaven's local BM25 knowledge base is a
separate retrieval component.

## In-process checkpoint example

Given an existing StateGraph builder:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "local-review-1"}}
result = graph.invoke(initial_state, config=config)
snapshot = graph.get_state(config)
```

InMemorySaver loses its data when the process exits. Reusing a thread ID selects
that thread's checkpoints; a different ID addresses a different thread. Durable
storage requires a separately configured persistent backend. Choose local storage
and protect its files if prompts or source code must stay on the device.

## CodeHaven implementation boundary

CodeHaven currently compiles without a checkpointer or store. Its conversation
state and retrieval index live in memory for the current run. Reading these notes
does not enable saved sessions, durable memory, or automatic resumption. Adding
those features requires an implementation change and an explicit storage policy.
