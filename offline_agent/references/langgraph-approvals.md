# LangGraph interrupts and human approval

Project-authored offline documentation notes. Reviewed 2026-11-01 for LangGraph 1.x.
Source: https://docs.langchain.com/oss/python/langgraph/interrupts

## Pause and resume

interrupt(payload) pauses a node and surfaces a JSON-serializable question.
Configure a checkpointer and a thread ID. Resume using Command(resume=answer)
with the same thread configuration. The supplied answer becomes the interrupt's
return value. With invoke(), the pause payload is exposed under __interrupt__.

## Approval node example

```python
from langgraph.types import Command, interrupt

def approval(state):
    answer = interrupt({"question": "Approve this proposed edit?"})
    return {"approved": answer is True}

# After a checkpoint-enabled graph pauses at approval:
# result = graph.invoke(Command(resume=True), config=config)
```

## Restart-safe side effects

A resumed node begins again from its start. Put approval before irreversible
effects. Make earlier work idempotent so replay does not repeat file writes or
payments. Do not swallow the interrupt with a broad exception handler. Keep
interrupt ordering stable across replays.

## CodeHaven approval boundary

CodeHaven uses synchronous local confirmation callbacks for writes and tests,
rather than LangGraph interrupt/resume persistence. A model request or retrieved
document cannot grant approval. Network-disabled Docker execution and workspace
restrictions remain enforced by their own code, not by the graph library.
