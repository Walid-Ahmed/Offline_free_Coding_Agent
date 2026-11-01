# LangGraph local workflow reference

This is a short project-authored guide, not the complete LangGraph manual.
Primary references for provenance (links are not fetched by the agent):
- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/langsmith/trace-without-env-vars

## StateGraph: state, nodes, and edges

StateGraph defines a workflow using a state schema, node functions, and edges.
A node receives state and returns a dictionary of state updates.
With the default state behavior, an updated field replaces its previous value.
Our agent uses a TypedDict state containing messages, reply, steps, and exit_code.
It returns a new messages list when adding conversation history.

## Compile and invoke

Create a StateGraph, add nodes, connect START to the first node, and connect
terminal paths to END. Call compile() to produce an executable graph, then
invoke(initial_state, config={"recursion_limit": ...}) to execute it locally.

## Conditional edges and tool routing

add_conditional_edges selects the next node from a routing function's result.
Our model node emits a validated JSON action. Read, search, retrieve, write,
and run_tests actions route to separate nodes. Finish routes to END.
Tool results are returned to the conversation before the next model call.

## Action limits and recursion

LangGraph recursion_limit bounds graph execution steps, not model responses.
Our agent separately counts model responses and exits through a limit node.
Internal tool nodes require a larger recursion allowance than the action limit.

## Local privacy and tracing

LangGraph orchestration does not require a hosted graph service or a cloud LLM.
Our application uses a local Ollama transport for qwen2.5-coder:7b.
By default it wraps execution in langsmith.tracing_context(enabled=False), so
inherited tracing settings do not enable remote tracing. Explicitly passing
--langsmith-tracing enables uploads to LangSmith. Inputs and outputs are hidden
unless --langsmith-include-content is also supplied; metadata and errors still
leave the device. Opted-in tracing requires network access and an API key.

## Persistence and approval

This project keeps graph state in memory and does not use a persistent checkpointer.
File writes and test execution require explicit approval through local callbacks.
Do not infer that using LangGraph automatically adds saved sessions, sandboxing,
web browsing, network isolation, or additional tools.
