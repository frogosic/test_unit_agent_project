# Agentic Unit Test Generator

A multi-agent system that autonomously generates pytest unit tests from Python source code. Built as an exploration of agentic AI architecture — not a production tool, but a deliberate exercise in designing systems where LLMs make real decisions.

---

## What it does

Point it at a Python file or directory. It discovers source files, designs structured test suites, generates pytest code, validates it by actually running pytest, and retries with error feedback if tests fail. The result is a set of test files written to a mirrored `tests/` directory.

```bash
python main.py
# Which directory should I inspect for unit test generation? game/core/memory.py
```

---

## Architecture

The system is built around the **GAME framework** (Goals, Actions, Memory, Environment) — a lightweight agentic pattern where each agent has explicit goals, a registry of callable actions, and a memory of its conversation history.

### Agent pipeline

```
CoordinatorAgent
├── FileOpsAgent          — discovers and reads Python source files
├── TestDesignAgent       — analyzes source code, designs structured test scenarios
└── TestWritingAgent      — generates pytest code, validates via execution
```

**CoordinatorAgent** runs a two-phase approach:
1. **Plan** — delegates to `FileOpsAgent` to discover files, builds a lightweight plan containing file paths only
2. **Execute** — processes each file deterministically via specialist agents

This separation is intentional. The coordinator never holds source code in memory — only file paths. Each specialist fetches its own content via `read_file`. This avoids context explosion when processing directories.

**FileOpsAgent** navigates the filesystem autonomously. It decides which files are relevant, reads them, and stops when done. Genuinely agentic — the LLM decides the traversal strategy.

**TestDesignAgent** reads the source file, consults a `prompt_expert` action for domain expertise, and produces a structured `TestDesignResult` with typed test targets and scenarios.

**TestWritingAgent** reads the source file, receives the structured design, generates pytest code, and calls a terminal action to submit it. The generated code is validated by actually running pytest in a subprocess — if tests fail, the agent retries with the full error output as context.

---

## Key design decisions

### ActionContext and delegation tracking

Every agent run receives an `ActionContext` — an immutable-ish context object that propagates through the delegation chain. It tracks:

- Current agent name
- Delegation depth and path
- Visited agents (loop detection)
- File security policy
- Agent registry

When one agent delegates to another, `spawn_delegated_child()` creates a new context with incremented depth and updated path. This makes delegation auditable and prevents infinite loops at the infrastructure level.

### AgentRegistry and permission graph

Agents are registered by name. Delegation permissions are explicit — the coordinator can call specialists, but specialists cannot call each other. `require_can_call()` enforces this at runtime.

### Validation in the execution layer

Test validation doesn't happen after the agent finishes — it happens inside `_execute_tool_calls` when the terminal action fires. If pytest fails, the tool result is replaced with the error output and the loop continues. The LLM sees the failure and retries.

This means validation works regardless of how the agent is invoked — directly via `run_and_parse` or through the delegation chain.

### Planning over pure LLM orchestration

An earlier version used the coordinator LLM to drive the entire pipeline via `call_agent` tool calls. It worked for single files but broke on directories — passing full source code through LLM tool call JSON caused encoding failures and context explosion.

The planning system solves this by making the coordinator's LLM job small: discover files, build a plan. Execution is deterministic Python. The agentic behavior lives where it belongs — in the specialists that reason about code.

---

## Project structure

```
game/
├── actions/          — callable actions (file ops, result returns, expert prompting)
├── agents/           — CoordinatorAgent, FileOpsAgent, TestDesignAgent, TestWritingAgent
├── bootstrap/        — system assembly, action registries, decorator-based action registration
├── core/             — BaseAgent, ActionContext, AgentRegistry, Environment, LLM, Memory
├── languages/        — ToolCallingAgentLanguage (prompt construction, tool call parsing)
├── models/           — typed data models (SourceFile, TestDesignResult, GeneratedTestFile, etc.)
├── policies/         — FileSecurityPolicy (path traversal protection)
└── services/         — file writer, pytest runner/validator
```

---

## What was learned

**Agentic architecture requires honest scoping.** The more structured the output contract, the less room there is for agent autonomy. `TestWritingAgent` needs to produce valid Python —> that constraint naturally pushes toward deterministic validation rather than open-ended generation.

**Context is the hard problem.** LLMs don't loop over lists. Passing large payloads through tool call arguments breaks JSON encoding. The planning system —> lightweight references, self-sufficient specialists —> emerged from hitting these limits, not from upfront design.

**Retry without feedback is useless.** Early retry logic just re-ran the same prompt and got the same result. Passing the full pytest output as context made retries actually productive.

**Validation belongs in the execution layer, not the parsing layer.** Moving pytest validation into `_execute_tool_calls` meant it fired regardless of how the agent was invoked —> a small architectural decision with a large practical impact.

---

## What this is NOT

This is a scaffolding project that does just the minimum. It does not:
- Use a vector store or RAG for codebase context
- Have persistent memory across runs
- Scale to large codebases without modification
- Guarantee test correctness — only test syntax validity and execution

---

## Stack

- **LiteLLM** — LLM abstraction (OpenAI GPT-4o-mini by default)
- **pytest** — test execution and validation
- Python 3.12

---

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# add OPENAI_API_KEY to .env
python main.py
```
