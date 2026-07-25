# Phase 1c: Selection and Interrupts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** `docs/plans/01-project-foundation-and-db.md` and `docs/plans/02-langgraph-trip-search-flow.md` are complete and committed.

**Goal:** Build the first `interrupt()` gate (`await_selection`) and the node that runs immediately after it (`finalize_itinerary`), which regenerates the itinerary using the traveler's actual chosen flight/hotel rather than the preliminary guess from plan 02.

**Architecture/Tech Stack:** see plan 01.

## Global Constraints

See `docs/plans/01-project-foundation-and-db.md` §Global Constraints. Most relevant here: this is the first of the two `interrupt()` gates (spec §5.2) — the graph pauses here and resumes only when the frontend submits a selection.

## Files This Plan Touches

```
backend/app/graph/nodes/
  await_selection.py
  finalize_itinerary.py
backend/tests/
  test_node_await_selection.py
  test_node_finalize_itinerary.py
```
(Full Phase 1 tree: see plan 01 §Full Phase 1 File Structure.)

---

### Task 1: Node — `await_selection`

**Files:**
- Create: `backend/app/graph/nodes/await_selection.py`
- Test: `backend/tests/test_node_await_selection.py`

**Consumes:** `interrupt` from `langgraph.types`; `state["shortlist"]` (plan 02 Task 2).
**Produces:** `await_selection(state, deps) -> dict` setting `selected_flights`, `selected_hotels` from the resume payload — the pause point where a human picks options.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_await_selection.py`:
```python
from app.graph.nodes.await_selection import await_selection


def test_await_selection_returns_resumed_choice(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.await_selection.interrupt",
        lambda payload: {"flights": ["f1"], "hotels": ["h1"]},
    )
    state = {"shortlist": {"0": {"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}}}
    result = await_selection(state, None)
    assert result["selected_flights"] == ["f1"]
    assert result["selected_hotels"] == ["h1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_await_selection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/await_selection.py`:
```python
from langgraph.types import interrupt

from app.models.state import GraphState


def await_selection(state: GraphState, deps) -> dict:
    selection = interrupt({"shortlist": state["shortlist"]})
    return {
        "selected_flights": selection["flights"],
        "selected_hotels": selection["hotels"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_await_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/await_selection.py backend/tests/test_node_await_selection.py
git commit -m "feat: add await_selection interrupt node"
```

---

### Task 2: Node — `finalize_itinerary`

**Files:**
- Create: `backend/app/graph/nodes/finalize_itinerary.py`
- Test: `backend/tests/test_node_finalize_itinerary.py`

**Consumes:** `NodeDeps`, `make_deps` (plan 01); `state["itinerary"]` (plan 02 Task 4), `state["selected_flights"]`/`state["selected_hotels"]` (Task 1).
**Produces:** `finalize_itinerary(state, deps) -> dict` setting an updated `state["itinerary"]` that incorporates the actual selected flight times/hotel locations.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_finalize_itinerary.py`:
```python
import json

import pytest

from app.graph.nodes.finalize_itinerary import finalize_itinerary
from app.models.domain import Itinerary
from factories import make_deps


@pytest.mark.asyncio
async def test_finalize_itinerary_incorporates_selected_options():
    preliminary = Itinerary(days=[{"day_index": 0, "date": "2026-09-01", "activities": ["Louvre"]}])
    canned = json.dumps({"days": [
        {"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm", "Check in", "Louvre"]}
    ]})
    deps = make_deps(llm_responses=[canned])
    state = {
        "itinerary": preliminary,
        "selected_flights": [{"id": "f1", "arrive_at": "2026-09-01T18:00:00Z"}],
        "selected_hotels": [{"id": "h1", "name": "Fake Grand Hotel"}],
    }
    result = await finalize_itinerary(state, deps)
    assert "Arrive at 6pm" in result["itinerary"].days[0].activities
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_finalize_itinerary.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/finalize_itinerary.py`:
```python
import json

from app.errors import LLMRateLimitError, LLMTimeoutError, with_retries
from app.graph.deps import NodeDeps
from app.models.domain import Itinerary
from app.models.state import GraphState

_SYSTEM_PROMPT = (
    "Revise this preliminary itinerary to incorporate the traveler's actual "
    "selected flight arrival times and hotel location. Respond with JSON only, "
    "same shape as the input."
)


async def finalize_itinerary(state: GraphState, deps: NodeDeps) -> dict:
    payload = {
        "preliminary_itinerary": state["itinerary"].model_dump(mode="json"),
        "selected_flights": state["selected_flights"],
        "selected_hotels": state["selected_hotels"],
    }
    raw = await with_retries(
        lambda: deps.llm.complete(system=_SYSTEM_PROMPT, prompt=json.dumps(payload)),
        exceptions=(LLMTimeoutError, LLMRateLimitError),
    )
    return {"itinerary": Itinerary(**json.loads(raw))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_finalize_itinerary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/finalize_itinerary.py backend/tests/test_node_finalize_itinerary.py
git commit -m "feat: add finalize_itinerary node"
```
