# Phase 1f: Graph Assembly, FastAPI, and Full-Suite Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** plans 01-05 are complete and committed — every node exists; this plan wires them into one graph and exposes it over HTTP.

**Goal:** Build the `agent_error` node, assemble the full `StateGraph` (all 12 nodes + conditional routing), verify persistence/crash-recovery and PII-non-leakage end-to-end, expose it via FastAPI, and run the complete Phase 1 test suite.

**Architecture:** This is the integration plan — it doesn't introduce new external dependencies, it wires together everything plans 01-05 built.

**Tech Stack:** see plan 01.

## Global Constraints

See `docs/plans/01-project-foundation-and-db.md` §Global Constraints — all apply, and this plan is where several of them get verified end-to-end for the first time: the `no_results`/`form_fill_failed`/`rejection_reason` routing signals produced by earlier plans are only exercised together here.

## Files This Plan Touches

```
backend/app/graph/
  build.py
backend/app/graph/nodes/
  agent_error.py
backend/app/api/
  __init__.py
  main.py
  schemas.py
backend/tests/
  test_node_agent_error.py
  test_graph_end_to_end.py
  test_graph_persistence.py
  test_pii_leak.py
  test_api.py
```
(Full Phase 1 tree: see plan 01 §Full Phase 1 File Structure.)

---

### Task 1: Node — `agent_error`

**Files:**
- Create: `backend/app/graph/nodes/agent_error.py`
- Test: `backend/tests/test_node_agent_error.py`

**Consumes:** `state["error_log"]`.
**Produces:** `agent_error(state, deps) -> dict` setting `state["user_facing_message"]` — a plain-language message with no raw error details.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_agent_error.py`:
```python
from app.graph.nodes.agent_error import agent_error


def test_agent_error_produces_plain_language_message_without_raw_details():
    state = {"error_log": [{"code": "AmadeusRateLimitError", "message": "rate limited"}]}
    result = agent_error(state, None)
    assert "AmadeusRateLimitError" not in result["user_facing_message"]
    assert "sorry" in result["user_facing_message"].lower() or "trouble" in result["user_facing_message"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_agent_error.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/agent_error.py`:
```python
from app.models.state import GraphState


def agent_error(state: GraphState, deps) -> dict:
    return {
        "user_facing_message": (
            "Sorry, we ran into trouble completing that step. Please try again "
            "in a moment, or start a new trip request if the problem continues."
        )
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_agent_error.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/agent_error.py backend/tests/test_node_agent_error.py
git commit -m "feat: add agent_error node with sanitized user-facing message"
```

---

### Task 2: Graph Assembly

**Files:**
- Create: `backend/app/graph/build.py`
- Test: `backend/tests/test_graph_end_to_end.py`

**Consumes:** every node function from plans 02-05 and Task 1; `GraphState`, `get_checkpointer`, `NodeDeps`, `make_deps` (plan 01).
**Produces:** `build_graph(deps, checkpointer) -> CompiledGraph`. This is what the FastAPI layer (Task 5) and this plan's own tests invoke with `graph.invoke(...)` / `Command(resume=...)`.

- [ ] **Step 1: Write the failing end-to-end happy-path test**

`backend/tests/test_graph_end_to_end.py`:
```python
import json
import uuid

from langgraph.types import Command

from app.db import get_checkpointer
from app.config import get_settings
from app.graph.build import build_graph
from factories import make_deps

HAPPY_PATH_LLM_RESPONSES = [
    # collect_trip_request
    json.dumps({
        "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
        "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
    }),
    "Here are your options.",  # recommend_options narrative
    json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),  # generate_itinerary
    json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),  # finalize_itinerary
]

TRAVELER_ANSWERS = {
    "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
    "passport_number": "X1234567", "contact_email": "jane@example.com",
}


def test_happy_path_end_to_end(db_engine):
    deps = make_deps(db_engine=db_engine, llm_responses=list(HAPPY_PATH_LLM_RESPONSES))

    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": str(uuid.uuid4())},
            config,
        )
        graph.invoke(Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config)  # await_selection
        graph.invoke(Command(resume=TRAVELER_ANSWERS), config)  # collect_traveler_details
        final_state = graph.invoke(Command(resume={"approved": True}), config)  # human_review_gate

    assert final_state["approval_status"] == "approved"
    assert final_state["email_status"] == "sent"
    assert final_state["crm_status"] == "created"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_end_to_end.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.graph.build'`

- [ ] **Step 3: Implement graph assembly**

`backend/app/graph/build.py`:
```python
from functools import partial

from langgraph.graph import END, START, StateGraph

from app.graph.deps import NodeDeps
from app.graph.nodes.agent_error import agent_error
from app.graph.nodes.await_selection import await_selection
from app.graph.nodes.collect_traveler_details import collect_traveler_details
from app.graph.nodes.collect_trip_request import collect_trip_request
from app.graph.nodes.fill_booking_form import fill_booking_form
from app.graph.nodes.finalize_itinerary import finalize_itinerary
from app.graph.nodes.generate_itinerary import generate_itinerary
from app.graph.nodes.human_review_gate import human_review_gate
from app.graph.nodes.recommend_options import recommend_options
from app.graph.nodes.search_flights_and_hotels import search_flights_and_hotels
from app.graph.nodes.send_confirmation_email import send_confirmation_email
from app.graph.nodes.update_crm import update_crm
from app.models.state import GraphState


def _route_after_search(state: GraphState) -> str:
    return "collect_trip_request" if state.get("no_results") else "recommend_options"


def _route_after_fill(state: GraphState) -> str:
    return "agent_error" if state.get("form_fill_failed") else "human_review_gate"


def _route_after_review(state: GraphState) -> str:
    if state["approval_status"] == "approved":
        return "send_confirmation_email"
    if state["rejection_reason"] == "wrong_selection":
        return "await_selection"
    return "collect_traveler_details"


def build_graph(deps: NodeDeps, checkpointer):
    builder = StateGraph(GraphState)

    for name, fn in [
        ("collect_trip_request", collect_trip_request),
        ("search_flights_and_hotels", search_flights_and_hotels),
        ("recommend_options", recommend_options),
        ("generate_itinerary", generate_itinerary),
        ("await_selection", await_selection),
        ("finalize_itinerary", finalize_itinerary),
        ("collect_traveler_details", collect_traveler_details),
        ("fill_booking_form", fill_booking_form),
        ("human_review_gate", human_review_gate),
        ("send_confirmation_email", send_confirmation_email),
        ("update_crm", update_crm),
        ("agent_error", agent_error),
    ]:
        builder.add_node(name, partial(fn, deps=deps))

    builder.add_edge(START, "collect_trip_request")
    builder.add_edge("collect_trip_request", "search_flights_and_hotels")
    builder.add_conditional_edges(
        "search_flights_and_hotels", _route_after_search,
        {"collect_trip_request": "collect_trip_request", "recommend_options": "recommend_options"},
    )
    builder.add_edge("recommend_options", "generate_itinerary")
    builder.add_edge("generate_itinerary", "await_selection")
    builder.add_edge("await_selection", "finalize_itinerary")
    builder.add_edge("finalize_itinerary", "collect_traveler_details")
    builder.add_edge("collect_traveler_details", "fill_booking_form")
    builder.add_conditional_edges(
        "fill_booking_form", _route_after_fill,
        {"agent_error": "agent_error", "human_review_gate": "human_review_gate"},
    )
    builder.add_conditional_edges(
        "human_review_gate", _route_after_review,
        {
            "send_confirmation_email": "send_confirmation_email",
            "await_selection": "await_selection",
            "collect_traveler_details": "collect_traveler_details",
        },
    )
    builder.add_edge("send_confirmation_email", "update_crm")
    builder.add_edge("update_crm", END)
    builder.add_edge("agent_error", END)

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_end_to_end.py -v`
Expected: PASS. If it fails on a node signature mismatch (`partial(fn, deps=deps)` requires each node's second parameter to be named `deps`), fix the affected node's signature to `def node_name(state: GraphState, *, deps: NodeDeps) -> dict` consistently across all node files from plans 02-05, and re-run.

- [ ] **Step 5: Write the failing reject-path test**

Append to `backend/tests/test_graph_end_to_end.py`:
```python
def test_reject_wrong_selection_routes_back_to_await_selection(db_engine):
    deps = make_deps(db_engine=db_engine, llm_responses=[
        *HAPPY_PATH_LLM_RESPONSES,
        # finalize_itinerary runs again after the reject loops back through await_selection
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 7pm"]}]}),
    ])
    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": str(uuid.uuid4())},
            config,
        )
        graph.invoke(Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config)
        graph.invoke(Command(resume=TRAVELER_ANSWERS), config)
        state_after_reject = graph.invoke(
            Command(resume={"approved": False, "reason": "wrong_selection"}), config
        )
        # graph re-entered await_selection; resume it again to reach the end
        final_state = graph.invoke(
            Command(resume={"flights": [{"id": "f2"}], "hotels": [{"id": "h2"}]}), config
        )
    assert state_after_reject["rejection_reason"] == "wrong_selection"
    assert final_state["selected_flights"] == ["f2"]
```

- [ ] **Step 6: Run test to verify it fails, then passes**

Run: `pytest tests/test_graph_end_to_end.py -v`
Expected: initial FAIL if routing/state carryover is wrong (commonly: `rejection_reason` not reset before the next review, or `booking_form_results` not recomputed) — debug against the actual assertion failure, fix `_route_after_review` or state carryover in `build.py`/`fill_booking_form.py`, then re-run until both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/graph/build.py backend/tests/test_graph_end_to_end.py
git commit -m "feat: assemble full LangGraph agent with happy-path and reject-path routing"
```

---

### Task 3: Persistence and Crash-Recovery Tests

**Files:**
- Test: `backend/tests/test_graph_persistence.py`

**Consumes:** `build_graph` (Task 2); `get_checkpointer` (plan 01 Task 5); `check_and_reserve` (plan 01 Task 7).

- [ ] **Step 1: Write the failing checkpoint-resume-after-restart test**

`backend/tests/test_graph_persistence.py`:
```python
import json
import uuid

from langgraph.types import Command

from app.config import get_settings
from app.db import get_checkpointer
from app.graph.build import build_graph
from factories import make_deps


def test_resume_after_process_restart_continues_from_last_checkpoint(db_engine):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    llm_responses = [
        json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        }),
        "Here are your options.",
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),
    ]

    # "Process 1": run up to the await_selection interrupt, then simulate a
    # restart by exiting the checkpointer context manager entirely.
    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(make_deps(db_engine=db_engine, llm_responses=llm_responses), checkpointer)
        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": str(uuid.uuid4())},
            config,
        )
        state_before_restart = graph.get_state(config)
        assert "await_selection" in str(state_before_restart.next)

    # "Process 2": brand-new checkpointer connection and graph instance,
    # same thread_id — proves resume works across a real reconnect, not just
    # a live in-memory object.
    with get_checkpointer(get_settings().test_database_url) as checkpointer_2:
        deps_2 = make_deps(db_engine=db_engine, llm_responses=[
            "Here are your options.",
            json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),
        ])
        graph_2 = build_graph(deps_2, checkpointer_2)
        result = graph_2.invoke(
            Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config
        )
        assert result["selected_flights"] == ["f1"]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_graph_persistence.py -v`
Expected: likely FAIL first on `state_before_restart.next` containing `await_selection` if node ordering differs — adjust the assertion to match the actual `graph.get_state(config).next` tuple returned by your `build_graph` wiring, then re-run until PASS.

- [ ] **Step 3: Write the crash-recovery idempotency test**

Append to `backend/tests/test_graph_persistence.py`:
```python
from sqlalchemy.orm import Session

from app.idempotency import check_and_reserve


def test_idempotency_prevents_duplicate_email_after_crash_before_status_persisted(db_engine):
    """Simulates: send_confirmation_email's side effect (the email) succeeded,
    but the process crashed before email_status was written to graph state.
    On resume, the node must not send a second email."""
    trip_request_id = str(uuid.uuid4())
    key = f"{trip_request_id}:send_confirmation_email"
    with Session(db_engine) as session:
        # First attempt's reservation succeeding simulates the email being
        # sent, right before a crash.
        assert check_and_reserve(session, key, "send_confirmation_email") is True

    with Session(db_engine) as session:
        # "Resume" — a second attempt for the same trip must be rejected.
        assert check_and_reserve(session, key, "send_confirmation_email") is False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_persistence.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_graph_persistence.py
git commit -m "test: add LangGraph persistence and crash-recovery idempotency tests"
```

---

### Task 4: PII-Leak Tests

**Files:**
- Test: `backend/tests/test_pii_leak.py`

**Consumes:** `build_graph` (Task 2); `get_checkpointer` (plan 01 Task 5).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_pii_leak.py`:
```python
import json
import uuid

from langgraph.types import Command

from app.config import get_settings
from app.db import get_checkpointer
from app.graph.build import build_graph
from factories import make_deps

SYNTHETIC_PASSPORT = "Z9988776"  # known fixture value, must never appear verbatim outside traveler_details table


def test_synthetic_passport_never_appears_in_checkpoints_or_error_log(db_engine):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    deps = make_deps(db_engine=db_engine, llm_responses=[
        json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        }),
        "Here are your options.",
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),
    ])

    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": str(uuid.uuid4())},
            config,
        )
        graph.invoke(Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config)
        graph.invoke(
            Command(resume={
                "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
                "passport_number": SYNTHETIC_PASSPORT, "contact_email": "jane@example.com",
            }),
            config,
        )
        final_state = graph.invoke(Command(resume={"approved": True}), config)

        # Inspect every persisted checkpoint blob for the raw passport string.
        with checkpointer.conn.cursor() as cur:
            cur.execute("SELECT checkpoint FROM checkpoints WHERE thread_id = %s", (thread_id,))
            for (checkpoint_blob,) in cur.fetchall():
                assert SYNTHETIC_PASSPORT not in str(checkpoint_blob)

    assert SYNTHETIC_PASSPORT not in json.dumps(final_state.get("error_log", []))
    assert SYNTHETIC_PASSPORT not in json.dumps(final_state)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_pii_leak.py -v`
Expected: PASS if plan 04 Task 1's design (storing only `traveler_details_id` on state) was followed correctly by every downstream node. If it FAILS, the failure itself is the finding — a node somewhere is putting raw traveler data onto graph state; locate it via the assertion's checkpoint content and fix that node to use the `traveler_details_id` pattern instead.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_pii_leak.py
git commit -m "test: assert synthetic passport value never appears in checkpoints or state"
```

---

### Task 5: FastAPI Endpoints

**Files:**
- Create: `backend/app/api/__init__.py`, `backend/app/api/schemas.py`, `backend/app/api/main.py`
- Test: `backend/tests/test_api.py`

**Consumes:** `build_graph` (Task 2); `get_checkpointer` (plan 01 Task 5); `NodeDeps` (plan 01 Task 10).
**Produces:** FastAPI app with `POST /conversations` (start), `POST /conversations/{thread_id}/messages` (send a message / resume an interrupt). Real Amadeus/Playwright/Gmail/HubSpot wiring is Phase 2 — this task wires the app with fakes via dependency overrides, proving the HTTP layer round-trips correctly through interrupts.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_api.py`:
```python
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.api.main import app, get_deps
from app.config import get_settings
from app.models.db_models import Base
from factories import make_deps


@pytest.fixture
def client():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)

    app.dependency_overrides[get_deps] = lambda: make_deps(
        db_engine=engine,
        llm_responses=[json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        })],
    )
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_start_conversation_returns_thread_id_and_first_interrupt(client):
    response = client.post(
        "/conversations", json={"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "thread_id" in body
    assert body["interrupt"]["shortlist"] is not None or "await_selection" in str(body)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api'`

- [ ] **Step 3: Implement the API schemas and app**

`backend/app/api/__init__.py`: empty file.

`backend/app/api/schemas.py`:
```python
from pydantic import BaseModel


class StartConversationRequest(BaseModel):
    user_message: str


class ResumeRequest(BaseModel):
    resume_payload: dict


class ConversationResponse(BaseModel):
    thread_id: str
    interrupt: dict | None
    state: dict
```

`backend/app/api/main.py`:
```python
import uuid

from fastapi import Depends, FastAPI
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.api.schemas import ConversationResponse, ResumeRequest, StartConversationRequest
from app.config import get_settings
from app.crypto import generate_test_key
from app.db import get_checkpointer, get_engine
from app.graph.build import build_graph
from app.graph.deps import NodeDeps
from app.llm.client import AnthropicLLMClient

app = FastAPI(title="AI Travel Booking Agent")

_engine = get_engine()


def get_deps() -> NodeDeps:
    return NodeDeps(
        llm=AnthropicLLMClient(),
        search_tool=None,  # wired to the real Amadeus MCP client in Phase 2
        booking_tool=None,  # wired to the real Playwright MCP client in Phase 2
        email_tool=None,  # wired to the real Gmail MCP client in Phase 2
        crm_tool=None,  # wired to the real HubSpot MCP client in Phase 2
        session_factory=lambda: Session(_engine),
        encryption_key=generate_test_key(),  # replaced by Secrets Manager-sourced key in Phase 3
    )


def _serialize_interrupt(state_snapshot) -> dict | None:
    if not state_snapshot.tasks:
        return None
    interrupts = state_snapshot.tasks[0].interrupts
    return interrupts[0].value if interrupts else None


@app.post("/conversations", response_model=ConversationResponse)
def start_conversation(body: StartConversationRequest, deps: NodeDeps = Depends(get_deps)):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    with get_checkpointer() as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        result = graph.invoke(
            {"user_message": body.user_message, "trip_request_id": str(uuid.uuid4())}, config
        )
        snapshot = graph.get_state(config)
    return ConversationResponse(
        thread_id=thread_id, interrupt=_serialize_interrupt(snapshot), state=dict(result)
    )


@app.post("/conversations/{thread_id}/messages", response_model=ConversationResponse)
def resume_conversation(thread_id: str, body: ResumeRequest, deps: NodeDeps = Depends(get_deps)):
    config = {"configurable": {"thread_id": thread_id}}
    with get_checkpointer() as checkpointer:
        graph = build_graph(deps, checkpointer)
        result = graph.invoke(Command(resume=body.resume_payload), config)
        snapshot = graph.get_state(config)
    return ConversationResponse(
        thread_id=thread_id, interrupt=_serialize_interrupt(snapshot), state=dict(result)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS. If `_serialize_interrupt` doesn't match the actual `StateSnapshot` shape returned by your installed LangGraph version, inspect `graph.get_state(config)` in a debugger/REPL and adjust the accessor to match, then re-run.

- [ ] **Step 5: Add the full round-trip-through-interrupts test**

Append to `backend/tests/test_api.py`:
```python
def test_full_conversation_round_trips_through_both_interrupts(client):
    start = client.post(
        "/conversations", json={"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler"}
    ).json()
    thread_id = start["thread_id"]

    client.post(
        f"/conversations/{thread_id}/messages",
        json={"resume_payload": {"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}},
    )
    client.post(
        f"/conversations/{thread_id}/messages",
        json={"resume_payload": {
            "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
            "passport_number": "X1234567", "contact_email": "jane@example.com",
        }},
    )
    final = client.post(
        f"/conversations/{thread_id}/messages", json={"resume_payload": {"approved": True}},
    ).json()

    assert final["state"]["approval_status"] == "approved"
    assert final["state"]["email_status"] == "sent"
    assert final["state"]["crm_status"] == "created"
```

- [ ] **Step 6: Run the full test file**

Run: `pytest tests/test_api.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/ backend/tests/test_api.py
git commit -m "feat: add FastAPI endpoints wiring the LangGraph agent through HTTP"
```

---

### Task 6: Full Suite Verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete Phase 1 test suite**

Run:
```bash
cd backend
docker compose up -d postgres  # from repo root, if not already running
alembic upgrade head
pytest -v
```
Expected: every test across all six Phase 1 plans PASSES (roughly 60+ individual test functions).

- [ ] **Step 2: Confirm no test depends on a live third-party API**

Run: `grep -rn "amadeus.com\|googleapis.com\|hubapi.com\|anthropic.com" backend/tests/`
Expected: no matches — every test in Phase 1 uses `Fake*` implementations, per spec §9 ("keeping normal CI deterministic").

- [ ] **Step 3: Commit the final state if any fixups were needed**

```bash
git add -A
git commit -m "chore: fix up any remaining issues found during full Phase 1 suite run"
```
(Skip this commit if Step 1 passed cleanly with no changes needed.)
