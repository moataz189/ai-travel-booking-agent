# Phase 1d: Traveler PII and Booking Form — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** plans 01-03 are complete and committed.

**Goal:** Build the three most safety-sensitive nodes in the whole graph: collecting and encrypting traveler PII, filling the (fake, for now) booking form once per leg, and the second `interrupt()` gate where a human reviews the filled form before anything downstream happens.

**Architecture/Tech Stack:** see plan 01.

## Global Constraints

See `docs/plans/01-project-foundation-and-db.md` §Global Constraints. Most relevant here:
- Traveler PII (passport/ID number, DOB) is never stored directly on graph state — only a `traveler_details_id` foreign key (spec §6.1). This plan is where that boundary is actually built and must hold.
- `fill_booking_form` deliberately never retries — form-fill failures route to a dedicated failure state instead (spec §5.2/§7).
- `human_review_gate` covers **all legs** of a multi-leg trip in a single combined decision (spec §5.2, node 9); `fill_booking_form` loops once per leg.
- `rejection_reason` (`wrong_selection` | `wrong_traveler_details`) drives reject-routing, wired up in plan 06.

## Files This Plan Touches

```
backend/app/graph/nodes/
  collect_traveler_details.py
  fill_booking_form.py
  human_review_gate.py
backend/tests/
  test_node_collect_traveler_details.py
  test_node_fill_booking_form.py
  test_node_human_review_gate.py
```
(Full Phase 1 tree: see plan 01 §Full Phase 1 File Structure.)

---

### Task 1: Node — `collect_traveler_details`

**Files:**
- Create: `backend/app/graph/nodes/collect_traveler_details.py`
- Test: `backend/tests/test_node_collect_traveler_details.py`

**Consumes:** `interrupt`; `TravelerDetailRecord` (plan 01 Task 4); `encrypt_field` (plan 01 Task 3); `NodeDeps.session_factory`/`encryption_key`, `make_deps` (plan 01 Task 10).
**Produces:** `collect_traveler_details(state, deps) -> dict` setting only `state["traveler_details_id"]` — never raw traveler fields on state (spec §6.1). This is the boundary every later node (`fill_booking_form`, `send_confirmation_email`, `update_crm`) relies on: they look up traveler data by id, they never receive it via graph state.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_collect_traveler_details.py`:
```python
import uuid

from sqlalchemy.orm import Session

from app.crypto import decrypt_field, generate_test_key
from app.graph.nodes.collect_traveler_details import collect_traveler_details
from app.models.db_models import TravelerDetailRecord, TripRequestRecord
from factories import make_deps


def test_collect_traveler_details_stores_encrypted_and_returns_only_id(db_engine, monkeypatch):
    key = generate_test_key()
    trip_id = str(uuid.uuid4())
    with Session(db_engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        session.commit()

    monkeypatch.setattr(
        "app.graph.nodes.collect_traveler_details.interrupt",
        lambda payload: {
            "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
            "passport_number": "X1234567", "contact_email": "jane@example.com",
        },
    )
    deps = make_deps(db_engine=db_engine, encryption_key=key)
    result = collect_traveler_details({"trip_request_id": trip_id}, deps)

    assert set(result.keys()) == {"traveler_details_id"}
    assert "passport_number" not in result
    assert "full_name" not in result

    with Session(db_engine) as session:
        record = session.get(TravelerDetailRecord, result["traveler_details_id"])
        assert decrypt_field(record.passport_number_enc, key) == "X1234567"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_collect_traveler_details.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/collect_traveler_details.py`:
```python
from langgraph.types import interrupt

from app.crypto import encrypt_field
from app.graph.deps import NodeDeps
from app.models.db_models import TravelerDetailRecord
from app.models.state import GraphState


def collect_traveler_details(state: GraphState, deps: NodeDeps) -> dict:
    answers = interrupt({"ask_for": "traveler_details"})

    record = TravelerDetailRecord(
        trip_request_id=state["trip_request_id"],
        full_name=answers["full_name"],
        date_of_birth_enc=encrypt_field(answers["date_of_birth"], deps.encryption_key),
        passport_number_enc=encrypt_field(answers["passport_number"], deps.encryption_key),
        contact_email=answers["contact_email"],
    )
    with deps.session_factory() as session:
        session.add(record)
        session.commit()
        traveler_details_id = record.id

    return {"traveler_details_id": traveler_details_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_collect_traveler_details.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/collect_traveler_details.py backend/tests/test_node_collect_traveler_details.py
git commit -m "feat: add collect_traveler_details node storing only encrypted PII"
```

---

### Task 2: Node — `fill_booking_form`

**Files:**
- Create: `backend/app/graph/nodes/fill_booking_form.py`
- Test: `backend/tests/test_node_fill_booking_form.py`

**Consumes:** `NodeDeps`, `make_deps`, `make_traveler_record` (plan 01); `TravelerDetailRecord`, `decrypt_field`; `PlaywrightTimeoutError`/`sanitize_error` — retry is deliberately **not** used here (form-fill failures route to a dedicated failure state, not a blind retry).
**Produces:** `fill_booking_form(state, deps) -> dict` setting `booking_form_results` (list, one per leg) or `{"form_fill_failed": True, "error_log": [...]}`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_fill_booking_form.py`:
```python
import pytest

from app.graph.nodes.fill_booking_form import fill_booking_form
from factories import make_deps, make_traveler_record


@pytest.mark.asyncio
async def test_fill_booking_form_loops_once_per_leg(db_engine):
    traveler_id = make_traveler_record(db_engine)
    deps = make_deps(db_engine=db_engine, booking_behavior="success")
    state = {
        "traveler_details_id": traveler_id,
        "selected_flights": [{"id": "f1"}, {"id": "f2"}],
        "selected_hotels": [{"id": "h1"}, {"id": "h2"}],
    }
    result = await fill_booking_form(state, deps)
    assert len(result["booking_form_results"]) == 2
    assert result["booking_form_results"][0]["leg_index"] == 0
    assert result["booking_form_results"][1]["leg_index"] == 1


@pytest.mark.asyncio
async def test_fill_booking_form_failure_does_not_retry_and_sets_flag(db_engine):
    traveler_id = make_traveler_record(db_engine)
    deps = make_deps(db_engine=db_engine, booking_behavior="failure")
    state = {
        "traveler_details_id": traveler_id,
        "selected_flights": [{"id": "f1"}],
        "selected_hotels": [{"id": "h1"}],
        "error_log": [],
    }
    result = await fill_booking_form(state, deps)
    assert result["form_fill_failed"] is True
    assert result["error_log"][0]["code"] == "PlaywrightTimeoutError"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_fill_booking_form.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/fill_booking_form.py`:
```python
from app.crypto import decrypt_field
from app.errors import PlaywrightTimeoutError, sanitize_error
from app.graph.deps import NodeDeps
from app.models.db_models import TravelerDetailRecord
from app.models.state import GraphState


async def fill_booking_form(state: GraphState, deps: NodeDeps) -> dict:
    with deps.session_factory() as session:
        record = session.get(TravelerDetailRecord, state["traveler_details_id"])
        traveler = {
            "full_name": record.full_name,
            "date_of_birth": decrypt_field(record.date_of_birth_enc, deps.encryption_key),
            "passport_number": decrypt_field(record.passport_number_enc, deps.encryption_key),
            "contact_email": record.contact_email,
        }

    results = []
    for leg_index, (flight, hotel) in enumerate(
        zip(state["selected_flights"], state["selected_hotels"])
    ):
        try:
            leg_result = await deps.booking_tool.fill_leg(
                leg_index=leg_index, flight=flight, hotel=hotel, traveler=traveler
            )
        except PlaywrightTimeoutError as exc:
            error_log = state.get("error_log", []) + [sanitize_error(exc)]
            return {"form_fill_failed": True, "error_log": error_log}
        results.append({"leg_index": leg_index, **leg_result})

    return {"booking_form_results": results, "form_fill_failed": False}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_fill_booking_form.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/fill_booking_form.py backend/tests/test_node_fill_booking_form.py
git commit -m "feat: add fill_booking_form node with per-leg loop and no blind retry"
```

---

### Task 3: Node — `human_review_gate`

**Files:**
- Create: `backend/app/graph/nodes/human_review_gate.py`
- Test: `backend/tests/test_node_human_review_gate.py`

**Consumes:** `interrupt`; `state["booking_form_results"]` (Task 2).
**Produces:** `human_review_gate(state, deps) -> dict` setting `approval_status` and `rejection_reason`. This gate approves the filled mock forms and downstream notification/CRM actions — **not a real booking or payment**, neither of which exist anywhere in this system (spec §5.2, node 9).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_human_review_gate.py`:
```python
from app.graph.nodes.human_review_gate import human_review_gate


def test_approve_sets_approved_status_and_no_reason(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.human_review_gate.interrupt", lambda payload: {"approved": True}
    )
    result = human_review_gate({"booking_form_results": []}, None)
    assert result == {"approval_status": "approved", "rejection_reason": None}


def test_reject_with_wrong_selection_sets_reason(monkeypatch):
    monkeypatch.setattr(
        "app.graph.nodes.human_review_gate.interrupt",
        lambda payload: {"approved": False, "reason": "wrong_selection"},
    )
    result = human_review_gate({"booking_form_results": []}, None)
    assert result == {"approval_status": "rejected", "rejection_reason": "wrong_selection"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_human_review_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/human_review_gate.py`:
```python
from langgraph.types import interrupt

from app.models.state import GraphState


def human_review_gate(state: GraphState, deps) -> dict:
    decision = interrupt({"booking_form_results": state["booking_form_results"]})
    if decision["approved"]:
        return {"approval_status": "approved", "rejection_reason": None}
    return {"approval_status": "rejected", "rejection_reason": decision["reason"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_human_review_gate.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/human_review_gate.py backend/tests/test_node_human_review_gate.py
git commit -m "feat: add human_review_gate node with rejection_reason routing signal"
```
