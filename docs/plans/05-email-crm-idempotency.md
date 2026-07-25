# Phase 1e: Email, CRM, Idempotency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** plans 01-04 are complete and committed — this plan uses `check_and_reserve` (plan 01 Task 7) and looks up traveler contact info via `TravelerDetailRecord` (plan 01 Task 4), set by `collect_traveler_details` (plan 04 Task 1).

**Goal:** Build the two downstream side-effect nodes that run after human approval — sending the confirmation email and writing the trip to the CRM — both made idempotent so a retry after a crash can never double-send or double-create.

**Architecture/Tech Stack:** see plan 01.

## Global Constraints

See `docs/plans/01-project-foundation-and-db.md` §Global Constraints. Most relevant here: both nodes are idempotent via a stored idempotency key derived from the trip/session ID (spec §7) — this is what prevents a duplicate email/CRM record if the process crashes after the side effect succeeds but before its status is persisted (verified in plan 06's crash-recovery tests).

## Files This Plan Touches

```
backend/app/graph/nodes/
  send_confirmation_email.py
  update_crm.py
backend/tests/
  test_node_send_confirmation_email.py
  test_node_update_crm.py
```
(Full Phase 1 tree: see plan 01 §Full Phase 1 File Structure.)

---

### Task 1: Node — `send_confirmation_email`

**Files:**
- Create: `backend/app/graph/nodes/send_confirmation_email.py`
- Test: `backend/tests/test_node_send_confirmation_email.py`

**Consumes:** `NodeDeps`, `make_deps`, `make_traveler_record` (plan 01); `check_and_reserve` (plan 01 Task 7); `GmailSendError`/`with_retries` (plan 01 Task 6); `TravelerDetailRecord` looked up via `state["traveler_details_id"]` for the recipient's name/email (plaintext columns — not the encrypted PII fields).
**Produces:** `send_confirmation_email(state, deps) -> dict` setting `email_status`, idempotent on `state["trip_request_id"]`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_send_confirmation_email.py`:
```python
import uuid

import pytest

from app.graph.nodes.send_confirmation_email import send_confirmation_email
from app.models.domain import Itinerary
from factories import make_deps, make_traveler_record


def make_state(trip_id, traveler_id):
    return {
        "trip_request_id": trip_id,
        "traveler_details_id": traveler_id,
        "itinerary": Itinerary(days=[{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]),
    }


@pytest.mark.asyncio
async def test_send_confirmation_email_sends_once(db_engine):
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler_record(db_engine, trip_id=trip_id)
    deps = make_deps(db_engine=db_engine, email_behavior="success")

    result = await send_confirmation_email(make_state(trip_id, traveler_id), deps)
    assert result["email_status"] == "sent"
    assert len(deps.email_tool.sent) == 1
    assert deps.email_tool.sent[0]["to"] == "jane@example.com"


@pytest.mark.asyncio
async def test_send_confirmation_email_is_idempotent_on_retry(db_engine):
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler_record(db_engine, trip_id=trip_id)
    deps = make_deps(db_engine=db_engine, email_behavior="success")
    state = make_state(trip_id, traveler_id)

    await send_confirmation_email(state, deps)
    result = await send_confirmation_email(state, deps)  # simulated retry after crash
    assert result["email_status"] == "already_sent"
    assert len(deps.email_tool.sent) == 1  # not sent twice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_send_confirmation_email.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/send_confirmation_email.py`:
```python
from app.errors import GmailSendError, sanitize_error, with_retries
from app.graph.deps import NodeDeps
from app.idempotency import check_and_reserve
from app.models.db_models import TravelerDetailRecord
from app.models.state import GraphState


def _render_itinerary_summary(state: GraphState) -> str:
    lines = [
        f"Day {day.day_index + 1} ({day.date}): {', '.join(day.activities)}"
        for day in state["itinerary"].days
    ]
    return "\n".join(lines)


async def send_confirmation_email(state: GraphState, deps: NodeDeps) -> dict:
    key = f"{state['trip_request_id']}:send_confirmation_email"
    with deps.session_factory() as session:
        if not check_and_reserve(session, key, "send_confirmation_email"):
            return {"email_status": "already_sent"}
        traveler = session.get(TravelerDetailRecord, state["traveler_details_id"])
        recipient_email, recipient_name = traveler.contact_email, traveler.full_name

    try:
        await with_retries(
            lambda: deps.email_tool.send_email(
                to=recipient_email, subject="Your trip itinerary",
                body=f"Hi {recipient_name},\n\n{_render_itinerary_summary(state)}",
            ),
            exceptions=(GmailSendError,),
        )
    except GmailSendError as exc:
        error_log = state.get("error_log", []) + [sanitize_error(exc)]
        return {"email_status": "failed", "error_log": error_log}

    return {"email_status": "sent"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_send_confirmation_email.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/send_confirmation_email.py backend/tests/test_node_send_confirmation_email.py
git commit -m "feat: add idempotent send_confirmation_email node"
```

---

### Task 2: Node — `update_crm`

**Files:**
- Create: `backend/app/graph/nodes/update_crm.py`
- Test: `backend/tests/test_node_update_crm.py`

**Consumes:** `NodeDeps`, `make_deps`, `make_traveler_record` (plan 01); `check_and_reserve` (plan 01 Task 7); `HubSpotError`/`with_retries` (plan 01 Task 6); `TravelerDetailRecord`; `state["trip_request"]` for the trip summary.
**Produces:** `update_crm(state, deps) -> dict` setting `crm_status`, idempotent on `state["trip_request_id"]`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_update_crm.py`:
```python
import uuid

import pytest

from app.graph.nodes.update_crm import update_crm
from app.models.domain import TripRequest
from factories import make_deps, make_traveler_record


def make_state(trip_id, traveler_id):
    return {
        "trip_request_id": trip_id,
        "traveler_details_id": traveler_id,
        "trip_request": TripRequest(
            origin="JFK", destinations=["CDG"], depart_date="2026-09-01",
            return_date="2026-09-05", budget_usd=2000, traveler_count=1, preferences=[],
        ),
    }


@pytest.mark.asyncio
async def test_update_crm_creates_contact_and_trip_record_once(db_engine):
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler_record(db_engine, trip_id=trip_id)
    deps = make_deps(db_engine=db_engine, crm_behavior="success")
    state = make_state(trip_id, traveler_id)

    result = await update_crm(state, deps)
    assert result["crm_status"] == "created"
    assert len(deps.crm_tool.trip_records) == 1

    retry_result = await update_crm(state, deps)
    assert retry_result["crm_status"] == "already_created"
    assert len(deps.crm_tool.trip_records) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_update_crm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/update_crm.py`:
```python
from app.errors import HubSpotError, sanitize_error, with_retries
from app.graph.deps import NodeDeps
from app.idempotency import check_and_reserve
from app.models.db_models import TravelerDetailRecord
from app.models.state import GraphState


async def update_crm(state: GraphState, deps: NodeDeps) -> dict:
    key = f"{state['trip_request_id']}:update_crm"
    with deps.session_factory() as session:
        if not check_and_reserve(session, key, "update_crm"):
            return {"crm_status": "already_created"}
        traveler = session.get(TravelerDetailRecord, state["traveler_details_id"])
        contact_email, contact_full_name = traveler.contact_email, traveler.full_name

    trip_summary = {"destinations": state["trip_request"].destinations}

    try:
        contact_id = await with_retries(
            lambda: deps.crm_tool.upsert_contact(email=contact_email, full_name=contact_full_name),
            exceptions=(HubSpotError,),
        )
        await with_retries(
            lambda: deps.crm_tool.create_trip_record(contact_id=contact_id, trip_summary=trip_summary),
            exceptions=(HubSpotError,),
        )
    except HubSpotError as exc:
        error_log = state.get("error_log", []) + [sanitize_error(exc)]
        return {"crm_status": "failed", "error_log": error_log}

    return {"crm_status": "created"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_update_crm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/update_crm.py backend/tests/test_node_update_crm.py
git commit -m "feat: add idempotent update_crm node"
```
