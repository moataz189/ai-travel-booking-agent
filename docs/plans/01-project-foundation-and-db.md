# Phase 1a: Project Foundation and Database — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every cross-cutting piece of infrastructure the LangGraph agent's nodes depend on — settings, domain models, the `GraphState` schema, encryption, Postgres app tables + migrations, the LangGraph checkpointer, error/retry handling, idempotency, tool/LLM client protocols with fakes, and the shared `NodeDeps` bundle — before any graph node is written. Running `pytest` at the end proves this foundation in isolation.

**Architecture:** Every external dependency (LLM calls, Amadeus/Playwright/Gmail/HubSpot tool calls) sits behind a small `Protocol` interface with a real implementation (Phase 2) and a fake, deterministic implementation (built here) — nodes (later plans) are written once against these protocols and never change between phases, only which implementation is injected changes.

**Tech Stack:** Python 3.12, FastAPI, LangGraph + `langgraph-checkpoint-postgres`, SQLAlchemy 2.0 + Alembic, `psycopg` v3, Pydantic v2, `pydantic-settings`, `cryptography` (AES-GCM), `anthropic` SDK, pytest + pytest-asyncio, Docker Compose (local Postgres).

## Global Constraints (apply to this plan and every later Phase 1 plan)

- Never store traveler PII (passport/ID number, DOB) directly on LangGraph graph state — only a `traveler_details_id` foreign key (spec §5.3, §6.1).
- `error_log` entries are sanitized `{code, message}` pairs only — never PII, tokens, API keys, or raw upstream responses (spec §5.3).
- All MCP-tool-calling and LLM-calling nodes get bounded retries (3 attempts, exponential backoff) before routing to `agent_error` (spec §5.2, §7) — except `fill_booking_form`, which deliberately never retries.
- `send_confirmation_email` and `update_crm` are idempotent via a stored idempotency key derived from the trip/session ID (spec §7).
- Traveler PII columns (passport/ID number, DOB) are encrypted at the application layer before insert; the key is never a plaintext env var in real deployments (spec §6.1) — in Phase 1, tests use a locally generated test key.
- `human_review_gate` covers **all legs** of a multi-leg trip in one combined decision; `fill_booking_form` loops once per leg (spec §5.2, nodes 8-9).
- `rejection_reason` is one of `wrong_selection` | `wrong_traveler_details` and drives reject-routing (spec §5.2, node 9).
- No feature beyond what's listed — Phase 2 adds real MCP servers/mock site; Phase 1 only builds and tests the graph against fakes.

---

## Full Phase 1 File Structure

This is the complete file tree for all of Phase 1 (built across this plan and the five that follow it). Later plans reference this section instead of repeating it — each only lists the specific files it adds.

```
backend/
  pyproject.toml
  alembic.ini
  alembic/
    env.py
    versions/                       (migration files — this plan)
  app/
    __init__.py
    config.py                       (this plan)
    db.py                           (this plan)
    crypto.py                       (this plan)
    errors.py                       (this plan)
    idempotency.py                  (this plan)
    llm/
      __init__.py
      client.py                    (this plan)
      fake.py                      (this plan)
    tools/
      __init__.py
      protocol.py                  (this plan)
      fake.py                      (this plan)
    models/
      __init__.py
      domain.py                    (this plan)
      db_models.py                 (this plan)
      state.py                     (this plan)
    graph/
      __init__.py
      deps.py                      (this plan)
      build.py                     (plan 06-fastapi-and-error-handling)
      nodes/
        __init__.py
        collect_trip_request.py     (plan 02-langgraph-trip-search-flow)
        search_flights_and_hotels.py (plan 02)
        recommend_options.py        (plan 02)
        generate_itinerary.py       (plan 02)
        await_selection.py          (plan 03-selection-and-interrupts)
        finalize_itinerary.py       (plan 03)
        collect_traveler_details.py (plan 04-traveler-pii-and-booking-form)
        fill_booking_form.py        (plan 04)
        human_review_gate.py        (plan 04)
        send_confirmation_email.py  (plan 05-email-crm-idempotency)
        update_crm.py               (plan 05)
        agent_error.py              (plan 06-fastapi-and-error-handling)
    api/
      __init__.py
      main.py                      (plan 06)
      schemas.py                   (plan 06)
  tests/
    conftest.py                     (this plan: settings + db_engine fixtures)
    factories.py                    (this plan: make_deps + make_traveler_record helpers)
    test_config.py                  (this plan)
    test_domain_models.py           (this plan)
    test_crypto.py                  (this plan)
    test_db_models.py               (this plan)
    test_checkpointer.py            (this plan)
    test_errors.py                  (this plan)
    test_idempotency.py             (this plan)
    test_fake_tools.py              (this plan)
    test_llm_client.py              (this plan)
    test_node_*.py                  (one per node — plans 02-06)
    test_graph_end_to_end.py        (plan 06)
    test_graph_persistence.py       (plan 06)
    test_pii_leak.py                (plan 06)
    test_api.py                     (plan 06)
docker-compose.yml                 (this plan)
```

**Cross-plan dependency chain:** this plan → 02 → 03 → 04 → 05 → 06. Each later plan assumes everything in the earlier plans is committed and passing; none of them are meant to be built out of order. "Independently executable" means each plan is its own reviewable, testable unit — not that they can run in parallel.

**Shared test fixtures/helpers this plan produces (used by every later plan's tests):**
- `db_engine` (pytest fixture in `conftest.py`) — a real Postgres engine with all app tables created, torn down after each test.
- `make_deps(...)` (plain function in `factories.py`) — builds a `NodeDeps` with fakes; every node test in later plans uses this instead of redefining `NodeDeps(...)` inline.
- `make_traveler_record(db_engine, ...)` (plain function in `factories.py`) — inserts a `TripRequestRecord` + `TravelerDetailRecord` and returns the traveler's id; used by any test needing an existing traveler in the DB.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `docker-compose.yml`
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_config.py`

**Produces:** `app.config.Settings` (a `pydantic-settings.BaseSettings` subclass with `database_url`, `test_database_url`, `traveler_pii_encryption_key`, `anthropic_api_key`, `anthropic_model: str = "claude-sonnet-5"`), loaded via `get_settings()`.

- [ ] **Step 1: Create the Python project file**

`backend/pyproject.toml`:
```toml
[project]
name = "travel-agent-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "langgraph>=0.2.60",
    "langgraph-checkpoint-postgres>=2.0.13",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg[binary]>=3.2",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "cryptography>=43.0",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create the local Postgres compose file**

`docker-compose.yml` (repo root):
```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: travel_agent
      POSTGRES_PASSWORD: travel_agent
      POSTGRES_DB: travel_agent
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

Run: `docker compose up -d postgres`
Expected: container starts; `docker compose ps` shows `postgres` as `running (healthy)` within ~10s.

- [ ] **Step 3: Write the settings module**

`backend/app/config.py`:
```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://travel_agent:travel_agent@localhost:5432/travel_agent"
    test_database_url: str = "postgresql+psycopg://travel_agent:travel_agent@localhost:5432/travel_agent_test"
    traveler_pii_encryption_key: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`backend/app/__init__.py`: empty file.

- [ ] **Step 4: Write the failing test**

`backend/tests/test_config.py`:
```python
from app.config import get_settings


def test_settings_default_model_is_claude_sonnet_5():
    assert get_settings().anthropic_model == "claude-sonnet-5"


def test_settings_database_url_is_postgres():
    assert get_settings().database_url.startswith("postgresql+psycopg://")
```

- [ ] **Step 5: Install dependencies and run the test**

Run:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_config.py -v
```
Expected: both tests PASS (no external services needed).

- [ ] **Step 6: Create the test fixtures file (expanded by Task 10 once the DB models exist)**

`backend/tests/conftest.py`:
```python
import pytest

from app.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()
```

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/app/__init__.py backend/app/config.py \
        backend/tests/conftest.py backend/tests/test_config.py docker-compose.yml
git commit -m "chore: scaffold backend project with settings and local Postgres"
```

---

### Task 2: Domain Models and Graph State

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/domain.py`
- Create: `backend/app/models/state.py`
- Test: `backend/tests/test_domain_models.py`

**Produces:** `TripRequest`, `FlightOption`, `HotelOption`, `Itinerary`, `ItineraryDay`, `TravelerDetails` (Pydantic models in `domain.py`); `GraphState`, `BookingFormResult` (TypedDicts in `state.py`) — the shared vocabulary every later task imports.

- [ ] **Step 1: Write the failing test for domain models**

`backend/tests/test_domain_models.py`:
```python
import pytest
from pydantic import ValidationError

from app.models.domain import TripRequest, FlightOption, TravelerDetails


def test_trip_request_requires_at_least_one_destination():
    with pytest.raises(ValidationError):
        TripRequest(
            origin="JFK", destinations=[], depart_date="2026-09-01", return_date="2026-09-10",
            budget_usd=2000, traveler_count=1, preferences=[],
        )


def test_trip_request_accepts_multi_leg_destinations():
    trip = TripRequest(
        origin="JFK", destinations=["CDG", "FCO"], depart_date="2026-09-01",
        return_date="2026-09-10", budget_usd=3000, traveler_count=2, preferences=["museums"],
    )
    assert trip.destinations == ["CDG", "FCO"]


def test_flight_option_rejects_negative_price():
    with pytest.raises(ValidationError):
        FlightOption(
            id="f1", origin="JFK", destination="CDG", depart_at="2026-09-01T18:00:00Z",
            arrive_at="2026-09-02T07:00:00Z", price_usd=-100, carrier="AF", stops=0,
        )


def test_traveler_details_holds_no_encryption_in_domain_model():
    # domain model is the in-memory shape used before persistence; the
    # encrypted-at-rest concern is handled by db_models/crypto, not here.
    traveler = TravelerDetails(
        full_name="Jane Doe", date_of_birth="1990-01-01",
        passport_number="X1234567", contact_email="jane@example.com",
    )
    assert traveler.passport_number == "X1234567"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_domain_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Implement the domain models**

`backend/app/models/__init__.py`: empty file.

`backend/app/models/domain.py`:
```python
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    origin: str
    destinations: list[str] = Field(min_length=1)
    depart_date: date
    return_date: date
    budget_usd: float = Field(gt=0)
    traveler_count: int = Field(gt=0)
    preferences: list[str] = Field(default_factory=list)

    @field_validator("return_date")
    @classmethod
    def return_after_depart(cls, v: date, info):
        depart = info.data.get("depart_date")
        if depart and v < depart:
            raise ValueError("return_date must be on or after depart_date")
        return v


class FlightOption(BaseModel):
    id: str
    origin: str
    destination: str
    depart_at: datetime
    arrive_at: datetime
    price_usd: float = Field(gt=0)
    carrier: str
    stops: int = Field(ge=0)


class HotelOption(BaseModel):
    id: str
    destination: str
    name: str
    check_in: date
    check_out: date
    price_usd_total: float = Field(gt=0)
    rating: float = Field(ge=0, le=5)


class ItineraryDay(BaseModel):
    day_index: int = Field(ge=0)
    date: date
    activities: list[str]


class Itinerary(BaseModel):
    days: list[ItineraryDay]


class TravelerDetails(BaseModel):
    full_name: str
    date_of_birth: date
    passport_number: str
    contact_email: str


ApprovalStatus = Literal["pending", "approved", "rejected"]
RejectionReason = Literal["wrong_selection", "wrong_traveler_details"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_domain_models.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Implement the graph state module**

`backend/app/models/state.py`:
```python
from typing import Optional, TypedDict

from app.models.domain import ApprovalStatus, Itinerary, RejectionReason, TripRequest


class BookingFormResult(TypedDict):
    leg_index: int
    screenshot_ref: str
    filled_fields_summary: dict


class GraphState(TypedDict, total=False):
    user_message: str
    trip_request_id: str
    trip_request: TripRequest
    flight_result_ids: list[str]
    hotel_result_ids: list[str]
    shortlist: dict
    no_results: bool
    recommendation_narrative: str
    itinerary: Itinerary
    selected_flights: list[dict]
    selected_hotels: list[dict]
    traveler_details_id: str
    booking_form_results: list[BookingFormResult]
    form_fill_failed: bool
    approval_status: ApprovalStatus
    rejection_reason: Optional[RejectionReason]
    email_status: str
    crm_status: str
    user_facing_message: str
    retry_counts: dict[str, int]
    error_log: list[dict]
```

Note: `email_status`/`crm_status` are plain status strings (`"sent"`, `"already_sent"`, `"failed"`, `"created"`, `"already_created"`) — the idempotency key itself lives in the `idempotency_records` table (Task 7), not in this field.

- [ ] **Step 6: Add a test confirming `GraphState` accepts a partial update shape**

Append to `backend/tests/test_domain_models.py`:
```python
from app.models.state import GraphState


def test_graph_state_allows_partial_dict():
    partial: GraphState = {"approval_status": "pending", "error_log": []}
    assert partial["approval_status"] == "pending"
```

- [ ] **Step 7: Run the full test file again**

Run: `pytest tests/test_domain_models.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/models/ backend/tests/test_domain_models.py
git commit -m "feat: add domain models and LangGraph state schema"
```

---

### Task 3: Traveler PII Encryption Helper

**Files:**
- Create: `backend/app/crypto.py`
- Test: `backend/tests/test_crypto.py`

**Produces:** `encrypt_field(plaintext, key) -> bytes`, `decrypt_field(ciphertext, key) -> str`, `generate_test_key() -> bytes`. Used by Task 4's `TravelerDetailRecord` model and every later plan that touches traveler PII.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_crypto.py`:
```python
import pytest

from app.crypto import decrypt_field, encrypt_field, generate_test_key


def test_encrypt_then_decrypt_roundtrips():
    key = generate_test_key()
    ciphertext = encrypt_field("X1234567", key)
    assert ciphertext != b"X1234567"
    assert decrypt_field(ciphertext, key) == "X1234567"


def test_decrypt_fails_with_wrong_key():
    key1, key2 = generate_test_key(), generate_test_key()
    ciphertext = encrypt_field("X1234567", key1)
    with pytest.raises(Exception):
        decrypt_field(ciphertext, key2)


def test_same_plaintext_produces_different_ciphertext_each_time():
    key = generate_test_key()
    assert encrypt_field("X1234567", key) != encrypt_field("X1234567", key)  # random nonce
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_crypto.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.crypto'`

- [ ] **Step 3: Implement the crypto helper**

`backend/app/crypto.py`:
```python
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_SIZE = 12


def generate_test_key() -> bytes:
    """32-byte AES-256 key for tests/dev only. Real deployments source this
    from AWS Secrets Manager (spec.md §6.2), never generate it inline."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_field(plaintext: str, key: bytes) -> bytes:
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    return nonce + aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)


def decrypt_field(blob: bytes, key: bytes) -> str:
    aesgcm = AESGCM(key)
    nonce, ciphertext = blob[:_NONCE_SIZE], blob[_NONCE_SIZE:]
    return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_crypto.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/crypto.py backend/tests/test_crypto.py
git commit -m "feat: add AES-GCM encryption helper for traveler PII"
```

---

### Task 4: Postgres App Tables and Migrations

**Files:**
- Create: `backend/app/models/db_models.py`
- Create: `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/versions/0001_initial.py`
- Test: `backend/tests/test_db_models.py`

**Consumes:** `app.crypto.encrypt_field`/`decrypt_field` (Task 3).
**Produces:** ORM classes `TripRequestRecord`, `FlightResultRecord`, `HotelResultRecord`, `TravelerDetailRecord` (encrypted `passport_number_enc`, `date_of_birth_enc`), `BookingRecord`, `IdempotencyRecord`; `Base`; `app.db.get_engine()`, `app.db.get_session()`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_db_models.py`:
```python
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import decrypt_field, encrypt_field, generate_test_key
from app.models.db_models import Base, TravelerDetailRecord, TripRequestRecord


@pytest.fixture
def engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_schema_creates_from_empty_database(engine):
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        ).scalars().all()
    assert "trip_requests" in tables
    assert "traveler_details" in tables
    assert "idempotency_records" in tables


def test_traveler_detail_foreign_key_requires_existing_trip(engine):
    with Session(engine) as session:
        session.add(TravelerDetailRecord(
            id=str(uuid.uuid4()), trip_request_id=str(uuid.uuid4()),  # does not exist
            full_name="Jane Doe", date_of_birth_enc=b"x", passport_number_enc=b"x",
            contact_email="jane@example.com",
        ))
        with pytest.raises(Exception):
            session.commit()


def test_traveler_detail_encrypted_columns_roundtrip(engine):
    key = generate_test_key()
    with Session(engine) as session:
        trip = TripRequestRecord(id=str(uuid.uuid4()), raw_request={"origin": "JFK"})
        session.add(trip)
        session.flush()
        traveler = TravelerDetailRecord(
            id=str(uuid.uuid4()), trip_request_id=trip.id, full_name="Jane Doe",
            date_of_birth_enc=encrypt_field("1990-01-01", key),
            passport_number_enc=encrypt_field("X1234567", key),
            contact_email="jane@example.com",
        )
        session.add(traveler)
        session.commit()
        fetched = session.get(TravelerDetailRecord, traveler.id)
        assert decrypt_field(fetched.passport_number_enc, key) == "X1234567"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.db_models'`

- [ ] **Step 3: Implement the ORM models**

`backend/app/models/db_models.py`:
```python
import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class TripRequestRecord(Base):
    __tablename__ = "trip_requests"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    raw_request: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class FlightResultRecord(Base):
    __tablename__ = "flight_results"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    trip_request_id: Mapped[str] = mapped_column(ForeignKey("trip_requests.id"))
    leg_index: Mapped[int]
    payload: Mapped[dict] = mapped_column(JSON)


class HotelResultRecord(Base):
    __tablename__ = "hotel_results"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    trip_request_id: Mapped[str] = mapped_column(ForeignKey("trip_requests.id"))
    leg_index: Mapped[int]
    payload: Mapped[dict] = mapped_column(JSON)


class TravelerDetailRecord(Base):
    __tablename__ = "traveler_details"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    trip_request_id: Mapped[str] = mapped_column(ForeignKey("trip_requests.id"))
    full_name: Mapped[str]
    date_of_birth_enc: Mapped[bytes] = mapped_column(LargeBinary)
    passport_number_enc: Mapped[bytes] = mapped_column(LargeBinary)
    contact_email: Mapped[str]
    purge_after: Mapped[datetime | None] = mapped_column(default=None)


class BookingRecord(Base):
    __tablename__ = "booking_records"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    trip_request_id: Mapped[str] = mapped_column(ForeignKey("trip_requests.id"))
    leg_index: Mapped[int]
    filled_fields_summary: Mapped[dict] = mapped_column(JSON)
    screenshot_ref: Mapped[str]


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (UniqueConstraint("key", name="uq_idempotency_key"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String, index=True)
    operation: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
```

`backend/app/db.py`:
```python
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url)


def get_session(engine: Engine) -> Session:
    return Session(engine)
```

- [ ] **Step 4: Run test to verify it passes**

Requires the test database to exist first:
```bash
docker compose exec postgres psql -U travel_agent -c "CREATE DATABASE travel_agent_test;"
pytest tests/test_db_models.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Set up Alembic for real migrations (not just `create_all`)**

Run: `cd backend && alembic init alembic`

Edit `backend/alembic.ini`, set:
```ini
sqlalchemy.url = postgresql+psycopg://travel_agent:travel_agent@localhost:5432/travel_agent
```

`backend/alembic/env.py` — replace the generated `target_metadata = None` line, and make the connection URL overridable by `DATABASE_URL` (so tests can point Alembic at the test database without editing `alembic.ini`). Add this right after the existing `config = context.config` line:
```python
import os

from app.models.db_models import Base

target_metadata = Base.metadata

database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)
```

- [ ] **Step 6: Generate and inspect the initial migration**

Run: `alembic revision --autogenerate -m "initial schema"`
Expected: creates `backend/alembic/versions/0001_initial.py` (rename from the autogenerated hash-suffixed filename) with `create_table` calls for all six tables from Step 3.

- [ ] **Step 7: Apply the migration to the dev database and verify**

Run:
```bash
alembic upgrade head
docker compose exec postgres psql -U travel_agent -d travel_agent -c "\dt"
```
Expected: lists all six tables.

- [ ] **Step 8: Add a migration-specific test against a genuinely empty database**

Append to `backend/tests/test_db_models.py`:
```python
import os
import subprocess

from sqlalchemy import create_engine, text


def test_alembic_migration_applies_cleanly_to_empty_database():
    test_url = get_settings().test_database_url
    # Ensure a clean slate: drop and recreate the public schema so this test
    # proves migrations work against an empty database, not a re-apply.
    engine = create_engine(test_url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()

    result = subprocess.run(
        ["alembic", "upgrade", "head"], cwd="backend",
        env={**os.environ, "DATABASE_URL": test_url}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(test_url)
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        ).scalars().all()
    assert "trip_requests" in tables
    engine.dispose()
```

- [ ] **Step 9: Run the full test file**

Run: `pytest tests/test_db_models.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/app/models/db_models.py backend/app/db.py backend/alembic.ini \
        backend/alembic/ backend/tests/test_db_models.py
git commit -m "feat: add Postgres app tables and Alembic migrations"
```

---

### Task 5: LangGraph Postgres Checkpointer

**Files:**
- Modify: `backend/app/db.py`
- Test: `backend/tests/test_checkpointer.py`

**Produces:** `app.db.get_checkpointer(url) -> PostgresSaver` (context-manager factory). Used by plan 06's graph assembly and persistence tests.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_checkpointer.py`:
```python
from app.config import get_settings
from app.db import get_checkpointer


def test_checkpointer_setup_creates_langgraph_tables():
    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        with checkpointer.conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                "AND tablename LIKE 'checkpoint%'"
            )
            tables = [row[0] for row in cur.fetchall()]
    assert "checkpoints" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpointer.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_checkpointer' from 'app.db'`

- [ ] **Step 3: Implement the checkpointer factory**

Append to `backend/app/db.py`:
```python
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver


@contextmanager
def get_checkpointer(url: str | None = None):
    with PostgresSaver.from_conn_string(url or get_settings().database_url) as saver:
        yield saver
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_checkpointer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db.py backend/tests/test_checkpointer.py
git commit -m "feat: add LangGraph Postgres checkpointer factory"
```

---

### Task 6: Error Handling Infrastructure

**Files:**
- Create: `backend/app/errors.py`
- Test: `backend/tests/test_errors.py`

**Produces:** exception classes `ToolCallError`, `AmadeusRateLimitError`, `AmadeusTimeoutError`, `PlaywrightTimeoutError`, `GmailSendError`, `HubSpotError`, `LLMTimeoutError`, `LLMRateLimitError` (all subclass `ToolCallError`, carry `.code: str`); `async def with_retries(fn, *, exceptions, max_attempts=3, backoff_base=0.5)` — `fn` is a zero-arg callable that may return a plain value or a coroutine (awaited internally, so callers always `await with_retries(...)` regardless of whether the wrapped call is sync or async); `sanitize_error(exc) -> dict` returning `{"code", "message"}` only. Used by every tool/LLM-calling node in later plans.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_errors.py`:
```python
import pytest

from app.errors import AmadeusRateLimitError, ToolCallError, sanitize_error, with_retries


@pytest.mark.asyncio
async def test_with_retries_succeeds_on_second_attempt_sync_callable():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise AmadeusRateLimitError("rate limited")
        return "ok"

    result = await with_retries(flaky, exceptions=(AmadeusRateLimitError,), max_attempts=3, backoff_base=0)
    assert result == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_with_retries_succeeds_on_second_attempt_async_callable():
    calls = {"count": 0}

    async def flaky_async():
        calls["count"] += 1
        if calls["count"] < 2:
            raise AmadeusRateLimitError("rate limited")
        return "ok"

    result = await with_retries(flaky_async, exceptions=(AmadeusRateLimitError,), max_attempts=3, backoff_base=0)
    assert result == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_with_retries_raises_after_max_attempts():
    def always_fails():
        raise AmadeusRateLimitError("still limited")

    with pytest.raises(AmadeusRateLimitError):
        await with_retries(always_fails, exceptions=(AmadeusRateLimitError,), max_attempts=3, backoff_base=0)


def test_sanitize_error_never_includes_raw_upstream_payload():
    exc = AmadeusRateLimitError(
        "rate limited", raw_upstream_response={"secret_token": "abc123", "traveler_passport": "X1"},
    )
    sanitized = sanitize_error(exc)
    assert sanitized == {"code": "AmadeusRateLimitError", "message": "rate limited"}
    assert "secret_token" not in str(sanitized)
    assert "X1" not in str(sanitized)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.errors'`

- [ ] **Step 3: Implement the error module**

`backend/app/errors.py`:
```python
import asyncio
import inspect
from typing import Callable, TypeVar

T = TypeVar("T")


class ToolCallError(Exception):
    code = "ToolCallError"

    def __init__(self, message: str, *, raw_upstream_response: dict | None = None):
        super().__init__(message)
        self.message = message
        # Deliberately not read by sanitize_error — kept only for local
        # debugging via logging middleware that itself must not forward it
        # into error_log/metrics/traces.
        self._raw_upstream_response = raw_upstream_response


class AmadeusRateLimitError(ToolCallError):
    code = "AmadeusRateLimitError"


class AmadeusTimeoutError(ToolCallError):
    code = "AmadeusTimeoutError"


class PlaywrightTimeoutError(ToolCallError):
    code = "PlaywrightTimeoutError"


class GmailSendError(ToolCallError):
    code = "GmailSendError"


class HubSpotError(ToolCallError):
    code = "HubSpotError"


class LLMTimeoutError(ToolCallError):
    code = "LLMTimeoutError"


class LLMRateLimitError(ToolCallError):
    code = "LLMRateLimitError"


async def with_retries(
    fn: Callable[[], T], *, exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3, backoff_base: float = 0.5,
) -> T:
    """Calls fn() and, if it returns a coroutine, awaits it. Retries on the
    given exceptions with exponential backoff. Always called as
    `await with_retries(...)`, whether fn is sync or async, so node code
    doesn't need two different retry helpers."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            if inspect.isawaitable(result):
                return await result
            return result
        except exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise
            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
    raise last_exc  # pragma: no cover - unreachable, satisfies type checker


def sanitize_error(exc: ToolCallError) -> dict:
    return {"code": exc.code, "message": exc.message}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_errors.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/errors.py backend/tests/test_errors.py
git commit -m "feat: add typed tool/LLM exceptions, retry helper, and error sanitizer"
```

---

### Task 7: Idempotency Helper

**Files:**
- Create: `backend/app/idempotency.py`
- Test: `backend/tests/test_idempotency.py`

**Consumes:** `IdempotencyRecord` (Task 4).
**Produces:** `check_and_reserve(session, key, operation) -> bool` — `True` and records the key if not already present (safe to proceed), `False` if already reserved (already done, skip). Used by the email/CRM nodes in plan 05.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_idempotency.py`:
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.idempotency import check_and_reserve
from app.models.db_models import Base


@pytest.fixture
def session():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_first_reservation_succeeds(session):
    assert check_and_reserve(session, "trip-1:send_email", "send_email") is True


def test_duplicate_reservation_is_rejected(session):
    assert check_and_reserve(session, "trip-1:send_email", "send_email") is True
    assert check_and_reserve(session, "trip-1:send_email", "send_email") is False


def test_different_keys_do_not_collide(session):
    assert check_and_reserve(session, "trip-1:send_email", "send_email") is True
    assert check_and_reserve(session, "trip-1:update_crm", "update_crm") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_idempotency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.idempotency'`

- [ ] **Step 3: Implement the idempotency helper**

`backend/app/idempotency.py`:
```python
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.db_models import IdempotencyRecord


def check_and_reserve(session: Session, key: str, operation: str) -> bool:
    """Attempts to reserve `key`. Returns True if this is the first time this
    key has been reserved (caller should proceed), False if it was already
    reserved (caller should skip — the side effect already happened)."""
    session.add(IdempotencyRecord(key=key, operation=operation))
    try:
        session.commit()
        return True
    except IntegrityError:
        session.rollback()
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_idempotency.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/idempotency.py backend/tests/test_idempotency.py
git commit -m "feat: add idempotency check-and-reserve helper"
```

---

### Task 8: Tool Client Protocol and Fake Implementations

**Files:**
- Create: `backend/app/tools/__init__.py`, `backend/app/tools/protocol.py`, `backend/app/tools/fake.py`
- Test: `backend/tests/test_fake_tools.py`

**Consumes:** `FlightOption`, `HotelOption` (Task 2); exceptions (Task 6).
**Produces:** `Protocol` classes `FlightHotelSearchTool`, `BookingFormTool`, `EmailTool`, `CRMTool`; `FakeFlightHotelSearchTool`, `FakeBookingFormTool`, `FakeEmailTool`, `FakeCRMTool` — each constructed with a `behavior` literal for deterministic test scenarios. Phase 2 adds real MCP-backed implementations of the same protocols without touching node code.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_fake_tools.py`:
```python
import pytest

from app.errors import AmadeusRateLimitError, GmailSendError
from app.tools.fake import FakeFlightHotelSearchTool, FakeEmailTool


@pytest.mark.asyncio
async def test_fake_search_tool_success_returns_options():
    flights, hotels = await FakeFlightHotelSearchTool(behavior="success").search(
        origin="JFK", destination="CDG", depart_date="2026-09-01"
    )
    assert len(flights) > 0
    assert len(hotels) > 0


@pytest.mark.asyncio
async def test_fake_search_tool_empty_returns_no_options():
    flights, hotels = await FakeFlightHotelSearchTool(behavior="empty").search(
        origin="JFK", destination="CDG", depart_date="2026-09-01"
    )
    assert flights == []
    assert hotels == []


@pytest.mark.asyncio
async def test_fake_search_tool_rate_limited_raises():
    with pytest.raises(AmadeusRateLimitError):
        await FakeFlightHotelSearchTool(behavior="rate_limited").search(
            origin="JFK", destination="CDG", depart_date="2026-09-01"
        )


@pytest.mark.asyncio
async def test_fake_email_tool_records_sent_emails():
    tool = FakeEmailTool(behavior="success")
    await tool.send_email(to="jane@example.com", subject="Your trip", body="Details...")
    assert tool.sent == [{"to": "jane@example.com", "subject": "Your trip", "body": "Details..."}]


@pytest.mark.asyncio
async def test_fake_email_tool_failure_raises():
    with pytest.raises(GmailSendError):
        await FakeEmailTool(behavior="failure").send_email(to="jane@example.com", subject="x", body="y")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fake_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.tools'`

- [ ] **Step 3: Implement the protocol definitions**

`backend/app/tools/__init__.py`: empty file.

`backend/app/tools/protocol.py`:
```python
from typing import Protocol

from app.models.domain import FlightOption, HotelOption


class FlightHotelSearchTool(Protocol):
    async def search(
        self, *, origin: str, destination: str, depart_date: str
    ) -> tuple[list[FlightOption], list[HotelOption]]: ...


class BookingFormTool(Protocol):
    async def fill_leg(
        self, *, leg_index: int, flight: dict, hotel: dict, traveler: dict
    ) -> dict: ...  # returns {"screenshot_ref": str, "filled_fields_summary": dict}


class EmailTool(Protocol):
    async def send_email(self, *, to: str, subject: str, body: str) -> None: ...


class CRMTool(Protocol):
    async def upsert_contact(self, *, email: str, full_name: str) -> str: ...
    async def create_trip_record(self, *, contact_id: str, trip_summary: dict) -> str: ...
```

- [ ] **Step 4: Implement the fake tools**

`backend/app/tools/fake.py`:
```python
from typing import Literal

from app.errors import AmadeusRateLimitError, GmailSendError, HubSpotError, PlaywrightTimeoutError
from app.models.domain import FlightOption, HotelOption

SearchBehavior = Literal["success", "empty", "rate_limited", "timeout"]
SimpleBehavior = Literal["success", "failure"]


class FakeFlightHotelSearchTool:
    def __init__(self, behavior: SearchBehavior = "success"):
        self.behavior = behavior

    async def search(self, *, origin, destination, depart_date):
        if self.behavior == "rate_limited":
            raise AmadeusRateLimitError("Amadeus rate limit exceeded")
        if self.behavior == "timeout":
            raise AmadeusRateLimitError("Amadeus request timed out")
        if self.behavior == "empty":
            return [], []
        flights = [FlightOption(
            id=f"flight-{destination}-1", origin=origin, destination=destination,
            depart_at=f"{depart_date}T18:00:00Z", arrive_at=f"{depart_date}T23:00:00Z",
            price_usd=450.0, carrier="AF", stops=0,
        )]
        hotels = [HotelOption(
            id=f"hotel-{destination}-1", destination=destination, name="Fake Grand Hotel",
            check_in=depart_date, check_out=depart_date, price_usd_total=600.0, rating=4.2,
        )]
        return flights, hotels


class FakeBookingFormTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior

    async def fill_leg(self, *, leg_index, flight, hotel, traveler):
        if self.behavior == "failure":
            raise PlaywrightTimeoutError(f"Timed out filling form for leg {leg_index}")
        return {
            "screenshot_ref": f"fake://screenshot/leg-{leg_index}",
            "filled_fields_summary": {
                "flight_id": flight.get("id"), "hotel_id": hotel.get("id"),
                "traveler_name": traveler.get("full_name"),
            },
        }


class FakeEmailTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior
        self.sent: list[dict] = []

    async def send_email(self, *, to, subject, body):
        if self.behavior == "failure":
            raise GmailSendError("Gmail API send failed")
        self.sent.append({"to": to, "subject": subject, "body": body})


class FakeCRMTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior
        self.contacts: dict[str, str] = {}
        self.trip_records: list[dict] = []

    async def upsert_contact(self, *, email, full_name):
        if self.behavior == "failure":
            raise HubSpotError("HubSpot upsert_contact failed")
        return self.contacts.setdefault(email, f"contact-{len(self.contacts) + 1}")

    async def create_trip_record(self, *, contact_id, trip_summary):
        if self.behavior == "failure":
            raise HubSpotError("HubSpot create_trip_record failed")
        record_id = f"trip-record-{len(self.trip_records) + 1}"
        self.trip_records.append({"id": record_id, "contact_id": contact_id, **trip_summary})
        return record_id
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fake_tools.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/tools/ backend/tests/test_fake_tools.py
git commit -m "feat: add tool client protocols and fake implementations"
```

---

### Task 9: LLM Client Wrapper

**Files:**
- Create: `backend/app/llm/__init__.py`, `backend/app/llm/client.py`, `backend/app/llm/fake.py`
- Test: `backend/tests/test_llm_client.py`

**Consumes:** `get_settings` (Task 1); `LLMTimeoutError`/`LLMRateLimitError` (Task 6).
**Produces:** `Protocol` class `LLMClient` with `async def complete(self, *, system, prompt) -> str`; `AnthropicLLMClient` (real, wraps `anthropic.AsyncAnthropic`); `FakeLLMClient` (queued/canned responses, or a configured error). Used by every LLM-calling node in later plans.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_llm_client.py`:
```python
import pytest

from app.errors import LLMRateLimitError
from app.llm.fake import FakeLLMClient


@pytest.mark.asyncio
async def test_fake_llm_client_returns_queued_response():
    result = await FakeLLMClient(responses=["hello from fake LLM"]).complete(system="s", prompt="hi")
    assert result == "hello from fake LLM"


@pytest.mark.asyncio
async def test_fake_llm_client_cycles_through_multiple_responses():
    client = FakeLLMClient(responses=["first", "second"])
    assert await client.complete(system="s", prompt="p") == "first"
    assert await client.complete(system="s", prompt="p") == "second"


@pytest.mark.asyncio
async def test_fake_llm_client_can_simulate_rate_limit():
    client = FakeLLMClient(responses=[], raise_error=LLMRateLimitError("rate limited"))
    with pytest.raises(LLMRateLimitError):
        await client.complete(system="s", prompt="p")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm'`

- [ ] **Step 3: Implement the LLM client module**

`backend/app/llm/__init__.py`: empty file.

`backend/app/llm/client.py`:
```python
from typing import Protocol

from anthropic import AsyncAnthropic

from app.config import get_settings


class LLMClient(Protocol):
    async def complete(self, *, system: str, prompt: str) -> str: ...


class AnthropicLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._model = model or settings.anthropic_model

    async def complete(self, *, system: str, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

`backend/app/llm/fake.py`:
```python
class FakeLLMClient:
    def __init__(self, responses: list[str], raise_error: Exception | None = None):
        self._responses = list(responses)
        self._raise_error = raise_error
        self.calls: list[dict] = []

    async def complete(self, *, system: str, prompt: str) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        if self._raise_error is not None:
            raise self._raise_error
        return self._responses.pop(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/llm/ backend/tests/test_llm_client.py
git commit -m "feat: add LLM client protocol with real Anthropic and fake implementations"
```

---

### Task 10: NodeDeps Bundle and Shared Test Fixtures

**Files:**
- Create: `backend/app/graph/__init__.py`, `backend/app/graph/nodes/__init__.py`, `backend/app/graph/deps.py`
- Modify: `backend/tests/conftest.py`
- Create: `backend/tests/factories.py`
- Test: `backend/tests/test_node_deps.py`

**Consumes:** `LLMClient` (Task 9); tool protocols (Task 8); `Session` (Task 4).
**Produces:** `NodeDeps` dataclass — every graph node's second argument, used by all five later plans; `db_engine` pytest fixture (real Postgres engine, all tables created/dropped per test); `make_deps(...)` and `make_traveler_record(...)` helpers, so no later plan's tests need to redefine `NodeDeps(...)` construction or a DB engine fixture from scratch.

- [ ] **Step 1: Define the shared `NodeDeps` bundle**

`backend/app/graph/__init__.py`: empty file.
`backend/app/graph/nodes/__init__.py`: empty file.

`backend/app/graph/deps.py`:
```python
from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.llm.client import LLMClient
from app.tools.protocol import BookingFormTool, CRMTool, EmailTool, FlightHotelSearchTool


@dataclass
class NodeDeps:
    llm: LLMClient
    search_tool: FlightHotelSearchTool
    booking_tool: BookingFormTool
    email_tool: EmailTool
    crm_tool: CRMTool
    session_factory: Callable[[], Session]
    encryption_key: bytes
```

- [ ] **Step 2: Write the failing test for the shared fixtures**

`backend/tests/test_node_deps.py`:
```python
from factories import make_deps, make_traveler_record


def test_make_deps_builds_a_usable_nodedeps(db_engine):
    deps = make_deps(db_engine=db_engine, llm_responses=["hello"])
    assert deps.encryption_key == b"0" * 32
    with deps.session_factory() as session:
        assert session is not None


def test_make_traveler_record_inserts_and_returns_id(db_engine):
    traveler_id = make_traveler_record(db_engine)
    assert traveler_id is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_node_deps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'factories'` (and `fixture 'db_engine' not found` until Step 4 lands).

- [ ] **Step 4: Add the `db_engine` fixture to conftest.py**

Update `backend/tests/conftest.py` (append to the existing file from Task 1):
```python
from sqlalchemy import create_engine

from app.models.db_models import Base


@pytest.fixture
def db_engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()
```

- [ ] **Step 5: Implement the shared factories**

`backend/tests/factories.py`:
```python
import uuid

from sqlalchemy.orm import Session

from app.crypto import encrypt_field, generate_test_key
from app.graph.deps import NodeDeps
from app.llm.fake import FakeLLMClient
from app.models.db_models import TravelerDetailRecord, TripRequestRecord
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


def make_deps(
    *, db_engine=None, llm_responses: list[str] | None = None, llm_error: Exception | None = None,
    search_behavior: str = "success", booking_behavior: str = "success",
    email_behavior: str = "success", crm_behavior: str = "success", encryption_key: bytes | None = None,
) -> NodeDeps:
    """Builds a NodeDeps wired entirely to fakes. Every node test in plans
    02-06 uses this instead of constructing NodeDeps(...) inline."""
    return NodeDeps(
        llm=FakeLLMClient(responses=llm_responses or [], raise_error=llm_error),
        search_tool=FakeFlightHotelSearchTool(behavior=search_behavior),
        booking_tool=FakeBookingFormTool(behavior=booking_behavior),
        email_tool=FakeEmailTool(behavior=email_behavior),
        crm_tool=FakeCRMTool(behavior=crm_behavior),
        session_factory=(lambda: Session(db_engine)) if db_engine is not None else (lambda: None),
        encryption_key=encryption_key or b"0" * 32,
    )


def make_traveler_record(
    db_engine, trip_id: str | None = None, key: bytes | None = None,
    full_name: str = "Jane Doe", passport_number: str = "X1234567",
) -> str:
    """Inserts a TripRequestRecord + TravelerDetailRecord, returns the traveler's id."""
    trip_id = trip_id or str(uuid.uuid4())
    key = key or generate_test_key()
    with Session(db_engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        traveler = TravelerDetailRecord(
            trip_request_id=trip_id, full_name=full_name,
            date_of_birth_enc=encrypt_field("1990-01-01", key),
            passport_number_enc=encrypt_field(passport_number, key),
            contact_email="jane@example.com",
        )
        session.add(traveler)
        session.commit()
        return traveler.id
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_node_deps.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Run the entire foundation test suite**

Run: `pytest -v`
Expected: every test from Tasks 1-10 PASSES (roughly 25 test functions).

- [ ] **Step 8: Commit**

```bash
git add backend/app/graph/deps.py backend/app/graph/__init__.py backend/app/graph/nodes/ \
        backend/tests/conftest.py backend/tests/factories.py backend/tests/test_node_deps.py
git commit -m "feat: add NodeDeps bundle and shared test fixtures/factories"
```
