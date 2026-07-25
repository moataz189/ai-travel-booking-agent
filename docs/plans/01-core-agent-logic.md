# Phase 1: Core Agent Logic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and fully test the LangGraph agent (all 12 workflow nodes, both `interrupt()` gates, retry/error handling, encrypted traveler data, idempotency) against fake tool/LLM clients and a real local Postgres — no real MCP servers, no infra, no cloud. Running `pytest` at the end of this phase proves the entire workflow logic in isolation.

**Architecture:** A FastAPI app hosts a LangGraph `StateGraph` with a Postgres checkpointer. Every external dependency (LLM calls, Amadeus/Playwright/Gmail/HubSpot tool calls) sits behind a small `Protocol` interface with a real implementation (Phase 2) and a fake, deterministic implementation (this phase) — nodes are written once and never change between phases, only which implementation is injected changes.

**Tech Stack:** Python 3.12, FastAPI, LangGraph + `langgraph-checkpoint-postgres`, SQLAlchemy 2.0 + Alembic, `psycopg` v3, Pydantic v2, `pydantic-settings`, `cryptography` (AES-GCM), `anthropic` SDK, pytest + pytest-asyncio, Docker Compose (local Postgres).

## Global Constraints

- Never store traveler PII (passport/ID number, DOB) directly on LangGraph graph state — only a `traveler_details_id` foreign key (spec §5.3, §6.1).
- `error_log` entries are sanitized `{code, message}` pairs only — never PII, tokens, API keys, or raw upstream responses (spec §5.3).
- All MCP-tool-calling and LLM-calling nodes get bounded retries (3 attempts, exponential backoff) before routing to `agent_error` (spec §5.2, §7).
- `send_confirmation_email` and `update_crm` are idempotent via a stored idempotency key derived from the trip/session ID (spec §7).
- Traveler PII columns (passport/ID number, DOB) are encrypted at the application layer before insert; the key is never a plaintext env var in real deployments (spec §6.1) — in this phase, tests use a locally generated test key.
- `human_review_gate` covers **all legs** of a multi-leg trip in one combined decision; `fill_booking_form` loops once per leg (spec §5.2, nodes 8-9).
- `rejection_reason` is one of `wrong_selection` | `wrong_traveler_details` and drives reject-routing (spec §5.2, node 9).
- No feature beyond what's listed below — Phase 2 adds real MCP servers/mock site; this phase only builds and tests the graph against fakes.

---

## File Structure

```
backend/
  pyproject.toml
  alembic.ini
  alembic/
    env.py
    versions/                       (migration files, created by Task 4)
  app/
    __init__.py
    config.py                       (Settings via pydantic-settings)
    db.py                           (SQLAlchemy engine/session + checkpointer factory)
    crypto.py                       (AES-GCM encrypt/decrypt for traveler PII)
    errors.py                      (typed exceptions, retry decorator, error_log sanitizer)
    idempotency.py                  (idempotency check-and-set helper)
    llm/
      __init__.py
      client.py                    (LLMClient Protocol + AnthropicLLMClient)
      fake.py                      (FakeLLMClient)
    tools/
      __init__.py
      protocol.py                  (Protocol classes + I/O Pydantic models for every tool call)
      fake.py                      (Fake implementations, configurable behavior)
    models/
      __init__.py
      domain.py                    (Pydantic domain models: TripRequest, FlightOption, HotelOption, Itinerary, TravelerDetails)
      db_models.py                 (SQLAlchemy ORM models: app tables)
      state.py                     (GraphState TypedDict)
    graph/
      __init__.py
      build.py                     (StateGraph assembly + conditional edges)
      nodes/
        __init__.py
        collect_trip_request.py
        search_flights_and_hotels.py
        recommend_options.py
        generate_itinerary.py
        await_selection.py
        finalize_itinerary.py
        collect_traveler_details.py
        fill_booking_form.py
        human_review_gate.py
        send_confirmation_email.py
        update_crm.py
        agent_error.py
    api/
      __init__.py
      main.py                      (FastAPI app + endpoints)
      schemas.py                   (request/response models)
  tests/
    conftest.py
    test_domain_models.py
    test_crypto.py
    test_db_models.py
    test_checkpointer.py
    test_errors.py
    test_idempotency.py
    test_fake_tools.py
    test_llm_client.py
    test_node_collect_trip_request.py
    test_node_search_flights_and_hotels.py
    test_node_recommend_options.py
    test_node_generate_itinerary.py
    test_node_await_selection.py
    test_node_finalize_itinerary.py
    test_node_collect_traveler_details.py
    test_node_fill_booking_form.py
    test_node_human_review_gate.py
    test_node_send_confirmation_email.py
    test_node_update_crm.py
    test_graph_end_to_end.py
    test_graph_persistence.py
    test_pii_leak.py
    test_api.py
docker-compose.yml                 (local Postgres for dev + tests)
```

**Interface summary (who produces/consumes what, across tasks):**
- `app/models/domain.py` (Task 2) is imported by nearly everything — domain types are the shared vocabulary.
- `app/models/state.py` (Task 2) `GraphState` is the TypedDict every node function receives and returns a partial update of.
- `app/tools/protocol.py` (Task 8) defines the `Protocol` classes every node calls; `app/tools/fake.py` (Task 8) is what tests inject.
- `app/llm/client.py` (Task 9) defines `LLMClient` Protocol; `app/llm/fake.py` is what tests inject.
- `app/errors.py` (Task 6) `with_retries()` decorator/helper is used by every tool-calling and LLM-calling node.
- `app/idempotency.py` (Task 7) `check_and_reserve(key: str, db: Session) -> bool` is used by nodes 10 and 11 (`send_confirmation_email`, `update_crm`).
- `app/graph/nodes/*.py` (Tasks 10-20) each export one function `def <node_name>(state: GraphState, deps: NodeDeps) -> dict`, assembled by `app/graph/build.py` (Task 22).

---

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `docker-compose.yml`
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (a `pydantic-settings.BaseSettings` subclass with `database_url: str`, `test_database_url: str`, `traveler_pii_encryption_key: str`, `anthropic_api_key: str`, `anthropic_model: str = "claude-sonnet-5"`), loaded via `get_settings()`.

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
    settings = get_settings()
    assert settings.anthropic_model == "claude-sonnet-5"


def test_settings_database_url_is_postgres():
    settings = get_settings()
    assert settings.database_url.startswith("postgresql+psycopg://")
```

- [ ] **Step 5: Install dependencies and run the test**

Run:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_config.py -v
```
Expected: both tests PASS (no external services needed for this test).

- [ ] **Step 6: Create the shared test fixtures file (empty for now, filled in by later tasks)**

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

**Interfaces:**
- Consumes: nothing (pure data models).
- Produces: `TripRequest`, `FlightOption`, `HotelOption`, `Itinerary`, `ItineraryDay`, `TravelerDetails` (Pydantic models in `domain.py`); `GraphState`, `BookingFormResult` (TypedDicts in `state.py`). Every later task imports from here.

- [ ] **Step 1: Write the failing test for domain models**

`backend/tests/test_domain_models.py`:
```python
import pytest
from pydantic import ValidationError

from app.models.domain import TripRequest, FlightOption, TravelerDetails


def test_trip_request_requires_at_least_one_destination():
    with pytest.raises(ValidationError):
        TripRequest(
            origin="JFK",
            destinations=[],
            depart_date="2026-09-01",
            return_date="2026-09-10",
            budget_usd=2000,
            traveler_count=1,
            preferences=[],
        )


def test_trip_request_accepts_multi_leg_destinations():
    trip = TripRequest(
        origin="JFK",
        destinations=["CDG", "FCO"],
        depart_date="2026-09-01",
        return_date="2026-09-10",
        budget_usd=3000,
        traveler_count=2,
        preferences=["museums"],
    )
    assert trip.destinations == ["CDG", "FCO"]


def test_flight_option_rejects_negative_price():
    with pytest.raises(ValidationError):
        FlightOption(
            id="f1",
            origin="JFK",
            destination="CDG",
            depart_at="2026-09-01T18:00:00Z",
            arrive_at="2026-09-02T07:00:00Z",
            price_usd=-100,
            carrier="AF",
            stops=0,
        )


def test_traveler_details_holds_no_encryption_in_domain_model():
    # domain model is the in-memory shape used before persistence; the
    # encrypted-at-rest concern is handled by db_models/crypto, not here.
    traveler = TravelerDetails(
        full_name="Jane Doe",
        date_of_birth="1990-01-01",
        passport_number="X1234567",
        contact_email="jane@example.com",
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

Note: `email_status`/`crm_status` are plain status strings (`"sent"`, `"already_sent"`, `"failed"`, `"created"`, `"already_created"`), not dicts — this corrects an inconsistency from the initial draft of this field (an idempotency key is tracked separately, in the `idempotency_records` table via Task 7, not embedded in this field's value).

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
git add backend/app/models/
git add backend/tests/test_domain_models.py
git commit -m "feat: add domain models and LangGraph state schema"
```

---

### Task 3: Traveler PII Encryption Helper

**Files:**
- Create: `backend/app/crypto.py`
- Test: `backend/tests/test_crypto.py`

**Interfaces:**
- Produces: `encrypt_field(plaintext: str, key: bytes) -> bytes`, `decrypt_field(ciphertext: bytes, key: bytes) -> str`, `generate_test_key() -> bytes`. Used by Task 4's `TravelerDetail` ORM model and Task 16's `collect_traveler_details` node.

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
    key1 = generate_test_key()
    key2 = generate_test_key()
    ciphertext = encrypt_field("X1234567", key1)
    with pytest.raises(Exception):
        decrypt_field(ciphertext, key2)


def test_same_plaintext_produces_different_ciphertext_each_time():
    key = generate_test_key()
    a = encrypt_field("X1234567", key)
    b = encrypt_field("X1234567", key)
    assert a != b  # random nonce per encryption
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
    from AWS Secrets Manager (see spec.md section 6.2), never generate it
    inline."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_field(plaintext: str, key: bytes) -> bytes:
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext


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
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/versions/0001_initial.py`
- Test: `backend/tests/test_db_models.py`

**Interfaces:**
- Consumes: `app.crypto.encrypt_field`/`decrypt_field` (Task 3).
- Produces: SQLAlchemy ORM classes `TripRequestRecord`, `FlightResultRecord`, `HotelResultRecord`, `TravelerDetailRecord` (encrypted `passport_number_enc`, `date_of_birth_enc` columns), `BookingRecord`, `IdempotencyRecord`; `Base` declarative base; `app.db.get_engine(url: str)`, `app.db.get_session(engine)`.

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
    settings = get_settings()
    engine = create_engine(settings.test_database_url)
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
        bad_traveler = TravelerDetailRecord(
            id=str(uuid.uuid4()),
            trip_request_id=str(uuid.uuid4()),  # does not exist
            full_name="Jane Doe",
            date_of_birth_enc=b"x",
            passport_number_enc=b"x",
            contact_email="jane@example.com",
        )
        session.add(bad_traveler)
        with pytest.raises(Exception):
            session.commit()


def test_traveler_detail_encrypted_columns_roundtrip(engine):
    key = generate_test_key()
    with Session(engine) as session:
        trip = TripRequestRecord(id=str(uuid.uuid4()), raw_request={"origin": "JFK"})
        session.add(trip)
        session.flush()

        traveler = TravelerDetailRecord(
            id=str(uuid.uuid4()),
            trip_request_id=trip.id,
            full_name="Jane Doe",
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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

`backend/alembic/env.py` (replace the generated `target_metadata = None` line, and make the connection URL overridable by the `DATABASE_URL` env var so tests can point Alembic at the test database without editing `alembic.ini`):
```python
import os

from app.models.db_models import Base

target_metadata = Base.metadata

database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)
```
Add this block right after the existing `config = context.config` line that the `alembic init` template already generated.

- [ ] **Step 6: Generate and inspect the initial migration**

Run: `alembic revision --autogenerate -m "initial schema"`
Expected: creates `backend/alembic/versions/0001_initial.py` (exact filename may include a hash prefix — rename it to `0001_initial.py` for readability) containing `create_table` calls for all six tables from Step 3.

- [ ] **Step 7: Apply the migration to the dev database and verify**

Run:
```bash
alembic upgrade head
docker compose exec postgres psql -U travel_agent -d travel_agent -c "\dt"
```
Expected: lists all six tables (`trip_requests`, `flight_results`, `hotel_results`, `traveler_details`, `booking_records`, `idempotency_records`).

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
        ["alembic", "upgrade", "head"],
        cwd="backend",
        env={**os.environ, "DATABASE_URL": test_url},
        capture_output=True,
        text=True,
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

**Interfaces:**
- Consumes: `app.config.get_settings` (Task 1).
- Produces: `app.db.get_checkpointer(url: str) -> PostgresSaver` (context-manager-friendly factory). Used by Task 22 (`graph/build.py`) and Task 23 (persistence tests).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_checkpointer.py`:
```python
from app.config import get_settings
from app.db import get_checkpointer


def test_checkpointer_setup_creates_langgraph_tables():
    settings = get_settings()
    with get_checkpointer(settings.test_database_url) as checkpointer:
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
    conn_string = url or get_settings().database_url
    with PostgresSaver.from_conn_string(conn_string) as saver:
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

**Interfaces:**
- Produces: exception classes `ToolCallError`, `AmadeusRateLimitError`, `AmadeusTimeoutError`, `PlaywrightTimeoutError`, `GmailSendError`, `HubSpotError`, `LLMTimeoutError`, `LLMRateLimitError` (all subclass `ToolCallError`, each carrying `.code: str`); `async def with_retries(fn, *, exceptions, max_attempts=3, backoff_base=0.5) -> Any` helper — `fn` is a zero-arg callable that may return either a plain value or a coroutine (it awaits coroutines internally, so callers always `await with_retries(...)` regardless of whether the wrapped call is sync or async); `sanitize_error(exc: ToolCallError) -> dict` returning `{"code": ..., "message": ...}` with no exception args beyond the fixed message. Used by every tool/LLM-calling node (Tasks 10-20, except Task 17 which deliberately never retries — see spec §5.2/§7) and by `agent_error` node.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_errors.py`:
```python
import pytest

from app.errors import (
    AmadeusRateLimitError,
    ToolCallError,
    sanitize_error,
    with_retries,
)


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
        "rate limited",
        raw_upstream_response={"secret_token": "abc123", "traveler_passport": "X1"},
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
    fn: Callable[[], T],
    *,
    exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
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

**Interfaces:**
- Consumes: `IdempotencyRecord` (Task 4), `app.db.get_session` (Task 4).
- Produces: `check_and_reserve(session: Session, key: str, operation: str) -> bool` — returns `True` and records the key if not already present (safe to proceed), returns `False` if the key already exists (already done, skip). Used by Task 19 (`send_confirmation_email`) and Task 20 (`update_crm`).

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
    record = IdempotencyRecord(key=key, operation=operation)
    session.add(record)
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
- Create: `backend/app/tools/__init__.py`
- Create: `backend/app/tools/protocol.py`
- Create: `backend/app/tools/fake.py`
- Test: `backend/tests/test_fake_tools.py`

**Interfaces:**
- Consumes: `app.models.domain.FlightOption`, `HotelOption` (Task 2); `app.errors.AmadeusRateLimitError`, `PlaywrightTimeoutError`, `GmailSendError`, `HubSpotError` (Task 6).
- Produces: `Protocol` classes `FlightHotelSearchTool`, `BookingFormTool`, `EmailTool`, `CRMTool` (each with typed async methods); `FakeFlightHotelSearchTool`, `FakeBookingFormTool`, `FakeEmailTool`, `FakeCRMTool` — each constructed with a `behavior: Literal["success", "empty", "rate_limited", "timeout"]` for deterministic test scenarios. These are what nodes in Tasks 11, 17, 19, 20 call; Phase 2 adds real MCP-backed implementations of the same protocols without touching node code.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_fake_tools.py`:
```python
import pytest

from app.errors import AmadeusRateLimitError, GmailSendError
from app.tools.fake import FakeFlightHotelSearchTool, FakeEmailTool


@pytest.mark.asyncio
async def test_fake_search_tool_success_returns_options():
    tool = FakeFlightHotelSearchTool(behavior="success")
    flights, hotels = await tool.search(origin="JFK", destination="CDG", depart_date="2026-09-01")
    assert len(flights) > 0
    assert len(hotels) > 0


@pytest.mark.asyncio
async def test_fake_search_tool_empty_returns_no_options():
    tool = FakeFlightHotelSearchTool(behavior="empty")
    flights, hotels = await tool.search(origin="JFK", destination="CDG", depart_date="2026-09-01")
    assert flights == []
    assert hotels == []


@pytest.mark.asyncio
async def test_fake_search_tool_rate_limited_raises():
    tool = FakeFlightHotelSearchTool(behavior="rate_limited")
    with pytest.raises(AmadeusRateLimitError):
        await tool.search(origin="JFK", destination="CDG", depart_date="2026-09-01")


@pytest.mark.asyncio
async def test_fake_email_tool_records_sent_emails():
    tool = FakeEmailTool(behavior="success")
    await tool.send_email(to="jane@example.com", subject="Your trip", body="Details...")
    assert tool.sent == [{"to": "jane@example.com", "subject": "Your trip", "body": "Details..."}]


@pytest.mark.asyncio
async def test_fake_email_tool_failure_raises():
    tool = FakeEmailTool(behavior="failure")
    with pytest.raises(GmailSendError):
        await tool.send_email(to="jane@example.com", subject="x", body="y")
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
    async def upsert_contact(self, *, email: str, full_name: str) -> str: ...  # returns contact id
    async def create_trip_record(self, *, contact_id: str, trip_summary: dict) -> str: ...  # returns record id
```

- [ ] **Step 4: Implement the fake tools**

`backend/app/tools/fake.py`:
```python
from typing import Literal

from app.errors import (
    AmadeusRateLimitError,
    GmailSendError,
    HubSpotError,
    PlaywrightTimeoutError,
)
from app.models.domain import FlightOption, HotelOption

SearchBehavior = Literal["success", "empty", "rate_limited", "timeout"]
SimpleBehavior = Literal["success", "failure"]


class FakeFlightHotelSearchTool:
    def __init__(self, behavior: SearchBehavior = "success"):
        self.behavior = behavior

    async def search(
        self, *, origin: str, destination: str, depart_date: str
    ) -> tuple[list[FlightOption], list[HotelOption]]:
        if self.behavior == "rate_limited":
            raise AmadeusRateLimitError("Amadeus rate limit exceeded")
        if self.behavior == "timeout":
            raise AmadeusRateLimitError("Amadeus request timed out")
        if self.behavior == "empty":
            return [], []
        flights = [
            FlightOption(
                id=f"flight-{destination}-1",
                origin=origin,
                destination=destination,
                depart_at=f"{depart_date}T18:00:00Z",
                arrive_at=f"{depart_date}T23:00:00Z",
                price_usd=450.0,
                carrier="AF",
                stops=0,
            )
        ]
        hotels = [
            HotelOption(
                id=f"hotel-{destination}-1",
                destination=destination,
                name="Fake Grand Hotel",
                check_in=depart_date,
                check_out=depart_date,
                price_usd_total=600.0,
                rating=4.2,
            )
        ]
        return flights, hotels


class FakeBookingFormTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior

    async def fill_leg(
        self, *, leg_index: int, flight: dict, hotel: dict, traveler: dict
    ) -> dict:
        if self.behavior == "failure":
            raise PlaywrightTimeoutError(f"Timed out filling form for leg {leg_index}")
        return {
            "screenshot_ref": f"fake://screenshot/leg-{leg_index}",
            "filled_fields_summary": {
                "flight_id": flight.get("id"),
                "hotel_id": hotel.get("id"),
                "traveler_name": traveler.get("full_name"),
            },
        }


class FakeEmailTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior
        self.sent: list[dict] = []

    async def send_email(self, *, to: str, subject: str, body: str) -> None:
        if self.behavior == "failure":
            raise GmailSendError("Gmail API send failed")
        self.sent.append({"to": to, "subject": subject, "body": body})


class FakeCRMTool:
    def __init__(self, behavior: SimpleBehavior = "success"):
        self.behavior = behavior
        self.contacts: dict[str, str] = {}
        self.trip_records: list[dict] = []

    async def upsert_contact(self, *, email: str, full_name: str) -> str:
        if self.behavior == "failure":
            raise HubSpotError("HubSpot upsert_contact failed")
        contact_id = self.contacts.setdefault(email, f"contact-{len(self.contacts) + 1}")
        return contact_id

    async def create_trip_record(self, *, contact_id: str, trip_summary: dict) -> str:
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
git add backend/app/tools/
git add backend/tests/test_fake_tools.py
git commit -m "feat: add tool client protocols and fake implementations"
```

---

### Task 9: LLM Client Wrapper

**Files:**
- Create: `backend/app/llm/__init__.py`
- Create: `backend/app/llm/client.py`
- Create: `backend/app/llm/fake.py`
- Test: `backend/tests/test_llm_client.py`

**Interfaces:**
- Consumes: `app.config.get_settings` (Task 1); `app.errors.LLMTimeoutError`, `LLMRateLimitError` (Task 6).
- Produces: `Protocol` class `LLMClient` with `async def complete(self, *, system: str, prompt: str) -> str`; `AnthropicLLMClient` (real, wraps `anthropic.AsyncAnthropic`); `FakeLLMClient` (returns a queued/canned response, or raises a configured error). Used by Tasks 10, 12, 13, 15.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_llm_client.py`:
```python
import pytest

from app.errors import LLMRateLimitError
from app.llm.fake import FakeLLMClient


@pytest.mark.asyncio
async def test_fake_llm_client_returns_queued_response():
    client = FakeLLMClient(responses=["hello from fake LLM"])
    result = await client.complete(system="you are a travel agent", prompt="hi")
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
            model=self._model,
            max_tokens=1024,
            system=system,
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
git add backend/app/llm/
git add backend/tests/test_llm_client.py
git commit -m "feat: add LLM client protocol with real Anthropic and fake implementations"
```

---

### Task 10: Node — `collect_trip_request`

**Files:**
- Create: `backend/app/graph/__init__.py`
- Create: `backend/app/graph/nodes/__init__.py`
- Create: `backend/app/graph/deps.py`
- Create: `backend/app/graph/nodes/collect_trip_request.py`
- Test: `backend/tests/test_node_collect_trip_request.py`

**Interfaces:**
- Consumes: `GraphState` (Task 2), `TripRequest` (Task 2), `LLMClient` (Task 9), `with_retries`/`LLMTimeoutError`/`LLMRateLimitError` (Task 6).
- Produces: `NodeDeps` dataclass (bundles `llm`, `search_tool`, `booking_tool`, `email_tool`, `crm_tool`, `db_session_factory`, `encryption_key` — every node function's second argument, defined once here and reused by all remaining node tasks); `collect_trip_request(state, deps) -> dict`. This node calls `interrupt()` in a loop each time a required field is missing, re-entering on each resume until `TripRequest` validates — this is how multi-turn conversational field collection is realized on top of LangGraph's single-invocation-per-node model (spec §5.2 node 1 says "loops back on missing/invalid fields"; this is the concrete mechanism).

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

- [ ] **Step 2: Write the failing test**

`backend/tests/test_node_collect_trip_request.py`:
```python
import pytest

from app.graph.deps import NodeDeps
from app.graph.nodes.collect_trip_request import collect_trip_request
from app.llm.fake import FakeLLMClient
from app.tools.fake import (
    FakeBookingFormTool,
    FakeCRMTool,
    FakeEmailTool,
    FakeFlightHotelSearchTool,
)


def make_deps(llm_responses):
    return NodeDeps(
        llm=FakeLLMClient(responses=llm_responses),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: None,
        encryption_key=b"0" * 32,
    )


@pytest.mark.asyncio
async def test_collect_trip_request_extracts_valid_request_from_first_message():
    deps = make_deps(
        llm_responses=[
            '{"origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01", '
            '"return_date": "2026-09-10", "budget_usd": 2000, "traveler_count": 1, '
            '"preferences": ["museums"]}'
        ]
    )
    state = {"user_message": "I want to fly JFK to Paris Sept 1-10, budget $2000, 1 traveler"}
    result = await collect_trip_request(state, deps)
    assert result["trip_request"].destinations == ["CDG"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_node_collect_trip_request.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.graph'`

- [ ] **Step 4: Implement the node**

`backend/app/graph/nodes/collect_trip_request.py`:
```python
import json

from langgraph.types import interrupt

from app.errors import LLMRateLimitError, LLMTimeoutError, with_retries
from app.graph.deps import NodeDeps
from app.models.domain import TripRequest
from app.models.state import GraphState

_SYSTEM_PROMPT = (
    "Extract a structured trip request from the user's message. Respond with "
    "JSON only, matching this shape: {origin, destinations: [...], depart_date, "
    "return_date, budget_usd, traveler_count, preferences: [...]}. If a required "
    "field is missing, respond with {\"missing_field\": \"<field name>\"}."
)


async def collect_trip_request(state: GraphState, deps: NodeDeps) -> dict:
    user_message = state.get("user_message", "")
    while True:
        raw = await with_retries(
            lambda: deps.llm.complete(system=_SYSTEM_PROMPT, prompt=user_message),
            exceptions=(LLMTimeoutError, LLMRateLimitError),
        )
        parsed = json.loads(raw)
        if "missing_field" in parsed:
            user_message = interrupt({"ask_for": parsed["missing_field"]})
            continue
        try:
            trip_request = TripRequest(**parsed)
        except Exception as exc:
            user_message = interrupt({"validation_error": str(exc)})
            continue
        return {"trip_request": trip_request, "error_log": state.get("error_log", [])}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_node_collect_trip_request.py -v`
Expected: PASS.

- [ ] **Step 6: Add a test for the missing-field interrupt loop**

Append to `backend/tests/test_node_collect_trip_request.py`:
```python
@pytest.mark.asyncio
async def test_collect_trip_request_reprompts_on_missing_field(monkeypatch):
    deps = make_deps(
        llm_responses=[
            '{"missing_field": "budget_usd"}',
            '{"origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01", '
            '"return_date": "2026-09-10", "budget_usd": 2000, "traveler_count": 1, '
            '"preferences": []}',
        ]
    )
    # Simulate LangGraph resuming interrupt() with the user's follow-up answer.
    monkeypatch.setattr(
        "app.graph.nodes.collect_trip_request.interrupt", lambda payload: "my budget is $2000"
    )
    state = {"user_message": "I want to fly to Paris"}
    result = await collect_trip_request(state, deps)
    assert result["trip_request"].budget_usd == 2000
```

- [ ] **Step 7: Run the full test file**

Run: `pytest tests/test_node_collect_trip_request.py -v`
Expected: both tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/graph/deps.py backend/app/graph/__init__.py backend/app/graph/nodes/
git add backend/tests/test_node_collect_trip_request.py
git commit -m "feat: add collect_trip_request node with interrupt-driven field collection"
```

---

### Task 11: Node — `search_flights_and_hotels`

**Files:**
- Create: `backend/app/graph/nodes/search_flights_and_hotels.py`
- Test: `backend/tests/test_node_search_flights_and_hotels.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `FlightResultRecord`/`HotelResultRecord`/`TripRequestRecord` (Task 4), `AmadeusRateLimitError`/`with_retries` (Task 6).
- Produces: `search_flights_and_hotels(state, deps) -> dict` setting `flight_result_ids`, `hotel_result_ids`, `shortlist` (or a `no_results: True` marker consumed by the graph's conditional edge in Task 22).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_search_flights_and_hotels.py`:
```python
import pytest

from app.graph.deps import NodeDeps
from app.graph.nodes.search_flights_and_hotels import search_flights_and_hotels
from app.llm.fake import FakeLLMClient
from app.models.domain import TripRequest
from app.tools.fake import (
    FakeBookingFormTool,
    FakeCRMTool,
    FakeEmailTool,
    FakeFlightHotelSearchTool,
)


def make_deps(search_behavior="success"):
    return NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(behavior=search_behavior),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: None,
        encryption_key=b"0" * 32,
    )


def trip_request():
    return TripRequest(
        origin="JFK",
        destinations=["CDG", "FCO"],
        depart_date="2026-09-01",
        return_date="2026-09-10",
        budget_usd=3000,
        traveler_count=1,
        preferences=[],
    )


@pytest.mark.asyncio
async def test_search_populates_result_ids_per_leg():
    deps = make_deps()
    state = {"trip_request": trip_request()}
    result = await search_flights_and_hotels(state, deps)
    assert len(result["flight_result_ids"]) == 2  # one per destination leg
    assert len(result["hotel_result_ids"]) == 2
    assert result.get("no_results") is not True


@pytest.mark.asyncio
async def test_search_empty_sets_no_results_flag():
    deps = make_deps(search_behavior="empty")
    state = {"trip_request": trip_request()}
    result = await search_flights_and_hotels(state, deps)
    assert result["no_results"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_search_flights_and_hotels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.graph.nodes.search_flights_and_hotels'`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/search_flights_and_hotels.py`:
```python
from app.errors import AmadeusRateLimitError, with_retries
from app.graph.deps import NodeDeps
from app.models.state import GraphState


async def search_flights_and_hotels(state: GraphState, deps: NodeDeps) -> dict:
    trip_request = state["trip_request"]
    flight_result_ids: list[str] = []
    hotel_result_ids: list[str] = []
    shortlist: dict = {}

    for leg_index, destination in enumerate(trip_request.destinations):
        origin = trip_request.origin if leg_index == 0 else trip_request.destinations[leg_index - 1]

        flights, hotels = await with_retries(
            lambda: deps.search_tool.search(
                origin=origin, destination=destination, depart_date=str(trip_request.depart_date)
            ),
            exceptions=(AmadeusRateLimitError,),
        )
        if not flights or not hotels:
            return {"no_results": True, "error_log": state.get("error_log", [])}

        flight_result_ids.extend(f.id for f in flights)
        hotel_result_ids.extend(h.id for h in hotels)
        shortlist[str(leg_index)] = {
            "flights": [f.model_dump(mode="json") for f in flights],
            "hotels": [h.model_dump(mode="json") for h in hotels],
        }

    return {
        "flight_result_ids": flight_result_ids,
        "hotel_result_ids": hotel_result_ids,
        "shortlist": shortlist,
        "no_results": False,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_search_flights_and_hotels.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/search_flights_and_hotels.py
git add backend/tests/test_node_search_flights_and_hotels.py
git commit -m "feat: add search_flights_and_hotels node with no_results routing signal"
```

---

### Task 12: Node — `recommend_options`

**Files:**
- Create: `backend/app/graph/nodes/recommend_options.py`
- Test: `backend/tests/test_node_recommend_options.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `state["shortlist"]` (Task 11).
- Produces: `recommend_options(state, deps) -> dict` setting `state["recommendation_narrative"]` (LLM-generated text) while leaving `shortlist`'s numeric fields untouched (spec §6.4 — numeric facts are never LLM-generated).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_recommend_options.py`:
```python
import pytest

from app.graph.deps import NodeDeps
from app.graph.nodes.recommend_options import recommend_options
from app.llm.fake import FakeLLMClient
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.mark.asyncio
async def test_recommend_options_narrates_without_altering_prices():
    shortlist = {
        "0": {
            "flights": [{"id": "f1", "price_usd": 450.0, "carrier": "AF", "stops": 0}],
            "hotels": [{"id": "h1", "price_usd_total": 600.0, "rating": 4.2}],
        }
    }
    deps = NodeDeps(
        llm=FakeLLMClient(responses=["Flight AF for $450 direct, hotel rated 4.2 for $600."]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: None,
        encryption_key=b"0" * 32,
    )
    state = {"shortlist": shortlist}
    result = await recommend_options(state, deps)
    assert result["shortlist"] == shortlist  # untouched — numeric data is not LLM-regenerated
    assert "450" in result["recommendation_narrative"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_recommend_options.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/recommend_options.py`:
```python
from app.errors import LLMRateLimitError, LLMTimeoutError, with_retries
from app.graph.deps import NodeDeps
from app.models.state import GraphState

_SYSTEM_PROMPT = (
    "You narrate flight/hotel shortlists for a traveler. Use only the numbers "
    "given to you verbatim — never invent or alter a price, time, or rating."
)


async def recommend_options(state: GraphState, deps: NodeDeps) -> dict:
    shortlist = state["shortlist"]

    narrative = await with_retries(
        lambda: deps.llm.complete(system=_SYSTEM_PROMPT, prompt=str(shortlist)),
        exceptions=(LLMTimeoutError, LLMRateLimitError),
    )
    return {"shortlist": shortlist, "recommendation_narrative": narrative}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_recommend_options.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/recommend_options.py
git add backend/tests/test_node_recommend_options.py
git commit -m "feat: add recommend_options node with fact-preserving narration"
```

---

### Task 13: Node — `generate_itinerary`

**Files:**
- Create: `backend/app/graph/nodes/generate_itinerary.py`
- Test: `backend/tests/test_node_generate_itinerary.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `Itinerary`/`ItineraryDay` (Task 2), `state["trip_request"]`.
- Produces: `generate_itinerary(state, deps) -> dict` setting `state["itinerary"]` (preliminary, before flight/hotel selection).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_generate_itinerary.py`:
```python
import json

import pytest

from app.graph.deps import NodeDeps
from app.graph.nodes.generate_itinerary import generate_itinerary
from app.llm.fake import FakeLLMClient
from app.models.domain import TripRequest
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.mark.asyncio
async def test_generate_itinerary_builds_one_day_per_night():
    trip_request = TripRequest(
        origin="JFK", destinations=["CDG"], depart_date="2026-09-01",
        return_date="2026-09-03", budget_usd=2000, traveler_count=1, preferences=["museums"],
    )
    canned = json.dumps({"days": [
        {"day_index": 0, "date": "2026-09-01", "activities": ["Louvre"]},
        {"day_index": 1, "date": "2026-09-02", "activities": ["Eiffel Tower"]},
    ]})
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[canned]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: None,
        encryption_key=b"0" * 32,
    )
    state = {"trip_request": trip_request}
    result = await generate_itinerary(state, deps)
    assert len(result["itinerary"].days) == 2
    assert result["itinerary"].days[0].activities == ["Louvre"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_generate_itinerary.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the node**

`backend/app/graph/nodes/generate_itinerary.py`:
```python
import json

from app.errors import LLMRateLimitError, LLMTimeoutError, with_retries
from app.graph.deps import NodeDeps
from app.models.domain import Itinerary
from app.models.state import GraphState

_SYSTEM_PROMPT = (
    "Build a preliminary day-by-day itinerary for the traveler's destination(s) "
    "and preferences. Respond with JSON only: {\"days\": [{\"day_index\", \"date\", "
    "\"activities\": [...]}]}."
)


async def generate_itinerary(state: GraphState, deps: NodeDeps) -> dict:
    trip_request = state["trip_request"]

    raw = await with_retries(
        lambda: deps.llm.complete(system=_SYSTEM_PROMPT, prompt=trip_request.model_dump_json()),
        exceptions=(LLMTimeoutError, LLMRateLimitError),
    )
    itinerary = Itinerary(**json.loads(raw))
    return {"itinerary": itinerary}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_generate_itinerary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/graph/nodes/generate_itinerary.py
git add backend/tests/test_node_generate_itinerary.py
git commit -m "feat: add generate_itinerary node"
```

---

### Task 14: Node — `await_selection`

**Files:**
- Create: `backend/app/graph/nodes/await_selection.py`
- Test: `backend/tests/test_node_await_selection.py`

**Interfaces:**
- Consumes: `interrupt` from `langgraph.types`, `state["shortlist"]`.
- Produces: `await_selection(state, deps) -> dict` setting `selected_flights`, `selected_hotels` from the resume payload.

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
git add backend/app/graph/nodes/await_selection.py
git add backend/tests/test_node_await_selection.py
git commit -m "feat: add await_selection interrupt node"
```

---

### Task 15: Node — `finalize_itinerary`

**Files:**
- Create: `backend/app/graph/nodes/finalize_itinerary.py`
- Test: `backend/tests/test_node_finalize_itinerary.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `state["itinerary"]`, `state["selected_flights"]`, `state["selected_hotels"]`.
- Produces: `finalize_itinerary(state, deps) -> dict` setting an updated `state["itinerary"]` incorporating actual selected flight times/hotel locations.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_finalize_itinerary.py`:
```python
import json

import pytest

from app.graph.deps import NodeDeps
from app.graph.nodes.finalize_itinerary import finalize_itinerary
from app.llm.fake import FakeLLMClient
from app.models.domain import Itinerary
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.mark.asyncio
async def test_finalize_itinerary_incorporates_selected_options():
    preliminary = Itinerary(days=[{"day_index": 0, "date": "2026-09-01", "activities": ["Louvre"]}])
    canned = json.dumps({"days": [
        {"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm", "Check in", "Louvre"]}
    ]})
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[canned]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: None,
        encryption_key=b"0" * 32,
    )
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
git add backend/app/graph/nodes/finalize_itinerary.py
git add backend/tests/test_node_finalize_itinerary.py
git commit -m "feat: add finalize_itinerary node"
```

---

### Task 16: Node — `collect_traveler_details`

**Files:**
- Create: `backend/app/graph/nodes/collect_traveler_details.py`
- Test: `backend/tests/test_node_collect_traveler_details.py`

**Interfaces:**
- Consumes: `interrupt`, `TravelerDetailRecord` (Task 4), `encrypt_field` (Task 3), `NodeDeps.session_factory`/`encryption_key`.
- Produces: `collect_traveler_details(state, deps) -> dict` setting only `state["traveler_details_id"]` — never raw traveler fields on state (spec §6.1).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_collect_traveler_details.py`:
```python
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import decrypt_field, generate_test_key
from app.graph.deps import NodeDeps
from app.graph.nodes.collect_traveler_details import collect_traveler_details
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base, TravelerDetailRecord, TripRequestRecord
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_collect_traveler_details_stores_encrypted_and_returns_only_id(engine, monkeypatch):
    key = generate_test_key()
    trip_id = str(uuid.uuid4())
    with Session(engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        session.commit()

    monkeypatch.setattr(
        "app.graph.nodes.collect_traveler_details.interrupt",
        lambda payload: {
            "full_name": "Jane Doe",
            "date_of_birth": "1990-01-01",
            "passport_number": "X1234567",
            "contact_email": "jane@example.com",
        },
    )
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: Session(engine),
        encryption_key=key,
    )
    state = {"trip_request_id": trip_id}
    result = collect_traveler_details(state, deps)

    assert "passport_number" not in result
    assert "full_name" not in result
    assert set(result.keys()) == {"traveler_details_id"}

    with Session(engine) as session:
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
git add backend/app/graph/nodes/collect_traveler_details.py
git add backend/tests/test_node_collect_traveler_details.py
git commit -m "feat: add collect_traveler_details node storing only encrypted PII"
```

---

### Task 17: Node — `fill_booking_form`

**Files:**
- Create: `backend/app/graph/nodes/fill_booking_form.py`
- Test: `backend/tests/test_node_fill_booking_form.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `TravelerDetailRecord` (Task 4), `decrypt_field` (Task 3), `PlaywrightTimeoutError`/`with_retries` — retry is deliberately NOT used here per spec §5.2/§7 (form-fill failures route to a dedicated failure state, not blind retry).
- Produces: `fill_booking_form(state, deps) -> dict` setting `booking_form_results` (list, one per leg) or `{"form_fill_failed": True, "error_log": [...]}`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_fill_booking_form.py`:
```python
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import encrypt_field, generate_test_key
from app.graph.deps import NodeDeps
from app.graph.nodes.fill_booking_form import fill_booking_form
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base, TravelerDetailRecord, TripRequestRecord
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def make_traveler(engine, key):
    trip_id = str(uuid.uuid4())
    with Session(engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        traveler = TravelerDetailRecord(
            trip_request_id=trip_id,
            full_name="Jane Doe",
            date_of_birth_enc=encrypt_field("1990-01-01", key),
            passport_number_enc=encrypt_field("X1234567", key),
            contact_email="jane@example.com",
        )
        session.add(traveler)
        session.commit()
        return traveler.id


@pytest.mark.asyncio
async def test_fill_booking_form_loops_once_per_leg(engine):
    key = generate_test_key()
    traveler_id = make_traveler(engine, key)
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(behavior="success"),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: Session(engine),
        encryption_key=key,
    )
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
async def test_fill_booking_form_failure_does_not_retry_and_sets_flag(engine):
    key = generate_test_key()
    traveler_id = make_traveler(engine, key)
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(behavior="failure"),
        email_tool=FakeEmailTool(),
        crm_tool=FakeCRMTool(),
        session_factory=lambda: Session(engine),
        encryption_key=key,
    )
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
git add backend/app/graph/nodes/fill_booking_form.py
git add backend/tests/test_node_fill_booking_form.py
git commit -m "feat: add fill_booking_form node with per-leg loop and no blind retry"
```

---

### Task 18: Node — `human_review_gate`

**Files:**
- Create: `backend/app/graph/nodes/human_review_gate.py`
- Test: `backend/tests/test_node_human_review_gate.py`

**Interfaces:**
- Consumes: `interrupt`, `state["booking_form_results"]`.
- Produces: `human_review_gate(state, deps) -> dict` setting `approval_status` and `rejection_reason`.

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
git add backend/app/graph/nodes/human_review_gate.py
git add backend/tests/test_node_human_review_gate.py
git commit -m "feat: add human_review_gate node with rejection_reason routing signal"
```

---

### Task 19: Node — `send_confirmation_email`

**Files:**
- Create: `backend/app/graph/nodes/send_confirmation_email.py`
- Test: `backend/tests/test_node_send_confirmation_email.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `check_and_reserve` (Task 7), `GmailSendError`/`with_retries` (Task 6), `TravelerDetailRecord` (Task 4) — looked up via `state["traveler_details_id"]` to get the recipient's name/email (not sensitive fields, stored as plaintext columns per Task 4).
- Produces: `send_confirmation_email(state, deps) -> dict` setting `email_status`, idempotent on `state["trip_request_id"]`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_send_confirmation_email.py`:
```python
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import encrypt_field, generate_test_key
from app.graph.deps import NodeDeps
from app.graph.nodes.send_confirmation_email import send_confirmation_email
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base, TravelerDetailRecord, TripRequestRecord
from app.models.domain import Itinerary
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def make_traveler(engine, trip_id):
    key = generate_test_key()
    with Session(engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        traveler = TravelerDetailRecord(
            trip_request_id=trip_id,
            full_name="Jane Doe",
            date_of_birth_enc=encrypt_field("1990-01-01", key),
            passport_number_enc=encrypt_field("X1234567", key),
            contact_email="jane@example.com",
        )
        session.add(traveler)
        session.commit()
        return traveler.id


def make_state(trip_id, traveler_id):
    return {
        "trip_request_id": trip_id,
        "traveler_details_id": traveler_id,
        "itinerary": Itinerary(days=[{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]),
    }


@pytest.mark.asyncio
async def test_send_confirmation_email_sends_once(engine):
    email_tool = FakeEmailTool()
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=email_tool,
        crm_tool=FakeCRMTool(),
        session_factory=lambda: Session(engine),
        encryption_key=b"0" * 32,
    )
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler(engine, trip_id)
    result = await send_confirmation_email(make_state(trip_id, traveler_id), deps)
    assert result["email_status"] == "sent"
    assert len(email_tool.sent) == 1
    assert email_tool.sent[0]["to"] == "jane@example.com"


@pytest.mark.asyncio
async def test_send_confirmation_email_is_idempotent_on_retry(engine):
    email_tool = FakeEmailTool()
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=email_tool,
        crm_tool=FakeCRMTool(),
        session_factory=lambda: Session(engine),
        encryption_key=b"0" * 32,
    )
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler(engine, trip_id)
    state = make_state(trip_id, traveler_id)
    await send_confirmation_email(state, deps)
    result = await send_confirmation_email(state, deps)  # simulated retry after crash
    assert result["email_status"] == "already_sent"
    assert len(email_tool.sent) == 1  # not sent twice
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
    lines = [f"Day {day.day_index + 1} ({day.date}): {', '.join(day.activities)}" for day in state["itinerary"].days]
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
                to=recipient_email,
                subject="Your trip itinerary",
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
git add backend/app/graph/nodes/send_confirmation_email.py
git add backend/tests/test_node_send_confirmation_email.py
git commit -m "feat: add idempotent send_confirmation_email node"
```

---

### Task 20: Node — `update_crm`

**Files:**
- Create: `backend/app/graph/nodes/update_crm.py`
- Test: `backend/tests/test_node_update_crm.py`

**Interfaces:**
- Consumes: `NodeDeps` (Task 10), `check_and_reserve` (Task 7), `HubSpotError`/`with_retries` (Task 6), `TravelerDetailRecord` (Task 4) — looked up via `state["traveler_details_id"]`; `state["trip_request"]` for the trip summary.
- Produces: `update_crm(state, deps) -> dict` setting `crm_status`, idempotent on `state["trip_request_id"]`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_node_update_crm.py`:
```python
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.crypto import encrypt_field, generate_test_key
from app.graph.deps import NodeDeps
from app.graph.nodes.update_crm import update_crm
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base, TravelerDetailRecord, TripRequestRecord
from app.models.domain import TripRequest
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def make_traveler(engine, trip_id):
    key = generate_test_key()
    with Session(engine) as session:
        session.add(TripRequestRecord(id=trip_id, raw_request={}))
        traveler = TravelerDetailRecord(
            trip_request_id=trip_id,
            full_name="Jane Doe",
            date_of_birth_enc=encrypt_field("1990-01-01", key),
            passport_number_enc=encrypt_field("X1234567", key),
            contact_email="jane@example.com",
        )
        session.add(traveler)
        session.commit()
        return traveler.id


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
async def test_update_crm_creates_contact_and_trip_record_once(engine):
    crm_tool = FakeCRMTool()
    deps = NodeDeps(
        llm=FakeLLMClient(responses=[]),
        search_tool=FakeFlightHotelSearchTool(),
        booking_tool=FakeBookingFormTool(),
        email_tool=FakeEmailTool(),
        crm_tool=crm_tool,
        session_factory=lambda: Session(engine),
        encryption_key=b"0" * 32,
    )
    trip_id = str(uuid.uuid4())
    traveler_id = make_traveler(engine, trip_id)
    state = make_state(trip_id, traveler_id)

    result = await update_crm(state, deps)
    assert result["crm_status"] == "created"
    assert len(crm_tool.trip_records) == 1

    retry_result = await update_crm(state, deps)
    assert retry_result["crm_status"] == "already_created"
    assert len(crm_tool.trip_records) == 1
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
git add backend/app/graph/nodes/update_crm.py
git add backend/tests/test_node_update_crm.py
git commit -m "feat: add idempotent update_crm node"
```

---

### Task 21: Node — `agent_error`

**Files:**
- Create: `backend/app/graph/nodes/agent_error.py`
- Test: `backend/tests/test_node_agent_error.py`

**Interfaces:**
- Consumes: `state["error_log"]`.
- Produces: `agent_error(state, deps) -> dict` setting `state["user_facing_message"]` — a plain-language message with no raw error details.

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

### Task 22: Graph Assembly

**Files:**
- Create: `backend/app/graph/build.py`
- Test: `backend/tests/test_graph_end_to_end.py`

**Interfaces:**
- Consumes: every node function from Tasks 10-21, `GraphState` (Task 2), `get_checkpointer` (Task 5), `NodeDeps` (Task 10).
- Produces: `build_graph(deps: NodeDeps, checkpointer) -> CompiledGraph` via `build_graph`. This is what Phase 2's FastAPI layer (and this phase's own `test_graph_end_to_end.py`) invokes with `graph.invoke(...)` / `Command(resume=...)`.

- [ ] **Step 1: Write the failing end-to-end happy-path test**

`backend/tests/test_graph_end_to_end.py`:
```python
import json
import uuid

import pytest
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_checkpointer
from app.graph.build import build_graph
from app.graph.deps import NodeDeps
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def db_engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_happy_path_end_to_end(db_engine):
    llm_responses = [
        # collect_trip_request
        json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        }),
        # recommend_options narrative
        "Here are your options.",
        # generate_itinerary
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),
        # finalize_itinerary
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),
    ]
    deps = NodeDeps(
        llm=FakeLLMClient(responses=llm_responses),
        search_tool=FakeFlightHotelSearchTool(behavior="success"),
        booking_tool=FakeBookingFormTool(behavior="success"),
        email_tool=FakeEmailTool(behavior="success"),
        crm_tool=FakeCRMTool(behavior="success"),
        session_factory=lambda: Session(db_engine),
        encryption_key=b"0" * 32,
    )

    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        trip_request_id = str(uuid.uuid4())
        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": trip_request_id},
            config,
        )
        # resume await_selection
        graph.invoke(
            Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config
        )
        # resume human_review_gate (after collect_traveler_details interrupt + fill_booking_form ran)
        graph.invoke(
            Command(resume={
                "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
                "passport_number": "X1234567", "contact_email": "jane@example.com",
            }),
            config,
        )
        final_state = graph.invoke(Command(resume={"approved": True}), config)

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
    if state.get("no_results"):
        return "collect_trip_request"
    return "recommend_options"


def _route_after_fill(state: GraphState) -> str:
    if state.get("form_fill_failed"):
        return "agent_error"
    return "human_review_gate"


def _route_after_review(state: GraphState) -> str:
    if state["approval_status"] == "approved":
        return "send_confirmation_email"
    if state["rejection_reason"] == "wrong_selection":
        return "await_selection"
    return "collect_traveler_details"


def build_graph(deps: NodeDeps, checkpointer):
    builder = StateGraph(GraphState)

    builder.add_node("collect_trip_request", partial(collect_trip_request, deps=deps))
    builder.add_node("search_flights_and_hotels", partial(search_flights_and_hotels, deps=deps))
    builder.add_node("recommend_options", partial(recommend_options, deps=deps))
    builder.add_node("generate_itinerary", partial(generate_itinerary, deps=deps))
    builder.add_node("await_selection", partial(await_selection, deps=deps))
    builder.add_node("finalize_itinerary", partial(finalize_itinerary, deps=deps))
    builder.add_node("collect_traveler_details", partial(collect_traveler_details, deps=deps))
    builder.add_node("fill_booking_form", partial(fill_booking_form, deps=deps))
    builder.add_node("human_review_gate", partial(human_review_gate, deps=deps))
    builder.add_node("send_confirmation_email", partial(send_confirmation_email, deps=deps))
    builder.add_node("update_crm", partial(update_crm, deps=deps))
    builder.add_node("agent_error", partial(agent_error, deps=deps))

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
Expected: PASS. If it fails on a node signature mismatch (`partial(fn, deps=deps)` requires each node function's second parameter to be named `deps`), fix the affected node's signature to `def node_name(state: GraphState, *, deps: NodeDeps) -> dict` consistently across all node files from Tasks 10-21, and re-run.

- [ ] **Step 5: Write the failing reject-path test**

Append to `backend/tests/test_graph_end_to_end.py`:
```python
def test_reject_wrong_selection_routes_back_to_await_selection(db_engine):
    llm_responses = [
        json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        }),
        "Here are your options.",
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),
        # finalize_itinerary runs again after the reject loops back through await_selection
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 7pm"]}]}),
    ]
    deps = NodeDeps(
        llm=FakeLLMClient(responses=llm_responses),
        search_tool=FakeFlightHotelSearchTool(behavior="success"),
        booking_tool=FakeBookingFormTool(behavior="success"),
        email_tool=FakeEmailTool(behavior="success"),
        crm_tool=FakeCRMTool(behavior="success"),
        session_factory=lambda: Session(db_engine),
        encryption_key=b"0" * 32,
    )
    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        trip_request_id = str(uuid.uuid4())

        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": trip_request_id},
            config,
        )
        graph.invoke(Command(resume={"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}), config)
        graph.invoke(
            Command(resume={
                "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
                "passport_number": "X1234567", "contact_email": "jane@example.com",
            }),
            config,
        )
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
Expected: initial FAIL if routing/state carryover is wrong (commonly: `rejection_reason` not reset before the next review, or `booking_form_results` not recomputed) — debug against the actual assertion failure, fix `_route_after_review` or node state carryover in `build.py`/`fill_booking_form.py` accordingly, then re-run until both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/graph/build.py backend/tests/test_graph_end_to_end.py
git commit -m "feat: assemble full LangGraph agent with happy-path and reject-path routing"
```

---

### Task 23: Persistence and Crash-Recovery Tests

**Files:**
- Test: `backend/tests/test_graph_persistence.py`

**Interfaces:**
- Consumes: `build_graph` (Task 22), `get_checkpointer` (Task 5), `check_and_reserve` (Task 7).

- [ ] **Step 1: Write the failing checkpoint-resume-after-restart test**

`backend/tests/test_graph_persistence.py`:
```python
import json
import uuid

import pytest
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_checkpointer
from app.graph.build import build_graph
from app.graph.deps import NodeDeps
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def db_engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def make_deps(db_engine, llm_responses):
    return NodeDeps(
        llm=FakeLLMClient(responses=llm_responses),
        search_tool=FakeFlightHotelSearchTool(behavior="success"),
        booking_tool=FakeBookingFormTool(behavior="success"),
        email_tool=FakeEmailTool(behavior="success"),
        crm_tool=FakeCRMTool(behavior="success"),
        session_factory=lambda: Session(db_engine),
        encryption_key=b"0" * 32,
    )


def test_resume_after_process_restart_continues_from_last_checkpoint(db_engine):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    trip_request_id = str(uuid.uuid4())

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
        graph = build_graph(make_deps(db_engine, llm_responses), checkpointer)
        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": trip_request_id},
            config,
        )
        state_before_restart = graph.get_state(config)
        assert "await_selection" in str(state_before_restart.next)

    # "Process 2": brand-new checkpointer connection and graph instance,
    # same thread_id — proves resume works across a real reconnect, not just
    # a live in-memory object.
    with get_checkpointer(get_settings().test_database_url) as checkpointer_2:
        graph_2 = build_graph(
            make_deps(db_engine, ["Here are your options.",
                                  json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]})]),
            checkpointer_2,
        )
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
from app.idempotency import check_and_reserve


def test_idempotency_prevents_duplicate_email_after_crash_before_status_persisted(db_engine):
    """Simulates: send_confirmation_email's side effect (the email) succeeded,
    but the process crashed before email_status was written to graph state.
    On resume, the node must not send a second email."""
    with Session(db_engine) as session:
        trip_request_id = str(uuid.uuid4())
        key = f"{trip_request_id}:send_confirmation_email"
        assert check_and_reserve(session, key, "send_confirmation_email") is True
        # ^ this simulates the first attempt's reservation succeeding and the
        # email being sent, right before a crash.

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

### Task 24: PII-Leak Tests

**Files:**
- Test: `backend/tests/test_pii_leak.py`

**Interfaces:**
- Consumes: `build_graph` (Task 22), `get_checkpointer` (Task 5).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_pii_leak.py`:
```python
import json
import uuid

import pytest
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_checkpointer
from app.graph.build import build_graph
from app.graph.deps import NodeDeps
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool

SYNTHETIC_PASSPORT = "Z9988776"  # known fixture value, must never appear verbatim outside traveler_details table


@pytest.fixture
def db_engine():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_synthetic_passport_never_appears_in_checkpoints_or_error_log(db_engine):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    trip_request_id = str(uuid.uuid4())
    llm_responses = [
        json.dumps({
            "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
            "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
        }),
        "Here are your options.",
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive"]}]}),
        json.dumps({"days": [{"day_index": 0, "date": "2026-09-01", "activities": ["Arrive at 6pm"]}]}),
    ]
    deps = NodeDeps(
        llm=FakeLLMClient(responses=llm_responses),
        search_tool=FakeFlightHotelSearchTool(behavior="success"),
        booking_tool=FakeBookingFormTool(behavior="success"),
        email_tool=FakeEmailTool(behavior="success"),
        crm_tool=FakeCRMTool(behavior="success"),
        session_factory=lambda: Session(db_engine),
        encryption_key=b"0" * 32,
    )
    with get_checkpointer(get_settings().test_database_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(deps, checkpointer)
        graph.invoke(
            {"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler", "trip_request_id": trip_request_id},
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
Expected: PASS if Task 16's design (storing only `traveler_details_id` on state) was followed correctly by every downstream node. If it FAILS, the failure itself is the finding — a node somewhere is putting raw traveler data onto graph state; locate it via the assertion's checkpoint content and fix that node to use the `traveler_details_id` pattern instead.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_pii_leak.py
git commit -m "test: assert synthetic passport value never appears in checkpoints or state"
```

---

### Task 25: FastAPI Endpoints

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/schemas.py`
- Create: `backend/app/api/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `build_graph` (Task 22), `get_checkpointer` (Task 5), `NodeDeps` (Task 10).
- Produces: FastAPI app with `POST /conversations` (start), `POST /conversations/{thread_id}/messages` (send a message / resume an interrupt), `GET /conversations/{thread_id}/state` (poll current state/interrupt payload). Real Amadeus/Playwright/Gmail/HubSpot wiring is Phase 2 — this task wires the app with fakes via dependency overrides, proving the HTTP layer round-trips correctly through interrupts.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_api.py`:
```python
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.main import app, get_deps
from app.config import get_settings
from app.graph.deps import NodeDeps
from app.llm.fake import FakeLLMClient
from app.models.db_models import Base
from app.tools.fake import FakeBookingFormTool, FakeCRMTool, FakeEmailTool, FakeFlightHotelSearchTool


@pytest.fixture
def client():
    engine = create_engine(get_settings().test_database_url)
    Base.metadata.create_all(engine)

    def override_deps():
        return NodeDeps(
            llm=FakeLLMClient(responses=[
                json.dumps({
                    "origin": "JFK", "destinations": ["CDG"], "depart_date": "2026-09-01",
                    "return_date": "2026-09-05", "budget_usd": 2000, "traveler_count": 1, "preferences": [],
                }),
            ]),
            search_tool=FakeFlightHotelSearchTool(behavior="success"),
            booking_tool=FakeBookingFormTool(behavior="success"),
            email_tool=FakeEmailTool(behavior="success"),
            crm_tool=FakeCRMTool(behavior="success"),
            session_factory=lambda: Session(engine),
            encryption_key=b"0" * 32,
        )

    app.dependency_overrides[get_deps] = override_deps
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_start_conversation_returns_thread_id_and_first_interrupt(client):
    response = client.post(
        "/conversations",
        json={"user_message": "JFK to Paris Sept 1-5, $2000, 1 traveler"},
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
    settings = get_settings()
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
def resume_conversation(
    thread_id: str, body: ResumeRequest, deps: NodeDeps = Depends(get_deps)
):
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

    after_selection = client.post(
        f"/conversations/{thread_id}/messages",
        json={"resume_payload": {"flights": [{"id": "f1"}], "hotels": [{"id": "h1"}]}},
    ).json()

    after_traveler_details = client.post(
        f"/conversations/{thread_id}/messages",
        json={"resume_payload": {
            "full_name": "Jane Doe", "date_of_birth": "1990-01-01",
            "passport_number": "X1234567", "contact_email": "jane@example.com",
        }},
    ).json()

    final = client.post(
        f"/conversations/{thread_id}/messages",
        json={"resume_payload": {"approved": True}},
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
git add backend/app/api/
git add backend/tests/test_api.py
git commit -m "feat: add FastAPI endpoints wiring the LangGraph agent through HTTP"
```

---

### Task 26: Full Suite Verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run:
```bash
cd backend
docker compose up -d postgres  # from repo root, if not already running
alembic upgrade head
pytest -v
```
Expected: all tests across all 25 preceding tasks PASS (roughly 60+ individual test functions).

- [ ] **Step 2: Confirm no test depends on a live third-party API**

Run: `grep -rn "amadeus.com\|googleapis.com\|hubapi.com\|anthropic.com" backend/tests/`
Expected: no matches — every test in this phase uses `Fake*` implementations, per spec §9 ("keeping normal CI deterministic").

- [ ] **Step 3: Commit the final state if any fixups were needed**

```bash
git add -A
git commit -m "chore: fix up any remaining issues found during full Phase 1 suite run"
```

(Skip this commit if Step 1 passed cleanly with no changes needed.)
