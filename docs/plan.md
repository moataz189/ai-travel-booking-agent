# AI Travel Planning and Booking Agent — Implementation Plan

**Companion document:** [`docs/spec.md`](spec.md) contains the full design specification
(architecture, security model, error handling, testing strategy, risks, alternatives
considered). This document is the implementation roadmap: what gets built, in what order, what
each task produces, how tasks depend on each other, and how to verify each one is done.

## 1. Overview

### 1.1 Problem Statement

Manually planning a multi-leg trip — comparing flights and hotels across sites, building a
day-by-day itinerary, and filling in booking forms — is tedious. This project builds an AI
agent that automates search, comparison, and itinerary planning end-to-end, while keeping the
one irreversible/financial step (payment) strictly under human control.

### 1.2 Objectives

1. A full working demo: a user states trip constraints in conversation, receives ranked
   flight/hotel options (including multi-leg trips), receives a personalized itinerary, picks
   options, the agent fills a mock booking form via browser automation, the flow stops for
   **explicit human approval**, a confirmation email is sent, and the trip is logged to a CRM.
2. A structurally safe human-in-the-loop pattern for an agent operating adjacent to a
   payment-shaped flow — safety enforced by graph topology and mock-site design, not by
   trusting model behavior.
3. Real infrastructure: a self-managed Kubernetes cluster on AWS EC2 (kubeadm), provisioned by
   Terraform and Packer, deployed via GitOps (Argo CD), observed via Prometheus/Grafana, built
   and shipped via CI/CD.

### 1.3 Non-Goals

No real payment processing (ever, not even later); no real booking-website integration (mock
site only); no multi-user accounts/authentication; no mobile app; no dedicated
points-of-interest API (itinerary content is LLM-generated); no highly-available control plane
(single node, explicitly accepted limitation). Full rationale in `spec.md` §3.

### 1.4 Build Order

Seven phases, strictly sequential — each assumes every prior phase is implemented and its
tests pass:

1. **Core Agent Logic** — the LangGraph agent, fully tested against fakes and a real local
   Postgres. No real MCP servers, no infrastructure.
2. **Real Integrations & Mock Booking Site** — fakes replaced with real sandboxed
   integrations; still local.
3. **Infrastructure Provisioning** — Terraform/Packer/kubeadm cluster on AWS.
4. **Kubernetes Deployment** — application services deployed to the cluster.
5. **CI/CD and GitOps** — Argo CD becomes the sole applier of manifests.
6. **Observability** — Prometheus/Grafana and application-level metrics.
7. **Production Validation** — full end-to-end verification on the real AWS environment.

---

## 2. Global Constraints

These apply to every task below; each task's acceptance criteria assume them.

- Never store traveler PII (passport/ID number, DOB) directly on LangGraph graph state — only
  a `traveler_details_id` foreign key (spec §5.3, §6.1).
- `error_log` entries are sanitized `{code, message}` pairs only — never PII, tokens, API keys,
  or raw upstream responses (spec §5.3).
- All MCP-tool-calling and LLM-calling nodes get bounded retries (3 attempts, exponential
  backoff) before routing to `agent_error` (spec §5.2, §7) — except `fill_booking_form`, which
  deliberately never retries.
- `send_confirmation_email` and `update_crm` are idempotent via a stored idempotency key
  derived from the trip/session ID (spec §7).
- Traveler PII columns (passport/ID number, DOB) are encrypted at the application layer before
  insert; the key is never a plaintext env var in real deployments (spec §6.1).
- `human_review_gate` covers **all legs** of a multi-leg trip in one combined decision;
  `fill_booking_form` loops once per leg (spec §5.2, nodes 8-9).
- `rejection_reason` is one of `wrong_selection` | `wrong_traveler_details` and drives
  reject-routing (spec §5.2, node 9).
- The mock booking site has no payment field or endpoint anywhere in its codebase — a
  structural absence, not a policy the agent is trusted to follow (spec §6.5).

---

## 3. Phase 1 — Core Agent Logic

**Implements:** the LangGraph `StateGraph` (all 12 workflow nodes + `agent_error`), both
`interrupt()` gates, Postgres persistence (app tables + LangGraph checkpointer), encrypted
traveler PII, idempotent side effects, sanitized error handling, and a FastAPI HTTP layer —
tested end to end against fake tool/LLM clients and a real local Postgres.

**Tech stack:** Python 3.12, FastAPI, LangGraph + `langgraph-checkpoint-postgres`, SQLAlchemy
2.0 + Alembic, `psycopg` v3, Pydantic v2, `pydantic-settings`, `cryptography` (AES-GCM),
`anthropic` SDK, pytest + pytest-asyncio, Docker Compose (local Postgres).

### 3.1 Sub-Phase 1a — Project Foundation and Database

Every cross-cutting piece of infrastructure the graph's nodes depend on, built before any node
exists.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Project scaffolding | `pyproject.toml`, `app/config.py` (`Settings`), `docker-compose.yml`, `tests/conftest.py` | — | `pytest tests/test_config.py` passes; `docker compose up -d postgres` runs a healthy container. |
| 2 | Domain models and graph state | `app/models/domain.py` (`TripRequest`, `FlightOption`, `HotelOption`, `Itinerary`, `TravelerDetails`), `app/models/state.py` (`GraphState`, `BookingFormResult`) | Task 1 | Domain model tests pass, including multi-leg `TripRequest` validation and negative-price rejection; `GraphState` accepts a partial-dict update. |
| 3 | Traveler PII encryption helper | `app/crypto.py` (`encrypt_field`, `decrypt_field`, `generate_test_key`) | Task 1 | Encrypt→decrypt roundtrips; wrong key fails to decrypt; same plaintext produces different ciphertext each call (random nonce). |
| 4 | Postgres app tables and migrations | `app/models/db_models.py` (ORM models incl. `TravelerDetailRecord`, `IdempotencyRecord`), `app/db.py`, `alembic/` | Tasks 2, 3 | Schema creates from an empty database; FK constraint rejects a traveler row for a nonexistent trip; encrypted columns roundtrip; `alembic upgrade head` applies cleanly to a freshly-recreated empty schema. |
| 5 | LangGraph Postgres checkpointer | `app/db.py` (`get_checkpointer`) | Task 1 | `checkpointer.setup()` creates the `checkpoints` table. |
| 6 | Error handling infrastructure | `app/errors.py` (typed exceptions, `with_retries`, `sanitize_error`) | Task 1 | Retry succeeds on a transient failure (sync and async callables) and raises after max attempts; `sanitize_error` never includes raw upstream payload fields. |
| 7 | Idempotency helper | `app/idempotency.py` (`check_and_reserve`) | Task 4 | First reservation of a key succeeds; a duplicate reservation of the same key is rejected; different keys don't collide. |
| 8 | Tool client protocol and fakes | `app/tools/protocol.py` (`FlightHotelSearchTool`, `BookingFormTool`, `EmailTool`, `CRMTool`), `app/tools/fake.py` | Tasks 2, 6 | Each fake tool supports a `behavior` flag (`success`/`empty`/`rate_limited`/`failure`) and raises the correct typed exception. |
| 9 | LLM client wrapper | `app/llm/client.py` (`LLMClient` protocol, `AnthropicLLMClient`), `app/llm/fake.py` | Task 6 | Fake LLM client returns queued responses in order and can simulate a rate-limit error. |
| 10 | `NodeDeps` bundle and shared test fixtures | `app/graph/deps.py` (`NodeDeps`), `tests/conftest.py` (`db_engine` fixture), `tests/factories.py` (`make_deps`, `make_traveler_record`) | Tasks 4, 8, 9 | Full foundation `pytest` suite (~25 tests) passes with zero failures. |

### 3.2 Sub-Phase 1b — LangGraph Trip Search Flow

The pipeline from a raw user message through a ranked, itinerary-attached shortlist.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Node: `collect_trip_request` | `app/graph/nodes/collect_trip_request.py` | 1a complete | Extracts a valid `TripRequest` from a well-formed message; loops via `interrupt()` and re-validates when a required field is reported missing. |
| 2 | Node: `search_flights_and_hotels` | `app/graph/nodes/search_flights_and_hotels.py` | Task 1 | Populates `flight_result_ids`/`hotel_result_ids`/`shortlist` per leg for a multi-leg trip; sets `no_results: True` when a leg returns nothing (consumed by Phase 1f's routing). |
| 3 | Node: `recommend_options` | `app/graph/nodes/recommend_options.py` | Task 2 | Narrative text is LLM-generated, but the numeric shortlist data it's shown alongside is returned untouched — never regenerated by the LLM. |
| 4 | Node: `generate_itinerary` | `app/graph/nodes/generate_itinerary.py` | Task 1 | Produces a preliminary day-by-day itinerary from the trip request (before any flight/hotel is selected). |

### 3.3 Sub-Phase 1c — Selection and Interrupts

The first `interrupt()` gate, and the node that runs immediately after it.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Node: `await_selection` | `app/graph/nodes/await_selection.py` | 1b complete | Pauses via `interrupt()`; on resume, sets `selected_flights`/`selected_hotels` from the resume payload. |
| 2 | Node: `finalize_itinerary` | `app/graph/nodes/finalize_itinerary.py` | Task 1 | Regenerates the itinerary incorporating the actual selected flight arrival times and hotel. |

### 3.4 Sub-Phase 1d — Traveler PII and Booking Form

The three most safety-sensitive nodes in the graph.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Node: `collect_traveler_details` | `app/graph/nodes/collect_traveler_details.py` | 1c complete | Return value contains **only** `traveler_details_id` — never raw name/DOB/passport fields; the encrypted record is verifiable in Postgres via the encryption helper. |
| 2 | Node: `fill_booking_form` | `app/graph/nodes/fill_booking_form.py` | Task 1 | Loops once per leg, producing one `booking_form_results` entry per leg; on a tool failure, does **not** retry — sets `form_fill_failed: True` and a sanitized `error_log` entry instead. |
| 3 | Node: `human_review_gate` | `app/graph/nodes/human_review_gate.py` | Task 2 | Presents all legs in one combined `interrupt()`; on resume, sets `approval_status` and, on rejection, the correct `rejection_reason` for both reject reasons. |

### 3.5 Sub-Phase 1e — Email, CRM, Idempotency

The two downstream side-effect nodes that run after human approval.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Node: `send_confirmation_email` | `app/graph/nodes/send_confirmation_email.py` | 1d complete | Sends once; calling it again with the same `trip_request_id` returns `email_status: "already_sent"` without sending a second email. |
| 2 | Node: `update_crm` | `app/graph/nodes/update_crm.py` | Task 1 | Creates the contact/trip record once; calling it again with the same `trip_request_id` returns `crm_status: "already_created"` without creating a duplicate. |

### 3.6 Sub-Phase 1f — Graph Assembly, FastAPI, and Full Verification

Wires every node built so far into one graph and exposes it over HTTP.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Node: `agent_error` | `app/graph/nodes/agent_error.py` | 1e complete | Produces a plain-language message with no raw error codes/details. |
| 2 | Graph assembly | `app/graph/build.py` (`build_graph`, conditional routing) | Task 1 | Happy-path `graph.invoke()` sequence (trip request → both interrupts resumed → approval) ends with `approval_status == "approved"`, `email_status == "sent"`, `crm_status == "created"`; a reject-at-review-gate sequence with `rejection_reason == "wrong_selection"` correctly routes back to `await_selection` and completes on a second resume. |
| 3 | Persistence and crash-recovery tests | `tests/test_graph_persistence.py` | Task 2 | Resuming via a brand-new checkpointer connection (simulating a process restart) continues correctly from the last checkpoint; an idempotency key correctly prevents a duplicate email/CRM write after a simulated crash between side effect and status persistence. |
| 4 | PII-leak tests | `tests/test_pii_leak.py` | Task 2 | A synthetic passport value planted in a test run is asserted absent, verbatim, from every persisted checkpoint row, from `error_log`, and from the final state dump. |
| 5 | FastAPI endpoints | `app/api/main.py`, `app/api/schemas.py` | Task 2 | `POST /conversations` and `POST /conversations/{thread_id}/messages` round-trip a full conversation through both interrupts via HTTP, ending in the same assertions as Task 2. |
| 6 | Full suite verification | — (verification only) | Tasks 1-5 | Full Phase 1 `pytest` suite (~60+ tests across all sub-phases) passes with zero failures; no test file references a live third-party API host (`grep` check). |

---

## 4. Phase 2 — Real Integrations & Mock Booking Site

**Implements:** replaces every fake tool client from Phase 1 with a real, sandboxed
implementation of the same `Protocol` interface — no node code changes. Adds the mock booking
site itself.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Amadeus MCP server | `mcp-servers/amadeus/` implementing `FlightHotelSearchTool` against the Amadeus sandbox API | Phase 1 complete | Contract tests (mocked/recorded responses) pass in default CI; a separate, non-default job exercises the real sandbox. |
| 2 | Gmail MCP server | `mcp-servers/gmail/` implementing `EmailTool` (OAuth2, "Testing" publishing status) | Phase 1 complete | Mocked send test passes in default CI; a separate job sends a real test email against the sandbox account. |
| 3 | HubSpot MCP server | `mcp-servers/hubspot/` implementing `CRMTool` | Phase 1 complete | Mocked upsert-contact/create-trip-record tests pass in default CI; a separate job exercises the real free-tier account. |
| 4 | Playwright MCP integration | Docker image built from Microsoft's official Playwright MCP source, pinned to a specific version/commit, implementing `BookingFormTool` | Phase 1 complete | Image build is reproducible from the pinned reference; no unpinned `latest` tag anywhere in the build. |
| 5 | Mock booking site | `mock-booking-site/` — traveler-detail form, seat/room extras, a "Proceed to Payment" screen with no functioning control or endpoint behind it | — | Component tests validate the form; a dedicated test asserts the codebase contains no payment field, route, or endpoint anywhere. |
| 6 | Wire real tool implementations into `NodeDeps` | Update to `app/graph/deps.py` / DI wiring (env-flag or config-driven choice of real vs. fake) | Tasks 1-5 | No changes required to any node file from Phase 1 — only which concrete class is injected changes. |
| 7 | Playwright automation against the real mock site | End-to-end automation test, single-leg and multi-leg | Tasks 4, 5, 6 | Automation stops at the pre-payment screen on every leg; masked review screenshots contain no raw passport/DOB values. |
| 8 | Local Docker Compose integration | `docker-compose.yml` extension wiring frontend, backend, all MCP servers, mock site, Postgres | Tasks 1-7 | The full conversational flow (identical assertions to Phase 1f Task 2/5) runs against real sandboxed APIs and the real mock site, completing successfully. |

---

## 5. Phase 3 — Infrastructure Provisioning

**Implements:** a self-managed Kubernetes cluster on AWS, provisioned entirely by Terraform and
Packer, with network isolation and secrets management wired in.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Packer node AMI | `infra/packer/` — kubeadm, kubelet, container runtime pre-installed | Phase 2 complete | `packer build` produces a bootable AMI usable by Terraform. |
| 2 | Terraform AWS networking + state backend | `infra/terraform/` — VPC, subnets, security groups; S3 backend with native locking (`use_lockfile = true`, no DynamoDB table) | Task 1 | `terraform plan`/`apply` provisions networking from empty AWS infrastructure; state locking prevents concurrent applies. |
| 3 | Terraform cluster instances | EC2 control-plane (1 node) + worker instances from the Task 1 AMI; control plane initializes via `user_data` (`kubeadm init`); workers retrieve a short-lived `kubeadm join` command from AWS SSM Parameter Store at boot | Task 2 | `kubectl get nodes` shows the control plane and all workers `Ready`; no Terraform `remote-exec` is used anywhere. |
| 4 | NetworkPolicies | Manifests: frontend→backend only; backend→{MCP servers, Postgres}; Playwright MCP→mock site only; Postgres←backend only; default-deny elsewhere | Task 3 | A conformance check confirms Playwright MCP cannot reach Postgres, and the mock site cannot reach anything but Playwright MCP. |
| 5 | External Secrets Operator | Operator installation + at least one working `ExternalSecret` sourced from AWS Secrets Manager | Task 3 | A Kubernetes Secret materializes from AWS Secrets Manager at runtime; no secret value is ever committed to the GitOps repo (only the `ExternalSecret` reference object is). |

---

## 6. Phase 4 — Kubernetes Deployment

**Implements:** every application service deployed to the cluster via Kustomize, manually
applied and verified before GitOps (Phase 5) takes over.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Kustomize base manifests | `k8s/base/` — one manifest set per service (frontend, backend, mock booking site, each MCP server, Postgres) | Phase 3 complete | `kustomize build k8s/base` succeeds with no errors. |
| 2 | `dev` overlay | `k8s/overlays/dev/` | Task 1 | `kubectl apply -k k8s/overlays/dev` deploys every service successfully. |
| 3 | Manual deployment verification | — (verification only) | Task 2 | The frontend is reachable and completes a full conversation end-to-end against the in-cluster backend and MCP servers. |
| 4 | `prod` overlay scaffold | `k8s/overlays/prod/` (parameterized, not yet driving real traffic) | Task 1 | `kustomize build k8s/overlays/prod` succeeds with no errors. |

---

## 7. Phase 5 — CI/CD and GitOps

**Implements:** a CI pipeline that builds and tags images, and Argo CD as the sole applier of
manifests to the cluster from this point forward.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | GitHub Actions pipeline | `.github/workflows/` — lint → type-check → unit tests → build → push to registry | Phase 4 complete | A commit to `main` triggers the pipeline and produces a pushed image, gated on all checks passing. |
| 2 | Automatic `dev` tag bump | CI step updating the `dev` overlay's image tag | Task 1 | A successful pipeline run updates the `dev` overlay's manifest tag automatically, with no manual edit. |
| 3 | Argo CD installation | Argo CD deployed to the cluster; an `Application` resource per service | Phase 4 complete | Argo CD syncs the `dev` overlay's current manifests to the cluster with no manual `kubectl apply`. |
| 4 | `dev` → `prod` promotion path | A separate, explicit PR-based process for bumping the `prod` overlay's tag | Tasks 2, 3 | Promoting to `prod` requires an explicit PR merge — never automatic, never via CI applying directly to the cluster. |

---

## 8. Phase 6 — Observability

**Implements:** Prometheus/Grafana plus application-level metrics giving visibility into both
infrastructure and the agent's own workflow behavior.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Prometheus + Grafana deployment | Kustomize/Helm manifests, deployed via the Phase 5 GitOps path | Phase 5 complete | Prometheus is scraping targets; Grafana is reachable and connected to it as a data source. |
| 2 | Infrastructure exporters | `postgres_exporter`, `node-exporter`, `kube-state-metrics` | Task 1 | Each exporter's metrics are visible in Prometheus. |
| 3 | Application-level metrics instrumentation | FastAPI backend exports: workflow completions/failures, interrupt counts per gate, retry counts per node, per-node duration, MCP tool call latency, downstream side-effect status | Task 1 | Metrics appear in Prometheus during a live conversation run; a check confirms metrics/traces never contain traveler PII (same sanitization boundary as `error_log`). |
| 4 | Grafana dashboards | Dashboards for request latency/error rate, per-node duration, MCP call success/failure rate, cluster health | Tasks 2, 3 | Dashboards populate from real traffic during a test conversation. |
| 5 | Alerting rules | Alertmanager rules for high error rate or repeated booking-flow node failures | Task 3 | A synthetic failure injection triggers the corresponding alert. |

---

## 9. Phase 7 — Production Validation / Demo

**Implements:** final end-to-end verification of the complete workflow against the real AWS
`prod` environment.

| # | Task | Files/Outputs | Depends on | Acceptance criteria |
|---|---|---|---|---|
| 1 | Deploy `prod` overlay | `prod` overlay live via Argo CD | Phases 1-6 complete | All services running in the `prod` namespace/overlay, `Ready`. |
| 2 | Happy-path production run | — (verification only) | Task 1 | A complete conversation (trip request through CRM update) runs successfully against `prod`. |
| 3 | Reject-at-review-gate production run | — (verification only) | Task 1 | A complete conversation that rejects at `human_review_gate` and completes on retry runs successfully against `prod`. |
| 4 | Dashboard/alert verification | — (verification only) | Tasks 2, 3 | Grafana shows the corresponding metrics/dashboard activity in real time during both runs. |
| 5 | AWS-specific integration verification | — (verification only) | Task 1 | IAM, EBS, and External Secrets Operator ↔ Secrets Manager integration — which cannot be verified in `kind` (spec §9) — are confirmed working in the real AWS environment. |

---

## 10. Cross-Phase Dependencies

**Sequencing:** phases run strictly in order — 1a→1b→1c→1d→1e→1f→2→3→4→5→6→7. None are
parallelizable; each assumes every prior phase's tests pass (see each phase's task table for
the specific reason in each case).

**External accounts/services required:**

| Account/service | Required starting | Purpose |
|---|---|---|
| Amadeus for Developers (sandbox) | Phase 2 | Flight/hotel search |
| Gmail account + OAuth2 client (Testing mode) | Phase 2 | Confirmation email sending |
| HubSpot (free tier) | Phase 2 | CRM contact/trip records |
| AWS account with billing enabled | Phase 3 | EC2, VPC, S3 (Terraform state), Secrets Manager |
| AWS IAM user/role | Phase 3 | Terraform provisioning permissions |
| Container registry (e.g. ECR) | Phase 5 | CI image push target |

---

## 11. How Later Phases Get Detailed Further

Phase 1's task tables above are implementation-ready. Phases 2-7 are specified at the same
task granularity but will be refined with concrete sub-tasks (files, exact commands, test
code) immediately before each phase starts, so the detail reflects what was actually built in
the prior phases rather than an upfront guess — the objectives, task ordering, dependencies,
and acceptance criteria in this document are expected to remain accurate regardless, since
they describe required outcomes, not incidental implementation choices.
