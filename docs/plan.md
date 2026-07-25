# AI Travel Planning and Booking Agent — Project Plan

**Companion document:** [`docs/spec.md`](spec.md) contains the full design specification
(architecture, security model, error handling, testing strategy, risks). This document is the
standalone project plan — timeline, milestones, phase objectives, deliverables, dependencies,
and acceptance criteria — and does not require opening the detailed sub-plans in `docs/plans/`
to understand project scope, sequencing, or what "done" means for each phase.

## 1. Project Overview

### 1.1 Problem Statement

Manually planning a multi-leg trip — comparing flights and hotels across sites, building a
day-by-day itinerary, and filling in booking forms — is tedious. This project builds an AI
agent that automates search, comparison, and itinerary planning end-to-end, while keeping the
one irreversible/financial step (payment) strictly under human control.

### 1.2 Project Objectives

1. Deliver a full working demo: a user states trip constraints in conversation, receives
   ranked flight/hotel options (including multi-leg trips), receives a personalized itinerary,
   picks options, the agent fills a mock booking form via browser automation, the flow stops
   for **explicit human approval**, a confirmation email is sent, and the trip is logged to a
   CRM.
2. Demonstrate a structurally safe human-in-the-loop pattern for an agent operating adjacent to
   a payment-shaped flow — safety enforced by graph topology and mock-site design, not by
   trusting model behavior.
3. Demonstrate real infrastructure engineering: a self-managed Kubernetes cluster on AWS EC2
   (kubeadm), provisioned by Terraform and Packer, deployed via GitOps (Argo CD), observed via
   Prometheus/Grafana, built and shipped via CI/CD.

### 1.3 Non-Goals

No real payment processing (ever, not even later); no real booking-website integration (mock
site only); no multi-user accounts/authentication; no mobile app; no dedicated
points-of-interest API (itinerary content is LLM-generated); no highly-available control plane
(single node, explicitly accepted limitation). Full rationale in `spec.md` §3.

### 1.4 Overall Success Criteria

The project is complete when: the full conversational flow runs against real sandboxed
third-party APIs (Amadeus, Gmail, HubSpot) and a purpose-built mock booking site; the entire
stack is deployed on a self-managed Kubernetes cluster on AWS via Terraform/Packer; GitOps
(Argo CD) is the sole applier of manifests; CI/CD builds and tags images automatically;
Prometheus/Grafana show live application and infrastructure metrics; and a full end-to-end run
(happy path + reject-at-review-gate path) has been demonstrated on the `prod` environment.

---

## 2. Timeline Summary

**Basis:** relative duration in working weeks, assuming a substantial part-time pace of
roughly 20-25 hours/week (approximated below as 5 working days/week at 4-5 hrs/day). These are
estimates for planning purposes, not calendar-locked commitments — see §6 for schedule risks
that could shift them. Re-anchor to real calendar dates once a start date is fixed.

| Phase | Weeks | Cumulative |
|---|---|---|
| 1 — Core Agent Logic (sub-phases 1a-1f) | 6 weeks | Weeks 1-6 |
| 2 — Real Integrations & Mock Booking Site | 4 weeks | Weeks 7-10 |
| 3 — Infrastructure Provisioning | 3 weeks | Weeks 11-13 |
| 4 — Kubernetes Deployment | 1 week | Week 14 |
| 5 — CI/CD and GitOps | 1.5 weeks | Weeks 15-16 (first half) |
| 6 — Observability | 1.5 weeks | Weeks 16 (second half)-17 |
| 7 — Production Validation / Demo | 1 week | Week 18 |
| **Total** | **~18 weeks (~4.5 months)** | |

### Phase 1 Sub-Phase Breakdown (working days, 6 weeks total)

| Sub-phase | Days | Cumulative |
|---|---|---|
| 1a — Project Foundation & Database | 7 days | Week 1 – Week 2 (day 2) |
| 1b — LangGraph Trip Search Flow | 5 days | Week 2 (day 3) – Week 3 (day 2) |
| 1c — Selection and Interrupts | 3 days | Week 3 (days 3-5) |
| 1d — Traveler PII and Booking Form | 5 days | Week 4 |
| 1e — Email, CRM, Idempotency | 3 days | Week 5 (days 1-3) |
| 1f — Graph Assembly, FastAPI, Full Verification | 7 days | Week 5 (days 4-5) – Week 6 |

---

## 3. Milestones

| # | Milestone | Marks the end of | Demonstrable outcome |
|---|---|---|---|
| M1 | Core agent proven in isolation | Phase 1 | Full `pytest` suite green (~60+ tests) for the complete LangGraph agent (12 nodes, both `interrupt()` gates, encrypted PII, idempotent side effects) running against fakes and a real local Postgres. No real external services yet. |
| M2 | Full local demo, real integrations | Phase 2 | A person can run the complete conversation locally (Docker Compose) against real Amadeus/Gmail/HubSpot sandboxes and a purpose-built mock booking site, with Playwright filling the form and stopping at the review screen. |
| M3 | Cloud infrastructure live | Phase 3 | A self-managed Kubernetes cluster (kubeadm) is running on AWS EC2, provisioned entirely by Terraform + Packer, with NetworkPolicies and External Secrets Operator functioning. |
| M4 | Application running in-cluster | Phase 4 | All services (frontend, backend, mock site, MCP servers, Postgres) deployed to the cluster via Kustomize `dev` overlay, manually verified. |
| M5 | GitOps live | Phase 5 | A merge to `main` results in Argo CD automatically syncing the updated image to the `dev` overlay, with no manual `kubectl apply`. |
| M6 | Observability operational | Phase 6 | Grafana dashboards show live application metrics (workflow completions/failures, interrupt/retry counts, MCP latency) and infrastructure metrics, sourced from a working Prometheus scrape. |
| M7 | **Production validation — final deliverable** | Phase 7 | A complete end-to-end run (happy path, and a reject-at-review-gate path) executes successfully against the real AWS `prod` environment, with the run visible live on the Grafana dashboards. |

---

## 4. Phase-by-Phase Plan

### Phase 1 — Core Agent Logic

**Overall objective:** build and fully test the LangGraph agent — all 12 workflow nodes, both
`interrupt()` gates, encrypted traveler PII, idempotent side effects, sanitized error
handling — against fake tool/LLM clients and a real local Postgres. No real MCP servers, no
infrastructure, no cloud.

**Detailed plans:** `docs/plans/01-project-foundation-and-db.md` through
`docs/plans/06-fastapi-and-error-handling.md` (six sub-phases, executed in order).

#### 1a — Project Foundation and Database (7 days)

- **Objective:** every cross-cutting piece of infrastructure the graph's nodes depend on —
  settings, domain models, `GraphState` schema, encryption, Postgres tables + migrations, the
  LangGraph checkpointer, error/retry handling, idempotency, tool/LLM client protocols with
  fakes, and the shared `NodeDeps` bundle.
- **Deliverables:** `backend/app/{config,crypto,errors,idempotency,db}.py`; `app/models/`
  (domain models, ORM models, `GraphState`); `app/llm/`, `app/tools/` (protocols + fakes);
  `app/graph/deps.py`; Alembic migrations; `docker-compose.yml`; shared test
  fixtures/factories (`tests/conftest.py`, `tests/factories.py`).
- **Dependencies:** none (first sub-phase). External: a local Docker installation.
- **Acceptance criteria:** `pytest -v` passes for every foundation test file (~25 tests) with
  zero failures; `alembic upgrade head` succeeds against a freshly-recreated empty schema;
  `docker compose up -d postgres` runs a healthy container.

#### 1b — LangGraph Trip Search Flow (5 days)

- **Objective:** the pipeline from a raw user message through a ranked, itinerary-attached
  shortlist — `collect_trip_request`, `search_flights_and_hotels`, `recommend_options`,
  `generate_itinerary`.
- **Deliverables:** four node implementations under `app/graph/nodes/`; one test file per node.
- **Dependencies:** 1a complete.
- **Acceptance criteria:** all four node test files pass in isolation against fakes; numeric
  data (price/time/rating) shown in `recommend_options`'s output is asserted to be untouched
  from the typed shortlist, never regenerated by the LLM.

#### 1c — Selection and Interrupts (3 days)

- **Objective:** the first `interrupt()` gate (`await_selection`) and the node that
  regenerates the itinerary against the traveler's actual choice (`finalize_itinerary`).
- **Deliverables:** two node implementations; two test files.
- **Dependencies:** 1b complete.
- **Acceptance criteria:** `await_selection` correctly pauses and resumes with a selection
  payload (verified via monkeypatched `interrupt()`); `finalize_itinerary` incorporates the
  selected flight/hotel data into the itinerary output.

#### 1d — Traveler PII and Booking Form (5 days)

- **Objective:** the three most safety-sensitive nodes — collecting and encrypting traveler
  PII (`collect_traveler_details`), filling the booking form once per leg with no blind retry
  (`fill_booking_form`), and the second `interrupt()` gate where a human reviews the filled
  form before anything downstream happens (`human_review_gate`).
- **Deliverables:** three node implementations; three test files.
- **Dependencies:** 1c complete.
- **Acceptance criteria:** a test asserts the node's return value contains **only**
  `traveler_details_id`, never raw name/DOB/passport fields (spec §6.1); a test confirms
  `fill_booking_form` does not retry on failure and instead sets `form_fill_failed`; a test
  confirms `human_review_gate` sets `rejection_reason` correctly for both reject reasons.

#### 1e — Email, CRM, Idempotency (3 days)

- **Objective:** the two downstream side-effect nodes that run after human approval —
  `send_confirmation_email` and `update_crm` — both idempotent.
- **Deliverables:** two node implementations; two test files.
- **Dependencies:** 1d complete.
- **Acceptance criteria:** a test proves calling either node twice with the same
  `trip_request_id` results in exactly one email sent / one CRM record created, not two.

#### 1f — Graph Assembly, FastAPI, and Full Verification (7 days)

- **Objective:** the `agent_error` node; full `StateGraph` assembly with conditional routing;
  persistence/crash-recovery verification; PII-non-leakage verification; the FastAPI HTTP
  layer; full Phase 1 suite verification.
- **Deliverables:** `app/graph/build.py`; `app/api/`; end-to-end/persistence/PII-leak/API test
  files.
- **Dependencies:** 1a-1e complete (this sub-phase wires together every node built so far).
- **Acceptance criteria (all must hold):**
  - A happy-path `graph.invoke()` sequence (trip request → both interrupts resumed → approval)
    ends with `approval_status == "approved"`, `email_status == "sent"`,
    `crm_status == "created"`.
  - A reject-at-review-gate sequence with `rejection_reason == "wrong_selection"` correctly
    routes back to `await_selection` and completes on a second resume.
  - Resuming a graph via a brand-new checkpointer connection (simulating a process restart)
    continues correctly from the last checkpoint.
  - A synthetic passport value planted in a test run is asserted absent, verbatim, from every
    persisted checkpoint row, from `error_log`, and from the final state dump.
  - The full Phase 1 `pytest` suite (all six sub-phases, ~60+ tests) passes with zero
    failures, and no test file references a live third-party API host.

---

### Phase 2 — Real Integrations & Mock Booking Site (4 weeks)

- **Objective:** replace fakes with real, sandboxed integrations — custom MCP servers for
  Amadeus, Gmail, and HubSpot; the official Playwright MCP server (pinned build); a
  purpose-built mock booking site with no payment field or endpoint anywhere in its code.
  Everything still runs locally via Docker Compose.
- **Deliverables:** `mcp-servers/amadeus/`, `mcp-servers/gmail/`, `mcp-servers/hubspot/`
  (custom MCP servers implementing the `FlightHotelSearchTool`/`EmailTool`/`CRMTool` protocols
  from Phase 1); a pinned Playwright MCP server build implementing `BookingFormTool`;
  `mock-booking-site/` (a small web app: traveler-detail form, seat/room extras, a "Proceed to
  Payment" screen with no functioning control behind it); Playwright automation tests against
  the real mock site; a `docker-compose.yml` extension wiring all of this together.
- **Dependencies:** Phase 1 complete (real implementations must satisfy the same `Protocol`
  interfaces the fakes did — no node code changes). External: an Amadeus for Developers
  sandbox account/API key, a Gmail account with API access enabled (OAuth2 client, "Testing"
  publishing status), a HubSpot free-tier account/API key.
- **Acceptance criteria:** the full conversational flow (identical to Phase 1's end-to-end
  test) runs against real sandboxed APIs instead of fakes and completes successfully; a
  Playwright test confirms automation stops at the pre-payment screen on every leg of a
  multi-leg booking; a test confirms the mock site's codebase contains no payment field or
  endpoint at all; live-sandbox integration tests run as a separate, non-default CI job (per
  spec §9) so the default suite stays deterministic.

---

### Phase 3 — Infrastructure Provisioning (3 weeks)

- **Objective:** Terraform + Packer provision AWS networking and a reusable Kubernetes node
  AMI; a self-managed Kubernetes cluster (kubeadm) is stood up (control plane via `user_data`,
  workers retrieving a short-lived join command from AWS SSM Parameter Store); NetworkPolicies
  and the External Secrets Operator are wired in.
- **Deliverables:** `infra/terraform/` (VPC, subnets, security groups, EC2 instances, S3
  backend with native locking); `infra/packer/` (node AMI definition); NetworkPolicy manifests
  (frontend→backend, backend→{MCP servers, Postgres}, Playwright MCP→mock site only,
  Postgres←backend only, default-deny elsewhere); External Secrets Operator installation and
  at least one working `ExternalSecret` pulling from AWS Secrets Manager.
- **Dependencies:** Phase 2 complete (needed to know exactly which secrets/services must be
  provisioned for). External: an AWS account with billing enabled, an IAM user/role with
  sufficient permissions.
- **Acceptance criteria:** `terraform apply` provisions the full cluster from empty AWS
  infrastructure; `kubectl get nodes` shows the control plane and all workers `Ready`; a
  `NetworkPolicy` conformance check confirms Playwright MCP cannot reach Postgres; an
  `ExternalSecret` resource successfully materializes a Kubernetes Secret sourced from AWS
  Secrets Manager, with no secret value ever committed to the GitOps repo.

---

### Phase 4 — Kubernetes Deployment (1 week)

- **Objective:** deploy every application service to the cluster via Kustomize overlays
  (`dev` first), manually applied to validate manifests before GitOps takes over.
- **Deliverables:** `k8s/base/` and `k8s/overlays/dev/` Kustomize manifests for every service
  (frontend, backend, mock booking site, each MCP server, Postgres); namespace layout.
- **Dependencies:** Phase 3 complete (cluster must exist).
- **Acceptance criteria:** `kubectl apply -k k8s/overlays/dev` deploys every service
  successfully; the frontend is reachable and can complete a full conversation end-to-end
  against the in-cluster backend and MCP servers; `kustomize build` succeeds with no errors for
  both the `dev` and (still-empty-shell) `prod` overlays.

---

### Phase 5 — CI/CD and GitOps (1.5 weeks)

- **Objective:** a GitHub Actions pipeline (lint → type-check → unit tests → build → push →
  tag-bump) feeds Argo CD, which becomes the **sole** applier of manifests to the cluster from
  this point forward.
- **Deliverables:** `.github/workflows/` CI pipeline; Argo CD installation and an `Application`
  resource per service; the promotion path (`main` merge auto-updates the `dev` overlay tag;
  promotion to `prod` is a separate, explicit PR).
- **Dependencies:** Phase 4 complete (manifests must already exist and be known-good).
- **Acceptance criteria:** a commit to `main` results in a new image being built, pushed, and
  its tag updated in the `dev` overlay automatically, with Argo CD syncing the change to the
  cluster with no manual `kubectl apply`; CI never applies directly to the cluster (verified by
  inspecting the workflow's permissions/steps).

---

### Phase 6 — Observability (1.5 weeks)

- **Objective:** Prometheus and Grafana deployed; application-level metrics instrumented
  (workflow completions/failures, interrupt counts per gate, retry counts per node, per-node
  duration, MCP tool call latency, downstream side-effect status); dashboards and lightweight
  alerting built out.
- **Deliverables:** Prometheus + Grafana Helm/Kustomize deployment; `postgres_exporter`,
  `node-exporter`, `kube-state-metrics`; FastAPI backend instrumentation exporting the
  application-level metrics listed above; Grafana dashboards; Alertmanager rules for high error
  rate or booking-flow node failures.
- **Dependencies:** Phase 5 complete (so dashboard/alerting config can itself be deployed via
  GitOps rather than applied manually).
- **Acceptance criteria:** Grafana shows live request latency/error-rate, per-node duration,
  and MCP call success/failure panels populated from real traffic; a test/manual check confirms
  metrics and traces never contain traveler PII (same sanitization boundary as `error_log`); an
  alert fires (verified with a synthetic failure injection) when a booking-flow node fails
  repeatedly.

---

### Phase 7 — Production Validation / Demo (1 week)

- **Objective:** full end-to-end verification of the complete workflow running on the real AWS
  Kubernetes environment (`prod` overlay) — both the happy path and the reject-at-review-gate
  path — with monitoring dashboards and alerts confirmed to reflect live activity during the
  run.
- **Deliverables:** a recorded/observed full run against `prod`; a final verification checklist
  confirming every acceptance criterion from Phases 1-6 still holds in the `prod` environment
  specifically (some, like AWS IAM/EBS/External Secrets integration, can only be verified here
  — see spec §9).
- **Dependencies:** Phases 1-6 complete.
- **Acceptance criteria:** a complete happy-path conversation (trip request through CRM update)
  and a complete reject-at-review-gate conversation both run successfully against `prod`;
  Grafana shows the corresponding metrics/dashboard activity in real time during the run; this
  is the terminal milestone (M7) and the point at which the project is considered complete.

---

## 5. Cross-Phase Dependencies

**Sequencing:** phases run strictly in order — 1a→1b→1c→1d→1e→1f→2→3→4→5→6→7. None are
parallelizable; each assumes every prior phase is committed and passing (see §4 per-phase
Dependencies for the specific reason in each case).

**External accounts/services needed, and by when:**

| Needed by | Account/service | Purpose |
|---|---|---|
| Phase 2 | Amadeus for Developers (sandbox) | Flight/hotel search |
| Phase 2 | Gmail account + OAuth2 client (Testing mode) | Confirmation email sending |
| Phase 2 | HubSpot (free tier) | CRM contact/trip records |
| Phase 3 | AWS account with billing enabled | EC2, VPC, S3 (Terraform state), Secrets Manager |
| Phase 3 | AWS IAM user/role | Terraform provisioning permissions |
| Phase 5 | Container registry (e.g. ECR) | CI image push target |

Setting these up is not assumed instantaneous — see §6.

---

## 6. Schedule Risks

Full technical risk register is in `spec.md` §10; the subset most likely to affect **this
timeline** specifically:

- **Third-party account/OAuth setup friction** (Gmail's OAuth consent-screen requirements,
  Amadeus sandbox provisioning) could delay the start of Phase 2 — mitigated by starting
  account setup in parallel with the tail end of Phase 1, not waiting until Phase 2 begins.
- **kubeadm/Packer/SSM bootstrap complexity** (Phase 3) is nontrivial compared to a managed
  EKS setup — if it stalls past the 3-week estimate, the documented fallback is a temporary
  `kind` cluster to keep Phase 4 unblocked while infrastructure work continues in parallel.
- **Scope creep across many subsystems** is the single biggest risk to the overall 18-week
  estimate — mitigated by the phase structure itself: each phase ships working, independently
  verifiable software, so a slip in one phase doesn't block demonstrating progress on the
  ones before it.

---

## 7. How This Plan Is Maintained

Phase 1's six sub-plans (`docs/plans/01-*.md` through `docs/plans/06-*.md`) already contain
full bite-sized TDD detail and are ready to execute. Phases 2-7 do not have detailed sub-plans
yet, by design — each will be written via the same planning process immediately before that
phase starts, so it reflects what was actually learned building the prior phases rather than
an upfront guess. When each phase's detailed sub-plan is written, a link will be added to the
corresponding section in §4 above; the objectives, deliverables, dependencies, and acceptance
criteria in this document should remain accurate regardless, since they describe outcomes
rather than implementation steps.
