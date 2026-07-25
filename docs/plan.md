# AI Travel Planning and Booking Agent — Implementation Plan Index

This indexes the implementation plans derived from [`docs/spec.md`](spec.md). The spec's
Roadmap (§11) defines 7 phases; each becomes its own plan document, written in full
bite-sized TDD detail immediately before that phase begins (see rationale below).

Phase 1 itself was further split into six sequential sub-plans once its single draft plan
reached ~4,000 lines and 26 tasks — too large to review or execute safely as one unit. Each
sub-plan is independently reviewable and produces its own passing test suite, but they still
build on each other in order (01 → 02 → 03 → 04 → 05 → 06); they are not parallelizable.

## Phase 1 Sub-Plans (execute in this order)

| # | Plan | Status | Scope |
|---|---|---|---|
| 1a | [`docs/plans/01-project-foundation-and-db.md`](plans/01-project-foundation-and-db.md) | **Ready to execute** | Settings, domain models, `GraphState` schema, AES-GCM encryption, Postgres app tables + Alembic migrations, LangGraph Postgres checkpointer, error/retry handling, idempotency helper, tool/LLM client protocols + fakes, `NodeDeps` bundle, shared test fixtures/factories. |
| 1b | [`docs/plans/02-langgraph-trip-search-flow.md`](plans/02-langgraph-trip-search-flow.md) | Ready (after 1a) | Nodes: `collect_trip_request`, `search_flights_and_hotels`, `recommend_options`, `generate_itinerary`. |
| 1c | [`docs/plans/03-selection-and-interrupts.md`](plans/03-selection-and-interrupts.md) | Ready (after 1b) | Nodes: `await_selection` (first `interrupt()` gate), `finalize_itinerary`. |
| 1d | [`docs/plans/04-traveler-pii-and-booking-form.md`](plans/04-traveler-pii-and-booking-form.md) | Ready (after 1c) | Nodes: `collect_traveler_details` (encrypted PII boundary), `fill_booking_form` (per-leg loop, no retry), `human_review_gate` (second `interrupt()` gate). |
| 1e | [`docs/plans/05-email-crm-idempotency.md`](plans/05-email-crm-idempotency.md) | Ready (after 1d) | Nodes: `send_confirmation_email`, `update_crm` — both idempotent. |
| 1f | [`docs/plans/06-fastapi-and-error-handling.md`](plans/06-fastapi-and-error-handling.md) | Ready (after 1e) | Node: `agent_error`; full `StateGraph` assembly + routing; persistence/crash-recovery tests; PII-leak tests; FastAPI HTTP layer; full Phase 1 suite verification. |

Together, 1a-1f cover everything originally scoped for Phase 1: LangGraph agent (all 12
workflow nodes + `agent_error`), both `interrupt()` gates, Postgres persistence, encrypted
traveler PII, idempotent email/CRM nodes, sanitized error handling, and a FastAPI HTTP layer
— all tested against fake tool/LLM clients, no real MCP servers, no infra.

## Later Phases

| Phase | Plan | Status | Scope |
|---|---|---|---|
| 2 | *not yet written* | Planned | Real MCP servers (Amadeus, Gmail, HubSpot — custom; Playwright — pinned build of Microsoft's official server) wired in against sandboxes; mock booking site built and Playwright-automated end to end; still running locally/Docker Compose. |
| 3 | *not yet written* | Planned | Terraform + Packer provision AWS networking and a kubeadm node AMI; self-managed Kubernetes cluster stood up (control plane via `user_data`, workers via SSM-delivered join command); NetworkPolicies and External Secrets Operator wired in. |
| 4 | *not yet written* | Planned | All application services deployed to the cluster via Kustomize overlays (`dev` first), manually applied to validate manifests before GitOps takes over. |
| 5 | *not yet written* | Planned | GitHub Actions pipeline (lint/test/build/tag-bump) wired to Argo CD, which becomes the sole applier of manifests from that point forward. |
| 6 | *not yet written* | Planned | Prometheus/Grafana deployed; application-level metrics instrumented (workflow completions/failures, interrupt/retry counts, node duration, MCP latency, side-effect status); dashboards and lightweight alerting built out. |
| 7 | *not yet written* | Planned | Full end-to-end verification of the complete workflow on the real AWS Kubernetes environment (`prod` overlay), happy path and reject-at-review-gate path, with monitoring dashboards and alerts confirmed live. |

## Why only Phase 1 is fully detailed right now

Fully detailing all 7 phases in bite-sized TDD steps before any code exists would mean
writing a large amount of content — especially for the infra phases — based on assumptions
that are likely to shift once Phase 1 is actually built. Each subsequent phase's plan (2-7)
will be written via the `writing-plans` skill immediately before that phase starts, so it
reflects what was actually learned and built in the prior phases rather than upfront guesses.
This also matches how the spec's own Roadmap (§11) is structured: phases build on each other
sequentially, not in parallel.

## Execution

Start with [`docs/plans/01-project-foundation-and-db.md`](plans/01-project-foundation-and-db.md)
and proceed through 02 → 03 → 04 → 05 → 06 in order — each assumes the previous ones are
committed and passing. Each sub-plan document has the full task-by-task detail, execution
method options (subagent-driven vs. inline), and the global constraints every task must satisfy.
