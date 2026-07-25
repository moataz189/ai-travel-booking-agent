# AI Travel Planning and Booking Agent — Implementation Plan Index

This indexes the implementation plans derived from [`docs/spec.md`](spec.md). The spec's
Roadmap (§11) defines 7 phases; each becomes its own plan document, written in full
bite-sized TDD detail immediately before that phase begins (see rationale below).

## Phases

| Phase | Plan | Status | Scope |
|---|---|---|---|
| 1 | [`docs/plans/01-core-agent-logic.md`](plans/01-core-agent-logic.md) | **Ready to execute** | LangGraph agent (all 12 workflow nodes + `agent_error`), both `interrupt()` gates, Postgres app tables + Alembic migrations, LangGraph Postgres checkpointer, encrypted traveler PII, idempotent email/CRM nodes, error handling, FastAPI HTTP layer — all tested against fake tool/LLM clients. No real MCP servers, no infra. |
| 2 | *not yet written* | Planned | Real MCP servers (Amadeus, Gmail, HubSpot — custom; Playwright — pinned build of Microsoft's official server) wired in against sandboxes; mock booking site built and Playwright-automated end to end; still running locally/Docker Compose. |
| 3 | *not yet written* | Planned | Terraform + Packer provision AWS networking and a kubeadm node AMI; self-managed Kubernetes cluster stood up (control plane via `user_data`, workers via SSM-delivered join command); NetworkPolicies and External Secrets Operator wired in. |
| 4 | *not yet written* | Planned | All application services deployed to the cluster via Kustomize overlays (`dev` first), manually applied to validate manifests before GitOps takes over. |
| 5 | *not yet written* | Planned | GitHub Actions pipeline (lint/test/build/tag-bump) wired to Argo CD, which becomes the sole applier of manifests from that point forward. |
| 6 | *not yet written* | Planned | Prometheus/Grafana deployed; application-level metrics instrumented (workflow completions/failures, interrupt/retry counts, node duration, MCP latency, side-effect status); dashboards and lightweight alerting built out. |
| 7 | *not yet written* | Planned | Full end-to-end verification of the complete workflow on the real AWS Kubernetes environment (`prod` overlay), happy path and reject-at-review-gate path, with monitoring dashboards and alerts confirmed live. |

## Why only Phase 1 is fully detailed right now

Fully detailing all 7 phases in bite-sized TDD steps before any code exists would mean
writing a large amount of content — especially for the infra phases — based on assumptions
that are likely to shift once Phase 1 is actually built. Each subsequent phase's plan will be
written via the `writing-plans` skill immediately before that phase starts, so it reflects
what was actually learned and built in the prior phases rather than upfront guesses. This
also matches how the spec's own Roadmap (§11) is structured: phases build on each other
sequentially, not in parallel.

## Execution

Phase 1 is ready. See [`docs/plans/01-core-agent-logic.md`](plans/01-core-agent-logic.md) for
the full task-by-task plan, execution method options (subagent-driven vs. inline), and the
global constraints every task must satisfy.
