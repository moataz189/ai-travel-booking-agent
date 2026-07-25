# AI Travel Planning and Booking Agent — Implementation Plan

This document is the complete, step-by-step implementation plan for building the system
described in [`docs/spec.md`](spec.md). It is organized as a sequential list of implementation
tasks, grouped into seven phases executed in order. Every task follows the same structure:
what is implemented, the main files/components affected, the tests or verification performed,
the completion criteria, and its dependency on earlier tasks. `spec.md` remains the design
document (architecture, rationale, security model, risks); this document does not repeat it.

## Constraints Governing Every Task

- Traveler PII (passport/ID number, date of birth) is never stored directly on LangGraph graph
  state — only a `traveler_details_id` reference (spec §5.3, §6.1).
- `error_log` entries are sanitized `{code, message}` pairs only — never PII, tokens, API keys,
  or raw upstream responses (spec §5.3).
- Every MCP-tool-calling and LLM-calling node retries a bounded number of times with backoff
  before routing to the error-handling node, except the booking-form-fill node, which never
  retries automatically (spec §5.2, §7).
- The email-confirmation and CRM-update nodes are idempotent via a stored idempotency key
  (spec §7).
- Traveler PII columns are encrypted at the application layer before insert (spec §6.1).
- The human-review gate covers all legs of a multi-leg trip in one combined decision; the
  booking-form-fill node loops once per leg (spec §5.2).
- The mock booking site contains no payment field or endpoint anywhere in its codebase (spec
  §6.5).

## Test Plan

The project is verified through the following test categories, applied throughout the tasks
below:

- **Unit tests** — pure logic (validation, ranking, encryption, idempotency key generation)
  with no external dependency.
- **Node-level tests** — each LangGraph node in isolation, using a fake LLM client and fake
  tool clients (mocked Amadeus/Gmail/HubSpot/Playwright), covering success, empty-result,
  error, and timeout cases.
- **MCP contract tests** — each custom MCP server's tool schemas validated against
  malformed/oversized/non-allowlisted input, using recorded or mocked upstream responses.
- **MCP transport integration tests** — real MCP protocol communication (stdio/HTTP) between
  the backend and each MCP server, exercised against sandbox credentials in a non-default test
  job so the default suite stays deterministic.
- **Graph integration tests** — full `StateGraph` runs (happy path and reject-at-review-gate
  path) through both `interrupt()` gates.
- **Persistence and crash-recovery tests** — checkpoint creation/resume across a simulated
  process restart, and idempotency-key protection against duplicate side effects after a
  simulated crash.
- **Security tests** — PII-non-leakage assertions (checkpoints, logs, metrics, traces,
  screenshots) and NetworkPolicy conformance checks.
- **Playwright/UI tests** — mock booking site form automation and frontend interaction.
- **Infrastructure tests** — `terraform validate`/`plan`, `kustomize build`, and cluster
  smoke tests.
- **End-to-end tests** — full conversation runs against the deployed `dev` and `prod`
  environments.

---

## Phase 1 — Core Agent Logic (Backend, Frontend, and Local Persistence)

### Task 1.1: Project Scaffolding and Configuration

- **Implements:** the Python backend project (dependency manifest, virtual environment,
  application settings loaded from environment variables), and a local Postgres instance for
  development and testing.
- **Files/components:** `backend/pyproject.toml`, `backend/app/config.py`,
  `docker-compose.yml`, `backend/tests/conftest.py`.
- **Tests/verification:** unit tests confirming settings load with correct defaults and
  correct types.
- **Completion criteria:** the backend package installs cleanly; the test suite runs; a local
  Postgres container starts and accepts connections.
- **Dependency:** none.

### Task 1.2: Domain Models and LangGraph State Schema

- **Implements:** the Pydantic domain models (trip request, flight option, hotel option,
  itinerary, traveler details) and the LangGraph `GraphState` schema shared by every node.
- **Files/components:** `backend/app/models/domain.py`, `backend/app/models/state.py`.
- **Tests/verification:** unit tests for field validation (including multi-leg trip requests
  and rejection of invalid values) and for `GraphState` accepting partial updates.
- **Completion criteria:** all domain-model and state-schema tests pass.
- **Dependency:** Task 1.1.

### Task 1.3: Traveler PII Encryption Utilities

- **Implements:** application-layer AES-GCM encryption/decryption for traveler PII fields.
- **Files/components:** `backend/app/crypto.py`.
- **Tests/verification:** unit tests for encrypt/decrypt round-trip, rejection with an
  incorrect key, and non-deterministic ciphertext for identical plaintext.
- **Completion criteria:** all encryption utility tests pass.
- **Dependency:** Task 1.1.

### Task 1.4: Postgres Application Schema and Migrations

- **Implements:** the application's Postgres tables (trip requests, flight/hotel search
  results, encrypted traveler details, booking records, idempotency records) and their Alembic
  migration history.
- **Files/components:** `backend/app/models/db_models.py`, `backend/alembic/`.
- **Tests/verification:** schema-creation test against an empty database; foreign-key
  constraint test; encrypted-column round-trip test; migration-application test against a
  freshly reset schema.
- **Completion criteria:** `alembic upgrade head` applies cleanly to an empty database and all
  schema tests pass.
- **Dependency:** Tasks 1.2, 1.3.

### Task 1.5: LangGraph Postgres Checkpointer Integration

- **Implements:** the LangGraph checkpointer backed by Postgres, used for conversation state
  persistence and resumability.
- **Files/components:** `backend/app/db.py`.
- **Tests/verification:** a test confirming checkpointer setup creates the required tables.
- **Completion criteria:** checkpointer setup succeeds against the local database.
- **Dependency:** Task 1.1.

### Task 1.6: Error Handling, Retry, and Idempotency Infrastructure

- **Implements:** typed exceptions for tool and LLM failures, a retry helper with exponential
  backoff applied to tool/LLM calls, sanitized error formatting for `error_log`, and an
  idempotency check-and-reserve helper for downstream side effects. Together these implement
  the retry, fallback, and clear-error-response behavior required of the agent; graceful
  termination on unrecoverable errors is completed in Task 1.14.
- **Files/components:** `backend/app/errors.py`, `backend/app/idempotency.py`.
- **Tests/verification:** retry-succeeds-after-transient-failure and
  retry-raises-after-max-attempts tests (for both synchronous and asynchronous calls);
  sanitize-never-leaks-raw-payload test; idempotency reservation and duplicate-rejection tests.
- **Completion criteria:** all error-handling and idempotency tests pass.
- **Dependency:** Task 1.1.

### Task 1.7: Tool Client Protocols, Fake Implementations, and LLM Client Wrapper

- **Implements:** the abstract tool-client interfaces for flight/hotel search, booking-form
  fill, email sending, and CRM updates; deterministic fake implementations of each, configurable
  to simulate success, empty results, rate limiting, and failure; and an LLM client wrapper
  (real Anthropic-backed implementation plus a fake with queued/canned responses) used by every
  LLM-calling node.
- **Files/components:** `backend/app/tools/protocol.py`, `backend/app/tools/fake.py`,
  `backend/app/llm/client.py`, `backend/app/llm/fake.py`.
- **Tests/verification:** unit tests confirming each fake tool's configurable behaviors and
  correct exception raising; unit tests confirming the fake LLM client returns queued responses
  and can simulate errors.
- **Completion criteria:** all tool-client and LLM-client tests pass.
- **Dependency:** Task 1.6.

### Task 1.8: Agent System Prompts, Capabilities, and Boundaries

- **Implements:** the system prompts supplied to the LLM for each LLM-calling node (trip-request
  extraction, option narration, itinerary generation and finalization), and a written definition
  of the agent's capabilities and boundaries: what actions the agent may take (search, narrate,
  draft itineraries, fill a booking form, request human approval) and what it must never do
  (fill or submit any payment information, proceed past the human review gate without explicit
  approval, alter numeric price/time/rating data shown to the user, or act on requests outside
  travel planning).
- **Files/components:** `backend/app/graph/prompts.py` (system prompt constants), a short
  boundaries document co-located with the code (e.g. a module-level docstring or
  `backend/app/graph/BOUNDARIES.md`).
- **Tests/verification:** for each stated boundary, confirmation that it is enforced by a
  structural mechanism (a graph edge, an absent route, or an absent site feature) rather than by
  the prompt text alone — cross-checked against the graph routing implemented in Task 1.15 and
  the mock site implemented in Phase 2.
- **Completion criteria:** every documented boundary maps to a specific structural enforcement
  point, verifiable by inspecting the graph's conditional edges once Task 1.15 is complete.
- **Dependency:** Task 1.7.

### Task 1.9: LangGraph Nodes — Trip Intake, Search, and Recommendation

- **Implements:** the `collect_trip_request` node (interrupt-driven conversational field
  collection, looping until the trip request validates), `search_flights_and_hotels` (per-leg
  search with a `no_results` signal for downstream routing), `recommend_options` (LLM narration
  that never alters the underlying numeric data), and `generate_itinerary` (preliminary
  itinerary before any selection is made).
- **Files/components:** `backend/app/graph/nodes/collect_trip_request.py`,
  `backend/app/graph/nodes/search_flights_and_hotels.py`,
  `backend/app/graph/nodes/recommend_options.py`,
  `backend/app/graph/nodes/generate_itinerary.py`.
- **Tests/verification:** node-level tests against fake tools/LLM covering successful
  extraction, the missing-field interrupt loop, per-leg result population, the `no_results`
  signal on empty search results, and confirmation that narrated numeric data matches the
  typed shortlist exactly.
- **Completion criteria:** all four node test files pass in isolation.
- **Dependency:** Task 1.8.

### Task 1.10: LangGraph Nodes — Selection Interrupt and Itinerary Finalization

- **Implements:** the `await_selection` node (the first `interrupt()` gate, pausing for the
  user's flight/hotel choice) and `finalize_itinerary` (regenerating the itinerary using the
  actual selected flight times and hotel).
- **Files/components:** `backend/app/graph/nodes/await_selection.py`,
  `backend/app/graph/nodes/finalize_itinerary.py`.
- **Tests/verification:** a test confirming `await_selection` pauses and resumes correctly with
  a selection payload; a test confirming the finalized itinerary incorporates the selected
  options.
- **Completion criteria:** both node test files pass.
- **Dependency:** Task 1.9.

### Task 1.11: LangGraph Nodes — Traveler PII Collection and Booking Form Automation

- **Implements:** `collect_traveler_details` (encrypts traveler PII before storage and returns
  only a reference id onto graph state) and `fill_booking_form` (loops once per leg, filling the
  booking form via the tool client, with no automatic retry on failure).
- **Files/components:** `backend/app/graph/nodes/collect_traveler_details.py`,
  `backend/app/graph/nodes/fill_booking_form.py`.
- **Tests/verification:** a test asserting the node's return value contains only the traveler
  reference id, never raw name/DOB/passport values, with the encrypted database record verified
  independently; tests confirming the per-leg fill loop and the no-retry failure path.
- **Completion criteria:** both node test files pass, including the PII-boundary assertion.
- **Dependency:** Task 1.10.

### Task 1.12: LangGraph Node — Human Review Gate

- **Implements:** `human_review_gate`, the second `interrupt()` gate, presenting all legs of the
  booking in one combined decision and routing on approval or rejection (with a specific reject
  reason).
- **Files/components:** `backend/app/graph/nodes/human_review_gate.py`.
- **Tests/verification:** tests confirming approval sets the approved status with no reject
  reason, and rejection sets the correct reject reason for each possible reason value.
- **Completion criteria:** the node test file passes for both approval and both rejection
  paths.
- **Dependency:** Task 1.11.

### Task 1.13: LangGraph Nodes — Confirmation Email and CRM Update

- **Implements:** `send_confirmation_email` and `update_crm`, the two downstream side-effect
  nodes that run after approval, both idempotent via the Task 1.6 idempotency helper.
- **Files/components:** `backend/app/graph/nodes/send_confirmation_email.py`,
  `backend/app/graph/nodes/update_crm.py`.
- **Tests/verification:** tests confirming each node performs its side effect exactly once, and
  that invoking the node again with the same trip identifier is a no-op that reports the side
  effect as already completed.
- **Completion criteria:** both node test files pass, including the duplicate-invocation case.
- **Dependency:** Task 1.12.

### Task 1.14: LangGraph Node — Graceful Error Termination

- **Implements:** `agent_error`, the node reached when a bounded retry sequence is exhausted or
  the booking-form fill fails; produces a plain-language message with no raw error codes,
  stack traces, or upstream payload details, completing the retry/fallback/graceful-termination
  requirement started in Task 1.6.
- **Files/components:** `backend/app/graph/nodes/agent_error.py`.
- **Tests/verification:** a test confirming the produced message contains no raw error code or
  technical detail.
- **Completion criteria:** the node test file passes.
- **Dependency:** Task 1.13.

### Task 1.15: LangGraph Graph Assembly and Conditional Routing

- **Implements:** the complete `StateGraph`, wiring every node from Tasks 1.9-1.14 with
  conditional edges: no-results routing back to trip intake, form-fill-failure routing to
  graceful error termination, and review-gate routing to either the email/CRM nodes (on
  approval) or back to selection or traveler-detail collection (on rejection, depending on
  reject reason).
- **Files/components:** `backend/app/graph/build.py`.
- **Tests/verification:** a full happy-path graph run (trip request through both interrupts to
  approval) verified against expected end state; a full reject-at-review-gate run verified to
  route back correctly and complete on a second resume; a persistence test resuming the graph
  through a new checkpointer connection to simulate a process restart; a security test
  confirming a planted synthetic PII value never appears verbatim in any persisted checkpoint,
  in `error_log`, or in the final state.
- **Completion criteria:** all graph-level tests pass, including the persistence and
  PII-non-leakage tests.
- **Dependency:** Task 1.14.

### Task 1.16: FastAPI HTTP API

- **Implements:** HTTP endpoints to start a conversation and to resume it by submitting a
  message or an interrupt response, exposing clear, sanitized error responses to the client on
  failure.
- **Files/components:** `backend/app/api/main.py`, `backend/app/api/schemas.py`.
- **Tests/verification:** an API test starting a conversation and receiving the first interrupt
  payload; an API test round-tripping a full conversation through both interrupts to
  completion; a test confirming a simulated node failure returns a sanitized error response
  rather than a raw exception.
- **Completion criteria:** all API tests pass.
- **Dependency:** Task 1.15.

### Task 1.17: React Chat Frontend

- **Implements:** a conversational web interface for trip input, flight/hotel selection, and
  the pre-submission review screen (showing the masked filled booking form with
  approve/reject controls), communicating with the FastAPI backend.
- **Files/components:** `frontend/` (React application: chat view, selection view, review
  view, API client module).
- **Tests/verification:** component tests for the chat input, selection list, and review
  screen; a manual verification pass driving a full conversation in a browser against the
  local backend.
- **Completion criteria:** a full conversation can be completed end to end through the browser
  UI against the local backend, ending with a visible confirmation summary.
- **Dependency:** Task 1.16.

### Task 1.18: Dockerfiles and Local Docker Compose Environment

- **Implements:** container images for the backend and frontend, and a Docker Compose
  configuration wiring the backend, frontend, and Postgres together for local development.
- **Files/components:** `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`
  (extended from Task 1.1).
- **Tests/verification:** a build verification for each Dockerfile; a Compose startup
  verification confirming the frontend can reach the backend and complete a conversation.
- **Completion criteria:** `docker compose up` brings up a working local stack (frontend,
  backend, Postgres) capable of completing a full conversation.
- **Dependency:** Task 1.17.

### Task 1.19: Phase 1 Verification

- **Implements:** no new functionality — full-suite verification of everything built in Phase
  1.
- **Files/components:** none (verification only).
- **Tests/verification:** the complete backend test suite (unit, node-level, graph, API) run
  together; a check confirming no test file depends on a live third-party API host.
- **Completion criteria:** the full Phase 1 test suite passes with zero failures.
- **Dependency:** Tasks 1.1-1.18.

---

## Phase 2 — Real Integrations and Mock Booking Site

### Task 2.1: Amadeus MCP Server

- **Implements:** a custom MCP server exposing flight-search and hotel-search tools backed by
  the Amadeus for Developers sandbox API, implementing the search protocol defined in Task 1.7.
- **Files/components:** `mcp-servers/amadeus/`.
- **Tests/verification:** contract tests validating tool input/output schemas against
  malformed, oversized, and non-allowlisted input, using recorded/mocked upstream responses.
- **Completion criteria:** all contract tests pass without requiring network access to the
  real Amadeus API.
- **Dependency:** Task 1.19.

### Task 2.2: Gmail MCP Server

- **Implements:** a custom MCP server exposing an email-sending tool backed by the Gmail API
  (OAuth2, Testing publishing status), implementing the email protocol defined in Task 1.7.
- **Files/components:** `mcp-servers/gmail/`.
- **Tests/verification:** contract tests using a mocked Gmail API response.
- **Completion criteria:** all contract tests pass without requiring network access to the real
  Gmail API.
- **Dependency:** Task 1.19.

### Task 2.3: HubSpot MCP Server

- **Implements:** a custom MCP server exposing contact-upsert and trip-record-creation tools
  backed by a HubSpot free-tier account, implementing the CRM protocol defined in Task 1.7.
- **Files/components:** `mcp-servers/hubspot/`.
- **Tests/verification:** contract tests using mocked HubSpot API responses.
- **Completion criteria:** all contract tests pass without requiring network access to the real
  HubSpot API.
- **Dependency:** Task 1.19.

### Task 2.4: Playwright MCP Integration

- **Implements:** integration of the official Playwright MCP server, built from Microsoft's
  published source and pinned to a specific version or commit, implementing the booking-form
  protocol defined in Task 1.7.
- **Files/components:** `mcp-servers/playwright/Dockerfile`.
- **Tests/verification:** a build verification confirming the image builds reproducibly from
  the pinned reference.
- **Completion criteria:** the image builds successfully with no unpinned `latest` reference
  anywhere in the build definition.
- **Dependency:** Task 1.19.

### Task 2.5: Mock Booking Website

- **Implements:** a standalone web application simulating an airline/hotel booking form
  (traveler details, seat/room extras) ending at a review screen with no functioning payment
  control or endpoint anywhere in its codebase.
- **Files/components:** `mock-booking-site/`.
- **Tests/verification:** component tests for form validation; a dedicated test asserting the
  codebase contains no payment field, route, or endpoint of any kind.
- **Completion criteria:** all component tests pass, including the no-payment-capability
  assertion.
- **Dependency:** Task 1.19.

### Task 2.6: Dockerfiles for MCP Servers and Mock Booking Site

- **Implements:** container images for the Amadeus, Gmail, and HubSpot MCP servers and the
  mock booking site.
- **Files/components:** `mcp-servers/amadeus/Dockerfile`, `mcp-servers/gmail/Dockerfile`,
  `mcp-servers/hubspot/Dockerfile`, `mock-booking-site/Dockerfile`.
- **Tests/verification:** a build verification for each Dockerfile.
- **Completion criteria:** every image builds successfully.
- **Dependency:** Tasks 2.1, 2.2, 2.3, 2.5.

### Task 2.7: Replace Fake Tool Clients with Real Implementations

- **Implements:** dependency wiring that selects the real MCP-backed tool clients (Tasks
  2.1-2.4) instead of the Phase 1 fakes, without changing any node implementation.
- **Files/components:** `backend/app/graph/deps.py` (dependency construction/wiring update).
- **Tests/verification:** a code-review-level check confirming no file under
  `backend/app/graph/nodes/` changed as part of this task.
- **Completion criteria:** the backend runs against real tool clients with zero node-level code
  changes.
- **Dependency:** Tasks 2.1, 2.2, 2.3, 2.4, 2.6.

### Task 2.8: MCP Transport Integration Tests

- **Implements:** tests exercising real MCP protocol communication (stdio or HTTP, as
  applicable) between the backend and each of the four MCP servers, independent of the graph.
- **Files/components:** `backend/tests/integration/test_mcp_transport.py`.
- **Tests/verification:** for each MCP server, a direct tool call over the real transport,
  run as a separate, non-default test job gated on sandbox credentials being present.
- **Completion criteria:** each MCP server responds correctly to a direct tool call over its
  real transport.
- **Dependency:** Task 2.7.

### Task 2.9: Agent-to-MCP-Server Integration Tests

- **Implements:** full graph runs that exercise real MCP tool calls end to end (search,
  booking-form fill, email, CRM) through the LangGraph agent, rather than through direct
  transport calls.
- **Files/components:** `backend/tests/integration/test_agent_mcp_integration.py`.
- **Tests/verification:** a full happy-path conversation run against the real Amadeus, Gmail,
  and HubSpot sandboxes and the real mock booking site, run as a separate, non-default test job.
- **Completion criteria:** the full conversation completes successfully against real sandboxed
  services.
- **Dependency:** Task 2.8.

### Task 2.10: Playwright Automation Tests Against the Real Mock Site

- **Implements:** browser automation tests validating the booking-form fill against the actual
  mock booking site (not a fake), for both single-leg and multi-leg trips.
- **Files/components:** `mock-booking-site/tests/test_playwright_automation.py`.
- **Tests/verification:** a headless run confirming automation stops at the pre-payment review
  screen on every leg, and that masked review screenshots contain no raw passport/DOB values.
- **Completion criteria:** the automation test suite passes for both single-leg and multi-leg
  cases.
- **Dependency:** Tasks 2.4, 2.5.

### Task 2.11: Local Docker Compose Integration of the Real Stack

- **Implements:** an extended Docker Compose configuration wiring the frontend, backend, all
  three custom MCP servers, the Playwright MCP server, the mock booking site, and Postgres into
  one locally runnable stack.
- **Files/components:** `docker-compose.yml`.
- **Tests/verification:** a manual/automated startup verification confirming every service
  reaches a healthy state and a full conversation can be completed through the stack.
- **Completion criteria:** `docker compose up` brings up the complete real-integration stack and
  a full conversation completes successfully.
- **Dependency:** Tasks 2.6, 2.7, 2.10.

### Task 2.12: Phase 2 Verification

- **Implements:** no new functionality — full-suite verification of everything built in Phase
  2.
- **Files/components:** none (verification only).
- **Tests/verification:** the default (mocked) test suite plus the separate live-sandbox test
  jobs from Tasks 2.8 and 2.9, both run to completion.
- **Completion criteria:** the default suite passes deterministically; the live-sandbox jobs
  pass when sandbox credentials are present.
- **Dependency:** Tasks 2.1-2.11.

---

## Phase 3 — Infrastructure Provisioning

### Task 3.1: AWS Service and IAM Inventory

- **Implements:** a documented inventory of every AWS service used by the project (EC2, VPC, S3
  for Terraform state, Secrets Manager, SSM Parameter Store, IAM) and the IAM policy required
  for Terraform to provision them.
- **Files/components:** `infra/terraform/README.md` (or equivalent), an IAM policy definition.
- **Tests/verification:** a review confirming the IAM policy grants exactly the permissions
  needed for the resources defined in Tasks 3.2-3.4, no broader.
- **Completion criteria:** the inventory and IAM policy are complete and an IAM
  user/role with this policy exists in the target AWS account.
- **Dependency:** Task 2.12.

### Task 3.2: Packer Node AMI

- **Implements:** a Packer build producing a Kubernetes node AMI with kubeadm, kubelet, and a
  container runtime pre-installed.
- **Files/components:** `infra/packer/`.
- **Tests/verification:** a `packer build` execution producing a usable AMI.
- **Completion criteria:** the AMI builds successfully and boots to a state where `kubeadm`,
  `kubelet`, and the container runtime are present and correctly versioned.
- **Dependency:** Task 3.1.

### Task 3.3: Terraform — AWS Networking and State Backend

- **Implements:** the VPC, subnets, and security groups for the cluster, and the Terraform
  remote state backend (S3 with native locking).
- **Files/components:** `infra/terraform/network.tf`, `infra/terraform/backend.tf`.
- **Tests/verification:** `terraform validate` and `terraform plan` against the empty target
  account; `terraform apply` provisioning the networking layer.
- **Completion criteria:** networking resources exist in AWS and Terraform state is stored
  remotely with locking confirmed to prevent concurrent applies.
- **Dependency:** Task 3.1.

### Task 3.4: Terraform — Cluster Instances and kubeadm Bootstrap

- **Implements:** EC2 instances for one control-plane node and the worker nodes, built from the
  Task 3.2 AMI; the control plane initializes via `user_data` running `kubeadm init`; workers
  retrieve a short-lived `kubeadm join` command from AWS SSM Parameter Store at boot. This is a
  self-managed cluster, not a managed EKS cluster.
- **Files/components:** `infra/terraform/cluster.tf`.
- **Tests/verification:** `terraform apply` provisioning the instances; a check that `kubectl
  get nodes` shows the control plane and every worker in `Ready` state.
- **Completion criteria:** the cluster is reachable via `kubectl` with all nodes `Ready`, and no
  Terraform `remote-exec` provisioner is used anywhere in the configuration.
- **Dependency:** Tasks 3.2, 3.3.

### Task 3.5: Kubernetes Namespace Layout

- **Implements:** separate `dev` and `prod` namespaces with distinct configuration per
  environment.
- **Files/components:** `k8s/base/namespaces.yaml`.
- **Tests/verification:** a check confirming both namespaces exist and are isolated from each
  other by the NetworkPolicies defined in Task 3.6.
- **Completion criteria:** both namespaces exist on the cluster.
- **Dependency:** Task 3.4.

### Task 3.6: NetworkPolicies

- **Implements:** default-deny NetworkPolicies with explicit allow rules: frontend to backend
  only; backend to the MCP servers and Postgres; the Playwright MCP server to the mock booking
  site only; Postgres accepting traffic from the backend only.
- **Files/components:** `k8s/base/network-policies.yaml`.
- **Tests/verification:** a conformance check confirming the Playwright MCP server cannot reach
  Postgres, and the mock booking site cannot reach anything but the Playwright MCP server.
- **Completion criteria:** all NetworkPolicy conformance checks pass.
- **Dependency:** Task 3.5.

### Task 3.7: External Secrets Operator and Secrets Manager Integration

- **Implements:** installation of the External Secrets Operator and at least one working
  `ExternalSecret` resource sourcing a value from AWS Secrets Manager into a Kubernetes Secret.
- **Files/components:** `k8s/base/external-secrets/`.
- **Tests/verification:** a check confirming the Kubernetes Secret materializes correctly at
  runtime, and that no secret value is present anywhere in the GitOps manifest repository
  (only the `ExternalSecret` reference object is committed).
- **Completion criteria:** the Secret is created and usable by a pod; a repository scan finds no
  committed plaintext secret value.
- **Dependency:** Task 3.6.

### Task 3.8: Infrastructure Verification

- **Implements:** no new functionality — verification of everything built in Phase 3.
- **Files/components:** none (verification only).
- **Tests/verification:** `terraform plan` producing no unexpected diff on a clean checkout;
  full node-readiness and NetworkPolicy conformance re-check.
- **Completion criteria:** the cluster is stable, reachable, and passes every check from Tasks
  3.4, 3.6, and 3.7.
- **Dependency:** Tasks 3.1-3.7.

---

## Phase 4 — Kubernetes Deployment

### Task 4.1: Kustomize Base Manifests

- **Implements:** base Kubernetes manifests (Deployment, Service) for every application service:
  frontend, backend, mock booking site, each of the three custom MCP servers, the Playwright MCP
  server, and Postgres.
- **Files/components:** `k8s/base/<service>/`.
- **Tests/verification:** `kustomize build k8s/base` succeeds with no errors.
- **Completion criteria:** the base build succeeds for every service.
- **Dependency:** Task 3.8.

### Task 4.2: ConfigMaps

- **Implements:** ConfigMaps for every service's non-secret configuration (e.g. log level,
  feature flags, external API base URLs).
- **Files/components:** `k8s/base/<service>/configmap.yaml`.
- **Tests/verification:** a check that each Deployment correctly mounts/consumes its
  ConfigMap.
- **Completion criteria:** every service starts successfully with its configuration sourced
  from a ConfigMap rather than hardcoded values.
- **Dependency:** Task 4.1.

### Task 4.3: Secret Wiring per Service

- **Implements:** per-service `ExternalSecret` resources (built on Task 3.7) supplying each
  service only the credentials it needs — least privilege between services.
- **Files/components:** `k8s/base/<service>/external-secret.yaml`.
- **Tests/verification:** a check confirming the Playwright MCP server and mock booking site
  receive no database or third-party API credentials, matching the least-privilege design.
- **Completion criteria:** every service that needs a secret receives exactly the secret(s) it
  needs, and no others.
- **Dependency:** Tasks 3.7, 4.1.

### Task 4.4: Liveness and Readiness Probes

- **Implements:** liveness and readiness probes for every service (HTTP health endpoints for
  the backend, frontend, and MCP servers; a connection check for Postgres).
- **Files/components:** `k8s/base/<service>/deployment.yaml` (probe configuration).
- **Tests/verification:** a check that a deliberately unhealthy container is correctly marked
  not-ready and restarted according to its liveness probe.
- **Completion criteria:** every service's probes correctly reflect its health state.
- **Dependency:** Task 4.1.

### Task 4.5: Resource Requests and Limits

- **Implements:** CPU and memory requests/limits for every Deployment, sized to each service's
  expected load.
- **Files/components:** `k8s/base/<service>/deployment.yaml` (resource configuration).
- **Tests/verification:** a check that every container spec declares both requests and limits,
  and that no pod is evicted under normal load in the `dev` overlay.
- **Completion criteria:** every Deployment has resource requests and limits defined.
- **Dependency:** Task 4.1.

### Task 4.6: Horizontal Pod Autoscaler

- **Implements:** a Horizontal Pod Autoscaler for the backend service, scaling on CPU
  utilization within a defined minimum/maximum replica range.
- **Files/components:** `k8s/base/backend/hpa.yaml`.
- **Tests/verification:** a load test confirming the backend scales out under sustained CPU
  load and scales back in after load subsides.
- **Completion criteria:** the HPA scales the backend Deployment correctly under a synthetic
  load test.
- **Dependency:** Task 4.5.

### Task 4.7: `dev` Overlay Deployment and Verification

- **Implements:** the `dev` Kustomize overlay and its manual deployment to the cluster's `dev`
  namespace.
- **Files/components:** `k8s/overlays/dev/`.
- **Tests/verification:** `kubectl apply -k k8s/overlays/dev`; a full conversation run against
  the deployed frontend confirming it reaches the in-cluster backend and MCP servers correctly.
- **Completion criteria:** every service deploys successfully to `dev` and a full conversation
  completes end to end against the in-cluster stack.
- **Dependency:** Tasks 4.2, 4.3, 4.4, 4.5, 4.6.

### Task 4.8: `prod` Overlay Definition

- **Implements:** the `prod` Kustomize overlay, parameterized for the production namespace and
  configuration, not yet carrying live traffic.
- **Files/components:** `k8s/overlays/prod/`.
- **Tests/verification:** `kustomize build k8s/overlays/prod` succeeds with no errors.
- **Completion criteria:** the `prod` overlay builds successfully and differs from `dev` only in
  environment-specific configuration (namespace, replica counts, secret references).
- **Dependency:** Task 4.7.

### Task 4.9: Deployment Verification

- **Implements:** no new functionality — verification of everything built in Phase 4.
- **Files/components:** none (verification only).
- **Tests/verification:** re-run of the Task 4.7 end-to-end conversation check plus a review
  confirming every service has probes, resource limits, and correctly scoped secrets.
- **Completion criteria:** all Phase 4 checks pass together on a freshly applied `dev`
  namespace.
- **Dependency:** Tasks 4.1-4.8.

---

## Phase 5 — CI/CD and GitOps

### Task 5.1: GitHub Actions — Tests on Every Pull Request

- **Implements:** a GitHub Actions workflow that runs on every pull request: linting,
  type-checking, and the full unit/node-level test suite from Phase 1 and the contract tests
  from Phase 2.
- **Files/components:** `.github/workflows/pr-checks.yml`.
- **Tests/verification:** a test pull request confirming the workflow triggers automatically
  and reports pass/fail status on the PR.
- **Completion criteria:** every pull request automatically runs the full check suite before it
  can be merged.
- **Dependency:** Task 4.9.

### Task 5.2: Test Result Reporting

- **Implements:** structured test-result reporting surfaced directly on the pull request (test
  counts, pass/fail summary, and failure details for any failing test).
- **Files/components:** `.github/workflows/pr-checks.yml` (test-report step/action).
- **Tests/verification:** a test pull request with an intentionally failing test, confirming
  the failure is clearly reported on the PR rather than only visible in raw log output.
- **Completion criteria:** test results are visible in a clear, human-readable form directly on
  the pull request.
- **Dependency:** Task 5.1.

### Task 5.3: Image Build and Push

- **Implements:** a CI step building and pushing container images (backend, frontend, each MCP
  server, mock booking site) to a container registry, gated on the Task 5.1 checks passing.
- **Files/components:** `.github/workflows/build-and-push.yml`.
- **Tests/verification:** a merge to the main branch confirming an image is built and pushed
  with a traceable tag (e.g. commit SHA).
- **Completion criteria:** every service's image builds and pushes successfully on a merge to
  main.
- **Dependency:** Task 5.1.

### Task 5.4: Automatic `dev` Overlay Tag Update

- **Implements:** a CI step that updates the `dev` overlay's image tag to match the
  just-published image, with no manual edit.
- **Files/components:** `.github/workflows/build-and-push.yml` (tag-update step),
  `k8s/overlays/dev/`.
- **Tests/verification:** a merge to main confirming the `dev` overlay's manifest is updated
  automatically in the same pipeline run.
- **Completion criteria:** the `dev` overlay's image tag always reflects the latest successful
  build from main with no manual step.
- **Dependency:** Task 5.3.

### Task 5.5: Argo CD Installation and Application Resources

- **Implements:** Argo CD deployed to the cluster, with an `Application` resource per service
  pointed at the `dev` overlay, making Argo CD the sole applier of manifests from this point
  forward.
- **Files/components:** `k8s/argocd/` (Argo CD installation manifests, `Application`
  definitions).
- **Tests/verification:** a change to the `dev` overlay (from Task 5.4) confirmed to sync to the
  cluster automatically via Argo CD, with no manual `kubectl apply` performed.
- **Completion criteria:** Argo CD detects and applies every manifest change to `dev`
  automatically.
- **Dependency:** Tasks 4.9, 5.4.

### Task 5.6: `dev`-to-`prod` Promotion Workflow

- **Implements:** an explicit, PR-based process for promoting a validated `dev` image tag to
  the `prod` overlay — never automatic, never applied directly by CI.
- **Files/components:** `k8s/overlays/prod/` (promotion PR template/process
  documentation).
- **Tests/verification:** a promotion PR confirming the `prod` overlay updates only after
  explicit review and merge, and that Argo CD (not CI) performs the resulting cluster
  application.
- **Completion criteria:** promoting to `prod` requires an explicit reviewed PR merge, and the
  resulting deployment is applied by Argo CD.
- **Dependency:** Task 5.5.

### Task 5.7: CI/CD Pipeline Verification

- **Implements:** no new functionality — verification of everything built in Phase 5.
- **Files/components:** none (verification only).
- **Tests/verification:** an end-to-end pipeline run: pull request opened, checks run and
  reported, merge triggers build/push, `dev` overlay updates automatically, Argo CD syncs the
  change, and a promotion PR updates `prod`.
- **Completion criteria:** the entire pipeline completes without any manual cluster
  intervention outside the promotion PR itself.
- **Dependency:** Tasks 5.1-5.6.

---

## Phase 6 — Observability

### Task 6.1: Prometheus Deployment and Scrape Configuration

- **Implements:** Prometheus deployed to the cluster, configured to scrape every application
  service and the infrastructure exporters added in Task 6.2.
- **Files/components:** `k8s/base/monitoring/prometheus/`.
- **Tests/verification:** a check confirming Prometheus's target list shows every expected
  service as `up`.
- **Completion criteria:** Prometheus is running and successfully scraping all defined targets.
- **Dependency:** Task 5.7.

### Task 6.2: Infrastructure Exporters

- **Implements:** `node-exporter`, `kube-state-metrics`, and `postgres_exporter`, giving
  visibility into node/cluster health and database state.
- **Files/components:** `k8s/base/monitoring/exporters/`.
- **Tests/verification:** a check confirming metrics from each exporter appear in Prometheus.
- **Completion criteria:** all three exporters are running and their metrics are queryable in
  Prometheus.
- **Dependency:** Task 6.1.

### Task 6.3: Structured Logging for All Services

- **Implements:** structured (JSON) logging in the backend, each MCP server, and the mock
  booking site, with the same sanitization boundary as `error_log` — no traveler PII, tokens,
  or API keys ever logged.
- **Files/components:** a shared logging configuration module per service.
- **Tests/verification:** a test asserting a known synthetic PII value never appears in log
  output during a full conversation run.
- **Completion criteria:** every service produces structured logs retrievable via `kubectl
  logs`, and the PII-non-leakage test passes.
- **Dependency:** Task 5.7.

### Task 6.4: Application-Level Metrics Instrumentation

- **Implements:** metrics exported by the backend covering workflow completions and failures,
  interrupt counts per gate, retry counts per node, per-node execution duration, MCP tool call
  latency per server, LLM call latency and failure counts, and downstream side-effect
  (email/CRM) status.
- **Files/components:** `backend/app/observability/metrics.py`, instrumentation calls added at
  each node and tool/LLM call site.
- **Tests/verification:** a live conversation run confirming each metric appears in Prometheus
  with a plausible value; a check confirming no metric or trace attribute contains traveler
  PII.
- **Completion criteria:** all listed metrics are visible in Prometheus during a live run, and
  the PII-non-leakage check passes.
- **Dependency:** Task 6.1.

### Task 6.5: Grafana Service Dashboards

- **Implements:** Grafana dashboards for request latency/error rate, per-node duration, MCP call
  success/failure rate, and cluster/node health.
- **Files/components:** `k8s/base/monitoring/grafana/dashboards/`.
- **Tests/verification:** a live conversation run confirming each dashboard populates from real
  traffic.
- **Completion criteria:** every dashboard shows live data during a test run.
- **Dependency:** Tasks 6.2, 6.4.

### Task 6.6: Agent-Health Dashboard

- **Implements:** a dedicated dashboard summarizing agent-specific health: workflow
  completion/failure rate, interrupt-gate activity, retry rates by node, LLM call
  latency/failure rate, and MCP tool latency/failure rate by server.
- **Files/components:** `k8s/base/monitoring/grafana/dashboards/agent-health.json`.
- **Tests/verification:** a live conversation run, including a deliberately triggered failure,
  confirming the dashboard reflects both normal and failure activity correctly.
- **Completion criteria:** the agent-health dashboard accurately reflects a live conversation
  run in both the success and failure case.
- **Dependency:** Task 6.4.

### Task 6.7: Alerting Rules

- **Implements:** Alertmanager rules for elevated overall error rate, elevated request/node
  latency, repeated LLM call failures, and repeated MCP tool call timeouts.
- **Files/components:** `k8s/base/monitoring/alertmanager/rules.yaml`.
- **Tests/verification:** a synthetic failure injection for each alert category (forced LLM
  error, forced MCP timeout, forced high latency, forced node failure), confirming the
  corresponding alert fires.
- **Completion criteria:** all four alert categories fire correctly under synthetic injection
  and do not fire under normal operation.
- **Dependency:** Tasks 6.1, 6.4.

### Task 6.8: Observability Verification

- **Implements:** no new functionality — verification of everything built in Phase 6.
- **Files/components:** none (verification only).
- **Tests/verification:** a full conversation run reviewed against every dashboard and alert
  rule together.
- **Completion criteria:** metrics, logs, dashboards, and alerts all correctly reflect a single
  live conversation run, end to end.
- **Dependency:** Tasks 6.1-6.7.

---

## Phase 7 — Final Verification and Demo Preparation

### Task 7.1: Deploy `prod` Overlay

- **Implements:** the promotion of a validated image set to the `prod` overlay via the Task 5.6
  promotion workflow, applied to the cluster by Argo CD.
- **Files/components:** `k8s/overlays/prod/` (promoted state).
- **Tests/verification:** a check that every service in the `prod` namespace reaches `Ready`
  state.
- **Completion criteria:** the full application stack is running in `prod`.
- **Dependency:** Tasks 5.7, 6.8.

### Task 7.2: End-to-End Happy-Path Production Test

- **Implements:** no new functionality — a full conversation (trip request through CRM update)
  executed against the live `prod` environment.
- **Files/components:** none (verification only).
- **Tests/verification:** the conversation is run to completion through the deployed frontend
  and backend in `prod`.
- **Completion criteria:** the conversation completes successfully with a confirmation email
  sent and a CRM record created.
- **Dependency:** Task 7.1.

### Task 7.3: End-to-End Reject-at-Review-Gate Production Test

- **Implements:** no new functionality — a full conversation that is rejected at the human
  review gate and then completes successfully on retry, executed against `prod`.
- **Files/components:** none (verification only).
- **Tests/verification:** the conversation is run to completion, including the rejection and
  retry path, through the deployed frontend and backend in `prod`.
- **Completion criteria:** the rejection correctly routes back to the appropriate earlier step
  and the conversation completes successfully on retry.
- **Dependency:** Task 7.1.

### Task 7.4: AWS-Specific Integration Verification

- **Implements:** no new functionality — verification of AWS-specific behavior that cannot be
  validated outside the real AWS environment.
- **Files/components:** none (verification only).
- **Tests/verification:** confirmation that IAM permissions, EBS-backed persistent storage, and
  the External Secrets Operator's integration with AWS Secrets Manager all function correctly
  in `prod`.
- **Completion criteria:** all three AWS-specific integrations are confirmed working in `prod`.
- **Dependency:** Task 7.1.

### Task 7.5: Dashboard and Alert Verification During Live Runs

- **Implements:** no new functionality — confirmation that observability reflects the Task 7.2
  and 7.3 runs in real time.
- **Files/components:** none (verification only).
- **Tests/verification:** the Grafana dashboards (including the agent-health dashboard) and
  Prometheus/Alertmanager state are observed during both production runs.
- **Completion criteria:** both runs are visibly reflected on the dashboards in real time, with
  no unexpected alerts firing.
- **Dependency:** Tasks 7.2, 7.3, 6.8.

### Task 7.6: Presentation and Live-Demo Preparation

- **Implements:** the materials needed to present and demonstrate the completed project:
  a slide deck or written walkthrough covering the problem, architecture, and safety design; a
  live-demo script covering the happy path and the reject-at-review-gate path; and a recorded
  backup demo in case the live environment is unavailable during presentation.
- **Files/components:** presentation materials (slides/script), a recorded demo video.
- **Tests/verification:** a full rehearsal of the live demo against `prod`, confirmed to
  complete within the time available and to match the recorded backup.
- **Completion criteria:** the live demo has been rehearsed successfully at least once, and a
  working recorded backup exists.
- **Dependency:** Task 7.5.

### Task 7.7: Final Requirements Verification

- **Implements:** no new functionality — a final review confirming every requirement in the
  Requirements Traceability Matrix below is satisfied by a completed, passing task.
- **Files/components:** none (verification only).
- **Tests/verification:** a line-by-line check of the traceability matrix against the current
  state of the repository and the deployed `prod` environment.
- **Completion criteria:** every row in the Requirements Traceability Matrix is satisfied.
- **Dependency:** Task 7.6.

---

## Requirements Traceability Matrix

| Requirement | Task(s) |
|---|---|
| Project scaffolding and configuration | 1.1 |
| LangGraph agent implementation | 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15 |
| Agent system prompt, capabilities, and boundaries | 1.8 |
| Retries, fallbacks, graceful termination, and clear error responses | 1.6, 1.14, 1.16 |
| FastAPI HTTP API | 1.16 |
| React chat frontend | 1.17 |
| Postgres application data and LangGraph checkpointer | 1.4, 1.5 |
| Encrypted traveler PII handling | 1.3, 1.11 |
| Amadeus MCP server | 2.1 |
| Gmail MCP server | 2.2 |
| HubSpot MCP server | 2.3 |
| Playwright MCP integration | 2.4 |
| Real MCP transport integration tests | 2.8 |
| Mock booking website with no payment capability | 2.5 |
| Dockerfiles and local Docker Compose environment | 1.18, 2.6, 2.11 |
| Unit tests with mocked LLM and external services | 1.7, 1.9-1.14 |
| Integration tests between the agent and MCP servers | 2.9 |
| A clear test-plan section | Test Plan (above) |
| Terraform provisioning of all AWS resources | 3.3, 3.4 |
| Packer image creation | 3.2 |
| Self-managed Kubernetes on AWS EC2 using kubeadm, not EKS | 3.4 |
| Separate dev and prod namespaces and configuration | 3.5, 4.7, 4.8 |
| Liveness and readiness probes | 4.4 |
| Resource requests and limits | 4.5 |
| HPA | 4.6 |
| ConfigMaps | 4.2 |
| Secrets management | 3.7, 4.3 |
| AWS services used by the project | 3.1 |
| GitHub Actions tests on every pull request | 5.1 |
| Clear test-result reporting | 5.2 |
| Image build and push | 5.3 |
| Deployment to dev and prod | 4.7, 4.8, 5.6, 7.1 |
| Argo CD GitOps deployment | 5.5 |
| Metrics and logs from all services | 6.3, 6.4 |
| Prometheus and Grafana | 6.1, 6.5 |
| Alerts for errors, latency, LLM failures, and MCP tool timeouts | 6.7 |
| Agent-health dashboard | 6.6 |
| Final end-to-end tests | 7.2, 7.3 |
| Presentation and live-demo preparation | 7.6 |
