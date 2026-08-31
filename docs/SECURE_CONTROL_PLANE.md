# Secure multi-agent control plane

AgentSSOT's core job is durable memory. A shared task/control plane is an
optional layer and must not weaken that kernel. This document defines the
security boundary required before external agents, chat surfaces, or social
account identities are connected.

## Goal and non-goals

The goal is trustworthy coordination that reduces the time from an approved
task to a safe handoff. Success is measured by admission, revocation,
provenance, isolation, rollback, and handoff latency.

This layer does not optimize followers, traffic, engagement, or posting volume.
It does not let an agent approve its own work, grant authority, or widen its own
capabilities.

## Separate the planes

| Plane | Default | Responsibility |
|---|---|---|
| Memory data plane | Enabled on loopback | Scoped ingest, exact query, bounded recall, feedback |
| Operator gateway | Disabled | Authenticated command and status transport |
| Task control plane | Not yet production-enabled | Tasks, approvals, dispatch, state receipts |
| External adapters | Not yet production-enabled | Narrow identities such as Grokbot |

The gateway being reachable is not evidence that the task control plane is
safe. Each layer has its own enablement and acceptance gate.

## Principals and capabilities

Every actor receives its own API key and an explicit capability set. Shared
keys, wildcard namespaces, and identity-by-display-name are prohibited for
external adapters.

| Principal | May | Must never |
|---|---|---|
| Human operator | Approve, revoke, grant bounded capabilities | Delegate root authority implicitly |
| Orchestrator | Propose plans, dispatch approved work, collect receipts | Approve its own expansion or mint authority |
| Implementation worker | Read assigned context, modify an admitted worktree | Publish, deploy, or cross namespace without approval |
| Reviewer | Read evidence and issue a review receipt | Mutate the implementation under review |
| Grokbot / social staff | Read approved brand context; draft or queue proposals for named accounts | Approve, post, develop, deploy, grant access, arbitrate, or self-escalate |

Grokbot is always a low-authority social-staff identity. Capabilities are scoped
per brand and account, for example `brand:acme/account:main/draft`. A capability
for one account confers no access to another account, and a draft capability is
not a publish capability.

## Current memory-service admission

1. An admin creates or selects a namespace owner.
2. The admin issues a single-use enrollment token with a role, namespace list,
   expiry, and intended principal name.
3. The principal redeems the token once and receives a unique API key.
4. The principal proves its effective identity with `/whoami` before work is
   admitted.

This is authenticated admission to the memory service, not a complete external
control-plane admission protocol. The current schema does not persist an
immutable issuer-linked admission receipt or a general capability vocabulary.
Before any external adapter or task plane is enabled, it must add a durable
receipt containing the key ID, issuer ID, exact namespaces/capabilities, and
timestamps. Raw credentials must never be logged or included in that receipt.

Passphrase-based open enrollment is not shipped. It is not an appropriate path
for external agents.

## Namespace ownership

- Every production namespace has one accountable owner and a documented
  purpose.
- Keys receive the minimum reader/writer role and exact namespace list.
- Only an admin can grant or remove namespaces. The current `admin` role is
  global control-plane authority, not a namespace-scoped delegated-admin role.
- Revocation, role downgrade, and namespace removal must affect new requests
  on their next authorization check and already-open transports within the
  configured revalidation interval. The bcrypt-match cache must re-read current
  key state rather than cache authority.
- Cross-namespace retrieval is an explicit orchestrator action with a receipt,
  never an implicit search expansion.

## Gateway transport boundary

The experimental gateway is disabled by default. When enabled:

- only an authenticated admin authorized for the gateway namespace can mint a
  transport ticket;
- orchestration and dispatch remain disabled until the separate
  `GATEWAY_EXECUTION_ENABLED` production decision is made;
- tickets are short-lived, single-use, transport-scoped, and kept out of URLs;
- established WebSocket and SSE sessions periodically re-read the API key from
  the database and have a hard maximum lifetime;
- frames, message rate, and concurrent connections are bounded;
- session IDs are prefixed by the API-key ID so one principal cannot resume
  another principal's session; and
- the production container's owned entrypoint rejects Uvicorn multi-worker and
  unknown launch commands before server startup, and the application rejects
  direct launch without that entrypoint's safety attestation, because current
  ticket and abuse-limit state is in memory.

External low-authority identities do not receive an admin gateway ticket. They
require a separate adapter endpoint whose capability vocabulary cannot express
developer, approval, grant, deployment, or publication authority.

## Task, approval, and provenance receipts

The production task plane must use immutable, linked records:

- `TaskReceipt`: task ID, requester, scope, namespace, allowed tools, input
  source hashes, and expiry;
- `ApprovalReceipt`: approver identity, exact action/scope approved, artifact or
  revision hash, and expiry;
- `DispatchReceipt`: admitted worker identity, effective model, worktree, and
  capability subset;
- `StateReceipt`: terminal status, changed artifact hashes, tests, timing, and
  errors; and
- `ReviewReceipt`: independent reviewer, revision reviewed, findings, and
  acceptance decision.

Receipts are append-only. A task cannot approve itself, a worker cannot issue
its own review receipt, and an approval for one revision does not silently carry
forward to another. Memory recall results are evidence inputs, not authority.

The current write-ahead audit log is useful operational evidence but is
best-effort. It is not yet a sufficient approval ledger. Production control-plane
enablement remains gated until durable receipt storage and verification exist.

## Acceptance gate

Before enabling any external adapter or task execution surface, verify:

- gateway routes are absent by default and passphrase enrollment is absent in
  every configuration;
- API, database proxy, and admin UI bind to loopback unless explicitly changed;
- ticket authentication, expiry, one-time use, and cross-scope rejection;
- revocation of already-open WebSocket and SSE sessions;
- namespace and role downgrade propagation;
- frame, message-rate, connection, and worker-count limits;
- subprocess proof that the production launcher rejects `--workers 2` before
  Uvicorn starts;
- no credential in URLs, access logs, fixtures, or receipts;
- immutable task/approval/dispatch/state/review receipt linkage;
- a gateway-disabled canary after deployment; and
- a tested rollback that disables external surfaces without restoring a known
  vulnerable image.

## Rollback

The safe operational rollback is configuration-first:

1. Set `GATEWAY_ENABLED=false`.
2. Restore loopback bind hosts.
3. Restart onto a security-patched image.
4. Verify gateway/HUD and passphrase-enrollment routes are absent, invalid
   token redemption rejects, valid admin-issued token enrollment still works,
   and pre-restart transports are gone.

Do not roll back to an image that contains unauthenticated gateway routes.

## Time to safe handoff

Track the elapsed time from an approved `TaskReceipt` to a verified
`StateReceipt`, with separate time spent in admission, execution, review, and
operator approval. A faster handoff counts only when namespace checks, required
receipts, tests, and rollback evidence are complete.
