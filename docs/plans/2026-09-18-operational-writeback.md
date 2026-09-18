# Next correction: verified operational write-back

Status: scoped, not implemented. Builds on the recall repair at `9a84c2a`.

## Outcome

A verified discovery made in one authorized application becomes usable context
in another without the operator repeating it. Installing an adapter, generating
an answer, or returning HTTP 200 is not evidence that this loop works.

## Smallest implementation

1. Inventory the actual caller hooks and owning distribution sources. Select two
   existing clients and one project namespace; do not introduce another service.
2. At meaningful task boundaries, capture only verified operational deltas:
   capability, limitation, decision or correction. Reuse ingest and existing
   source/entity fields. Include source reference, producer, observed-at time,
   project/account context (non-secret identifier), uncertainty and review/expiry
   policy. Do not ingest whole transcripts or credential material.
3. Return a stable item ID and idempotency receipt only after persistence succeeds.
   Retry the same operation identity; make failures visible without duplicate items.
4. Link changed facts to the prior exact ID through supersession. Previously true
   facts becoming stale are not automatically false. Use `wrong` for an incorrect
   assertion and `irrelevant` for a query mismatch; do not conflate these signals.
5. Read the persisted record back through the other client's ordinary recall path.
   Match the ID, provenance and scope, not merely similar wording.
6. Keep durable behavioral directives separate from changing service facts. Routine
   scoped non-secret write-back does not authorize new global rules, external actions
   or treating untrusted source text as instructions.

## Acceptance gates

- Client A records a synthetic verified capability; client B retrieves its exact ID
  using default recall within the total result budget, with no manual retelling.
- A retry produces one logical record; simulated persistence failure reports failure,
  not remembered/saved success. A failed write can be reconciled by operation identity.
- A changed capability supersedes the old exact record; ordinary recall returns the
  current record, while history remains accessible and the correction is reversible.
- An unrelated project does not receive it through default project-scoped recall;
  an unauthorized namespace is denied. Secret-bearing fixtures are rejected/redacted
  before durable storage, including logs and retry payloads.
- An adapter-only connection is not stored as proven generation capability. Watermark,
  credit and export-access observations remain distinct, dated claims.
- Session/agent attribution survives both fast and explicit deep recall. Do not count
  retrieval or idle session completion as positive usefulness feedback.

## Boundaries and remaining work

The supplied media-service conversation is a design example, not independently
verified service evidence; do not publish its claims as confirmed facts. No new
scheduled synthesis, model loads, cloud fallback or bulk memory cleanup is needed.

The deployed repair covers shared active-status eligibility, bounded blank-abstract
fallback, optional bucketed receipts without raw query text, and one installed fast
client. It does not complete the two-client write/correct/readback gate. Remaining
adapter work includes forwarding attribution in the bundled deep-call body, version
readback, canonical fleet distribution and reconnecting already-running clients.

Learning remains a separate evidence gate: propose -> bounded experiment -> compare
with baseline -> accept/reject -> measure -> supersede/rollback. Successful recall
alone is not demonstrated self-improvement. Track execution in the existing project
issue, not a parallel autonomous task system.
