-- 008_review_attempts.sql — convergence guard for review-queue reclassification.
--
-- Problem: reclassify_low_conf left low-confidence rows `pending` with nothing
-- stamped, so items the classifier can't type re-enter the pool forever. A
-- drain loop over them never converges (it ran 3 days on 5 stuck items, 2026-06).
--
-- Fix: track per-row attempts so a row can reach a terminal state (dismissed as
-- "unclassifiable") after RECLASSIFY_MAX_ATTEMPTS. Additive, non-breaking.
-- Rollback: ALTER TABLE review_queue DROP COLUMN attempts;

ALTER TABLE review_queue
    ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;
