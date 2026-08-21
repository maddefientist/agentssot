"""Regression: /feedback query-mode must not silently rate an unrelated concept.

Before this, crud.create_concept_feedback took the argmax concept for ANY query
with no distance predicate, wrote a ConceptFeedback row, and (on 'wrong' + note)
also ingested a `Correction: ... (re: concept 'X')` knowledge item asserting an
association nobody made. It then returned the matched concept's own stored
`confidence` column as if it were a match score, so the misfire read as a
high-confidence success. ~18 unrelated 'wrong' signals piled onto one concept
that way.

These tests pin the three properties that kill that class:
  D2  a sub-threshold query resolves to NOTHING (no row, no correction)
  D3  the response reports the real match distance, distinct from `confidence`
  D4  a sub-threshold 'wrong' + note creates no correction knowledge item
"""
import pytest

from app import crud
from app.models import ConceptFeedback, KnowledgeItem


class _FakeConcept:
    def __init__(self, cid="c-1", title="Execute Feedback Loop Triad Verification"):
        self.id = cid
        self.title = title
        self.namespace = "claude-shared"
        self.confidence = 1.0  # deliberately high: the misleading D3 value
        self.confirming_agents = []
        self.tags = []


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeSession:
    """Minimal stand-in: records what would have been persisted."""

    def __init__(self, row):
        self._row = row
        self.added = []
        self.flushed = 0

    def execute(self, _stmt):
        return _FakeResult(self._row)

    def get(self, _model, _pk):
        return None

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flushed += 1


class _FakeEmbedder:
    is_available = True

    def embed_text(self, _text):
        return [0.0] * 768


@pytest.fixture(autouse=True)
def _no_profile_writes(monkeypatch):
    """update_profile_from_feedback needs a real DB; it is not under test."""
    monkeypatch.setattr(crud, "update_profile_from_feedback", lambda *a, **k: None)


def _call(distance, signal="wrong", note="Unrelated recall.", **kw):
    concept = _FakeConcept()
    session = _FakeSession((concept, distance))
    result = crud.create_concept_feedback(
        session=session,
        namespace="claude-shared",
        signal=signal,
        agent_key="device-test-writer",
        embedding_provider=_FakeEmbedder(),
        query="unrelated recall",
        note=note,
        **kw,
    )
    return result, session


def _kinds(session):
    return {
        "feedback": [o for o in session.added if isinstance(o, ConceptFeedback)],
        "knowledge": [o for o in session.added if isinstance(o, KnowledgeItem)],
    }


# ── D2: no relevance floor ────────────────────────────────────────────────

def test_subthreshold_query_records_nothing():
    """0.42 is past the floor. This is the exact shape of the historical sink:
    the phrasings that produced it measured 0.360 / 0.382 / 0.391."""
    result, session = _call(0.42)

    assert result["matched"] is False
    assert result["recorded"] is False
    assert result["concept_id"] == ""
    added = _kinds(session)
    assert added["feedback"] == [], "a sub-threshold query must not write a rating"
    assert added["knowledge"] == [], "a sub-threshold query must not write knowledge"


def test_subthreshold_response_names_the_nearest_candidate():
    """The caller must be able to recover: hand back the near-miss so it can
    re-send with an explicit id instead of guessing again."""
    result, _ = _call(0.42)
    assert result["nearest_concept_id"] == "c-1"
    assert result["nearest_concept_title"] == "Execute Feedback Loop Triad Verification"
    assert "knowledge_item_id" in result["detail"]


def test_above_threshold_query_still_records():
    """The floor must not break legitimate feedback. Genuine content-bearing
    queries measured <= 0.087 against the live corpus."""
    result, session = _call(0.05)
    assert result["matched"] is True
    assert result["recorded"] is True
    assert len(_kinds(session)["feedback"]) == 1


def test_threshold_is_configurable_and_enforced_at_the_boundary():
    assert _call(0.34)[0]["matched"] is True
    assert _call(0.36)[0]["matched"] is False
    # exactly at the floor is accepted
    assert _call(crud.FEEDBACK_MATCH_MAX_DISTANCE)[0]["matched"] is True
    # an explicit override wins over the module default
    assert _call(0.42, max_match_distance=0.5)[0]["matched"] is True
    assert _call(0.05, max_match_distance=0.01)[0]["matched"] is False


# ── D3: response masked the failure ───────────────────────────────────────

def test_confidence_is_not_a_match_score():
    """`confidence` is the concept's OWN stored column. A near-miss concept
    stored at 1.00 previously reported 1.00 and read as a perfect match."""
    result, _ = _call(0.30)
    assert result["confidence"] == 1.0          # the concept's own value
    assert result["match_distance"] == 0.30     # the real, separate signal
    assert result["match_confidence"] == pytest.approx(0.70)
    assert result["confidence"] != result["match_confidence"]


def test_id_resolved_feedback_reports_no_match_distance():
    """Nothing was fuzzy-matched, so there is no distance to report — the
    absence must be explicit rather than a fabricated 0.0."""
    concept = _FakeConcept()

    class _S(_FakeSession):
        def get(self, _model, _pk):
            return concept

    session = _S(None)
    result = crud.create_concept_feedback(
        session=session,
        namespace="claude-shared",
        signal="useful",
        agent_key="device-test-writer",
        embedding_provider=_FakeEmbedder(),
        concept_id="c-1",
    )
    assert result["resolved_by"] == "concept_id"
    assert result["match_distance"] is None
    assert result["match_confidence"] is None


# ── D4: self-reinforcing pollution loop ───────────────────────────────────

def test_subthreshold_wrong_with_note_creates_no_correction_item():
    _, session = _call(0.42, signal="wrong", note="Irrelevant to Muse closeout.")
    assert _kinds(session)["knowledge"] == []


def test_fuzzy_correction_does_not_assert_a_concept_association():
    """Above the floor a fuzzy 'wrong' is still recorded, but the caller never
    confirmed the target, so the correction must not manufacture a
    '(re: concept X)' claim that later recall will treat as fact."""
    _, session = _call(0.10, signal="wrong", note="ZeroClaw is deprecated.")
    items = _kinds(session)["knowledge"]
    assert len(items) == 1
    assert "re: concept" not in items[0].content
    assert items[0].content == "Correction: ZeroClaw is deprecated."
    assert "fuzzy-resolved" in items[0].tags


def test_id_resolved_correction_keeps_its_attribution():
    """When the caller named the concept, the association is real — keep it."""
    concept = _FakeConcept(title="Correct BrandForge Service Configuration")

    class _S(_FakeSession):
        def get(self, _model, _pk):
            return concept

    session = _S(None)
    crud.create_concept_feedback(
        session=session,
        namespace="claude-shared",
        signal="wrong",
        agent_key="device-test-writer",
        embedding_provider=_FakeEmbedder(),
        concept_id="c-1",
        note="BrandForge runs on hiveagent now.",
    )
    items = _kinds(session)["knowledge"]
    assert len(items) == 1
    assert "(re: concept 'Correct BrandForge Service Configuration')" in items[0].content
    assert "fuzzy-resolved" not in items[0].tags
