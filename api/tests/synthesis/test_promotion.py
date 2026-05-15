"""Unit tests for the Concept -> doctrine promotion eligibility logic.

The DB-coupled writer is smoke-tested in place against the running API
after deploy; these tests cover the pure-logic gate.
"""
from app.models import ConceptType
from app.synthesis.promotion import (
    CONFIDENCE_THRESHOLD,
    DOCTRINE_LOADOUT_PRIORITY,
    is_eligible,
    _build_ki,
)


class _FakeConcept:
    def __init__(self, **kw):
        self.id = kw.get("id", "00000000-0000-0000-0000-000000000001")
        self.namespace = kw.get("namespace", "default")
        self.type = kw.get("type", ConceptType.principle)
        self.confidence = kw.get("confidence", 0.9)
        self.tags = kw.get("tags", [])
        self.embedding = kw.get("embedding", [0.1] * 8)
        self.title = kw.get("title", "Test principle")
        self.content = kw.get("content", "Test content")


def test_eligible_principle_passes():
    assert is_eligible(_FakeConcept(type=ConceptType.principle, confidence=0.9)) is True


def test_eligible_mental_model_passes():
    assert is_eligible(_FakeConcept(type=ConceptType.mental_model, confidence=0.85)) is True


def test_skill_type_rejected():
    assert is_eligible(_FakeConcept(type=ConceptType.skill)) is False


def test_relationship_type_rejected():
    assert is_eligible(_FakeConcept(type=ConceptType.relationship)) is False


def test_low_confidence_rejected():
    assert is_eligible(_FakeConcept(confidence=CONFIDENCE_THRESHOLD - 0.01)) is False


def test_at_threshold_passes():
    assert is_eligible(_FakeConcept(confidence=CONFIDENCE_THRESHOLD)) is True


def test_superseded_tag_rejects():
    assert is_eligible(_FakeConcept(tags=["superseded", "other"])) is False


def test_null_embedding_rejects():
    assert is_eligible(_FakeConcept(embedding=None)) is False


def test_build_ki_carries_fields():
    c = _FakeConcept(
        title="Always verify",
        content="When in doubt, run the test before claiming completion.",
        tags=["operator-taught"],
    )
    ki = _build_ki(c)
    assert ki.memory_type == "doctrine"
    assert ki.loadout_priority == DOCTRINE_LOADOUT_PRIORITY
    assert ki.abstract == "Always verify"
    assert ki.source == "synthesis-promotion"
    assert ki.source_ref == str(c.id)
    assert "doctrine-promoted" in ki.tags
    assert "operator-taught" in ki.tags
    assert ki.verbatim is False
