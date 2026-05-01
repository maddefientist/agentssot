"""Supersession: same-tier same-entity items get marked superseded."""
from app.services.lifecycle import find_supersession_candidates


def _item(id_, type_, entity_refs, content, confidence=1.0):
    class M:
        pass
    o = M()
    o.id = id_
    o.memory_type = type_
    o.entity_refs = entity_refs
    o.content = content
    o.confidence = confidence
    o.superseded_by = None
    return o


def test_same_tier_same_entity_finds_candidates():
    new = _item("new", "command", ["e-unraid"], "ssh unraid -p 22 root@192.168.1.116")
    existing = [
        _item("old", "command", ["e-unraid"], "ssh unraid"),
        _item("other", "command", ["e-hari"], "ssh hari"),
    ]
    matches = find_supersession_candidates(new, existing)
    assert {m.id for m in matches} == {"old"}


def test_different_tier_no_match():
    new = _item("new", "command", ["e-unraid"], "ssh unraid")
    existing = [_item("rule1", "rule", ["e-unraid"], "Never access unraid")]
    assert find_supersession_candidates(new, existing) == []
