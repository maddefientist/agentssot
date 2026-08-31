"""Contradiction detector: new command/skill for entity X vs existing rule
mentioning X with negation patterns."""
from app.services.contradiction import detect_contradictions


def _rule(id_, content, entity_refs):
    class M:
        pass
    o = M()
    o.id = id_
    o.memory_type = "rule"
    o.content = content
    o.entity_refs = entity_refs
    return o


def test_negation_rule_contradicts_command():
    rules = [_rule("r1", "Never access storage-node - this host is OFF LIMITS", ["e-storage-node"])]
    matches = detect_contradictions(
        new_type="command",
        new_entity_refs=["e-storage-node"],
        existing_rules=rules,
    )
    assert {m.id for m in matches} == {"r1"}


def test_affirmative_rule_does_not_contradict():
    rules = [_rule("r1", "Always specify the namespace when querying", ["e-hive"])]
    matches = detect_contradictions(
        new_type="command",
        new_entity_refs=["e-hive"],
        existing_rules=rules,
    )
    assert matches == []


def test_unrelated_entity_does_not_contradict():
    rules = [_rule("r1", "Never access storage-node", ["e-storage-node"])]
    matches = detect_contradictions(
        new_type="command",
        new_entity_refs=["e-build-host"],
        existing_rules=rules,
    )
    assert matches == []
