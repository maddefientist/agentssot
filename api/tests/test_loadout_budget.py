"""Loadout budget pack honors loadout_priority and stops at token cap."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.loadout import pack_loadout


def _item(id_, type_, abstract, priority=0):
    return {
        "id": id_, "memory_type": type_, "abstract": abstract,
        "title": abstract[:40], "priority": priority,
    }


def test_budget_respects_priority_order():
    items = [
        _item("a", "command", "low priority", priority=1),
        _item("b", "command", "high priority", priority=10),
        _item("c", "rule", "med priority", priority=5),
    ]
    packed, overflow, used = pack_loadout(items, token_budget=50)
    assert packed[0]["id"] == "b"  # highest priority first
    assert packed[1]["id"] == "c"
    # 'a' may or may not fit depending on tokens; check ordering not exclusion


def test_budget_stops_at_cap():
    big = "x " * 100  # ~200 tokens
    items = [_item(f"id{i}", "skill", big, priority=10 - i) for i in range(5)]
    packed, overflow, used = pack_loadout(items, token_budget=300)
    assert used <= 300
    assert overflow >= 0
    assert len(packed) + overflow == 5


def test_empty_input():
    packed, overflow, used = pack_loadout([], token_budget=750)
    assert packed == [] and overflow == 0 and used == 0
