"""Cwd glob → entity_id resolution.

Entities have cwd_hints like ["/opt/agentssot", "~/.claude/plugins/hari-hive"].
Resolver matches a given cwd against any prefix in any entity's cwd_hints.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.loadout import resolve_cwd_entities


def test_exact_path_match():
    entities = [
        {"id": "e1", "slug": "agentssot", "cwd_hints": ["/opt/agentssot"]},
        {"id": "e2", "slug": "hive-plugin", "cwd_hints": ["/.claude/plugins/hari-hive"]},
    ]
    matched = resolve_cwd_entities("/opt/agentssot", entities)
    assert {e["slug"] for e in matched} == {"agentssot"}


def test_subdirectory_match():
    entities = [{"id": "e1", "slug": "agentssot", "cwd_hints": ["/opt/agentssot"]}]
    matched = resolve_cwd_entities("/opt/agentssot/api/app", entities)
    assert len(matched) == 1


def test_unrelated_cwd_returns_empty():
    entities = [{"id": "e1", "slug": "agentssot", "cwd_hints": ["/opt/agentssot"]}]
    matched = resolve_cwd_entities("/home/hari/elsewhere", entities)
    assert matched == []


def test_multiple_entities_can_match():
    entities = [
        {"id": "e1", "slug": "claude-config", "cwd_hints": ["/.claude"]},
        {"id": "e2", "slug": "hari-hive-plugin", "cwd_hints": ["/.claude/plugins/hari-hive"]},
    ]
    matched = resolve_cwd_entities("/home/hari/.claude/plugins/hari-hive", entities)
    assert {e["slug"] for e in matched} == {"claude-config", "hari-hive-plugin"}
