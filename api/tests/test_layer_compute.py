"""Layer compute: produces L0 abstract (≤50 tokens) and L1 summary (≤500 tokens).

Uses the same classifier output (already includes abstract/summary fields)
when available; falls back to a heuristic head-of-content when classifier
is unavailable.
"""
from app.llm.layer_compute import compute_layers


def test_classifier_output_used_when_present():
    classifier_out = {
        "abstract": "Restart Gluetun then qbit when port forwarding fails.",
        "summary": "When Gluetun loses VPN port forwarding, qBittorrent's listen_port "
                   "becomes stale. The fix is to restart Gluetun first, then qbit "
                   "(qbit shares Gluetun's netns).",
    }
    result = compute_layers("the original content", classifier_out)
    assert result["abstract"] == classifier_out["abstract"]
    assert result["summary"] == classifier_out["summary"]
    assert result["full_content"] == "the original content"


def test_token_caps_enforced():
    long_abs = "x " * 200  # ~400 tokens
    classifier_out = {"abstract": long_abs, "summary": "ok summary"}
    result = compute_layers("content", classifier_out)
    # Abstract truncated to ≤50 tokens (≈200 chars)
    assert len(result["abstract"]) <= 220


def test_fallback_heuristic_when_classifier_empty():
    classifier_out = {"abstract": None, "summary": None}
    content = "first sentence. second sentence. third sentence."
    result = compute_layers(content, classifier_out)
    assert result["abstract"]
    assert result["summary"]


def test_full_content_always_preserved():
    result = compute_layers("verbatim original", {"abstract": "ok", "summary": "ok"})
    assert result["full_content"] == "verbatim original"
