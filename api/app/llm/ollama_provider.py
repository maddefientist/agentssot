import logging

import httpx

from .base import LLMProvider, LLMProviderError

logger = logging.getLogger("agentssot.llm")


class OllamaLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 45):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="ollama",
            is_available=bool(base_url and model),
            unavailable_reason=None if (base_url and model) else "OLLAMA_BASE_URL or OLLAMA_CHAT_MODEL missing",
        )

    def summarize(self, transcript: str) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "Ollama LLM provider unavailable")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Summarize the agent session into key decisions and next steps. "
                        "Keep it concise and actionable."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
        }

        url = f"{self.base_url}/api/chat"

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
        except Exception as exc:
            raise LLMProviderError(f"Ollama chat request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"Ollama chat request failed with {response.status_code}: {response.text[:400]}")

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama chat response format was unexpected")
        return content.strip()

    def synthesize_concepts(
        self,
        facts: str,
        existing_concepts: str,
        model_override: str | None = None,
        fallback_model: str | None = None,
    ) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "Ollama LLM provider unavailable")

        model = model_override or self.model

        system_prompt = (
            "You are a knowledge synthesis engine. Review the provided facts and identify conceptual patterns.\n\n"
            "For each concept you identify, output a JSON object on its own line with these fields:\n"
            '- type: "mental_model" | "relationship" | "principle"\n'
            '- scope: "global" | "project" | "device"\n'
            "- scope_ref: project slug or device name if scoped, null if global\n"
            "- title: concise label (under 80 chars)\n"
            "- content: full description (2-5 sentences)\n"
            "- confidence: 0.0-1.0 based on evidence strength\n"
            "- matches_existing_id: UUID of existing concept if this reinforces/contradicts one, null otherwise\n"
            "- is_contradiction: true if this contradicts the matched existing concept, false if reinforcement\n\n"
            "Output ONLY valid JSON lines (one JSON object per line, no markdown, no commentary).\n"
            "If no meaningful concepts can be extracted, output an empty line.\n\n"
            "Rules:\n"
            "- Only extract concepts with clear evidence from multiple facts\n"
            "- Prefer specific, actionable knowledge over vague generalizations\n"
            "- Relationships should name specific entities (projects, hosts, tools)\n"
            "- Principles should be testable and falsifiable\n"
            "- Mental models should describe observable patterns"
        )

        user_content = f"=== NEW FACTS ===\n{facts}"
        if existing_concepts.strip():
            user_content += f"\n\n=== EXISTING CONCEPTS ===\n{existing_concepts}"

        payload = {
            "model": model,
            "stream": False,
            "options": {"num_ctx": 8192},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        url = f"{self.base_url}/api/chat"

        try:
            response = httpx.post(url, json=payload, timeout=120)
            if response.status_code >= 400:
                raise LLMProviderError(f"Ollama synthesis failed with {response.status_code}: {response.text[:400]}")
        except Exception as exc:
            if fallback_model and model != fallback_model:
                logger.warning("synthesis model %s failed, falling back to %s: %s", model, fallback_model, exc)
                payload["model"] = fallback_model
                try:
                    response = httpx.post(url, json=payload, timeout=120)
                    if response.status_code >= 400:
                        raise LLMProviderError(f"Fallback model also failed: {response.status_code}")
                except Exception as fallback_exc:
                    raise LLMProviderError(f"Both {model} and {fallback_model} failed: {fallback_exc}") from fallback_exc
            else:
                raise LLMProviderError(f"Ollama synthesis request failed: {exc}") from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMProviderError("Ollama synthesis response format was unexpected")
        return content.strip()
