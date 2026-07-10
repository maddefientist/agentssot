import httpx

from .base import LLMProvider, LLMProviderError


class OpenAILLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 45):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(
            provider_name="openai",
            is_available=bool(api_key),
            unavailable_reason=None if api_key else "OPENAI_API_KEY is not configured",
        )

    def summarize(self, transcript: str) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "OpenAI LLM provider unavailable")

        system_prompt = (
            "You are summarizing an autonomous agent session. "
            "Produce a concise distillation with key decisions and concrete next steps."
        )

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI chat request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"OpenAI chat request failed with {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI chat response format was unexpected") from exc

    def distill(self, transcript: str, model: str | None = None) -> str:
        if not self.is_available:
            raise LLMProviderError(self.unavailable_reason or "OpenAI LLM provider unavailable")

        system_prompt = (
            "You are a learning-intake distiller. Extract atomic, reusable lessons from the transcript.\n\n"
            "Output ONLY valid JSON lines, one JSON object per line. No markdown, no commentary.\n"
            "Each object MUST contain:\n"
            "- claim: concise actionable lesson or factual takeaway\n"
            "- citation: source anchor supporting the claim; use a timestamp for audio/video when available, otherwise a quote or paragraph anchor\n"
            '- memory_type: "skill" | "decision" | "fact"; default to "skill" for best-practices\n'
            "- confidence: number from 0.0 to 1.0\n\n"
            "Rules:\n"
            "- Prefer specific best-practices over generic summaries.\n"
            "- Each lesson must stand alone and cite evidence from the source.\n"
            "- Do not invent claims not supported by the transcript.\n"
            "- If no useful lessons exist, output an empty line."
        )

        payload = {
            "model": model or self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI distill request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMProviderError(f"OpenAI distill request failed with {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI distill response format was unexpected") from exc
