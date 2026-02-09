import re


def _split_hard(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars].strip())
        start += max_chars
    return [c for c in chunks if c]


def chunk_text_semantic(content: str, max_chars: int = 800) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return _split_hard(text, max_chars)

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            if not sentences:
                sentences = _split_hard(paragraph, max_chars)
            for sentence in sentences:
                if len(sentence) > max_chars:
                    flush()
                    chunks.extend(_split_hard(sentence, max_chars))
                    continue

                candidate = f"{current}\n{sentence}".strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    flush()
                    current = sentence
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush()
            current = paragraph

    flush()
    return chunks
