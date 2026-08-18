from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_client = None


def _get_client():
    """Lazy-init OpenAI client, dùng lại cho mọi technique (1 lần cho cả process)."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()
    return _client


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False,
          label: str = "enrichment") -> str | None:
    """1 call gpt-4o-mini. Không có API key hoặc call lỗi → None để caller fallback.

    json_mode dùng response_format json_object: model bị buộc trả JSON hợp lệ,
    không kèm ```json fence — json.loads() khỏi vỡ.
    """
    if not OPENAI_API_KEY:
        return None

    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,                             # enrichment cần ổn định, không sáng tạo
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI {label} failed: {type(e).__name__}: {e}")
        return None


def _extractive_summary(text: str, n_sentences: int = 2) -> str:
    """Fallback không cần API: lấy n câu đầu."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:n_sentences]).rstrip(".") + "." if sentences else text


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return text

    summary = _chat(
        "Tóm tắt đoạn văn sau bằng tiếng Việt, tối đa 2 câu và PHẢI ngắn hơn bản gốc. "
        "Chỉ trả về phần tóm tắt.",
        text, max_tokens=150, label="summarize",
    )
    # Summary dài hơn bản gốc thì không còn là summary → dùng extractive
    if not summary or len(summary) >= len(text):
        return _extractive_summary(text)
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip():
        return []

    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt mà đoạn văn có thể "
        "trả lời. Mỗi câu hỏi trên 1 dòng, không đánh số.",
        text, max_tokens=200, label="HyQA",
    )
    if raw:
        # LLM vẫn hay tự đánh số dù đã yêu cầu không → strip tiền tố "1. ", "- "
        questions = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
        if questions:
            return questions[:n_questions]

    # Extractive fallback: biến câu trần thuật thành câu hỏi thô
    import re

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return text

    context = _chat(
        "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu và "
        "nói về chủ đề gì. Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80, label="contextual",
    )
    if context:
        return f"{context}\n\n{text}"

    # Fallback: ít nhất gắn tên tài liệu để chunk không mất ngữ cảnh nguồn
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


_FALLBACK_METADATA = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if not text.strip():
        return dict(_FALLBACK_METADATA)

    raw = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", '
        '"entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150, json_mode=True, label="metadata",
    )
    if raw:
        import json as _json

        try:
            data = _json.loads(raw)
            if isinstance(data, dict):
                return data
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  OpenAI metadata JSON invalid: {e}")

    return dict(_FALLBACK_METADATA)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    raw = _chat(
        """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        max_tokens=400, json_mode=True, label="enrichment",
    )
    if raw:
        import json as _json

        try:
            data = _json.loads(raw)
            if isinstance(data, dict):
                return data
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  Enrichment JSON invalid: {e}")

    # Không có API key hoặc call lỗi → enrichment offline, vẫn giữ được ngữ cảnh nguồn
    return {
        "summary": _extractive_summary(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ {source}." if source else "",
        "metadata": dict(_FALLBACK_METADATA),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):             # console Windows mặc định cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"API key: {'có' if OPENAI_API_KEY else 'KHÔNG (chạy fallback)'}")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}\n")

    # Combined mode: 1 call/chunk thay vì 4 calls như trên
    combined = enrich_chunks([{"text": sample, "metadata": {"source": "so_tay_nhan_vien.md"}}])
    c = combined[0]
    print(f"Combined ({c.method}):")
    print(f"  summary:   {c.summary}")
    print(f"  questions: {c.hypothesis_questions}")
    print(f"  metadata:  {c.auto_metadata}")
    print(f"  enriched:  {c.enriched_text[:120]}...")
