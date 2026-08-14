from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llmfeeder.judge.lexical import LexicalJudge
from llmfeeder.models import Claim, ClaimResult, Evidence, SourceSpan, Verdict
from llmfeeder.retrieve import Candidate, CorpusIndex

__all__ = ["LLMJudge", "credential_available", "load_env_file", "resolve_credential"]

_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
_ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_VERSION = "2023-06-01"

_consent_shown = False

SYSTEM_PROMPT = """You are a strict citation checker. You are given one CLAIM and \
numbered PASSAGES taken from a source corpus. Decide whether the passages support the \
claim.

Rules:
- Judge only against the passages. Ignore anything you happen to know.
- "supported" means every part of the claim is stated in a passage.
- "partial" means the passage supports some of the claim but leaves part of it unstated.
- "contradicted" means a passage states something incompatible with the claim, such as \
a different figure, date or outcome.
- "unsupported" means no passage addresses the claim.
- The quote must be copied verbatim from the passage you chose. Never paraphrase it and \
never write a quote that is not present in the passage.

Reply with JSON only, no prose and no code fence:
{"verdict": "supported|partial|contradicted|unsupported", "passage": <number>, \
"quote": "<verbatim quote>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}"""


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _setting(name: str, env_file: str | Path = ".env") -> str | None:
    return os.environ.get(name) or load_env_file(env_file).get(name)


def resolve_credential(env_file: str | Path = ".env") -> tuple[str, str, str] | None:
    anthropic = _setting("ANTHROPIC_API_KEY", env_file)
    openai = _setting("OPENAI_API_KEY", env_file)
    base = _setting("LLMFEEDER_BASE_URL", env_file)

    if openai:
        return ("openai", openai, base or "https://api.openai.com/v1")
    if anthropic:
        return ("anthropic", anthropic, base or "https://api.anthropic.com")
    return None


def credential_available(env_file: str | Path = ".env") -> bool:
    return resolve_credential(env_file) is not None


def _consent_notice(provider: str, model: str) -> None:
    global _consent_shown
    if _consent_shown:
        return
    _consent_shown = True
    print(
        f"notice: the LLM judge is active. Claim text and the retrieved source "
        f"passages will be sent to {provider} ({model}) using your local key. "
        f"Use --judge lexical to keep everything on this machine.",
        file=sys.stderr,
    )


def _post(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return body


class LLMJudge:
    name = "llm"

    def __init__(
        self,
        *,
        fallback: LexicalJudge,
        model: str | None = None,
        supported: float = 0.75,
        partial: float = 0.45,
        env_file: str | Path = ".env",
        timeout: float = 60.0,
    ) -> None:
        self.fallback = fallback
        self.supported = supported
        self.partial = partial
        self.env_file = env_file
        self.timeout = timeout
        self._model_override = model or _setting("LLMFEEDER_MODEL", env_file)

    def bind(self, index: CorpusIndex) -> None:
        self.fallback.bind(index)

    def _complete(self, prompt: str) -> str:
        credential = resolve_credential(self.env_file)
        if credential is None:
            raise RuntimeError("no local LLM credential found")
        provider, key, base = credential

        if provider == "anthropic":
            model = self._model_override or _ANTHROPIC_DEFAULT_MODEL
            _consent_notice(provider, model)
            body = _post(
                f"{base.rstrip('/')}/v1/messages",
                {
                    "model": model,
                    "max_tokens": 512,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
                {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION},
                self.timeout,
            )
            blocks = body.get("content", [])
            return str(blocks[0].get("text", "")) if blocks else ""

        model = self._model_override or _OPENAI_DEFAULT_MODEL
        _consent_notice(provider, model)
        body = _post(
            f"{base.rstrip('/')}/chat/completions",
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            {"Authorization": f"Bearer {key}"},
            self.timeout,
        )
        choices = body.get("choices", [])
        return str(choices[0]["message"]["content"]) if choices else ""

    def judge(self, claim: Claim, candidates: list[Candidate]) -> ClaimResult:
        if not candidates:
            return self.fallback.judge(claim, candidates)

        prompt = self._prompt(claim, candidates)
        try:
            raw = self._complete(prompt)
            parsed = self._parse(raw)
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
            result: ClaimResult = self.fallback.judge(claim, candidates)
            return ClaimResult(
                claim=result.claim,
                verdict=result.verdict,
                score=result.score,
                evidence=result.evidence,
                rationale=f"LLM judge unavailable ({exc}); fell back to the lexical judge",
            )
        if parsed is None:
            result = self.fallback.judge(claim, candidates)
            return ClaimResult(
                claim=result.claim,
                verdict=result.verdict,
                score=result.score,
                evidence=result.evidence,
                rationale="LLM returned an unusable response; fell back to the lexical judge",
            )
        return self._assemble(claim, candidates, parsed)

    @staticmethod
    def _prompt(claim: Claim, candidates: list[Candidate]) -> str:
        passages = "\n\n".join(
            f"[{i + 1}] (source: {c.doc_id})\n{c.text}" for i, c in enumerate(candidates)
        )
        return f"CLAIM:\n{claim.text}\n\nPASSAGES:\n{passages}"

    @staticmethod
    def _parse(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "verdict" not in parsed:
            return None
        return parsed

    def _assemble(
        self, claim: Claim, candidates: list[Candidate], parsed: dict[str, Any]
    ) -> ClaimResult:
        try:
            verdict = Verdict(str(parsed.get("verdict", "")).strip().lower())
        except ValueError:
            verdict = Verdict.UNSUPPORTED

        try:
            index = int(parsed.get("passage", 1)) - 1
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(candidates) - 1))
        candidate = candidates[index]

        quote = str(parsed.get("quote", "") or "")
        span = self._locate(candidate, quote)
        confidence = self._confidence(parsed, verdict)
        rationale = str(parsed.get("rationale", "") or "").strip() or None

        evidence: tuple[Evidence, ...] = ()
        if verdict is not Verdict.UNSUPPORTED and span is not None:
            evidence = (
                Evidence(
                    span=span,
                    score=round(confidence, 4),
                    method="llm",
                    conflict=rationale if verdict is Verdict.CONTRADICTED else None,
                ),
            )
        elif verdict is not Verdict.UNSUPPORTED:
            rationale = (
                f"{rationale + '; ' if rationale else ''}"
                "the model's quote was not found verbatim in the passage, "
                "so no citation was recorded"
            )
            verdict = Verdict.PARTIAL if verdict is Verdict.SUPPORTED else verdict

        return ClaimResult(
            claim=claim,
            verdict=verdict,
            score=round(confidence, 4),
            evidence=evidence,
            rationale=rationale,
        )

    def _confidence(self, parsed: dict[str, Any], verdict: Verdict) -> float:
        try:
            raw = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            raw = 0.0
        raw = min(max(raw, 0.0), 1.0)
        if verdict is Verdict.SUPPORTED:
            return max(raw, self.supported)
        if verdict is Verdict.PARTIAL:
            return min(max(raw, self.partial), self.supported - 0.01)
        if verdict is Verdict.CONTRADICTED:
            return min(raw, self.partial - 0.01)
        return min(raw, 0.2)

    @staticmethod
    def _locate(candidate: Candidate, quote: str) -> SourceSpan | None:
        needle = quote.strip()
        if not needle:
            return None
        haystack = candidate.text
        offset = haystack.find(needle)
        if offset == -1:
            offset = haystack.lower().find(needle.lower())
        if offset == -1:
            collapsed = " ".join(needle.split())
            offset = " ".join(haystack.split()).find(collapsed)
            if offset == -1:
                return None
            offset = haystack.find(collapsed.split(" ", 1)[0])
            if offset == -1:
                return None
            needle = haystack[offset : offset + len(collapsed)]

        start = candidate.start + offset
        end = start + len(needle)
        document = candidate.document
        return SourceSpan(
            doc_id=document.doc_id,
            start=start,
            end=end,
            text=document.text[start:end],
            page=document.page_at(start),
        )
