import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from gateway.config import (
    ENABLE_SMART_ROUTING,
    OPUS_ROUTING_THRESHOLD,
    DEFAULT_MODEL,
    OPUS_MODEL,
    DATABASE_URL,
)
from gateway.logging_setup import log


@dataclass
class RoutingDecision:
    model: str
    score: float
    reasons: List[str]
    is_opus: bool


OPUS_KEYWORDS = [
    ("architect", 0.15),
    ("architecture", 0.15),
    ("design", 0.10),
    ("security", 0.12),
    ("migration", 0.12),
    ("refactor", 0.10),
    ("production", 0.10),
    ("incident", 0.12),
    ("postmortem", 0.12),
    ("kubernetes", 0.10),
    ("terraform", 0.10),
    ("infrastructure", 0.10),
    ("scalability", 0.10),
    ("performance", 0.08),
    ("optimization", 0.08),
    ("database schema", 0.12),
    ("api design", 0.10),
    ("system design", 0.12),
    ("distributed", 0.10),
    ("microservice", 0.10),
    ("breaking change", 0.12),
    ("backwards compat", 0.10),
    ("deprecat", 0.08),
]

SONNET_KEYWORDS = [
    ("simple", -0.10),
    ("quick", -0.08),
    ("small change", -0.10),
    ("typo", -0.12),
    ("formatting", -0.10),
    ("comment", -0.08),
    ("rename", -0.08),
]


def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
    return " ".join(texts)


def compute_keyword_score(text: str) -> Tuple[float, List[str]]:
    text_lower = text.lower()
    score = 0.0
    reasons = []

    for keyword, weight in OPUS_KEYWORDS:
        if keyword in text_lower:
            score += weight
            reasons.append(f"+{keyword}:{weight:.2f}")

    for keyword, weight in SONNET_KEYWORDS:
        if keyword in text_lower:
            score += weight
            reasons.append(f"{keyword}:{weight:.2f}")

    return score, reasons


def compute_complexity_score(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    text: str,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    if len(text) > 5000:
        addition = 0.15
        score += addition
        reasons.append(f"+long_context:{addition:.2f}")
    elif len(text) > 2000:
        addition = 0.08
        score += addition
        reasons.append(f"+medium_context:{addition:.2f}")

    if len(tools) > 8:
        addition = 0.15
        score += addition
        reasons.append(f"+many_tools:{addition:.2f}")
    elif len(tools) > 4:
        addition = 0.08
        score += addition
        reasons.append(f"+some_tools:{addition:.2f}")

    tool_uses = sum(1 for m in messages if m.get("tool_calls"))
    if tool_uses > 5:
        addition = 0.10
        score += addition
        reasons.append(f"+heavy_tool_use:{addition:.2f}")

    file_refs = len(re.findall(r"(?:\.py|\.js|\.ts|\.tsx|\.go|\.rs|\.java)\b", text))
    if file_refs > 10:
        addition = 0.12
        score += addition
        reasons.append(f"+many_files:{addition:.2f}")
    elif file_refs > 5:
        addition = 0.06
        score += addition
        reasons.append(f"+some_files:{addition:.2f}")

    return score, reasons


async def get_historical_score(project_id: Optional[int], model: str) -> float:
    if not DATABASE_URL or not project_id:
        return 0.0

    try:
        from gateway.db import get_session, ModelSuccessRate
        from sqlalchemy import select

        async with get_session() as session:
            result = await session.execute(
                select(ModelSuccessRate).where(
                    ModelSuccessRate.project_id == project_id,
                    ModelSuccessRate.model == model,
                )
            )
            rate = result.scalar_one_or_none()

            if rate:
                total = rate.success_count + rate.failure_count
                if total > 10:
                    success_ratio = rate.success_count / total
                    return (success_ratio - 0.5) * 0.2

    except Exception as e:
        log.warning("Failed to get historical score: %r", e)

    return 0.0


async def record_outcome(
    project_id: Optional[int],
    model: str,
    success: bool,
):
    if not DATABASE_URL or not project_id:
        return

    try:
        from gateway.db import get_session, ModelSuccessRate
        from sqlalchemy import select
        from datetime import datetime

        async with get_session() as session:
            result = await session.execute(
                select(ModelSuccessRate).where(
                    ModelSuccessRate.project_id == project_id,
                    ModelSuccessRate.model == model,
                )
            )
            rate = result.scalar_one_or_none()

            if rate:
                if success:
                    rate.success_count += 1
                else:
                    rate.failure_count += 1
                rate.last_updated = datetime.utcnow()
            else:
                rate = ModelSuccessRate(
                    project_id=project_id,
                    model=model,
                    success_count=1 if success else 0,
                    failure_count=0 if success else 1,
                )
                session.add(rate)

    except Exception as e:
        log.warning("Failed to record outcome: %r", e)


async def should_use_opus(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
) -> RoutingDecision:
    if explicit_model:
        model_lower = explicit_model.lower()
        if "opus" in model_lower or "high" in model_lower or "thinking" in model_lower:
            return RoutingDecision(
                model=OPUS_MODEL,
                score=1.0,
                reasons=["explicit_opus_request"],
                is_opus=True,
            )
        if "sonnet" in model_lower or "fast" in model_lower:
            return RoutingDecision(
                model=DEFAULT_MODEL,
                score=0.0,
                reasons=["explicit_sonnet_request"],
                is_opus=False,
            )

    if not ENABLE_SMART_ROUTING:
        return RoutingDecision(
            model=DEFAULT_MODEL,
            score=0.0,
            reasons=["smart_routing_disabled"],
            is_opus=False,
        )

    text = extract_text_from_messages(messages)

    keyword_score, keyword_reasons = compute_keyword_score(text)
    complexity_score, complexity_reasons = compute_complexity_score(messages, tools, text)

    historical_score = await get_historical_score(project_id, OPUS_MODEL)

    total_score = keyword_score + complexity_score + historical_score
    all_reasons = keyword_reasons + complexity_reasons

    if historical_score != 0:
        all_reasons.append(f"historical:{historical_score:.2f}")

    is_opus = total_score >= OPUS_ROUTING_THRESHOLD
    model = OPUS_MODEL if is_opus else DEFAULT_MODEL

    if is_opus:
        log.info(
            "Smart routing -> OPUS (score=%.2f, threshold=%.2f, reasons=%s)",
            total_score,
            OPUS_ROUTING_THRESHOLD,
            all_reasons[:5],
        )

    return RoutingDecision(
        model=model,
        score=total_score,
        reasons=all_reasons,
        is_opus=is_opus,
    )


async def route_model(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    project_id: Optional[int] = None,
    explicit_model: Optional[str] = None,
) -> str:
    decision = await should_use_opus(messages, tools, project_id, explicit_model)
    return decision.model
