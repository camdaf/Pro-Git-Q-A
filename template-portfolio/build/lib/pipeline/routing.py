from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    complex_keywords = [
        "explique", "compare", "analise", "projete",
        "descreva", "diferença", "diferenca", "porque", "por que",
    ]
    q = query.lower().strip()

    if any(w in q for w in complex_keywords):
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="palavra de alta complexidade detectada",
        )
    if len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="query curta e direta",
        )
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="default cheap-first",
    )


def make_client() -> OpenAI:
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()