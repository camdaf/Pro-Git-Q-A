from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader


def lookup_chapter(chapter: int) -> str:
    corpus_dir = Path(os.environ.get("CORPUS_DIR", "data/corpus"))
    hits: list[str] = []

    markers = [
        f"Chapter {chapter}",
        f"chapter {chapter}",
        f"CHAPTER {chapter}",
        f"Capítulo {chapter}",
        f"Capitulo {chapter}",
    ]

    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if any(m in text for m in markers):
                preview = text.strip()[:600]
                hits.append(f"[{pdf_path.name} — página {page_idx + 1}]\n{preview}")
                if len(hits) >= 3:
                    break
        if hits:
            break

    if not hits:
        return (
            f"Capítulo {chapter} não encontrado. "
            "Verifique número ou se livro foi indexado."
        )
    return "\n\n---\n\n".join(hits)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_chapter",
            "description": (
                "Retorna conteúdo introdutório do capítulo N do livro técnico indexado. "
                "Use quando usuário perguntar sobre capítulo específico, quiser navegar "
                "para seção do livro, ou mencionar 'capítulo', 'chapter' ou número de seção."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {
                        "type": "integer",
                        "description": "Número do capítulo (ex: 3 para Capítulo 3).",
                    }
                },
                "required": ["chapter"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "lookup_chapter": lookup_chapter,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' não registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"