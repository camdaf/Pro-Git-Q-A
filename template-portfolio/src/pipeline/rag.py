from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader


def _make_client() -> tuple[OpenAI, str]:
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


class RAGPipeline:

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def ingest_and_index(self) -> int:
        docs = []
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            reader = PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_idx + 1,
                    })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for doc in docs:
            for i, chunk in enumerate(splitter.split_text(doc["text"])):
                chunks.append({
                    "id": f"{doc['source']}-p{doc['page']}-c{i}",
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_idx": i,
                })

        BATCH = 50
        for start in range(0, len(chunks), BATCH):
            lote = chunks[start : start + BATCH]
            self.collection.add(
                ids=[c["id"] for c in lote],
                documents=[c["text"] for c in lote],
                metadatas=[
                    {"source": c["source"], "page": c["page"], "chunk_idx": c["chunk_idx"]}
                    for c in lote
                ],
            )
            print(f"indexados {min(start + BATCH, len(chunks))}/{len(chunks)} chunks")
            time.sleep(1)

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        result = self.collection.query(query_texts=[query], n_results=k)
        return [
            {
                "text": result["documents"][0][i],
                "source": result["metadatas"][0][i]["source"],
                "page": result["metadatas"][0][i]["page"],
                "distance": result["distances"][0][i],
            }
            for i in range(len(result["documents"][0]))
        ]

    def answer(self, question: str, k: int = 5) -> dict:
        hits = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join(
            f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": [(h["source"], h["page"]) for h in hits],
        }


PROMPT_TEMPLATE = """Voce e um assistente tecnico. Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
