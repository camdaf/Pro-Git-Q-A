from dotenv import load_dotenv
load_dotenv()

from src.pipeline.rag import build_rag_pipeline
from src.pipeline.cache import ExactCache, SemanticCache
from src.pipeline.routing import classify_complexity

pipeline = build_rag_pipeline()
exact = ExactCache()
semantic = SemanticCache(threshold=0.93)

queries = [
    'O que e rebase?',
    'O que e rebase?',
    'Me explica rebase',
    'Como funciona git bisect?',
    'Como funciona git bisect?',
    'Explica git bisect',
    'O que e refspec?',
    'O que e refspec?',
    'Me fala sobre refspec',
    'Como fazer merge?',
]

llm_calls = 0
cache_exact_hits = 0
cache_semantic_hits = 0
baseline_calls = len(queries)

print("BENCHMARK — 10 queries com paráfrases")

for q in queries:
    hit = exact.get(q)
    if hit:
        cache_exact_hits += 1
        print(f"[EXACT HIT]    {q}")
        continue

    hit = semantic.get(q)
    if hit:
        cache_semantic_hits += 1
        print(f"[SEMANTIC HIT] {q}")
        continue

    decision = classify_complexity(q)
    print(f"[LLM {decision.complexity.upper():7}] {q} → {decision.model}")
    result = pipeline.answer(q)
    exact.put(q, result['answer'])
    semantic.put(q, result['answer'])
    llm_calls += 1

total_hits = cache_exact_hits + cache_semantic_hits
reducao = (total_hits / baseline_calls) * 100
proporcao_llm = llm_calls / baseline_calls
llm_por_1k = proporcao_llm * 1000
quota_pct = (llm_por_1k / 1500) * 100

print()
print("=" * 50)
print("RESULTADO")
print("=" * 50)
print(f"Total queries:         {baseline_calls}")
print(f"Exact cache hits:      {cache_exact_hits}")
print(f"Semantic cache hits:   {cache_semantic_hits}")
print(f"LLM calls:             {llm_calls}")
print(f"Reducao de custo:      {reducao:.1f}%")
print(f"Meta (>=50%):          {'ATINGIDA ✓' if reducao >= 50 else 'NAO ATINGIDA ✗'}")
print()
print(f"LLM calls por 1k queries:       {llm_por_1k:.0f}")
print(f"Quota Gemini Free por 1k queries: {quota_pct:.1f}% da cota diaria")
