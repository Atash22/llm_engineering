# Agentic RAG: Simple RAG vs. Self-Correcting Retrieval

This document summarizes an extension to the Week 5 RAG evaluation exercise from the "LLM Engineering" course. It builds an Agentic RAG pipeline using LangGraph and benchmarks it against an already-optimized Simple RAG pipeline.

## Background

The Week 5 exercise builds a Retrieval-Augmented Generation (RAG) system for a fictional company, Insurellm, and evaluates it using:

- **Retrieval metrics**: Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (nDCG), keyword coverage
- **Answer quality metrics** (LLM-as-a-judge): Accuracy, Completeness, Relevance

The course's own benchmark: **0.9116 MRR, 0.9025 nDCG, 96.0% keyword coverage** on retrieval, and **4.62 / 4.35 / 4.84** (Accuracy / Completeness / Relevance) on answer quality.

## Phase 1 — Improving Simple RAG

Before building the agentic version, the baseline Simple RAG pipeline (`implementation/answer.py`) was improved with two changes:

1. **Cross-encoder re-ranking** — retrieval over-fetches 30 candidate chunks from vector search, then re-ranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2` to surface the most relevant chunk at rank 1, directly improving MRR/nDCG.
2. **Stronger, completeness-focused prompt** — the system prompt now explicitly requires exact preservation of names/titles/numbers from context and forbids irrelevant extra information, fixing a real completeness gap found during evaluation (a dropped last name in one test case).

**Result:** 5.00 / 5.00 / 5.00 on Accuracy / Completeness / Relevance on the tested question set — exceeding the course benchmark.

## Phase 2 — Building Agentic RAG

`implementation/agentic_answer.py` implements a self-correcting retrieval loop using **LangGraph**, rather than a fixed retrieve-then-generate pipeline.

### Architecture

- **retrieve**: fetches context using the same re-ranked retrieval as Simple RAG.
- **grade_documents**: an LLM call judges whether the retrieved context is sufficient to answer the question.
- **rewrite_query**: if not sufficient, the LLM rewrites the question into a better search query and retries retrieval.
- **generate**: produces the final answer once context is judged adequate (or after 3 attempts, as a safety cap).

This is the core "agentic" behavior — the system decides for itself whether to retrieve again, rather than following a single fixed pass.

## Phase 3 — Head-to-Head Comparison

A Gradio dashboard (`compare_app.py`, backed by `evaluation/compare.py`) runs both pipelines on the same test set and plots Accuracy, Completeness, and Relevance side by side.

### Results (10-question sample)

| Metric | Simple RAG | Agentic RAG |
|---|---|---|
| Accuracy | 4.90 | 4.90 |
| Completeness | 4.70 | 4.60 |
| Relevance | 5.00 | 5.00 |

## Key Finding

On straightforward, single-hop questions, **Simple RAG (with re-ranking + a strong prompt) performs on par with or slightly better than Agentic RAG**. This makes sense: the agentic self-correction loop adds the most value on ambiguous or multi-hop questions where a single retrieval pass genuinely misses relevant context — not on questions the base pipeline already retrieves well.

This is a more honest and instructive result than assuming "agentic is always better" — it shows that architecture complexity should match the actual failure mode you're trying to fix.

## Files

| File | Purpose |
|---|---|
| `implementation/answer.py` | Simple RAG pipeline with re-ranking + strong prompt |
| `implementation/agentic_answer.py` | Agentic RAG pipeline using LangGraph |
| `evaluation/eval.py` | Retrieval + answer evaluation, including `judge_answer` for scoring pre-generated answers |
| `evaluation/compare.py` | Runs both pipelines on the same test set for comparison |
| `compare_app.py` | Gradio dashboard visualizing the comparison |
| `app.py` | Main chat interface (Simple RAG) |

## Next Steps

- Test with harder multi-hop questions (e.g., ones requiring synthesis across multiple documents) where Agentic RAG is expected to outperform Simple RAG.
- Add cost/latency tracking to the comparison dashboard, since Agentic RAG makes more LLM calls per question.