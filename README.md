# GARDEN

GARDEN is an LLM-based SQL query rewriting system that combines three forms
of context: data-distribution evidence (Gate 1), Rule Dependency Graph (RDG)
guidance (Gate 2), and History-Based Decision Correction (HBDC).

This repository is a clean research snapshot. It contains the latest method
implementation, the knowledge used by the planner, benchmark query manifests,
and selected reproduction code. Generated results, run logs, model caches,
credentials, database files, and paper drafts are intentionally excluded.

## Repository layout

- `my_rewriter/`: GARDEN's online rewriting pipeline.
- `my_rewriter/candidate_selection/`: cost-only and HBDC selection policies.
- `my_rewriter/rdg_refinement/`: offline RDG transition collection/building.
- `CalciteRewrite/`: Java/Calcite rule application code.
- `knowledge/`: frozen RDG, HBDC, and distribution knowledge.
- `rag/`: RAG implementation and lightweight corpora.
- `workloads/`: schemas, SQL queries, and frozen evaluation manifests.
- `docs/ARTIFACT_MANIFEST.md`: included/excluded artifact inventory.

## Environment

The tested stack uses Python 3.10, Java 17, and PostgreSQL 14. Create the
environment with either:

```bash
conda env create -f environment.yml
conda activate rbot310
```

or install the Python dependencies into an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and provide credentials through environment
variables. Never commit `.env`.

## RAG knowledge asset

The Chroma index and its source corpus are distributed as a GitHub Release
asset because they are generated binary data. After the first release is
published, run:

```bash
GARDEN_RAG_ASSET_URL='<release-asset-url>' scripts/download_rag_knowledge.sh
```

The expected SHA-256 is recorded in the downloader. The archive extracts to
`rag/chroma_db/` and adds `rag/stackoverflow-rewrite-query-optimization.jsonl`.

## Smoke test

The offline smoke test does not call an LLM or database:

```bash
scripts/smoke_test.sh
```

An end-to-end run additionally requires PostgreSQL databases, model API
credentials, the RAG asset, and the Calcite Java artifacts described in
`docs/ARTIFACT_MANIFEST.md`. Set `GARDEN_ROOT` only when launching from a
nonstandard layout; otherwise the runner resolves the repository root itself.

## License

See `LICENSE`.
