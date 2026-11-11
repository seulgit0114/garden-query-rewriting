# Artifact manifest

## Included

- Latest Gate 1 distribution analysis and prompt evidence construction.
- Gate 2 online RDG closure, ordering, and post-transform rematching.
- Offline RDG construction code and the four frozen DSB transition shards.
- Final global RDG (`knowledge/rdg/global_rdg_s2_rho04.json`).
- HBDC training history, split, Calcite delta, and sequence override.
- Distribution pools used for TPC-H, DSB, Calcite Uniform/Zipf, and stress data.
- TPC-H, DSB, Calcite, and distribution-stress SQL/manifests.
- SQL equivalence verification and selected parameter-study code.

## Excluded

- Experiment outputs, latency logs, figures, paper drafts, scheduler state,
  database clusters, caches, and local machine profiles.
- API keys, `.env`, access tokens, and account-specific Git configuration.
- Third-party SQLSolver and Calcite JAR binaries. Obtain them from their
  upstream projects and keep them untracked.
- Chroma binary index and the large RAG source corpus. These are packaged as
  the GitHub Release asset `garden-rag-knowledge-v1.tar.gz`.

## RAG release asset

- File: `garden-rag-knowledge-v1.tar.gz`
- SHA-256: `e15deb0d85645943d1b74d8e92d10f1df51b63247c46858b30757d47da7ec127`
The release maintainer uploads this archive separately; it is not tracked in
Git history.
