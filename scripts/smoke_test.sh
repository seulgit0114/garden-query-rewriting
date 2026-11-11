#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m compileall -q "$ROOT/my_rewriter" "$ROOT/rag"
python - <<'PY'
import json
from pathlib import Path

root = Path(__import__('os').environ['PYTHONPATH'].split(':')[0])
rdg = json.loads((root / 'knowledge/rdg/global_rdg_s2_rho04.json').read_text())
assert rdg, 'RDG knowledge is empty'
for path in (
    root / 'knowledge/hbdc/mvm_v2/candidates.jsonl',
    root / 'knowledge/hbdc/calcite_log_mined_v2/delta_candidates.jsonl',
    root / 'knowledge/distribution/calcite10_pool.json',
    root / 'knowledge/distribution/calcite10zipf_pool.json',
):
    assert path.is_file() and path.stat().st_size > 0, path
print('offline smoke test: OK')
PY
