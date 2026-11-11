#!/usr/bin/env python3
"""B3 기회 테이블 생성 — [[A, B, hit, opp], ...]

opp(A,B) = A 를 적용한 상태들 중 "그때 B 가 비활성이었던" 횟수.
hit(A,B) = 그 기회들 중 B 가 실제로 활성이 된 횟수.

B 가 이미 활성이던 경우는 기회가 아니다 — A 가 무엇을 하든 B 는 원래 쓸 수
있었으므로 "A 가 B 를 여는가"에 대한 정보가 없다.

B5 표기 원칙: 아래 세 층위를 분리해 출력한다.
    raw observation       (query, A, B) 삼중항 건수
    unique directed edge  중복 제거한 (A, B) 쌍
    threshold 통과 edge   family 지지도 >= min_support
"""
from __future__ import annotations
import argparse, glob, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from srdg import manifest as mf   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trans-glob', default=str(HERE / 'data' / 'dsb_kfold' / 'transitions_*.jsonl'))
    ap.add_argument('--out', default=str(HERE / 'data' / 'opportunities.json'))
    ap.add_argument('--min-support', type=int, default=2)
    a = ap.parse_args(argv)

    rows = []
    for p in sorted(glob.glob(a.trans_glob)):
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get('newly_enabled') is not None]
    if not rows:
        print('전이 데이터 없음', file=sys.stderr); return 1

    universe = set()
    for r in rows:
        universe |= (set(r['enabled_before']) | set(r['newly_enabled'])
                     | set(r.get('enabled_after') or []))

    hit, opp = Counter(), Counter()
    fam = defaultdict(set)
    raw_obs = 0
    for r in rows:
        A = r['attempted_rule']
        before, new = set(r['enabled_before']), set(r['newly_enabled'])
        raw_obs += len(new)
        for B in universe:
            if B == A or B in before:
                continue                     # 이미 활성이면 기회가 아님
            opp[(A, B)] += 1
            if B in new:
                hit[(A, B)] += 1
                fam[(A, B)].add((r['workload'], r['query_family']))

    table = [[a_, b_, hit[(a_, b_)], o] for (a_, b_), o in sorted(opp.items())]
    mf.atomic_write_json(Path(a.out), table)

    uniq = {k for k, v in hit.items() if v > 0}
    passed = {k for k in uniq if len(fam[k]) >= a.min_support}
    print('B5 수치 층위 분리')
    print(f'  raw observation      (query, A, B) 삼중항   {raw_obs}')
    print(f'  unique directed edge (A, B) 중복제거        {len(uniq)}')
    print(f'  threshold 통과 edge  family 지지도 >= {a.min_support}    {len(passed)}')
    print(f'  기회 테이블 항목      (A, B) with opp>0      {len(table)}')
    print(f'  코퍼스: {len({r["query_id"] for r in rows})} 쿼리 / '
          f'{len({(r["workload"], r["query_family"]) for r in rows})} family')
    print(f'  기회 분포: 중앙 {statistics.median([o for _, _, _, o in table]):.0f} '
          f'최대 {max(o for _, _, _, o in table)}')
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
