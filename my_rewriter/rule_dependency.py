"""
rule_dependency.py

Module 2 (Rule Dependency Graph). RDG 게이팅, 위상정렬 기반 "권장 순서 골격"(soft 제약,
하드 제약 아님) 생성, 그리고 LLM이 그 골격을 실제로 따랐는지/벗어났는지를 사후에 코드로
객관적으로 판정하는 유틸을 모은다.

baseline(my_utils.py 등)은 읽기만 하고 절대 수정하지 않는다.

흐름 (distaware_rewrite.py에서 호출):
  1) load_rdg(): knowledge-base/rule_dependency_graph.json을 로드.
     아직 RDG 추출(1단계, explain_rule/*.jsonl -> 정형 태그 -> 엣지)이 안 끝나서 파일이
     없어도 빈 그래프로 안전하게 fallback한다 — 이 모듈 전체가 RDG 데이터 유무와 무관하게
     항상 에러 없이 동작해야 한다.
  2) rdg_gate(): operator-group 겹침 여부로 RDG 적용 여부를 게이팅한다
     (실측 검증: 152개 실로그 기준 recall=100%, precision=62.0%, 통과율=84.9%).
  3) build_skeleton(): 후보 규칙들의 유도 부분그래프(induced subgraph)에 대해
     Kahn's algorithm으로 위상정렬 layer를 계산한다. 순환(=RDG 추출 오류 가능성)이
     있으면 그 노드들은 강제 배치하지 않고 "unresolved"로 따로 분리한다.
  4) render_skeleton_for_prompt(): layer + 근거(reason)를 텍스트로 직렬화 —
     {rule_dependency_constraints} 슬롯에 주입될 내용. 이건 "하드 제약"이 아니라
     "강한 권장 골격"으로 프롬프트에 들어가며(RDG_GUIDANCE_SYS_NOTE 참고),
     LLM이 규칙 설명과 모순된다고 판단하면 벗어날 수 있다.
  5) check_skeleton_followed(): 최종 순서가 유도 부분그래프의 모든 선행 제약을 지켰는지
     코드로 객관적으로 판정한다 (LLM의 자기보고에 의존하지 않음 — "따랐다고 말했지만
     실제로는 안 따른" 경우를 거를 수 있음).
  6) closure_expand(): select_rules_dist가 고른 relevant_rules에서 출발해, RDG
     엣지를 타고 도달 가능한 규칙을 fixed-point(다중 hop)로 편입시킨다. 두 소스에서
     끌어온다 — (a) RAG의 match_calcite_rules가 "원본 쿼리" 트리에 이미 구조적으로
     매칭시켰지만 LLM이 걸러낸 규칙(candidate_pool), (b) match_calcite_rules는 원본
     트리에 대해 딱 1회만 매칭하므로 애초에 candidate_pool에 들어올 수조차 없었던,
     "선행 규칙 적용 후에만 매칭될" 규칙(전체 70개 카탈로그, load_generic_rule_
     descriptions()). (b)가 RDG closure 확장의 본래 목적 — 그 빈틈을 메우는 것이다.
     이 확장이 틀려도 안전하다 — Calcite HepPlanner는 실제 RelNode 패턴이 안 맞는
     규칙은 그냥 적용을 건너뛰므로, 편입된 규칙이 진짜 적용 가능한지는 항상
     execute_rewrite()가 최종 검증한다.
  7) load_generic_rule_descriptions(): explain_rule/calcite_rewrite_rules_structured.jsonl
     (RDG 추출 1단계와 동일한 소스)에서 규칙별 범용 conditions/transformations 설명을
     로드한다 — closure_expand()가 (b) 경로로 편입시킨 규칙은 이 쿼리에 대한 맞춤
     설명이 없으므로, 이 범용 설명을 rewrite 필드 대용으로 채운다.
"""
import json
import logging
import os
import typing as t
from collections import defaultdict, deque

from my_rewriter.my_utils import get_rule_sets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# [2026-09-14] RDG_PATH 환경변수로 그래프 파일을 바꿔 끼울 수 있게 한다.
# 기본값은 기존과 동일하므로 변수를 설정하지 않으면 동작이 전혀 바뀌지 않는다.
# 그래프 품질 ablation(예: 필터 제거 전/후 그래프 비교)을 할 때 파일을 실제로
# 바꿔치기하면 같은 파일을 읽는 다른 실행까지 영향을 받으므로, 경로를 주입하는
# 쪽이 안전하다.
DEFAULT_RDG_PATH = os.environ.get(
    'RDG_PATH',
    os.path.join(PROJECT_ROOT, 'knowledge-base', 'rule_dependency_graph.json'))
DEFAULT_EXPLAIN_RULE_PATH = os.path.join(PROJECT_ROOT, 'explain_rule', 'calcite_rewrite_rules_structured.jsonl')

# (from_rule, to_rule, reason) 튜플로 엣지를 표현한다.
Edge = t.Tuple[str, str, str]

# [2026-09-18] B7 — 프롬프트 폭증 방지 상한.
# 렌더링 결과가 규칙 설명으로 덮이면 LLM 이 쿼리 자체를 보지 못한다.
MAX_REASONS_PER_RULE = int(os.environ.get('RDG_MAX_REASONS_PER_RULE', '2'))
MAX_REASON_CHARS = int(os.environ.get('RDG_MAX_REASON_CHARS', '160'))
# closure_expand 가 한 쿼리에 추가할 수 있는 규칙 수 상한. 0 이면 무제한.
MAX_CLOSURE_ADDITIONS = int(os.environ.get('RDG_MAX_CLOSURE_ADDITIONS', '12'))
# B3 — 두 방향의 (평활) 확률 차이가 이보다 작으면 순서를 강제하지 않는다.
TIE_EPS = float(os.environ.get('RDG_TIE_EPS', '0.05'))


def load_opportunities(path: t.Optional[str] = None) -> t.Optional[dict]:
    """B3 기회 테이블 {(A,B): 기회 횟수}. 없으면 None (원시 횟수 점수로 동작).

    RDG_OPPORTUNITIES 환경변수로 경로를 준다. 파일은 [[A, B, hit, opp], ...]
    형식의 JSON 이며, 전이 수집 로그에서 생성한다.
    """
    path = path or os.environ.get('RDG_OPPORTUNITIES')
    if not path or not os.path.exists(path):
        return None
    try:
        raw = json.load(open(path))
        return {(a, b): int(o) for a, b, _h, o in raw}
    except Exception:                                      # noqa: BLE001
        logging.warning(f'[rule_dependency] 기회 테이블 로드 실패: {path}')
        return None


def load_rdg(path: str = DEFAULT_RDG_PATH) -> dict:
    """{"nodes": [...], "edges": [{"from":..., "to":..., "reason":...}]} 형식의 RDG를 로드한다.
    파일이 없으면(아직 1단계 추출 전이어도) 빈 그래프로 안전하게 반환한다.
    """
    if not os.path.exists(path):
        logging.warning(f"[rule_dependency] RDG 파일을 찾을 수 없음: {path} — 빈 그래프로 진행 "
                         f"(RDG 추출 1단계가 아직 안 끝났다면 정상적인 상태)")
        return {'nodes': [], 'edges': []}
    with open(path, 'r') as f:
        graph = json.load(f)
    graph.setdefault('nodes', [])
    graph.setdefault('edges', [])
    return graph


def load_generic_rule_descriptions(path: str = DEFAULT_EXPLAIN_RULE_PATH) -> t.Dict[str, str]:
    """explain_rule/calcite_rewrite_rules_structured.jsonl(RDG 추출 1단계와 동일한 소스)에서
    규칙별 범용 conditions/transformations 설명을 로드한다.

    closure_expand()가 "원본 쿼리엔 매칭 안 됐지만 RDG가 추정한" 규칙을 편입시킬 때,
    그 규칙엔 이 쿼리에 대한 맞춤 설명(match_calcite_rules가 만들어주는 query-specific
    rewrite 텍스트)이 없으므로 이 범용 설명을 대신 채운다. 파일이 없으면 빈 dict로
    안전하게 fallback한다.
    """
    if not os.path.exists(path):
        logging.warning(f"[rule_dependency] explain_rule 명세 파일을 찾을 수 없음: {path}")
        return {}
    descriptions: t.Dict[str, str] = {}
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            entries = obj['rewrite_rules_structured']
            multi = len(entries) > 1
            parts = [
                (f"Case {i}: " if multi else "") + f"Conditions: {e['conditions']} | Transformations: {e['transformations']}"
                for i, e in enumerate(entries, 1)
            ]
            descriptions[obj['name']] = '\n'.join(parts)
    return descriptions


def rdg_gate(relevant_rules: t.List[t.Dict[str, str]]) -> bool:
    """후보 규칙들 중 같은 operator group(JOIN/FILTER/AGGREGATE 등)에 2개 이상 몰려
    있으면 활성화한다. distribution-aware 게이트(Uniformity Score)와는 완전히 독립이며,
    추가 LLM 호출이 없어서 length 기반 게이트보다 더 엄격하게 걸어도 비용 부담이 없다.
    (실측 검증: recall=100%, precision=62.0%, 통과율=84.9%, 152개 실로그 기준)
    """
    rule_names = [r['name'] for r in relevant_rules]
    groups = get_rule_sets(rule_names)
    return any(len(v) >= 2 for v in groups.values())


def _induced_edges(rule_names: t.List[str], graph: dict,
                   enabled_at_s0: t.Optional[t.Iterable[str]] = None) -> t.List[Edge]:
    """전체 RDG 엣지 중 양 끝점이 모두 후보 규칙 집합 안에 있는 것만 추출(유도 부분그래프).

    [2026-09-17] B1 — enabled_at_s0 가 주어지면 "target 이 이 쿼리에서 이미 활성"인
    엣지를 제외한다.

    엣지 A->B 는 "다른 쿼리들에서는 A 를 적용해야 비로소 B 가 매칭되더라"는 관찰이다.
    그런데 지금 이 쿼리에서 B 가 원본 상태부터 이미 매칭된다면, B 를 쓰기 위해 A 를
    먼저 적용할 이유가 없다. 그 엣지를 선행 제약으로 쓰는 것은 이 쿼리에 대해서는
    근거가 없다.

    영향은 크다. dsb 76개 쿼리에서 측정한 결과:
        유도 엣지            1076 -> 1032  (-4%)
        최대 SCC 크기          22 -> 16
        진입차수 0 인 규칙 수     0 -> 6
        Kahn 이 첫 layer 를 만드는 쿼리   4/76 -> 76/76
    되살아난 시작점은 정확히 Enabled(S0) 와 일치했다. 원본에서 바로 쓸 수 있는 규칙이
    본래의 시작점인데, 근거 없는 화살표가 거기 꽂혀 "선행 조건이 있는 규칙"처럼
    보이게 만들고 있었다.

    enabled_at_s0=None 이면 기존 동작 그대로다.
    """
    name_set = set(rule_names)
    already = set(enabled_at_s0 or ())
    return [(e['from'], e['to'], e.get('reason', ''))
            for e in graph.get('edges', [])
            if e['from'] in name_set and e['to'] in name_set
            and e['to'] not in already]


def build_skeleton(rule_names: t.List[str], graph: dict,
                    enabled_at_s0: t.Optional[t.Iterable[str]] = None
                    ) -> t.Tuple[t.List[t.List[str]], t.List[Edge], t.List[str]]:
    """Kahn's algorithm으로 위상정렬 layer를 계산한다.

    반환: (layers, induced_edges, unresolved_rule_names)
      - layers: [[Layer1 규칙들], [Layer2 규칙들], ...] — 같은 layer 안은 순서 무관(독립)
      - induced_edges: 이번 쿼리 후보군에 한정된 엣지 목록 (사후 검증용으로도 재사용)
      - unresolved: 순환/모순으로 끝내 layer가 안 매겨진 노드 (RDG 추출 오류 의심 신호)
    """
    edges = _induced_edges(rule_names, graph, enabled_at_s0)
    indegree = {r: 0 for r in rule_names}
    adj = defaultdict(list)
    for src, dst, _ in edges:
        if src not in indegree or dst not in indegree:
            continue
        adj[src].append(dst)
        indegree[dst] += 1

    layers: t.List[t.List[str]] = []
    remaining = set(rule_names)
    frontier = deque(sorted(r for r in remaining if indegree[r] == 0))

    while frontier:
        layer = sorted(frontier)
        layers.append(layer)
        next_frontier = []
        for r in layer:
            remaining.discard(r)
            for nxt in adj[r]:
                if nxt in remaining:
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        next_frontier.append(nxt)
        frontier = deque(sorted(set(next_frontier)))

    unresolved = sorted(remaining)
    if unresolved:
        logging.warning(f"[rule_dependency] RDG 유도 부분그래프에 순환/모순 의심 — "
                         f"미해결 노드: {unresolved}")
    return layers, edges, unresolved


def _scc(nodes: t.List[str], edges: t.List[t.Tuple[str, str]]) -> t.Dict[str, int]:
    """Kosaraju 로 강결합 성분을 구한다. 반환: {규칙: 성분 id}.

    재귀 대신 명시적 스택을 쓴다 — 후보가 70개를 넘지 않아 깊이 문제는 없지만,
    파이프라인 안에서 도는 코드라 재귀 한계에 의존하지 않는 편이 안전하다.
    """
    adj, radj = defaultdict(list), defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)
    seen, order = set(), []
    for start in nodes:
        if start in seen:
            continue
        stack = [(start, iter(adj[start]))]
        seen.add(start)
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, iter(adj[nxt])))
                    break
            else:
                order.append(node)
                stack.pop()
    comp: t.Dict[str, int] = {}
    cid = 0
    for node in reversed(order):
        if node in comp:
            continue
        stack = [node]
        comp[node] = cid
        while stack:
            u = stack.pop()
            for v in radj[u]:
                if v not in comp:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def _intra_scc_order(members: t.List[str], weights: t.Dict[t.Tuple[str, str], int],
                     opportunities: t.Optional[t.Dict[t.Tuple[str, str], int]] = None,
                     strengths: t.Optional[t.Dict[t.Tuple[str, str], float]] = None
                     ) -> t.List[str]:
    """순환 성분 내부를 지지도 비대칭으로 정렬한다.

    기본 점수 (opportunities 없을 때)
        score(A) = sum_B [ w(A->B) - w(B->A) ]      (같은 성분 안에서만)

    기회 정규화 (B3, opportunities 주어졌을 때)
        p(A->B) = w(A->B) / opp(A->B)               라플라스 평활 적용
        score(A) = sum_B [ p(A->B) - p(B->A) ]

    왜 정규화가 필요한가
    --------------------
    원시 관찰 횟수에는 "A 를 몇 번 시도했나"가 섞여 있다. dsb 에서 실측한 예:

        AGGREGATE_PROJECT_MERGE -> UNION_TO_DISTINCT : 10회 / 기회 54회 = 18.5%
        UNION_TO_DISTINCT -> AGGREGATE_PROJECT_MERGE :  2회 / 기회  2회 = 100%

    원시 횟수로는 10 > 2 라 앞의 규칙이 앞서 보이지만, 성공률로는 반대다.
    양방향 쌍 149개 중 31개(21%)에서 방향이 뒤집혔다.

    다만 정규화만 하면 반대 편향이 생긴다 — 2/2 = 100% 는 신뢰할 수 없다.
    실제로 그 31건 중 11건은 한쪽 기회가 3 이하였고, 양쪽 다 기회 10 이상인
    것은 4건뿐이었다. 그래서 Beta(1,1) 평활을 적용하고, 두 방향의 평활 확률이
    거의 같으면(TIE_EPS 이내) 순서를 강제하지 않고 이름순으로 둔다.

    주의: SCC 이론이 주는 결과가 아니라 별도 휴리스틱이다. Eades-Lin-Smyth 의
    feedback-arc 휴리스틱이 쓰는 outdeg-indeg 발상에서 착안했으나, 그쪽은
    노드를 제거할 때마다 동적으로 재계산하므로 동일한 알고리즘은 아니다.
    """
    inside = set(members)

    def directed(a, b):
        if strengths is not None:
            return strengths.get((a, b), 0.0)
        w = weights.get((a, b), 0)
        if opportunities is None:
            return float(w)
        o = opportunities.get((a, b), 0)
        return (w + 1.0) / (o + 2.0)          # Beta(1,1)

    score = {}
    for a in members:
        total = 0.0
        for b in inside:
            if b == a:
                continue
            d = directed(a, b) - directed(b, a)
            if abs(d) >= TIE_EPS:
                total += d
        score[a] = total
    return sorted(members, key=lambda r: (-score[r], r))


def build_skeleton_scc(rule_names: t.List[str], graph: dict,
                       enabled_at_s0: t.Optional[t.Iterable[str]] = None,
                       opportunities: t.Optional[t.Dict[t.Tuple[str, str], int]] = None
                       ) -> dict:
    """SCC 축약 + 지지도 가중 순서.

    왜 위상정렬을 그대로 못 쓰는가
    -------------------------------
    필터를 제거해 관측이 정확해지자 유도 부분그래프가 촘촘해졌다. 후보 22개에
    유도 엣지 136개면 진입차수 0 인 노드가 없어 Kahn 은 첫 layer 조차 만들지
    못한다. 실측에서 21/21 쿼리가 layers=0 이었다.

    SCC 로 축약하면 그 결과는 정의상 항상 DAG 이므로 위상정렬이 실패할 수 없다.
    순환 밖 규칙은 제 순서를 받고, 순환 안은 별도 휴리스틱으로 정렬한다.

    반환
    ----
      layers        : [[규칙, ...], ...]  축약 DAG 의 위상 layer.
                      순환 성분은 내부를 점수 순으로 펼쳐 한 layer 에 담는다.
      edges         : 유도 부분그래프 엣지 (기존과 동일 형식)
      unresolved    : 어떤 layer 에도 못 들어간 규칙 (정상적으로는 비어야 한다)
      cyclic_groups : [[규칙, ...], ...]  크기 2 이상인 성분. 이 안의 순서는
                      확신도가 낮다는 표시이며, 프롬프트 렌더링이 이를 구분해
                      표기해야 한다 (한 layer 에 나열하면 '동순위'로 읽히므로).
    """
    edges = _induced_edges(rule_names, graph, enabled_at_s0)
    weights = {}
    strengths = {}
    for e in graph.get('edges', []):
        weights[(e['from'], e['to'])] = int(e.get('observed_count', 1) or 1)
        if 'structural_strength' in e:
            strengths[(e['from'], e['to'])] = float(e['structural_strength'])

    pairs = [(a, b) for a, b, _ in edges]
    comp = _scc(list(rule_names), pairs)
    for r in rule_names:                      # 고립 노드도 성분을 갖도록
        comp.setdefault(r, max(comp.values(), default=-1) + 1)

    members: t.Dict[int, t.List[str]] = defaultdict(list)
    for r in rule_names:
        members[comp[r]].append(r)

    # 축약 그래프 (성분 간 엣지만)
    cadj: t.Dict[int, set] = defaultdict(set)
    cindeg: t.Dict[int, int] = {c: 0 for c in members}
    for a, b in pairs:
        ca, cb = comp[a], comp[b]
        if ca != cb and cb not in cadj[ca]:
            cadj[ca].add(cb)
            cindeg[cb] += 1

    layers, cyclic = [], []
    remaining = set(members)
    frontier = sorted(c for c in remaining if cindeg[c] == 0)
    while frontier:
        layer: t.List[str] = []
        nxt: t.List[int] = []
        for c in frontier:
            mem = members[c]
            ordered = (_intra_scc_order(mem, weights, opportunities,
                                        strengths if strengths else None)
                       if len(mem) > 1 else mem)
            if len(mem) > 1:
                cyclic.append(ordered)
            layer.extend(ordered)
            remaining.discard(c)
        for c in frontier:
            for d in cadj[c]:
                if d in remaining:
                    cindeg[d] -= 1
                    if cindeg[d] == 0:
                        nxt.append(d)
        layers.append(layer)
        frontier = sorted(set(nxt))

    unresolved = sorted(r for c in remaining for r in members[c])
    if unresolved:
        logging.warning(f"[rule_dependency] SCC 축약 후에도 미해결 노드가 남았다 "
                        f"(축약 그래프가 DAG 여야 하므로 정상적으로는 발생하지 않는다): "
                        f"{unresolved}")
    return {'layers': layers, 'edges': edges, 'unresolved': unresolved,
            'cyclic_groups': cyclic}


def render_skeleton_for_prompt(layers: t.List[t.List[str]], edges: t.List[Edge],
                                unresolved: t.List[str],
                                cyclic_groups: t.Optional[t.List[t.List[str]]] = None) -> str:
    """layer + 근거를 프롬프트용 텍스트로 직렬화한다. 엣지가 하나도 없으면 빈 문자열
    (= 호출부에서 "(none)"으로 치환됨)을 반환한다.

    [2026-09-18] B4 — 순환 성분을 별도로 표기한다.

    `Layer 1: A, B, C` 는 기존 의미상 "세 규칙이 동순위"다. SCC 내부를 점수로
    정렬해 놓고 같은 형식으로 내보내면 그 순서가 전달되지 않는다. 그렇다고 별도
    layer 로 쪼개면 관측되지 않은 강한 선후 관계를 만들어내는 셈이 된다.

    그래서 순환 성분은 "preferred order, low confidence" 로 명시한다 — 권고는
    하되 확신도가 다르다는 것을 LLM 이 알 수 있게 한다.
    """
    if not edges:
        return ''
    cyc_index = {}
    for gi, g in enumerate(cyclic_groups or []):
        for r in g:
            cyc_index[r] = gi
    cyc_sets = [set(g) for g in (cyclic_groups or [])]

    deps_by_target: t.Dict[str, t.List[t.Tuple[str, str]]] = defaultdict(list)
    for src, dst, reason in edges:
        deps_by_target[dst].append((src, reason))

    lines = []
    for i, layer in enumerate(layers, 1):
        groups_here = {cyc_index[r] for r in layer if r in cyc_index}
        plain = [r for r in layer if r not in cyc_index]
        if plain:
            lines.append(f"Layer {i}: {', '.join(plain)}")
            for rule in plain:
                # [2026-09-18] B7 — 근거 줄 수와 길이를 제한한다.
                # 1076 엣지 그래프에서는 한 규칙이 선행 근거를 8개 넘게 갖고, reason
                # 이 규칙 설명 전문이라 렌더링 결과가 489K 문자까지 늘었다(옛 43엣지
                # 그래프의 34배). 프롬프트가 규칙 설명으로 덮이면 LLM 이 정작 쿼리를
                # 못 본다. 근거는 상위 몇 개만 있으면 목적을 달성한다.
                for src, reason in deps_by_target.get(rule, [])[:MAX_REASONS_PER_RULE]:
                    short = (reason or '')[:MAX_REASON_CHARS]
                    lines.append(f"    - {rule} requires {src} first (reason: {short})")
        for gi in sorted(groups_here):
            members = [r for r in layer if r in cyc_sets[gi]]
            lines.append(
                f"Layer {i} (mutually-enabling group — these rules can unlock each "
                f"other, so the order below is a preference, not a requirement; "
                f"confidence: low):")
            lines.append(f"    preferred order: {' -> '.join(members)}")
    if unresolved:
        lines.append(f"Unresolved (순환/모순 의심 — 자유롭게 배치 가능): {', '.join(unresolved)}")
    return '\n'.join(lines)


def check_skeleton_followed(final_seq: t.List[str], edges: t.List[Edge]
                             ) -> t.Tuple[bool, t.List[t.Tuple[str, str]]]:
    """최종 순서(final_seq)가 유도 부분그래프의 모든 "A가 B보다 먼저" 제약을 지켰는지
    코드로 객관적으로 판정한다 (LLM의 자기보고에 의존하지 않음).

    반환: (모두 지켰는가, 위반한 (선행규칙, 후행규칙) 쌍 목록)
    """
    pos = {rule: i for i, rule in enumerate(final_seq)}
    violations = []
    for src, dst, _ in edges:
        if src in pos and dst in pos and pos[src] > pos[dst]:
            violations.append((src, dst))
    return len(violations) == 0, violations


def extract_reasoning_text(llm_response: str, max_chars: int = 800) -> str:
    """LLM 응답에서 ```python 코드블록 이전의 자유서술(reasoning) 부분만 추출한다.
    RDG_GUIDANCE_SYS_NOTE가 "골격을 벗어나면 이유를 reasoning에 명시하라"고 지시하므로,
    벗어난 경우 이 텍스트에 그 이유가 들어있을 가능성이 높다.
    """
    text = llm_response.split('```python')[0].strip()
    return text[:max_chars]


def closure_expand(relevant_rules: t.List[t.Dict[str, str]], candidate_pool: t.List[t.Dict[str, str]],
                    graph: dict, full_rule_descriptions: t.Optional[t.Dict[str, str]] = None,
                    enabled_at_s0: t.Optional[t.Iterable[str]] = None
                    ) -> t.Tuple[t.List[t.Dict[str, str]], t.List[t.Tuple[str, str, str]]]:
    """select_rules_dist가 고른 relevant_rules에서 RDG 엣지를 타고 도달 가능한 규칙을
    편입시킨다 (fixed-point — 새로 편입된 규칙이 또 다른 규칙을 여는 다중 hop까지 반복 처리).

    두 가지 소스에서 끌어온다:
      (a) candidate_pool — RAG의 match_calcite_rules가 "원본 쿼리" 트리에 대해 이미
          구조적으로 매칭시켰지만 LLM(select_rules_dist)이 걸러낸 규칙. query-specific
          설명(rewrite)이 이미 있으므로 그대로 쓴다.
      (b) full_rule_descriptions — RDG 그래프의 전체 노드(70개 규칙 카탈로그) 중 (a)에
          없는 규칙. match_calcite_rules는 "원본 쿼리" 트리에 대해 딱 1회만 매칭하므로,
          "규칙 A를 먼저 적용해야 비로소 매칭되는" 규칙 B는 애초에 candidate_pool에
          들어올 수가 없다 — execute_rewrite()(HepPlanner)는 넘겨받은 rule_seq에 있는
          규칙만 시도하므로, B가 한 번도 후보에 안 들어가면 트리가 바뀐 뒤에도 B는
          영원히 시도되지 않는다. 이 빈틈을 메우는 게 closure 확장의 본래 목적이다.
          이 경우 explain_rule/*.jsonl의 범용 설명을 rewrite 필드로 대신 채운다.

    matched_pool=0 안전장치 (2026-07-02 analysis_report_baseline_vs_bothgates.md
    근본원인 분석 반영): "B가 실제 적용 가능한지는 execute_rewrite()가 검증하니 틀린
    추가는 무시될 뿐 안전하다"는 가정은 사실이 아니었다 — HepPlanner는 "B가 지금
    트리에 구조적으로 매칭되는가"만 보고 적용해버리므로, full_catalog로 추가된
    B가 실제로 적용까지 성공한 뒤 Calcite의 RelNode→SQL 재변환(un-parse) 단계에서
    PostgreSQL이 못 읽는 SQL(예: self-join 평탄화 시 컬럼 qualifier 누락으로 인한
    AmbiguousColumn)을 만들어내는 사례가 실측 확인됐다 (JOIN_ADD_REDUNDANT_SEMI_JOIN:
    실패 46회/성공 0회). 이 확장 pass에서 matched_pool 소스 추가가 단 하나도 없다면
    (=candidate_pool에 애초에 이 쿼리와 구조적으로 겹치는 게 전혀 없었다는 뜻),
    full_catalog 소스 추가는 "다른 쿼리들의 통계"에만 의존하는 무근거 추정이므로
    전부 제거한다. matched_pool 소스가 하나라도 있으면 그게 이 쿼리에 대한 RDG 신호의
    로컬 타당성을 뒷받침하는 것으로 보고 full_catalog 추가도 함께 유지한다.

    반환: (확장된 relevant_rules, [(enabling_rule, newly_added_rule, source), ...])
      source는 'matched_pool'(원래도 Calcite가 매칭했던 것) 또는
      'full_catalog'(원본엔 안 맞았으나 RDG가 추정한 것) 중 하나 — 로그에서 두 경우를
      구분해서 감사(audit)할 수 있게 한다. matched_pool=0 안전장치로 제거된 항목은
      반환값에 포함되지 않는다(호출부가 별도로 재조회할 필요 없음).
    """
    full_rule_descriptions = full_rule_descriptions or {}
    selected_names = {r['name'] for r in relevant_rules}
    pool_by_name = {r['name']: r for r in candidate_pool}

    enables: t.Dict[str, t.Set[str]] = defaultdict(set)
    edge_strength: t.Dict[t.Tuple[str, str], float] = {}
    edge_family_support: t.Dict[t.Tuple[str, str], int] = {}
    edge_observed_count: t.Dict[t.Tuple[str, str], int] = {}
    already = set(enabled_at_s0 or ())
    for e in graph.get('edges', []):
        if e['to'] in already:
            continue
        enables[e['from']].add(e['to'])
        edge_strength[(e['from'], e['to'])] = float(
            e.get('structural_strength', e.get('observed_count', 1)) or 0)
        edge_family_support[(e['from'], e['to'])] = int(
            e.get('family_support', 0) or 0)
        edge_observed_count[(e['from'], e['to'])] = int(
            e.get('observed_count', 0) or 0)

    expanded = list(relevant_rules)  # 원래 순서/객체 보존
    existing = {r['name'] for r in expanded}
    added_log: t.List[t.Tuple[str, str, str]] = []

    changed = True
    while changed:
        changed = False
        for name in sorted(selected_names):
            for target in sorted(enables.get(name, [])):
                if target in selected_names:
                    continue
                if target in pool_by_name:
                    selected_names.add(target)
                    expanded.append(pool_by_name[target])
                    existing.add(target)
                    added_log.append((name, target, 'matched_pool'))
                    changed = True
                elif target in full_rule_descriptions:
                    selected_names.add(target)
                    expanded.append({'name': target, 'rewrite': full_rule_descriptions[target]})
                    existing.add(target)
                    added_log.append((name, target, 'full_catalog'))
                    changed = True

    # Experimental family-balanced RDG: rank each target by its strongest
    # supporting incoming edge rather than the alphabetically first path that
    # happened to discover it.  Default behavior is unchanged unless enabled.
    if os.environ.get('RDG_STRENGTH_RANKING') == '1' and added_log:
        source_by_target: t.Dict[str, t.Tuple[str, str]] = {}
        for _old_source, target, source_kind in added_log:
            possible = [(
                edge_strength.get((src, target), 0.0),
                edge_family_support.get((src, target), 0),
                edge_observed_count.get((src, target), 0),
                src,
            ) for src in selected_names if (src, target) in edge_strength]
            best_source = max(possible, default=(0.0, 0, 0, _old_source))[3]
            source_by_target[target] = (best_source, source_kind)
        added_log = [(src, target, kind)
                     for target, (src, kind) in source_by_target.items()]

    # matched_pool=0 안전장치: 이 확장에서 matched_pool 소스 추가가 하나도 없으면
    # full_catalog 소스 추가(무근거 추정)를 전부 되돌린다.
    has_matched_pool_addition = any(source == 'matched_pool' for _, _, source in added_log)
    if not has_matched_pool_addition:
        dropped = {target for _, target, source in added_log if source == 'full_catalog'}
        if dropped:
            expanded = [r for r in expanded if r['name'] not in dropped]
            added_log = [e for e in added_log if e[2] != 'full_catalog']

    # [2026-09-18] B7 — 추가 개수 상한. 반드시 matched_pool 안전장치 **뒤**에 둔다.
    #
    # 앞에 두었더니 상한이 matched_pool 출처 항목을 잘라내 버렸고, 그러면 위의
    # "matched_pool 추가가 하나도 없으면 full_catalog 추가를 전부 되돌린다"가
    # 발동해 closure 결과가 0개가 됐다(파일럿에서 실측). 두 로직이 같은 리스트를
    # 두고 싸우지 않도록 순서를 고정한다.
    #
    # 자르는 기준은 family-balanced structural strength, independent-family support,
    # raw observation 순이다. 1076 엣지 그래프에서는 쿼리당
    # 60개 이상이 추가되는데 전체 규칙이 70개라, 상한이 없으면 그래프가 선별
    # 장치로 기능하지 못하고 프롬프트만 34배로 불어난다.
    if MAX_CLOSURE_ADDITIONS > 0 and len(added_log) > MAX_CLOSURE_ADDITIONS:
        # matched_pool(이 쿼리에 실제로 매칭됐던 규칙)을 full_catalog 추정보다
        # 우선 남긴다 — 근거가 더 강한 쪽이다.
        ranked = sorted(added_log,
                        key=lambda x: (0 if x[2] == 'matched_pool' else 1,
                                       -edge_strength.get((x[0], x[1]), 0.0),
                                       -edge_family_support.get((x[0], x[1]), 0),
                                       -edge_observed_count.get((x[0], x[1]), 0),
                                       x[1]))
        keep = ranked[:MAX_CLOSURE_ADDITIONS]
        keep_names = {t_ for _, t_, _ in keep}
        dropped = {t_ for _, t_, _ in added_log} - keep_names
        expanded = [r for r in expanded if r['name'] not in dropped]
        added_log = keep

    return expanded, added_log


def linearize_skeleton(original_seq: t.List[str], layers: t.List[t.List[str]],
                       cyclic_groups: t.Optional[t.List[t.List[str]]] = None
                       ) -> t.List[str]:
    """Apply component-layer precedence while preserving unconstrained order.

    This is used only by the opt-in RDG experiment.  Between independent nodes
    in one layer the LLM's original order is retained; inside an SCC the
    family-balanced preferred order is used.
    """
    layer_of = {name: i for i, layer in enumerate(layers) for name in layer}
    original_pos = {name: i for i, name in enumerate(original_seq)}
    cyclic_pos = {}
    for group in cyclic_groups or []:
        for i, name in enumerate(group):
            cyclic_pos[name] = i
    present = set(original_seq)
    ordered = sorted(original_seq, key=lambda name: (
        layer_of.get(name, len(layers)),
        0 if name in cyclic_pos else 1,
        cyclic_pos.get(name, original_pos[name]),
    ))
    # Graph/catalog mismatches never delete an LLM-selected rule.
    return ordered + [n for n in original_seq if n not in present]
