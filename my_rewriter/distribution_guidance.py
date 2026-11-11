"""
distribution_guidance.py

Module 1 (Distribution-Aware Optimization Advisor) + recipe 단계의 distribution-aware
변형 함수들을 모아 둔 모듈.

baseline(rag_rewrite.py / proposed_rewrite.py / my_utils.py)은 이 파일에서 "읽기만" 하고
절대 수정하지 않는다. 새로운 방법론에 필요한 모든 신규 로직은 여기에만 추가한다.

흐름:
  1) build_distribution_evidence(): Uniformity Score 계산 결과(per_column)를
     이 쿼리에 실제로 등장하는 테이블/컬럼만으로 압축한 "Evidence Pack" 생성.
     (LLM 호출 없음, DB 추가 조회 없음 — 이미 계산된 결과를 재사용)
  2) render_distribution_summary(): Evidence Pack에서 상위 role-aware skew 컬럼과
     MCV/NULL/n_distinct 의미를 고정 template으로 짧게 설명 (LLM/DB 호출 없음).
  3) gen_distribution_guidance(): Evidence Pack -> LLM 1회(+재시도 1회) 호출
     -> 구조화된 가이드 {summary, rewrite_hints, index_hints, planner_hints}
  4) render_guidance_for_prompt(): rewrite_hints/summary만 짧은 텍스트로 직렬화하여
     select_rules_dist / arrange_rules_dist / rearrange_rules_dist 의
     {distribution_guidance} 슬롯에 주입.
     index_hints/planner_hints는 Calcite 실행 경로가 없으므로 여기서 사용하지 않고,
     호출부(distaware_rewrite.py)에서 advisory 로그로만 남긴다.
  5) select_rules_dist/arrange_rule_sets_dist/arrange_rules_dist/rearrange_rules_dist:
     my_utils.py의 select_rules/arrange_rule_sets/arrange_rules/rearrange_rules와
     동일한 파싱 로직이지만, DIST_* 프롬프트(+distribution_guidance 슬롯)를 사용하는
     독립된 함수. my_utils.py의 기존 메서드는 변경하지 않는다.
"""
import json
import math
import re
import logging
import typing as t

from my_rewriter.my_utils import MyModel, chat, get_rule_sets
from my_rewriter.uniformity_score import ROLE_WEIGHTS, UniformityResult
from my_rewriter.rule_dependency import check_skeleton_followed, extract_reasoning_text

EMPTY_GUIDANCE = {'summary': '', 'rewrite_hints': [], 'index_hints': [], 'planner_hints': []}

PILOT_LOG_PREFIX = '[PILOT-CHECK]'
_PLANNER_RULE_DESC_CHARS = 600
_SUMMARY_MAX_COLUMNS = 3
_SUMMARY_MAX_MCV_VALUES = 2
_SUMMARY_MAX_CORRELATIONS = 1
_SUMMARY_MAX_CHARS = 1800


def _log_llm_call(fn_name: str, attempt: int, response: str, parsed_ok: bool, extra: str = '') -> None:
    """모든 distribution-aware LLM 호출 직후 한 줄로 남기는 공통 체크포인트 로그.
    grep 'PILOT-CHECK' 한 줄로 "어떤 함수가 LLM을 몇 번 호출했고, 응답을 받았고,
    파싱에 성공했는지"를 바로 확인할 수 있게 한다. 실제 메시지/응답 전체는
    my_utils.chat()이 이미 DEBUG 레벨로 남기므로 여기서는 길이/요약만 남긴다.
    """
    status = 'PARSED_OK' if parsed_ok else 'PARSE_FAILED'
    suffix = f' {extra}' if extra else ''
    logging.info(f'{PILOT_LOG_PREFIX} LLM call [{fn_name}] attempt={attempt} '
                 f'response_len={len(response)}chars status={status}{suffix}')


def build_distribution_evidence(result: UniformityResult, pool: t.Optional[dict] = None,
                                 top_k_mcv: int = 3) -> dict:
    """Uniformity Score 계산 결과(compute_uniformity_score의 반환값)에서, 이 쿼리에
    실제로 등장하는 테이블/컬럼에 대한 구조적 역할 + skew 통계만 추출한다.

    pool이 주어지면(`{db}_pool.json`을 미리 로드한 dict), per_column에는 없는
    most_common_values / join correlation 정보를 "이 쿼리에 등장하는 테이블/컬럼"으로만
    필터링해서 보강한다 (풀 전체를 넘기지 않음 — 토큰 절약).
    """
    evidence: dict = {
        'uniformity_score': round(result.score, 4),
        'query_complexity': result.query_complexity,
        'notes': result.notes,
        'columns': [
            {
                'table': table,
                'column': col,
                'roles': sorted(set(info['roles'])),
                'skew_score': info.get('skew_score'),
                'n_distinct': info.get('n_distinct'),
                'null_frac': info.get('null_frac'),
            }
            for (table, col), info in result.per_column.items()
        ],
    }

    if not pool:
        return evidence

    cols_in_query = {(c['table'], c['column']) for c in evidence['columns']}
    tables_in_query = {tbl for tbl, _ in cols_in_query}

    histogram = pool.get('histogram', {})
    mcv_facts = []
    for (table, col) in cols_in_query:
        h = histogram.get(table, {}).get(col)
        if h and h.get('most_common_vals'):
            mcv_facts.append({
                'table': table,
                'column': col,
                'most_common_vals': h['most_common_vals'][:top_k_mcv],
                'most_common_freqs': h.get('most_common_freqs', [])[:top_k_mcv],
            })
    if mcv_facts:
        evidence['most_common_values'] = mcv_facts

    join_facts = []
    for key, corr in pool.get('correlation', {}).get('pearson_join', {}).items():
        # key 형식: "table1.col1__table2.col2"
        try:
            left, right = key.split('__')
            t1, t2 = left.split('.')[0], right.split('.')[0]
        except ValueError:
            continue
        if t1 in tables_in_query and t2 in tables_in_query and corr is not None:
            join_facts.append({'join_pair': key, 'pearson_correlation': corr})
    if join_facts:
        evidence['join_correlations'] = join_facts

    return evidence


def _finite_float(value: t.Any) -> t.Optional[float]:
    """Return a finite float without letting malformed optional statistics escape."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_stat_value(value: t.Any, limit: int = 48) -> str:
    """Render an MCV value deterministically and keep a single value prompt-safe."""
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit - 1] + '\u2026'


def _normalized_roles(value: t.Any) -> t.List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    return sorted({str(role).upper() for role in values if str(role).strip()})


def _column_priority(column: dict) -> float:
    """Use Gate 1 weights as a bounded max-role proxy for summary ranking."""
    skew = _finite_float(column.get('skew_score'))
    if skew is None or not 0.0 <= skew <= 1.0:
        skew = 0.0
    roles = _normalized_roles(column.get('roles'))
    role_weight = max((ROLE_WEIGHTS.get(str(role).upper(), 0.0) for role in roles), default=0.0)
    return role_weight * skew


def render_distribution_summary(
    evidence: t.Optional[dict],
    max_columns: int = _SUMMARY_MAX_COLUMNS,
    max_mcv_values: int = _SUMMARY_MAX_MCV_VALUES,
    max_correlations: int = _SUMMARY_MAX_CORRELATIONS,
    max_chars: int = _SUMMARY_MAX_CHARS,
) -> str:
    """Explain structured evidence with a bounded deterministic template.

    The JSON remains the source of truth.  This renderer only exposes score
    direction, query roles, and prominent observed values so the planner does
    not have to infer project-specific score semantics.  It never recommends a
    concrete rewrite rule and never performs a DB or LLM call.
    """
    if not isinstance(evidence, dict) or not evidence:
        return ''
    if max_chars <= 0:
        return ''

    lines: t.List[str] = []
    uniformity = _finite_float(evidence.get('uniformity_score'))
    if uniformity is not None and not 0.0 <= uniformity <= 1.0:
        uniformity = None
    complexity = evidence.get('query_complexity')
    if uniformity is not None:
        query_line = (
            f'- Query-level: uniformity_score={uniformity:.4f} '
            '(lower means stronger role-aware non-uniformity)'
        )
        if complexity is not None:
            query_line += f'; query_complexity={complexity}'
        lines.append(query_line + '.')

    raw_columns = evidence.get('columns') or []
    ranked_columns = []
    for index, column in enumerate(raw_columns):
        if not isinstance(column, dict):
            continue
        table = str(column.get('table') or '?')
        name = str(column.get('column') or '?')
        skew = _finite_float(column.get('skew_score'))
        ranked_columns.append((
            -_column_priority(column),
            -(skew if skew is not None else -1.0),
            table,
            name,
            index,
            column,
        ))
    ranked_columns.sort(key=lambda item: item[:5])
    selected = ranked_columns[:max(0, max_columns)]

    mcv_by_column: t.Dict[t.Tuple[str, str], dict] = {}
    for fact in evidence.get('most_common_values') or []:
        if isinstance(fact, dict):
            key = (str(fact.get('table') or '?'), str(fact.get('column') or '?'))
            mcv_by_column[key] = fact

    for rank, item in enumerate(selected, start=1):
        _, _, table, name, _, column = item
        roles = _normalized_roles(column.get('roles'))
        roles_text = ', '.join(roles) if roles else 'UNKNOWN_ROLE'
        skew = _finite_float(column.get('skew_score'))
        if skew is not None and not 0.0 <= skew <= 1.0:
            skew = None
        priority = _column_priority(column)
        skew_text = f'{skew:.4f}' if skew is not None else 'unknown'
        lines.append(
            f'- Priority {rank}: {table}.{name}; roles=[{roles_text}]; '
            f'skew_score={skew_text} (higher means more skew); '
            f'role-weighted priority={priority:.4f}.'
        )

        observations: t.List[str] = []
        null_frac = _finite_float(column.get('null_frac'))
        if null_frac is not None and 0 < null_frac <= 1:
            observations.append(f'NULL fraction={100.0 * null_frac:.2f}%')

        n_distinct = _finite_float(column.get('n_distinct'))
        if n_distinct is not None:
            if -1.0 <= n_distinct < 0:
                observations.append(
                    f'n_distinct={n_distinct:g}, meaning about '
                    f'{100.0 * abs(n_distinct):.2f}% of table rows are distinct'
                )
            else:
                observations.append(f'n_distinct statistic={n_distinct:g}')

        mcv = mcv_by_column.get((table, name))
        if mcv and max_mcv_values > 0:
            values = list(mcv.get('most_common_vals') or [])
            freqs = list(mcv.get('most_common_freqs') or [])
            pairs = []
            for value, frequency in zip(values, freqs):
                freq = _finite_float(frequency)
                if freq is None or not 0.0 <= freq <= 1.0:
                    continue
                pairs.append(f'{_display_stat_value(value)}={100.0 * freq:.2f}%')
                if len(pairs) >= max_mcv_values:
                    break
            if pairs:
                observations.append('top MCV frequencies: ' + ', '.join(pairs))

        if observations:
            lines.append('  Observed: ' + '; '.join(observations) + '.')

    correlations = []
    for fact in evidence.get('join_correlations') or []:
        if not isinstance(fact, dict):
            continue
        corr = _finite_float(fact.get('pearson_correlation'))
        if corr is None or not -1.0 <= corr <= 1.0:
            continue
        correlations.append((-abs(corr), str(fact.get('join_pair') or '?'), corr))
    correlations.sort()
    for _, pair, corr in correlations[:max(0, max_correlations)]:
        lines.append(
            f'- Offline table-pair context: {pair} has Pearson correlation={corr:.4f}; '
            'this does not by itself prove correlation between the current join keys.'
        )

    if not lines:
        return ''

    bounded: t.List[str] = []
    current_length = 0
    for line in lines:
        added = len(line) + (1 if bounded else 0)
        if bounded and current_length + added > max_chars:
            break
        if not bounded and len(line) > max_chars:
            bounded.append(line[:max(0, max_chars - 1)] + '\u2026')
            break
        bounded.append(line)
        current_length += added
    return '\n'.join(bounded)


def render_guidance_for_prompt(guidance: dict) -> str:
    """guidance['rewrite_hints'] (+ summary)만 짧은 텍스트로 직렬화한다.
    index_hints/planner_hints는 의도적으로 제외 — Calcite rewrite 엔진은 DDL/GUC를
    실행할 경로가 없으므로 recipe 생성 프롬프트에 넣지 않는다 (advisory 로그로만 보존).
    """
    if not guidance or not guidance.get('rewrite_hints'):
        return ''
    lines = []
    if guidance.get('summary'):
        lines.append(f"- Summary: {guidance['summary']}")
    for h in guidance['rewrite_hints']:
        target = h.get('target', '')
        obs = h.get('observation', '')
        sug = h.get('suggestion', '')
        lines.append(f"- [{target}] {obs} -> {sug}")
    return '\n'.join(lines)


def _compact_rule_description(value: str, limit: int = _PLANNER_RULE_DESC_CHARS) -> str:
    """Keep planner prompts bounded without dropping candidate rule names."""
    compact = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(' ', 1)[0] + ' …'


def plan_rules_with_guidance_dist(
    model: MyModel,
    query: str,
    suggestions_str: str,
    candidate_rules: t.List[t.Dict[str, str]],
    distribution_evidence: t.Optional[dict] = None,
    rule_dependency_constraints: str = '',
) -> t.Tuple[dict, t.List[t.Dict[str, str]], t.List[str]]:
    """Generate guidance, select rules, and order them in one fast planning call.

    The gated pipeline previously used four sequential calls for guidance,
    relevance selection, rule-set grouping, and final ordering.  This returns
    the three runtime artifacts together while validating every rule name
    against the code-produced candidate pool.
    """
    if not candidate_rules:
        return dict(EMPTY_GUIDANCE), [], []

    by_name = {r['name'].upper(): r for r in candidate_rules}
    distribution_summary = render_distribution_summary(distribution_evidence)
    rules_str = '\n\n'.join(
        f'### Rule {r["name"]}:\n{_compact_rule_description(r["rewrite"])}'
        for r in candidate_rules
    ) or '(none)'
    system_content = (
        'You are a PostgreSQL/Apache Calcite rewrite planner. In one pass, '
        'summarize relevant distribution evidence, select only applicable '
        'candidate rules, and order them for execution. Return JSON only: '
        '{"guidance":{"summary":"...","rewrite_hints":['
        '{"target":"...","observation":"...","suggestion":"..."}],'
        '"index_hints":[],"planner_hints":[]},'
        '"selected_rules":["RULE"],"ordered_rules":["RULE"]}. '
        'Use only rule names from Candidate Rules. ordered_rules must be a '
        'permutation of selected_rules and should prefer a short safe plan.'
    )
    user_content = (
        f'SQL:\n{query}\n\nRewrite suggestions:\n{suggestions_str or "(none)"}'
        f'\n\nDistribution evidence:\n'
        f'{json.dumps(distribution_evidence or {}, indent=2, default=str)}'
    )
    if distribution_summary:
        system_content += (
            ' Treat Distribution evidence JSON as the source of truth. The '
            'deterministic summary only explains supplied statistics. Do not '
            'invent missing statistics, and do not select an inapplicable rule '
            'solely because skew is present.'
        )
        user_content += (
            f'\n\nDeterministic distribution summary:\n{distribution_summary}'
        )
    user_content += (
        f'\n\nRule dependency guidance:\n'
        f'{rule_dependency_constraints or "(none)"}'
        f'\n\nCandidate Rules:\n{rules_str}'
    )
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]
    prompt_chars = sum(len(str(m.get('content', ''))) for m in messages)
    logging.info(
        f'{PILOT_LOG_PREFIX} Integrated planner prompt: '
        f'rules={len(candidate_rules)} chars={prompt_chars} '
        f'per_rule_limit={_PLANNER_RULE_DESC_CHARS}'
    )
    response = chat(messages, getattr(model, 'fast_llm', None))
    block = re.search(r'```json\s*(.+?)\s*```', response or '', re.I | re.DOTALL)
    raw = block.group(1) if block else (response or '').strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed.get('selected_rules'), list) or not isinstance(
                parsed.get('ordered_rules'), list):
            raise ValueError('planner response lacks rule lists')
        guidance = parsed.get('guidance') or dict(EMPTY_GUIDANCE)
        for key, default in EMPTY_GUIDANCE.items():
            guidance.setdefault(key, default)
        selected_names = []
        for name in parsed.get('selected_rules', []):
            upper = str(name).upper()
            if upper in by_name and upper not in selected_names:
                selected_names.append(upper)
        ordered_names = []
        selected_set = set(selected_names)
        for name in parsed.get('ordered_rules', []):
            upper = str(name).upper()
            if upper in selected_set and upper not in ordered_names:
                ordered_names.append(upper)
        ordered_names.extend(name for name in selected_names if name not in ordered_names)
        selected = [by_name[name] for name in selected_names]
        _log_llm_call(
            'plan_rules_with_guidance_dist', 1, response, True,
            extra=f'selected={len(selected)}/{len(candidate_rules)}',
        )
        return guidance, selected, ordered_names
    except (TypeError, ValueError):
        pass

    # A malformed planner response must not turn into a false "no rules"
    # decision. Preserve correctness with a deterministic all-candidate plan;
    # do not issue another LLM call.
    fallback_names = list(by_name)
    _log_llm_call('plan_rules_with_guidance_dist', 1, response or '', False,
                  extra=f'fallback={len(fallback_names)}')
    return dict(EMPTY_GUIDANCE), list(candidate_rules), fallback_names


def plan_baseline_and_skew_rules(
    model: MyModel,
    query: str,
    suggestions_str: str,
    candidate_rules: t.List[t.Dict[str, str]],
    gate1_active: bool,
    distribution_evidence: t.Optional[dict] = None,
    rule_dependency_constraints: str = '',
) -> dict:
    """Preserve the current Gate1-OFF planner and add one active-only call.

    The first call is byte-for-byte the same planner invocation used by the
    current Gate1-OFF path: it receives an empty distribution-evidence object.
    Its selected rules and order are frozen as the baseline.  Gate1 activation
    permits exactly one additional invocation with distribution evidence; it
    never mutates or replaces the baseline artifact.

    This function only generates plans.  Final selection belongs to the
    asymmetric DCRP selector in
    ``candidate_selection.baseline_preserving_gate1`` after both SQL candidates
    have passed equivalence checking and EXPLAIN feature extraction.
    """
    baseline_guidance, baseline_rules, baseline_order = plan_rules_with_guidance_dist(
        model,
        query,
        suggestions_str,
        candidate_rules,
        distribution_evidence={},
        rule_dependency_constraints=rule_dependency_constraints,
    )
    baseline = {
        'guidance': baseline_guidance,
        'selected_rules': list(baseline_rules),
        'ordered_rules': list(baseline_order),
    }
    alternative = None
    extra_llm_calls = 0
    if gate1_active:
        extra_llm_calls = 1
        skew_guidance, skew_rules, skew_order = plan_rules_with_guidance_dist(
            model,
            query,
            suggestions_str,
            candidate_rules,
            distribution_evidence=distribution_evidence or {},
            rule_dependency_constraints=rule_dependency_constraints,
        )
        alternative = {
            'guidance': skew_guidance,
            'selected_rules': list(skew_rules),
            'ordered_rules': list(skew_order),
        }
    return {
        'baseline': baseline,
        'alternative': alternative,
        'gate1_active': bool(gate1_active),
        'extra_llm_calls': extra_llm_calls,
    }


def gen_distribution_guidance(model: MyModel, query: str, evidence: dict) -> dict:
    """Module 1 LLM 호출. ```json 블록을 파싱해 구조화된 guidance dict를 반환한다.
    파싱 실패 시 2회까지 재시도하고, 끝내 실패하면 EMPTY_GUIDANCE를 반환한다
    (호출부에서 빈 guidance를 안전하게 처리할 수 있도록).
    """
    messages = [
        {'role': 'system', 'content': model.GEN_DISTRIBUTION_GUIDANCE_SYS_PROMPT},
        {'role': 'user', 'content': model.GEN_DISTRIBUTION_GUIDANCE_USER_PROMPT.format(
            sql=query, evidence=json.dumps(evidence, indent=2, default=str))},
    ]
    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        m = re.search(r'```json\s*(.+?)\s*```', response, re.I | re.DOTALL)
        if m:
            try:
                guidance = json.loads(m.group(1))
                for key, default in EMPTY_GUIDANCE.items():
                    guidance.setdefault(key, default)
                _log_llm_call('gen_distribution_guidance', attempt, response, True,
                              extra=f"rewrite_hints={len(guidance['rewrite_hints'])} "
                                    f"index_hints={len(guidance['index_hints'])} "
                                    f"planner_hints={len(guidance['planner_hints'])}")
                return guidance
            except Exception:
                pass
        _log_llm_call('gen_distribution_guidance', attempt, response, False)
    logging.warning(f"Failed to parse distribution guidance: {response}")
    return dict(EMPTY_GUIDANCE)


# ── recipe 단계의 distribution-aware 변형 (my_utils.py 원본은 수정하지 않음) ──────

def select_rules_dist(model: MyModel, query: str, suggestions_str: str,
                       rules: t.List[t.Dict[str, str]], distribution_guidance: str,
                       rule_stat_priors: str = '') -> t.List[t.Dict[str, str]]:
    rules_str = '\n\n'.join([f'### Rule {r["name"]}:\n"""{r["rewrite"]}"""' for r in rules])
    dist_section = distribution_guidance or '(none)'
    if rule_stat_priors:
        dist_section = dist_section + '\n\n' + rule_stat_priors
    messages = [
        {'role': 'system', 'content': model.DIST_SELECT_RULES_SYS_PROMPT},
        {'role': 'user', 'content': model.DIST_SELECT_RULES_USER_PROMPT.format(
            sql=query, suggestions=suggestions_str, rules=rules_str,
            distribution_guidance=dist_section)},
    ]
    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        prefix, suffix = '```python', '```'
        start_idx, end_idx = response.find(prefix), response.rfind(suffix)
        if start_idx != -1 and end_idx != -1:
            list_str = response[start_idx + len(prefix): end_idx].strip()
            try:
                rule_names = [r.upper() for r in eval(list_str)]
                selected = [r for r in rules if r['name'] in rule_names]
                _log_llm_call('select_rules_dist', attempt, response, True,
                              extra=f"selected={len(selected)}/{len(rules)}")
                return selected
            except Exception:
                pass
        _log_llm_call('select_rules_dist', attempt, response, False)
    logging.warning(f"Failed to select relevant rules (dist): {response}")
    return []


def arrange_rule_sets_dist(model: MyModel, query: str, suggestions_str: str,
                            rule_names: t.List[str], rules_str: str,
                            distribution_guidance: str,
                            rule_stat_priors: str = '') -> t.List[t.List[str]]:
    rule_groups_dict = get_rule_sets(rule_names)
    if len(rule_groups_dict) == 0:
        return []
    rule_sets_str = '\n\n'.join([f'### {op} Operator Rules: ' + str(group).replace("'", '"')
                                  for op, group in rule_groups_dict.items()])
    dist_section = distribution_guidance or '(none)'
    if rule_stat_priors:
        dist_section = dist_section + '\n\n' + rule_stat_priors
    messages = [
        {'role': 'system', 'content': model.DIST_ARRANGE_RULE_SETS_SYS_PROMPT},
        {'role': 'user', 'content': model.DIST_ARRANGE_RULE_SETS_USER_PROMPT.format(
            sql=query, suggestions=suggestions_str, rule_sets=rule_sets_str, rules=rules_str,
            distribution_guidance=dist_section)},
    ]
    arranged_rule_sets: t.List[t.List[str]] = []
    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        res = re.findall(r'```python\s*(.+?)\s*```', response, re.I | re.DOTALL)
        for python_content in res:
            try:
                cur_rule_names = eval(python_content)
                arranged_rule_sets.append([r.upper() for r in cur_rule_names if r.upper() in rule_names])
            except Exception:
                pass
        if len(arranged_rule_sets) > 0:
            _log_llm_call('arrange_rule_sets_dist', attempt, response, True,
                          extra=f"rule_sets={len(arranged_rule_sets)}")
            return arranged_rule_sets
        _log_llm_call('arrange_rule_sets_dist', attempt, response, False)
    logging.warning(f"Failed to arrange selected rule sets (dist): {response}")
    return []


def _log_rdg_compliance(fn_name: str, final_seq: t.List[str], rdg_edges: t.Optional[list],
                         response: str) -> None:
    """RDG skeleton이 주어졌을 때, 최종 순서가 그 골격(선행 제약)을 실제로 따랐는지
    코드로 객관적으로 판정해서 한 줄로 남긴다. 벗어난 경우, LLM이 그 이유를 reasoning에
    설명하도록 RDG_GUIDANCE_SYS_NOTE가 지시해뒀으므로 그 텍스트도 같이 기록한다
    (LLM 자기보고에만 의존하지 않고, followed 여부 자체는 항상 코드로 재검증함).
    """
    if not rdg_edges:
        return
    followed, violations = check_skeleton_followed(final_seq, rdg_edges)
    log = {'followed': followed, 'violations': violations, 'n_edges_checked': len(rdg_edges)}
    if not followed:
        log['llm_reasoning'] = extract_reasoning_text(response)
    logging.info(f'{PILOT_LOG_PREFIX} RDG Skeleton Compliance [{fn_name}]: {log}')


def arrange_rules_dist(model: MyModel, query: str, suggestions_str: str,
                        selected_rules: t.List[t.Dict[str, str]], distribution_guidance: str,
                        rule_dependency_constraints: str = '',
                        rdg_edges: t.Optional[list] = None,
                        rule_stat_priors: str = '') -> t.List[str]:
    rules_str = '\n\n'.join([f'### Rule {r["name"]}:\n"""{r["rewrite"]}"""' for r in selected_rules])
    rule_names = [r['name'] for r in selected_rules]
    arranged_rule_sets = arrange_rule_sets_dist(
        model, query, suggestions_str, rule_names, rules_str, distribution_guidance, rule_stat_priors
    )
    logging.info(f'Arranged Rule Sets (dist): {arranged_rule_sets}')
    arranged_rule_sets_str = '\n\n'.join([f'### Rule Sequence {i+1}: ' + str(seq).replace("'", '"')
                                           for i, seq in enumerate(arranged_rule_sets)])
    dist_section = distribution_guidance or '(none)'
    if rule_stat_priors:
        dist_section = dist_section + '\n\n' + rule_stat_priors
    messages = [
        {'role': 'system', 'content': model.DIST_ARRANGE_RULES_SYS_PROMPT},
        {'role': 'user', 'content': model.DIST_ARRANGE_RULES_USER_PROMPT.format(
            sql=query, suggestions=suggestions_str, rules=rules_str, rule_sequences=arranged_rule_sets_str,
            distribution_guidance=dist_section,
            rule_dependency_constraints=rule_dependency_constraints or '(none)')},
    ]
    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        prefix, suffix = '```python', '```'
        start_idx, end_idx = response.find(prefix), response.rfind(suffix)
        if start_idx != -1 and end_idx != -1:
            list_str = response[start_idx + len(prefix): end_idx].strip()
            try:
                res = eval(list_str)
                format_res = [r.upper() for r in res if r.upper() in rule_names]
                if len(format_res) * 2 > len(selected_rules):
                    _log_llm_call('arrange_rules_dist', attempt, response, True,
                                  extra=f"arranged={len(format_res)}/{len(selected_rules)}")
                    _log_rdg_compliance('arrange_rules_dist', format_res, rdg_edges, response)
                    return format_res
            except Exception:
                pass
        _log_llm_call('arrange_rules_dist', attempt, response, False)
    logging.warning(f"Failed to arrange selected rules (dist): {response}")
    return rule_names


# ── Diverse Candidate Generation ─────────────────────────────────────────────

_DIVERSE_CANDIDATE_SYS_PROMPT = (
    "You are an expert in SQL query optimization and Apache Calcite rewrite rules. "
    "Given that a previous rule sequence failed to improve query cost, generate diverse "
    "alternative rule sequences. Each candidate must follow a distinct optimization "
    "strategy to maximize the probability that at least one candidate improves performance. "
    "Output only valid JSON as specified."
)

_DIVERSE_CANDIDATE_USER_PROMPT = """\
Query Rewrite Suggestions:
{suggestions}

Candidate Rules:
{rules}

Data-Driven Optimization Guidance:
{distribution_guidance}

RDG Dependency Skeleton:
{rule_dependency_constraints}

Previous Failed Candidate:
{failed_sequence}

Previous Used Rules:
{used_rules}

Failure Signal:
input_cost: {input_cost}
output_cost: {output_cost}
reason: output cost was worse than the original query, or EXPLAIN failed.

Generate {k} diverse alternative rule sequences.

Requirements:
- Each candidate must use only Candidate Rules listed above.
- Each candidate must respect hard dependency constraints when possible.
- Do not repeat the failed sequence.
- Make candidates strategically different:
  one conservative sequence (only rules strongly supported by suggestions)
  one filter/predicate-pushdown-first sequence
  one simplification-first sequence (reduce/remove/merge rules)
  one join/project exploration sequence (only if justified by the query structure)
- Avoid including rules that were unused in the failed candidate unless needed to unlock another rule.
- Prefer shorter sequences when uncertain.
- Do not use statistical priors to suppress exploration.

Output JSON:
{{
  "candidates": [
    {{
      "name": "conservative",
      "rules": ["RULE_NAME_1", "RULE_NAME_2"],
      "rationale": "short reason"
    }},
    {{
      "name": "filter_first",
      "rules": ["RULE_NAME_A"],
      "rationale": "short reason"
    }}
  ]
}}"""


def generate_diverse_candidates_dist(
    model: MyModel,
    query: str,
    suggestions_str: str,
    relevant_rules: t.List[t.Dict[str, str]],
    distribution_guidance: str,
    rule_dependency_constraints: str,
    failed_sequence: t.List[str],
    used_rules: t.List[str],
    input_cost: float,
    output_cost: float,
    k: int,
    rdg_edges: t.Optional[list] = None,
) -> t.List[t.Dict]:
    """
    이전 Calcite 실행이 cost 개선에 실패했을 때, K개의 다양한 대안 후보 규칙 시퀀스를
    LLM에게 생성하도록 요청한다.

    stat_prior는 이 함수에 주입하지 않는다 — 탐색 공간을 축소하지 않기 위해.
    stat_prior는 사후 risk scoring(_score_candidate_risk)에서만 사용된다.

    Returns: list of {'name': str, 'rules': [str, ...], 'rationale': str}
    """
    valid_rule_names = {r['name'].upper() for r in relevant_rules}
    rules_str = '\n'.join([f'- {r["name"]}' for r in relevant_rules])

    messages = [
        {'role': 'system', 'content': _DIVERSE_CANDIDATE_SYS_PROMPT},
        {'role': 'user', 'content': _DIVERSE_CANDIDATE_USER_PROMPT.format(
            suggestions=suggestions_str or '(none)',
            rules=rules_str,
            distribution_guidance=distribution_guidance or '(none)',
            rule_dependency_constraints=rule_dependency_constraints or '(none)',
            failed_sequence=str(failed_sequence),
            used_rules=str(used_rules),
            input_cost=round(input_cost, 4),
            output_cost=round(output_cost, 4),
            k=k,
        )},
    ]

    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        # JSON 블록 파싱 시도
        m = re.search(r'```json\s*(.+?)\s*```', response, re.I | re.DOTALL)
        raw_json = m.group(1) if m else response.strip()
        try:
            parsed = json.loads(raw_json)
            candidates = parsed.get('candidates', [])
            # 각 후보의 rules를 valid_rule_names로 필터링
            cleaned = []
            for cand in candidates:
                valid_seq = [r.upper() for r in cand.get('rules', []) if r.upper() in valid_rule_names]
                if valid_seq:
                    cleaned.append({
                        'name': cand.get('name', f'candidate_{len(cleaned)+1}'),
                        'rules': valid_seq,
                        'rationale': cand.get('rationale', ''),
                    })
            _log_llm_call(
                'generate_diverse_candidates_dist', attempt, response,
                parsed_ok=len(cleaned) > 0,
                extra=f'k_requested={k} k_valid={len(cleaned)}',
            )
            if cleaned:
                return cleaned
        except Exception:
            _log_llm_call('generate_diverse_candidates_dist', attempt, response, parsed_ok=False)

    logging.warning(f'{PILOT_LOG_PREFIX} Diverse candidate 생성 실패: {response[:200]}')
    return []


def rearrange_rules_dist(model: MyModel, query: str, suggestions_str: str,
                          selected_rules: t.List[t.Dict[str, str]], arranged_rules: t.List[str],
                          used_rules: t.List[str], distribution_guidance: str,
                          rule_dependency_constraints: str = '',
                          rdg_edges: t.Optional[list] = None,
                          rule_stat_priors: str = '') -> t.List[str]:
    rules_str = '\n\n'.join([f'### Rule {r["name"]}:\n"""{r["rewrite"]}"""' for r in selected_rules])
    rule_names = [r['name'] for r in selected_rules]
    unused_rules = [r for r in arranged_rules if r not in used_rules]
    dist_section = distribution_guidance or '(none)'
    if rule_stat_priors:
        dist_section = dist_section + '\n\n' + rule_stat_priors
    messages = [
        {'role': 'system', 'content': model.DIST_REARRANGE_RULES_SYS_PROMPT},
        {'role': 'user', 'content': model.DIST_REARRANGE_RULES_USER_PROMPT.format(
            sql=query, suggestions=suggestions_str, rules=rules_str,
            arranged_rules=str(arranged_rules).replace("'", '"'),
            used_rules=str(used_rules).replace("'", '"'),
            unused_rules=str(unused_rules).replace("'", '"'),
            distribution_guidance=dist_section,
            rule_dependency_constraints=rule_dependency_constraints or '(none)')},
    ]
    response = ''
    for attempt in range(1, 3):
        response = chat(messages, getattr(model, 'fast_llm', None))
        prefix, suffix = '```python', '```'
        start_idx, end_idx = response.find(prefix), response.rfind(suffix)
        if start_idx != -1 and end_idx != -1:
            list_str = response[start_idx + len(prefix): end_idx].strip()
            try:
                res = eval(list_str)
                format_res = [r.upper() for r in res if r.upper() in rule_names]
                if len(format_res) * 2 > len(selected_rules):
                    _log_llm_call('rearrange_rules_dist', attempt, response, True,
                                  extra=f"rearranged={len(format_res)}/{len(selected_rules)}")
                    _log_rdg_compliance('rearrange_rules_dist', format_res, rdg_edges, response)
                    return format_res
            except Exception:
                pass
        _log_llm_call('rearrange_rules_dist', attempt, response, False)
    logging.warning(f"Failed to re-arrange selected rules (dist): {response}")
    return rule_names
