"""
rule_stat_prior.py

Rule-Statistic Prior Matrix를 이용해 후보 규칙에 benefit/risk prior를 부여한다.

이 모듈은 empirical truth가 아니라 symbolic cost-aware prior를 계산한다.
즉, rule semantics와 relational operator cost driver로 초기화하고,
실험 로그로 calibration할 수 있는 구조를 유지한다.

API:
  load_rule_stat_matrix()
  build_query_stat_vector(query, schema, db_args, pool=None)
  score_rules_by_statistics(rules, query_features, matrix, rdg_added_names=None)
  render_rule_stat_priors_for_prompt(rule_priors, max_items=12)
"""
import json
import os
import re
import logging
import typing as t

import sqlglot
from sqlglot import exp

from my_rewriter.database import DBArgs, Database

MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'knowledge-base', 'rule_stat_feature_matrix.json'
)

_MATRIX_CACHE: t.Dict[str, dict] = {}

# feature level thresholds
_LEVEL_HIGH   = 'high'
_LEVEL_MEDIUM = 'medium'
_LEVEL_LOW    = 'low'
_LEVEL_UNK    = 'unknown'

# scoring constants
SCORE_PREFER     =  0.5
SCORE_ALLOW      = -0.3
SCORE_SOFT_ALLOW = -0.8

# RDG-added rules get a risk penalty multiplier
RDG_RISK_MULTIPLIER = 1.3


def load_rule_stat_matrix(path: t.Optional[str] = None) -> dict:
    """
    Load rule-stat matrix.

    By default this loads knowledge-base/rule_stat_feature_matrix.json.
    For calibrated experiments, set RULE_STAT_MATRIX_PATH or pass path
    explicitly so runtime scoring uses the calibrated relevance values.
    """
    global _MATRIX_CACHE
    matrix_path = path or os.environ.get('RULE_STAT_MATRIX_PATH') or MATRIX_PATH
    matrix_path = os.path.abspath(matrix_path)

    if matrix_path in _MATRIX_CACHE:
        return _MATRIX_CACHE[matrix_path]
    if not os.path.exists(matrix_path):
        logging.warning(f'[StatPrior] rule_stat_feature_matrix.json not found at {matrix_path}. '
                        'Run build_rule_stat_matrix.py first.')
        return {}
    with open(matrix_path, 'r') as f:
        matrix = json.load(f)
    _MATRIX_CACHE[matrix_path] = matrix
    logging.info(f'[StatPrior] Loaded rule_stat_feature_matrix: {len(matrix)} rules from {matrix_path}')
    return matrix


# ── Query Statistic Vector ────────────────────────────────────────────────────

def _count_from_tables(query: str) -> int:
    m = re.search(
        r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
        query, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return 0
    from_text = m.group(1)
    depth, flat = 0, []
    for ch in from_text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0:
            flat.append(ch)
    flat_text = ''.join(flat)
    n_commas = flat_text.count(',')
    n_joins = len(re.findall(r'\bJOIN\b', flat_text, re.IGNORECASE))
    return n_commas + n_joins + 1


def _level(value: t.Optional[float], low_thresh: float, high_thresh: float) -> str:
    if value is None:
        return _LEVEL_UNK
    if value >= high_thresh:
        return _LEVEL_HIGH
    if value >= low_thresh:
        return _LEVEL_MEDIUM
    return _LEVEL_LOW


def build_query_stat_vector(
    query: str,
    schema: str,
    db_args: DBArgs,
    pool: t.Optional[dict] = None,
    uniformity_result=None,
) -> dict:
    """
    query-level statistic feature vector를 계산한다.
    정밀도보다 방향성이 중요하다 — high/medium/low level이면 충분.

    Parameters
    ----------
    uniformity_result : UniformityResult, optional
        이미 계산된 경우 재사용해서 DB 재조회를 피한다.
    """
    features: dict = {}

    # ── join_width ────────────────────────────────────────────────────────────
    jw = _count_from_tables(query)
    features['join_width'] = {
        'value': jw,
        'level': _LEVEL_HIGH if jw >= 8 else (_LEVEL_MEDIUM if jw >= 4 else _LEVEL_LOW),
    }

    # ── sort_limit_pressure ───────────────────────────────────────────────────
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE))
    has_limit    = bool(re.search(r'\bLIMIT\b',    query, re.IGNORECASE))
    features['sort_limit_pressure'] = {
        'value': int(has_order_by and has_limit),
        'level': _LEVEL_HIGH if (has_order_by and has_limit) else _LEVEL_LOW,
    }

    # ── literal/constant/set-operation structural signals ────────────────────
    has_values = bool(re.search(r'\bVALUES\s*\(', query, re.IGNORECASE))
    value_tuples = len(re.findall(r'\bVALUES\s*\(|,\s*\(', query, re.IGNORECASE)) if has_values else 0
    features['value_tuple_count'] = {
        'value': value_tuples,
        'level': _LEVEL_HIGH if has_values and value_tuples <= 20 else (
            _LEVEL_MEDIUM if has_values else _LEVEL_UNK
        ),
        'evidence': ['VALUES clause detected'] if has_values else [],
    }

    has_setop = bool(re.search(r'\b(UNION|INTERSECT|EXCEPT|MINUS)\b', query, re.IGNORECASE))
    features['setop_branch_cardinality'] = {
        'level': _LEVEL_MEDIUM if has_setop else _LEVEL_UNK,
        'evidence': ['set operation detected'] if has_setop else [],
    }

    has_distinct = bool(re.search(r'\bDISTINCT\b', query, re.IGNORECASE))
    features['dedup_key_ndv'] = {
        'level': _LEVEL_HIGH if has_distinct else _LEVEL_UNK,
        'evidence': ['DISTINCT detected'] if has_distinct else [],
    }

    impossible_filter = bool(re.search(
        r'\bWHERE\b[^;]*(?:\bFALSE\b|1\s*=\s*0|0\s*=\s*1|\bNULL\s+IS\s+NOT\s+NULL\b)',
        query, re.IGNORECASE | re.DOTALL,
    ))
    features['filter_contradiction'] = {
        'level': _LEVEL_HIGH if impossible_filter else _LEVEL_UNK,
        'evidence': ['obvious contradictory filter detected'] if impossible_filter else [],
    }
    features['empty_input_certainty'] = {
        'level': _LEVEL_HIGH if impossible_filter else _LEVEL_UNK,
        'evidence': ['empty input inferred from contradiction'] if impossible_filter else [],
    }

    constant_order_key = bool(re.search(
        r'\bORDER\s+BY\s+(?:\d+|\'[^\']*\'|"[^"]*")\b',
        query, re.IGNORECASE,
    ))
    features['constant_expression_ratio'] = {
        'level': _LEVEL_HIGH if constant_order_key else _LEVEL_UNK,
        'evidence': ['constant ORDER BY key detected'] if constant_order_key else [],
    }

    # ── distinct_arg_ndv (proxy: COUNT DISTINCT present) ─────────────────────
    has_distinct_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*DISTINCT\b', query, re.IGNORECASE))
    features['distinct_arg_ndv'] = {
        'level': _LEVEL_HIGH if has_distinct_agg else _LEVEL_UNK,
        'evidence': ['DISTINCT aggregate detected'] if has_distinct_agg else [],
    }

    # ── Features derived from UniformityResult (per_column) ──────────────────
    if uniformity_result is not None:
        per_col = uniformity_result.per_column  # {(table, col): {...}}

        # mcv_skew: proxy from skew_score (s_conc component)
        skew_scores = [v.get('skew_score', 0) or 0 for v in per_col.values()]
        avg_skew = sum(skew_scores) / len(skew_scores) if skew_scores else 0
        features['mcv_skew'] = {
            'value': round(avg_skew, 3),
            'level': _level(avg_skew, 0.3, 0.6),
        }

        # histogram_skew: use uniformity score complement
        us = uniformity_result.score
        hist_skew = round(1.0 - us, 3)
        features['histogram_skew'] = {
            'value': hist_skew,
            'level': _level(hist_skew, 0.2, 0.5),
        }

        # null_fraction: max null_frac across all columns
        null_fracs = [v.get('null_frac', 0) or 0 for v in per_col.values()]
        max_null = max(null_fracs) if null_fracs else 0
        features['null_fraction'] = {
            'value': round(max_null, 3),
            'level': _level(max_null, 0.1, 0.3),
        }

        # filter_reduction_power: approximate from eq/range filter columns with skew
        filter_roles = {'EQ_FILTER', 'RANGE_FILTER'}
        filter_skews = [
            v.get('skew_score', 0) or 0
            for v in per_col.values()
            if filter_roles.intersection(set(v.get('roles', [])))
        ]
        frp = max(filter_skews) if filter_skews else 0
        features['filter_reduction_power'] = {
            'value': round(frp, 3),
            'level': _level(frp, 0.2, 0.5),
            'evidence': [f'{len(filter_skews)} filter column(s) with skew'],
        }

        # group_key_ndv: from GROUPBY_KEY columns n_distinct (normalized proxy)
        gbk_ndvs = [
            v.get('n_distinct', 0) or 0
            for v in per_col.values()
            if 'GROUPBY_KEY' in v.get('roles', [])
        ]
        if gbk_ndvs:
            # use log-ratio as proxy; normalize: > 1000 = high
            max_gbk = max(gbk_ndvs)
            gbk_level = _LEVEL_HIGH if max_gbk > 1000 else (
                _LEVEL_MEDIUM if max_gbk > 100 else _LEVEL_LOW
            )
            features['group_key_ndv'] = {'value': int(max_gbk), 'level': gbk_level}
        else:
            features['group_key_ndv'] = {'level': _LEVEL_UNK}

        # join_key_ndv: from JOIN_KEY columns n_distinct
        jk_ndvs = [
            v.get('n_distinct', 0) or 0
            for v in per_col.values()
            if 'JOIN_KEY' in v.get('roles', [])
        ]
        if jk_ndvs:
            min_jk = min(jk_ndvs)  # worst-case join key: lowest NDV
            jk_level = _LEVEL_HIGH if min_jk > 1000 else (
                _LEVEL_MEDIUM if min_jk > 50 else _LEVEL_LOW
            )
            features['join_key_ndv'] = {'value': int(min_jk), 'level': jk_level}
        else:
            features['join_key_ndv'] = {'level': _LEVEL_UNK}

    else:
        # Fallback: SQL-only approximation (no DB stats available)
        features['mcv_skew']             = {'level': _LEVEL_UNK}
        features['histogram_skew']       = {'level': _LEVEL_UNK}
        features['null_fraction']        = {'level': _LEVEL_UNK}
        features['filter_reduction_power'] = {'level': _LEVEL_UNK}
        features['group_key_ndv']        = {'level': _LEVEL_UNK}
        features['join_key_ndv']         = {'level': _LEVEL_UNK}

    # ── join_correlation: from pool if available ─────────────────────────────
    if pool:
        pearson = pool.get('correlation', {}).get('pearson_join', {})
        if pearson:
            corr_vals = [abs(v) for v in pearson.values() if v is not None]
            avg_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0
            features['join_correlation'] = {
                'value': round(avg_corr, 3),
                'level': _level(avg_corr, 0.3, 0.7),
            }
        else:
            features['join_correlation'] = {'level': _LEVEL_UNK}
    else:
        features['join_correlation'] = {'level': _LEVEL_UNK}

    # input_cardinality: use join_width as proxy (more tables → larger combined input)
    ic_level = _LEVEL_HIGH if jw >= 8 else (_LEVEL_MEDIUM if jw >= 4 else _LEVEL_LOW)
    features['input_cardinality'] = {'level': ic_level, 'proxy': f'join_width={jw}'}

    return features


# ── Scoring ───────────────────────────────────────────────────────────────────

def _feature_active(feat_val: dict) -> bool:
    """feature가 'high' 또는 known medium 이면 active로 간주한다."""
    level = feat_val.get('level', _LEVEL_UNK)
    return level in (_LEVEL_HIGH, _LEVEL_MEDIUM)


def _feature_high(feat_val: dict) -> bool:
    return feat_val.get('level', '') == _LEVEL_HIGH


def score_rules_by_statistics(
    rules: t.List[t.Dict[str, str]],
    query_features: dict,
    matrix: dict,
    rdg_added_names: t.Optional[t.Set[str]] = None,
) -> t.List[dict]:
    """
    각 rule에 대해 benefit/risk score를 계산하고 decision을 부여한다.

    Returns list of:
      {
        name, stat_score, decision, benefits, risks, guidance
      }
    """
    if rdg_added_names is None:
        rdg_added_names = set()

    results = []
    for rule in rules:
        name = rule['name'] if isinstance(rule, dict) else rule
        rule_name = name.upper()

        if rule_name not in matrix:
            # 매트릭스에 없는 규칙은 neutral allow
            results.append({
                'name': rule_name,
                'stat_score': 0.0,
                'decision': 'allow',
                'benefits': [],
                'risks': [],
                'guidance': f'{rule_name}: no prior in matrix, allowed by default.',
                'is_rdg_added': rule_name in rdg_added_names,
            })
            continue

        entry = matrix[rule_name]
        feat_specs = entry.get('features', {})

        score = 0.0
        benefits = []
        risks = []

        for feat_name, spec in feat_specs.items():
            direction = spec.get('direction', 'conditional')
            relevance = spec.get('relevance', 0.5)
            condition_text = spec.get('condition', '')

            qf = query_features.get(feat_name, {'level': _LEVEL_UNK})
            q_level = qf.get('level', _LEVEL_UNK)

            if q_level == _LEVEL_UNK:
                continue  # no evidence, skip

            if direction == 'benefit':
                if _feature_active(qf):
                    contrib = relevance * (1.5 if _feature_high(qf) else 1.0)
                    score += contrib
                    benefits.append(f'{feat_name}={q_level} (+{contrib:.2f})')
            elif direction == 'risk':
                if _feature_active(qf):
                    contrib = relevance * (1.5 if _feature_high(qf) else 1.0)
                    score -= contrib
                    risks.append(f'{feat_name}={q_level} (-{contrib:.2f})')
            elif direction == 'conditional':
                if _feature_high(qf):
                    # High conditional: treat as mild risk (uncertainty)
                    score -= relevance * 0.3
                    risks.append(f'{feat_name}={q_level} (conditional, -{relevance*0.3:.2f})')
                elif _feature_active(qf):
                    score += relevance * 0.3
                    benefits.append(f'{feat_name}={q_level} (conditional, +{relevance*0.3:.2f})')

        # RDG-added rules get risk penalty multiplier on the negative portion
        is_rdg = rule_name in rdg_added_names
        if is_rdg and score < 0:
            score *= RDG_RISK_MULTIPLIER

        # Normalize to [-1, 1] range (soft clip)
        score = max(-2.0, min(2.0, score))

        # Decision
        if score >= SCORE_PREFER:
            decision = 'prefer'
        elif score >= SCORE_ALLOW:
            decision = 'allow'
        elif score >= SCORE_SOFT_ALLOW:
            decision = 'soft_allow'
        else:
            decision = 'suppress'

        # Build guidance string
        guidance_parts = []
        if decision == 'prefer':
            guidance_parts.append(f'Prefer {rule_name}')
        elif decision == 'suppress':
            guidance_parts.append(f'Suppress {rule_name}')
        elif decision == 'soft_allow':
            guidance_parts.append(f'Soft-allow {rule_name}')
        else:
            guidance_parts.append(f'Allow {rule_name}')

        reasons = []
        if benefits:
            reasons.append(f'benefits=[{", ".join(benefits[:2])}]')
        if risks:
            reasons.append(f'risks=[{", ".join(risks[:2])}]')
        if is_rdg:
            reasons.append('RDG-added (higher risk weight)')
        if reasons:
            guidance_parts.append(': ' + '; '.join(reasons))

        results.append({
            'name': rule_name,
            'stat_score': round(score, 3),
            'decision': decision,
            'benefits': benefits,
            'risks': risks,
            'guidance': ''.join(guidance_parts),
            'is_rdg_added': is_rdg,
            'confidence': entry.get('confidence', 'low'),
        })

    return results


# ── Prompt Rendering ──────────────────────────────────────────────────────────

def render_rule_stat_priors_for_prompt(
    rule_priors: t.List[dict],
    max_items: int = 12,
) -> str:
    """
    LLM prompt에 주입할 짧은 Statistic-Aware Rule Priors 텍스트를 생성한다.
    suppress/prefer 를 먼저, neutral allow를 뒤에 배치해 LLM attention을 유도한다.
    """
    if not rule_priors:
        return ''

    # Sort: suppress first, then soft_allow, then prefer, then allow
    order = {'suppress': 0, 'soft_allow': 1, 'prefer': 2, 'allow': 3}
    sorted_priors = sorted(rule_priors, key=lambda x: order.get(x['decision'], 4))

    lines = ['Statistic-Aware Rule Priors (symbolic cost prior, not ground truth):']
    for p in sorted_priors[:max_items]:
        lines.append(f"  - {p['guidance']}")

    return '\n'.join(lines)
