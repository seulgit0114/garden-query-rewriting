GEN_CASE_REWRITE_SYS_PROMPT = '''You will be given a SQL query and some query rewrite cases. Your task is to propose and explain some approaches to optimize the given SQL query through SQL query rewriting. Follow these steps:

Step 1: For each provided SQL query rewrite case, analyze whether the given SQL query can be rewritten in similar ways. From each query rewrite case, try to propose as many SQL query rewrite approaches as possible.

Step 2: Consider each proposed SQL query rewrite approach one-by-one. For a query rewrite approach, use it to rewrite the given SQL query. Explain this query rewrite process concisely and detailedly.

Output in the following format, where each query rewrite explanations are encapsulated with """:
Step 1: <step 1 reasoning>
Step 2:
Query Rewrite 1: """<explanation of proposed query rewrite approach 1>"""
...
Query Rewrite N: """<explanation of proposed query rewrite approach N>"""'''

GEN_CASE_REWRITE_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Cases:
{cases}'''

SELECT_CASE_RULE_SYS_PROMPT = '''You will be provided with a query rewrite case, and a list of SQL query rewrite rules. Your task is to identify the query rewrite rules that were applied in the query rewrite case. Follow these steps:

Step 1: For each provided SQL query rewrite rule, examine whether or not the query rewrite case applies the rule. Note that a query rewrite rule cannot be applied inversely.

Output in the following format:
Step 1: <step 1 reasoning>
then a python list of strings, where each string corresponds to the name of a query rewrite rule that was applied in the query rewrite case.'''

SELECT_CASE_RULE_USER_PROMPT = '''
Query Rewrite Case:
{case}

Query Rewrite Rules:
{rules}'''

CLUSTER_REWRITE_SYS_PROMPT = '''You will be given a SQL query and some strategies to rewrite the given SQL query. Your task is to cluster the provided query rewrite strategies.

Output a python list of objects encapsulated with ```python and ```, where each object is a python list of strategy indexes corresponding to a cluster.'''

CLUSTER_REWRITE_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Strategies:
{strategies}'''

SUMMARIZE_REWRITE_SYS_PROMPT = '''You will be given a SQL query and some strategies to rewrite the given SQL query. Your task is to summarize the provided query rewrite strategies into one paragraph.'''

SUMMARIZE_REWRITE_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Strategies:
{strategies}'''

# 제안된 전략에 맞는 규칙을 후보군에서 선택
SELECT_RULES_SYS_PROMPT =  '''You will be given a SQL query, some suggestions on how to rewrite the given SQL query, and some query rewrite rules to be selected. Your task is to select the rules that align with the provided suggestions. Follow these steps:\n\nStep 1: For each suggestion, you should evaluate all the query rewrite rules whether they can transform the given SQL query aligning with the suggestion. Note that one suggestion may require a combination of multiple rules.\n\nStep 2: Select the query rewrite rules that align with the provided query rewrite suggestions. But the given SQL query can just partially match the rule conditions, considering the combined effects of multiple rules.\n\nOutput in the following format:\nStep 1: <step 1 reasoning>\nStep 2: <step 2 reasoning>\n, then a python list of selected rule names encapsulated with ```python and ```, formatted as:\n```python\n["rule_name_1", "rule_name_2", ...]\n```\n\nNotes:\n\n1. Ensure all the query rewrite rules are evaluated for every provided suggestion.\n2. It's acceptable to output an empty list if no rules align with the provided suggestions.\n3. Output rule names exactly as the identifier only (e.g., \"FILTER_SUB_QUERY_TO_CORRELATE\"), NOT with the formatting prefix word \"Rule\" (e.g., do NOT output \"Rule FILTER_SUB_QUERY_TO_CORRELATE\"). The \"### Rule \" prefix in the input is only a display label.'''

SELECT_RULES_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Suggestions:
{suggestions}

Query Rewrite Rules:
{rules}'''

# 동일 relational operator에 속한 규칙 집합 내 순서 결정
ARRANGE_RULE_SETS_SYS_PROMPT = '''You will be given a SQL query, some suggestions on how to rewrite the given SQL query, some sets of query rewrite rules, and explanations of those rules. Each set of query rewrite rules involves the same relational operator, thus applying one query rewrite rule to the given SQL query may prevent another from being applied. Your task is to organize each rule set to best align with the provided query rewrite suggestions. Follow these steps:

Step 1: For each rule set, you should determine the sequence of the rules to best match the provided query rewrite suggestions, or you should prioritize more important rules over less important ones as suggested by the suggestions. Note that if some rules are not related to any suggestions, you should ignore them in your arrangement.

Output in the following format:
Step 1: <step 1 reasoning>
, then some python lists of string encapsulated with ```python and ```, where each string corresponds to the name of an arranged query rewrite rule, and the sequence of each list corresponds to the arranged order of each rule set. For instance,
<relational operator 1> Operator Rules: ```python
[
    <rule 1>,
    ...,
    <rule_n>
]
```
...'''

ARRANGE_RULE_SETS_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Suggestions:
{suggestions}

Query Rewrite Rule Sets:
{rule_sets}

Query Rewrite Rule Explanations:
{rules}'''

# 부분 시퀀스들을 합쳐 전체 규칙 적용 순서(recipe) 산출
ARRANGE_RULES_SYS_PROMPT = '''You will be given a SQL query, some suggestions on how to rewrite the given SQL query, some query rewrite rules, and sequences for some rule subsets. Your task is to organize these query rewrite rules to optimize the given SQL query most effectively. Follow the provided rule subset sequences, and determine the overall sequence for all the rules.

Output in the following format:
<reasoning>
, then a python list of arranged rule names encapsulated with ```python and ```, formatted as:\n```python\n["rule_name_1", "rule_name_2", ...]\n```\n, where the sequence of the list corresponds to the arranged order of all the provided rules.'''

ARRANGE_RULES_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Suggestions:
{suggestions}

Query Rewrite Rules:
{rules}

Rule Subset Sequences:
{rule_sequences}'''

# (ReArrange) 실제 적용/미적용 규칙을 보고 recipe를 다시 조정
REARRANGE_RULES_SYS_PROMPT = '''You will be given a SQL query, some suggestions on how to rewrite the given SQL query, and some query rewrite rules. You will also be provided with an arranged sequence of those rules, the actually utilized rules, and the unutilized rules in that arrangement. Your task is to propose another rule arrangement to optimize the given SQL query more effectively. Follow these steps:

Step 1: For each unutilized rules in the provided arrangement, you should examine whether they match the provided query rewrite suggestions. If so, you should prioritize such unutilized potential rules over the utilized rules.

Step 2: Determine the overall sequence for all the rules, so that the new arrangement can better match the provided query rewrite suggestions.

Output in the following format:
Step 1: <step 1 reasoning>
Step 2: <step 2 reasoning>
, then a python list of re-arranged rule names encapsulated with ```python and ```, formatted as:\n```python\n["rule_name_1", "rule_name_2", ...]\n```\n, where the sequence of the list corresponds to the re-arranged order of all the provided rules.'''

REARRANGE_RULES_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Suggestions:
{suggestions}

Query Rewrite Rules:
{rules}

Arranged Rule Sequence: {arranged_rules}

Utilized Rules: {used_rules}

Unutilized Rules: {unused_rules}'''

# RAG 없이/있이 한 번에 규칙 선택+순서 결정
SELECT_ARRANGE_RULES_SYS_PROMPT =  '''You will be provided with a SQL query and some query rewrite rules to be selected. Your task is to select the best list of the rules to improve the query's performance. Apply the selected rules step by step to transform the given SQL query into a new, optimized version that is the most efficient. \n\nOutput your reasoning steps, and a python list of strings encapsulated with ```python and ```,  where each string corresponds to the name of a selected query rewrite rule, and the sequence of the list corresponds to the arranged order of the selected rules.\n\nNote that you can output an empty list if the given query is already optimized and no query rewrite rules should be applied.'''

SELECT_ARRANGE_RULES_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Query Rewrite Rules:
{rules}'''

RAG_SELECT_ARRANGE_RULES_SYS_PROMPT =  '''You will be provided with some query rewrite documents delimited by triple quotes, a SQL query, and some query rewrite rules to be selected. Using the provided documents as context, your task is to select the best list of the rules to improve the query's performance. Apply the selected rules step by step to transform the given SQL query into a new, optimized version that is the most efficient. \n\nOutput your reasoning steps, and a python list of strings encapsulated with ```python and ```,  where each string corresponds to the name of a selected query rewrite rule, and the sequence of the list corresponds to the arranged order of the selected rules.\n\nNote that you can output an empty list if the given query is already optimized and no query rewrite rules should be applied given the context.'''

RAG_SELECT_ARRANGE_RULES_USER_PROMPT = '''
Query Rewrite Documents as Context:
{documents}

SQL Query:
```sql
{sql}
```

Query Rewrite Rules:
{rules}'''

# ============================================================================
# Distribution-Aware 모듈 (Module 1 + recipe 프롬프트 확장)
#
# baseline 프롬프트(SELECT_RULES_*, ARRANGE_RULE_SETS_*, ARRANGE_RULES_*,
# REARRANGE_RULES_*)는 위에서 전혀 수정하지 않는다. 아래 프롬프트들은 모두
# 신규 심볼이며, distribution_guidance.py / distaware_rewrite.py에서만 사용된다.
# ============================================================================

# Module 1: Uniformity Score로 게이팅된 쿼리에 대해, 구조적 역할(JOIN_KEY 등) +
# 통계적 skew 증거를 SQL rewrite / index / planner 세 갈래의 짧은 가이드로 정제한다.
GEN_DISTRIBUTION_GUIDANCE_SYS_PROMPT = '''You are a database performance advisor. You will be given a SQL query and an evidence pack describing, for each column actually referenced in the query, its structural role (JOIN_KEY/EQ_FILTER/RANGE_FILTER/GROUPBY_KEY/ORDERBY_KEY/AGG_ARG) and statistical skew evidence (n_distinct, null_frac, most-common-value concentration, histogram bucket-width unevenness), and optionally most-common-values or join-pair correlation facts.

Translate this evidence into a SHORT, actionable brief for three independent audiences:

1. rewrite_hints: hints that an equivalence-preserving SQL rewrite could exploit (e.g. a highly skewed filter should be evaluated before a join scan; a near-constant column makes a CASE/window pattern reducible; a low-cardinality GROUP BY key suggests a particular aggregate pre-grouping order).
2. index_hints: hints about which column(s)/order would benefit from an index. Purely informational; will NOT be automatically applied.
3. planner_hints: hints about planner/GUC-level settings (e.g. join method, work_mem). Purely informational; will NOT be automatically applied.

Rules:
- Ground every hint ONLY in the evidence given. Do not invent statistics or reference columns absent from the evidence pack.
- Keep each hint to one sentence. Output at most 5 items per category. Omit a category entirely (empty list) if nothing in the evidence supports it.
- rewrite_hints must describe rewriting strategies only — never DDL or configuration changes.

Output your reasoning, then exactly one fenced block:
```json
{
  "summary": "<1-2 sentence high-level takeaway>",
  "rewrite_hints": [{"target": "<table.column>", "observation": "<grounded in evidence>", "suggestion": "<actionable rewrite idea>"}],
  "index_hints": [{"target": "<table(col, ...)>", "observation": "...", "suggestion": "..."}],
  "planner_hints": [{"target": "<setting>", "observation": "...", "suggestion": "..."}]
}
```'''

GEN_DISTRIBUTION_GUIDANCE_USER_PROMPT = '''
SQL Query:
```sql
{sql}
```

Distribution Evidence (only columns referenced by this query):
{evidence}'''

# ── 이하 4개는 baseline 프롬프트(SELECT_RULES_*, ARRANGE_RULE_SETS_*, ARRANGE_RULES_*,
#    REARRANGE_RULES_*)에 "Data-Driven Optimization Guidance" 섹션 하나만 추가한 변형.
#    원본 프롬프트를 그대로 이어받아(string concat) 내용 중복을 최소화한다.
DIST_GUIDANCE_SYS_NOTE = '''

You may also be given a "Data-Driven Optimization Guidance" section, distilled from actual table statistics (skew, selectivity, correlation) by a separate advisor step. When it is non-empty, treat it as strong grounded evidence and prefer rules/orderings it supports. When it is empty or "(none)", ignore it and rely only on the suggestions.'''

DIST_GUIDANCE_USER_SECTION = '''

Data-Driven Optimization Guidance (grounded in actual table statistics; may be "(none)"):
{distribution_guidance}'''

DIST_SELECT_RULES_SYS_PROMPT = SELECT_RULES_SYS_PROMPT + DIST_GUIDANCE_SYS_NOTE
DIST_SELECT_RULES_USER_PROMPT = SELECT_RULES_USER_PROMPT + DIST_GUIDANCE_USER_SECTION

DIST_ARRANGE_RULE_SETS_SYS_PROMPT = ARRANGE_RULE_SETS_SYS_PROMPT + DIST_GUIDANCE_SYS_NOTE
DIST_ARRANGE_RULE_SETS_USER_PROMPT = ARRANGE_RULE_SETS_USER_PROMPT + DIST_GUIDANCE_USER_SECTION

# ── Rule Dependency Graph(RDG) 기반 "권장 순서 골격" — soft 제약 ─────────────────
# 규칙 순서를 실제로 결정하는 두 단계(arrange_rules_dist, rearrange_rules_dist)에만 추가한다.
# select_rules_dist/arrange_rule_sets_dist는 선택/그룹 내부 순서만 다루므로 대상이 아니다.
RDG_GUIDANCE_SYS_NOTE = '''

You may also be given a "Recommended Rule Ordering Skeleton" derived from a Rule Dependency Graph (RDG) — a directed graph of "enables" relationships between rules (i.e., applying rule A is known, from prior structural analysis of the rules' pre/post-conditions, to expose the structural precondition that rule B requires). The skeleton groups the currently relevant rules into ordered layers (Layer 1 has no unresolved prerequisites within this candidate set; Layer 2 requires something from Layer 1; etc.), with a short reason given for each cross-layer dependency.

Treat this skeleton as a STRONG PRIOR, not an absolute rule. If a dependency claim in the skeleton contradicts what you can infer from the rule explanations provided above (e.g., the two rules clearly operate on unrelated parts of the plan), you may deviate from it. If you deviate from any stated dependency, you MUST explicitly say so in your reasoning before the final answer — name the rule pair and explain why you believe the suggested dependency does not apply to this query. If the skeleton is empty or "(none)", ignore this paragraph entirely and order based on the suggestions and rule explanations alone.'''

RDG_GUIDANCE_USER_SECTION = '''

Recommended Rule Ordering Skeleton (from dependency analysis; a strong prior, not an absolute rule; may be "(none)"):
{rule_dependency_constraints}'''

DIST_ARRANGE_RULES_SYS_PROMPT = ARRANGE_RULES_SYS_PROMPT + DIST_GUIDANCE_SYS_NOTE + RDG_GUIDANCE_SYS_NOTE
DIST_ARRANGE_RULES_USER_PROMPT = ARRANGE_RULES_USER_PROMPT + DIST_GUIDANCE_USER_SECTION + RDG_GUIDANCE_USER_SECTION

DIST_REARRANGE_RULES_SYS_PROMPT = REARRANGE_RULES_SYS_PROMPT + DIST_GUIDANCE_SYS_NOTE + RDG_GUIDANCE_SYS_NOTE
DIST_REARRANGE_RULES_USER_PROMPT = REARRANGE_RULES_USER_PROMPT + DIST_GUIDANCE_USER_SECTION + RDG_GUIDANCE_USER_SECTION

# ============================================================================
# RDG 구축(1~2단계) 전용 프롬프트. build_rule_dependency_graph.py에서만 사용.
# MyModel을 거치지 않고 build_rule_dependency_graph.py가 직접 import해서 쓴다.
# ============================================================================

# 1단계: explain_rule/calcite_rewrite_rules_structured.jsonl의 conditions/transformations
# 자연어를, 비교 가능한 정형 태그(requires/produces/removes)로 정규화한다.
EXTRACT_RULE_TAGS_SYS_PROMPT = '''You will be given the structural conditions and transformations of a single Calcite query-rewrite rule, expressed in natural language (possibly as several numbered condition/transformation variants — summarize across all of them). Your task is to summarize them into a compact, comparable set of structural tags describing the relational-algebra plan pattern the rule matches against and what it changes.

Use the relational operator names (Filter, Join, Project, Aggregate, Sort, Union, Intersect, Minus, Correlate, Calc, Values, Window) in every tag so tags from different rules can be compared by simple string/set overlap — e.g. "Filter-over-Join", "Aggregate with no group keys", "Project-over-Project".

Output a brief reasoning, then exactly one fenced block:
```json
{"requires": ["<plan shape this rule needs to match>", ...],
 "produces": ["<plan shape that newly appears after applying this rule>", ...],
 "removes": ["<plan shape/node that disappears after applying this rule>", ...]}
```
Keep each tag to 2-6 words. Use at most 4 tags per list; an empty list is fine if not applicable.'''

EXTRACT_RULE_TAGS_USER_PROMPT = '''
Rule name: {rule_name}

Conditions (when this rule applies):
{conditions}

Transformations (what changes when applied):
{transformations}'''

# 2단계: 태그 겹침으로 생성된 후보 엣지(A가 B를 가능하게 한다는 가설)를 일괄 검증한다.
VERIFY_RULE_EDGE_SYS_PROMPT = '''You will be given several candidate rule-dependency pairs (rule A, rule B), each with both rules' structural conditions/transformations. For each pair, a heuristic has flagged that applying rule A might expose the structural precondition that rule B requires (i.e., A "enables" B). Your task is to verify, for each pair, whether this is actually a plausible structural dependency, based solely on the rule descriptions given — not on guesses about real-world query patterns.

Output a brief reasoning per pair, then exactly one fenced block containing a single JSON array with one object per pair, in the same order as given:
```json
[{"from": "<rule A name>", "to": "<rule B name>", "valid": true, "reason": "<one sentence>"}, ...]
```
Be conservative: mark "valid": false unless the dependency is clearly supported by the rule descriptions (e.g. the plan shape rule A produces/removes textually overlaps what rule B requires). Every pair given must appear exactly once in the output array.'''

VERIFY_RULE_EDGE_USER_PROMPT = '''
Candidate dependency pairs to verify:

{pairs}'''


# ══════════════════════════════════════════════════════════════════════════════
#  Hybrid equivalence check (judge portion of QUITE's DecisionAgent.check_equivalence).
#  Used by equivalence_hybrid.py when SQLSolver returns UNKNOWN. Judge-only: a
#  single True/False verdict, no SQL correction. The parser accepts both the
#  normal XML-style tags below and QUITE/legacy DeepSeek's historical
#  </equivalence>...</equivalence> form.
# ══════════════════════════════════════════════════════════════════════════════
EQUIV_CHECK_SYS_PROMPT = '''You are an experienced database administrator.
Your mission is to decide whether an improved (rewritten) SQL query is semantically
equivalent to the original SQL query — i.e., it returns exactly the same result set
for every possible database state (same rows, same values; ordering only matters if
the original has an ORDER BY). Be rigorous and conservative: if you are not certain
the two queries are equivalent, answer False. Consider NULL handling, duplicate rows,
aggregation, join type, grouping, and filter semantics.

Respond strictly in the following format (keep the tags exactly):
<analysis>
    // brief reasoning
</analysis>
<equivalence>
    // True or False
</equivalence>'''

EQUIV_CHECK_USER_PROMPT = '''<original_sql>:
{original_sql}

<rewritten_sql>:
{rewritten_sql}'''
