# Calcite HBDC log-mined v2 감사 보고서

## 목적

기존 `mvm_v2`와 `hbdc_calcite_synthetic25_v1`을 수정하지 않고, 지금까지
축적된 Calcite 실행 로그에서 cost estimation error를 판정할 수 있는 strict
equivalence + measured latency 지식을 별도 버전으로 구축했다.

## 롤백 지점

- 변경 전 커밋: `9232f35f`
- 기존 HBDC 및 과거 campaign snapshot: 변경 없음

## 수집 및 정제

- modern GARDEN 관측: full-scale/SF1/formal equivalence와 실제 crossover
  latency가 모두 존재하는 행만 수집
- historical exhaustive crossover: 2026-10-13에 Uniform/Zipf 양쪽에서
  반복 측정한 Calcite-rule 후보를 수집
- micro-equivalence 32건: 전부 제외
- SQL pair + ordered rule sequence + distribution + evidence partition 기준 중복 제거
- 반복 측정의 개선 방향 일치율 2/3 이상, 절대 median 개선율 2% 이상만 delta에 포함
- 현재 평가 family와 normalized original SQL hash는 runtime prior에서 모두 제외

## 결과

| 항목 | 개수 |
|---|---:|
| Raw strict observations | 457 |
| Deduplicated candidates | 257 |
| Modern Calcite-44 held-out candidates | 140 |
| Historical cross-family candidates | 94 |
| Final HBDC delta | 88 |
| Families | 38 |
| TP / FP / FN / TN | 53 / 7 / 23 / 5 |

주요 cost-error 패턴은 다음과 같다.

| Rule | Support | Cost errors | Error rate |
|---|---:|---:|---:|
| AGGREGATE_PROJECT_MERGE | 38 | 14 | 36.8% |
| PROJECT_REMOVE | 21 | 7 | 33.3% |
| FILTER_PROJECT_TRANSPOSE | 6 | 4 | 66.7% |
| AGGREGATE_EXPAND_DISTINCT_AGGREGATES | 17 | 3 | 17.6% |
| FILTER_REDUCE_EXPRESSIONS | 12 | 3 | 25.0% |

## Leakage-resistant shadow replay

Modern Calcite-44 strict 후보 128건을 평가할 때 평가 family와 동일 original
SQL hash를 history에서 제외했다.

| Policy | Correct | Accuracy | Change vs. current HBDC |
|---|---:|---:|---:|
| Cost-only | 121/128 | 94.53% | -- |
| Current HBDC delta | 121/128 | 94.53% | -- |
| Log-mined v2 | 122/128 | 95.31% | +1 correct, 0 harmful flips |

개선된 한 건은 Zipf의 `testEmptyMinus`에서 cost가 거절한
`FILTER_VALUES_MERGE` 후보를 HBDC가 복구한 사례다. 해당 쿼리는 매우 짧아
전체 latency 합계 차이는 사실상 0이다. 따라서 이 결과는 end-to-end 성능
향상 근거가 아니라 cost-error knowledge coverage 개선 근거로만 사용한다.

## 적용 상태

- canonical Calcite runner의 기본 delta를 v2로 변경했다.
- 현재 DB에 맞게 `uniform`/`zipf` delta만 결합한다.
- query-family뿐 아니라 normalized original SQL hash도 제외한다.
- frozen campaign snapshot과 이미 실행 중인 job에는 영향을 주지 않는다.
- 실제 논문용 end-to-end 수치는 별도의 새 rewrite + equivalence + crossover
  latency campaign이 완료되기 전까지 기존 결과로 유지한다.
