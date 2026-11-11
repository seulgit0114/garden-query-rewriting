-- Scenario B-1: Aggregation + Cardinality Join Order (더 많은 규칙 적용 기회)
-- AggregateJoinTransposeRule: GROUP BY를 JOIN 전으로 이동 가능
-- FilterIntoJoinRule: brand/clerk 필터 pushdown
-- Skewed: clerk non-selective (45%) → brand 기반 최적화가 핵심
SELECT
    p.p_brand,
    COUNT(*)                          AS order_cnt,
    SUM(l.l_extendedprice)           AS total_revenue,
    AVG(l.l_discount)                AS avg_discount
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
JOIN part p     ON p.p_partkey  = l.l_partkey
WHERE o.o_clerk = 'Clerk#000000001'
  AND p.p_brand = 'Brand#13'
GROUP BY p.p_brand;
