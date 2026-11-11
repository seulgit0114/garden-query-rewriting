-- Scenario B-0: Cardinality 기반 Join Order 문제
-- Skewed DB: o_clerk='Clerk#000000001' → 45% 매칭 (non-selective)
-- Uniform DB: o_clerk='Clerk#000000001' → 0.1% 매칭 (selective)
-- p.p_brand='Brand#13' → 1% of part (selective in both)
--
-- 최적화 기회:
--   FILTER_INTO_JOIN: brand 필터를 part 조인 전에 밀어 넣기 (skewed에서 clerk보다 brand가 훨씬 selective)
--   JOIN_COMMUTE: part → lineitem → orders 순서로 join 재정렬
--
-- Baseline (분포 정보 없음): clerk 필터가 selective하다고 가정 → orders 먼저 → skewed에서 44K row 중간 결과
-- Distaware (분포 정보 있음): clerk=45% non-selective → brand 필터 우선 → 작은 중간 결과
SELECT
    o.o_orderkey,
    o.o_clerk,
    l.l_extendedprice,
    p.p_brand
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
JOIN part p     ON p.p_partkey  = l.l_partkey
WHERE o.o_clerk = 'Clerk#000000001'
  AND p.p_brand = 'Brand#13'
ORDER BY l.l_extendedprice DESC;
