-- Scenario D-1: EXISTS semi-join form (what JOIN_ADD_REDUNDANT_SEMI_JOIN produces)
-- Pre-filters lineitem by existence of matching order before the join.
-- This is the "after rule" form that the Calcite rule would generate.
-- Skewed: EXISTS set = 45K orders → lineitem 400K→180K, modest reduction → overhead > benefit.
-- Uniform: EXISTS set = 100 orders → lineitem 400K→400 rows, 1000× reduction → much faster.

SELECT l.l_orderkey,
       l.l_extendedprice,
       l.l_quantity,
       o.o_clerk,
       o.o_totalprice
FROM lineitem l
JOIN orders o ON o.o_orderkey = l.l_orderkey
WHERE o.o_clerk = 'Clerk#000000001'
  AND l.l_orderkey IN (
      SELECT o2.o_orderkey
      FROM orders o2
      WHERE o2.o_clerk = 'Clerk#000000001'
  )
ORDER BY l.l_extendedprice DESC;
