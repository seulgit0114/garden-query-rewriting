-- Scenario F-1: Correlated subquery form — what JOIN_TO_CORRELATE would produce (BAD for skewed)
-- Input in correlated form; pipeline should recognize and AVOID this pattern on skewed data.
-- Skewed: outer orders = 45,033 rows → each correlated subquery executes 45,033 times.
--   PG SubPlan: Seq Scan lineitem (400K) × 45,033 loops = ~18 minutes.
-- Uniform: outer orders = 100 rows → 100 subplan executions → ~3 seconds (manageable).
-- Distaware: guidance says "clerk=45% → 45K outer rows → decorrelate via PROJECT_SUB_QUERY_TO_CORRELATE"
-- Note: This is the WORST case form. Baseline should also decorrelate, but distaware confirms via hint.

SELECT o.o_orderkey,
       o.o_clerk,
       o.o_totalprice,
       (SELECT COUNT(l.l_linenumber)
        FROM lineitem l
        WHERE l.l_orderkey = o.o_orderkey)  AS lineitem_count,
       (SELECT SUM(l.l_extendedprice)
        FROM lineitem l
        WHERE l.l_orderkey = o.o_orderkey)  AS total_revenue
FROM orders o
WHERE o.o_clerk = 'Clerk#000000001'
ORDER BY o.o_totalprice DESC;
