-- Scenario F-0: LEFT JOIN + GROUP BY form — candidate where JOIN_TO_CORRELATE would be HARMFUL
-- Rule: converts the join to a correlated subquery (executed once per outer row).
-- CORRECT behavior: DO NOT apply JOIN_TO_CORRELATE when outer cardinality is high.
-- Skewed (clerk=45%, ~45,033 outer orders): converting to correlated → 45,033 subplan executions.
--   Each execution = full seq scan of lineitem (400K rows) → ~18 minutes total.
-- Uniform (clerk=0.1%, ~100 outer orders): correlated → 100 subplan executions → ~3 seconds.
-- Distaware: detects clerk=45% → 45K outer rows → BLOCK JOIN_TO_CORRELATE → fast hash join.
-- Baseline risk: without distribution info, may apply JOIN_TO_CORRELATE → catastrophic on skewed.

SELECT o.o_orderkey,
       o.o_clerk,
       o.o_totalprice,
       COUNT(l.l_linenumber)      AS lineitem_count,
       SUM(l.l_extendedprice)     AS total_revenue
FROM orders o
LEFT JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_clerk = 'Clerk#000000001'
GROUP BY o.o_orderkey, o.o_clerk, o.o_totalprice
ORDER BY o.o_totalprice DESC;
