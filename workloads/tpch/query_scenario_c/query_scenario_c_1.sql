-- Scenario C-1: Correlated Aggregate in WHERE (semi-join form)
-- Correlated aggregate subquery used as a comparison condition.
-- Distribution-aware hint: clerk is non-selective (45%) in skewed DB.
-- Optimize by computing aggregate once (GROUP BY) then filtering.
-- Rule candidates: FILTER_SUB_QUERY_TO_CORRELATE, FILTER_INTO_JOIN, AGGREGATE_PROJECT_MERGE

SELECT o.o_orderkey,
       o.o_clerk,
       o.o_totalprice
FROM orders o
WHERE o.o_clerk = 'Clerk#000000001'
  AND o.o_totalprice > (
      SELECT AVG(o2.o_totalprice)
      FROM orders o2
      WHERE o2.o_clerk = o.o_clerk
  )
ORDER BY o.o_totalprice DESC;
