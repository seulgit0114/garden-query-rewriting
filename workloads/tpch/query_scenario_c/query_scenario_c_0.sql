-- Scenario C-0: Correlated Aggregate Subquery in SELECT clause
-- Problem: PostgreSQL executes the subquery once per outer row (SubPlan per-row).
-- Skewed DB: clerk='Clerk#000000001' matches ~45,033 orders → subquery runs 45,033 times.
-- Uniform DB: clerk='Clerk#000000001' matches ~100 orders   → subquery runs 100 times.
-- Optimization: Rewrite as LEFT JOIN + GROUP BY → lineitem scanned only once.
-- Rule candidates: PROJECT_SUB_QUERY_TO_CORRELATE, FILTER_INTO_JOIN, AGGREGATE_PROJECT_MERGE

SELECT o.o_orderkey,
       o.o_clerk,
       o.o_totalprice,
       (SELECT SUM(l.l_extendedprice)
        FROM lineitem l
        WHERE l.l_orderkey = o.o_orderkey) AS total_line_revenue
FROM orders o
WHERE o.o_clerk = 'Clerk#000000001'
ORDER BY o.o_totalprice DESC;
