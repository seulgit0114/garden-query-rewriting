-- Scenario E-1: Subquery with LIMIT before JOIN (what SORT_JOIN_TRANSPOSE produces)
-- Explicitly forces PG to apply LIMIT on orders before joining with lineitem.
-- PG cannot push the outer LIMIT back into the subquery (subquery boundary is opaque).
-- Skewed: subquery returns top 10 orders → outer join processes only 10 × 4 = 40 lineitem rows.
-- Uniform: subquery returns top 10 of 100 matching orders → 40 lineitem rows.
-- This rewrite form is the target output when SORT_JOIN_TRANSPOSE is correctly applied.

SELECT top_o.o_orderkey,
       top_o.o_clerk,
       top_o.o_totalprice,
       l.l_extendedprice,
       l.l_quantity
FROM (
    SELECT o.o_orderkey, o.o_clerk, o.o_totalprice
    FROM orders o
    WHERE o.o_clerk = 'Clerk#000000001'
    ORDER BY o.o_totalprice DESC
    LIMIT 10
) top_o
JOIN lineitem l ON l.l_orderkey = top_o.o_orderkey;
