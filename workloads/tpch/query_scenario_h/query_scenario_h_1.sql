-- Scenario H-1: Expanded DISTINCT form (what AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN produces)
-- COUNT(DISTINCT x) expanded into: DISTINCT subquery → join → count.
-- This form is what the Calcite rule would generate to replace COUNT(DISTINCT).
-- For uniform (100 orders, 400 lineitem): the expansion adds two DISTINCT scans → overkill → slower.
-- For skewed (45K orders, 180K lineitem): the expansion may reduce peak memory for hash aggregate.
-- Distaware guidance for uniform: "small cardinality → expansion overhead > benefit → skip rule."

SELECT yrs.order_year,
       COUNT(dp.l_partkey)    AS distinct_parts_ordered,
       COUNT(do2.o_orderkey)  AS distinct_orders,
       rev.total_revenue
FROM (
    SELECT EXTRACT(YEAR FROM o.o_orderdate) AS order_year
    FROM orders o
    WHERE o.o_clerk = 'Clerk#000000001'
    GROUP BY EXTRACT(YEAR FROM o.o_orderdate)
) yrs
LEFT JOIN (
    SELECT EXTRACT(YEAR FROM o.o_orderdate) AS order_year,
           l.l_partkey
    FROM orders o
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE o.o_clerk = 'Clerk#000000001'
    GROUP BY EXTRACT(YEAR FROM o.o_orderdate), l.l_partkey
) dp ON dp.order_year = yrs.order_year
LEFT JOIN (
    SELECT EXTRACT(YEAR FROM o.o_orderdate) AS order_year,
           o.o_orderkey
    FROM orders o
    WHERE o.o_clerk = 'Clerk#000000001'
    GROUP BY EXTRACT(YEAR FROM o.o_orderdate), o.o_orderkey
) do2 ON do2.order_year = yrs.order_year
LEFT JOIN (
    SELECT EXTRACT(YEAR FROM o.o_orderdate) AS order_year,
           SUM(l.l_extendedprice) AS total_revenue
    FROM orders o
    JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE o.o_clerk = 'Clerk#000000001'
    GROUP BY EXTRACT(YEAR FROM o.o_orderdate)
) rev ON rev.order_year = yrs.order_year
GROUP BY yrs.order_year, rev.total_revenue
ORDER BY yrs.order_year;
