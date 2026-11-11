-- Scenario H-0: COUNT(DISTINCT) with JOIN — AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN target
-- Rule: expands COUNT(DISTINCT x) into a join-based computation using a DISTINCT subquery.
-- Expansion adds overhead for small cardinalities but may help for very large distinct sets.
-- Skewed (clerk=45%, ~45K orders × 4 lineitem avg = 180K lineitem rows):
--   COUNT(DISTINCT l_partkey): 180K rows → hash for distinct partkeys → potentially 100K distinct
--   Expanding with JOIN: DISTINCT(orderkey, partkey) → join → COUNT → may be more memory efficient.
-- Uniform (clerk=0.1%, ~100 orders × 4 = 400 lineitem rows):
--   COUNT(DISTINCT l_partkey): trivial 400 rows → any strategy fast.
--   Expanding with JOIN adds unnecessary overhead → WORSE.
-- Distaware: uniform → small count → DO NOT expand (avoid overhead).
--            skewed → large count (100K distinct) → expansion MAY help, but hash is also fine.

SELECT EXTRACT(YEAR FROM o.o_orderdate)  AS order_year,
       COUNT(DISTINCT l.l_partkey)        AS distinct_parts_ordered,
       COUNT(DISTINCT o.o_orderkey)       AS distinct_orders,
       SUM(l.l_extendedprice)             AS total_revenue
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_clerk = 'Clerk#000000001'
GROUP BY EXTRACT(YEAR FROM o.o_orderdate)
ORDER BY order_year;
