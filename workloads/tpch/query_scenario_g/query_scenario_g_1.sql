-- Scenario G-1: part JOIN first, then IN-filter (reordered version of G-0)
-- Forces join with part table BEFORE applying the IN-subquery semi-join on orders.
-- For skewed: part (Brand#13 ~100 rows) × lineitem (400K) → ~4K rows → IN filter (45K set) → ~1800
--   vs G-0: lineitem (400K) → IN (45K) → 180K → part → ~600 rows (but 180K intermediate!)
-- This explicit form bypasses SEMI_JOIN_JOIN_TRANSPOSE and shows the benefit of correct ordering.
-- On uniform: this form is suboptimal (IN filter should come first for high selectivity).

SELECT lp.l_orderkey,
       lp.l_extendedprice,
       lp.l_quantity,
       lp.p_brand,
       lp.p_name
FROM (
    SELECT l.l_orderkey,
           l.l_extendedprice,
           l.l_quantity,
           p.p_brand,
           p.p_name
    FROM lineitem l
    JOIN part p ON p.p_partkey = l.l_partkey
    WHERE p.p_brand = 'Brand#13'
) lp
WHERE lp.l_orderkey IN (
    SELECT o.o_orderkey
    FROM orders o
    WHERE o.o_clerk = 'Clerk#000000001'
)
ORDER BY lp.l_extendedprice DESC;
