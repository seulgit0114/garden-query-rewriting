-- Scenario G-0: lineitem JOIN part + IN-subquery on orders — SEMI_JOIN_JOIN_TRANSPOSE target
-- Rule: reorders semi-join (IN subquery) relative to the regular JOIN.
-- Current form: semi-join on orders happens BEFORE the part join at the planner level.
-- Skewed (clerk=45%, ~45K orders in IN set):
--   lineitem 400K → IN semi-join (45K set) → 180K rows → JOIN part (Brand#13, ~100 rows) → ~600 rows
--   Better: JOIN part first (400K→~4K filtered rows) → IN semi-join → ~1800 rows
-- Uniform (clerk=0.1%, ~100 orders in IN set):
--   lineitem 400K → IN semi-join (100 set) → 400 rows → JOIN part → ~4 rows  (FAST!)
--   Better to keep IN first (highly selective for uniform).
-- Distaware: clerk=45% → IN semi-join is NON-selective → SEMI_JOIN_JOIN_TRANSPOSE: join part first.

SELECT l.l_orderkey,
       l.l_extendedprice,
       l.l_quantity,
       p.p_brand,
       p.p_name
FROM lineitem l
JOIN part p ON p.p_partkey = l.l_partkey
WHERE p.p_brand = 'Brand#13'
  AND l.l_orderkey IN (
      SELECT o.o_orderkey
      FROM orders o
      WHERE o.o_clerk = 'Clerk#000000001'
  )
ORDER BY l.l_extendedprice DESC;
