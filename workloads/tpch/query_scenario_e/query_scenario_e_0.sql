-- Scenario E-0: JOIN + ORDER BY + LIMIT — candidate for SORT_JOIN_TRANSPOSE
-- Rule: pushes ORDER BY through JOIN so the sort is applied before joining.
-- PG cannot push LIMIT through JOIN automatically without this rewrite.
-- Skewed (clerk=45%, ~45K orders): join 400K lineitem × 45K orders → 180K rows → sort → LIMIT 10.
--   After rule: sort 45K orders → LIMIT 10 → join only 10 orders × ~4 lineitem = 40 rows.
--   Speedup: 180K vs 40 rows = 4500x reduction in join size.
-- Uniform (clerk=0.1%, ~100 orders): join 400 rows → sort → LIMIT 10.
--   After rule: sort 100 → LIMIT 10 → join 40 rows. Modest 10x improvement.
-- Distaware: detects 45K rows + LIMIT 10 → ratio 4500:1 → SORT_JOIN_TRANSPOSE is critical.

SELECT o.o_orderkey,
       o.o_clerk,
       o.o_totalprice,
       l.l_extendedprice,
       l.l_quantity
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_clerk = 'Clerk#000000001'
ORDER BY o.o_totalprice DESC
LIMIT 10;
