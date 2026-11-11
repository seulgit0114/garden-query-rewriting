-- Scenario D-0: JOIN query — candidate for JOIN_ADD_REDUNDANT_SEMI_JOIN
-- Rule: adds a semi-join pre-filter on the probe side (lineitem) before the main join.
-- On skewed DB (clerk=45%, ~45K orders): semi-join probe 400K→180K; overhead > gain → HURTS.
-- On uniform DB (clerk=0.1%, ~100 orders): semi-join probe 400K→400 rows → 1000× reduction → HELPS.
-- Distaware insight: detect selectivity → apply rule only for uniform (low selectivity).

SELECT l.l_orderkey,
       l.l_extendedprice,
       l.l_quantity,
       o.o_clerk,
       o.o_totalprice
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_clerk = 'Clerk#000000001'
ORDER BY l.l_extendedprice DESC;
