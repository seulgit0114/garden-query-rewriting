SELECT COUNT(*) AS matching_customers
FROM customer c
WHERE c.c_mktsegment = 'AUTOMOBILE'
  AND EXISTS (
    SELECT 1 FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey
    WHERE o.o_custkey = c.c_custkey AND l.l_discount >= 0.05
  );
