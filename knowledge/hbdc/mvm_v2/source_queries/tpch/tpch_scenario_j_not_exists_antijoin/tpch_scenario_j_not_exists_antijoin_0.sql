SELECT COUNT(*) AS matching_suppliers
FROM supplier s
WHERE s.s_nationkey = 1
  AND NOT EXISTS (
    SELECT 1 FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
    WHERE ps.ps_suppkey = s.s_suppkey AND p.p_size BETWEEN 1 AND 5
  );
