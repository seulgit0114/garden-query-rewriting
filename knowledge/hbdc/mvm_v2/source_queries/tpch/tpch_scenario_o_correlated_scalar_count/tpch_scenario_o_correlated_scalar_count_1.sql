SELECT COUNT(*) AS qualifying_parts
FROM part p
WHERE p.p_size = 15
  AND (SELECT COUNT(DISTINCT ps.ps_suppkey) FROM partsupp ps
       WHERE ps.ps_partkey=p.p_partkey AND ps.ps_availqty > 500) >= 3;
