SELECT COUNT(*) AS below_average_pairs
FROM partsupp ps
WHERE ps.ps_supplycost < (
  SELECT AVG(ps2.ps_supplycost) * 1.0
  FROM partsupp ps2 WHERE ps2.ps_partkey = ps.ps_partkey
);
