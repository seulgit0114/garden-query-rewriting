SELECT nation_key, SUM(amount) AS total_amount
FROM (
  SELECT c.c_nationkey AS nation_key, o.o_totalprice AS amount
  FROM customer c JOIN orders o ON o.o_custkey=c.c_custkey
  WHERE o.o_orderstatus='O'
  UNION ALL
  SELECT s.s_nationkey AS nation_key, ps.ps_supplycost*ps.ps_availqty AS amount
  FROM supplier s JOIN partsupp ps ON ps.ps_suppkey=s.s_suppkey
  WHERE ps.ps_availqty > 500
) u GROUP BY nation_key HAVING SUM(amount) > 2000000;
