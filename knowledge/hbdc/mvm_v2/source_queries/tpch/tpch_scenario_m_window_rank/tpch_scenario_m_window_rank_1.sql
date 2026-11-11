SELECT nation_name, supplier_name, total_value
FROM (
  SELECT n.n_name AS nation_name, s.s_name AS supplier_name,
         SUM(ps.ps_supplycost * ps.ps_availqty) AS total_value,
         DENSE_RANK() OVER (PARTITION BY n.n_name ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS rk
  FROM nation n JOIN supplier s ON s.s_nationkey=n.n_nationkey
       JOIN partsupp ps ON ps.ps_suppkey=s.s_suppkey
  WHERE ps.ps_availqty > 500
  GROUP BY n.n_name, s.s_name
) ranked WHERE rk <= 5;
