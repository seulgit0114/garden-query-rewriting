SELECT r.r_name, SUM(x.order_value) AS regional_value
FROM region r JOIN nation n ON n.n_regionkey=r.r_regionkey
JOIN customer c ON c.c_nationkey=n.n_nationkey
JOIN (
  SELECT o.o_custkey, o.o_orderkey, SUM(l.l_extendedprice*(1-l.l_discount)) AS order_value
  FROM orders o JOIN lineitem l ON l.l_orderkey=o.o_orderkey
  WHERE o.o_orderdate >= DATE '1998-01-01'
  GROUP BY o.o_custkey, o.o_orderkey
) x ON x.o_custkey=c.c_custkey
GROUP BY r.r_name HAVING SUM(x.order_value) > 3000000;
