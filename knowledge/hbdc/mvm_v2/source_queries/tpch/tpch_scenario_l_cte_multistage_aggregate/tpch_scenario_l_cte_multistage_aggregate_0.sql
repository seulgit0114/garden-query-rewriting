WITH revenue AS (
  SELECT l.l_suppkey, SUM(l.l_extendedprice * (1-l.l_discount)) AS rev
  FROM lineitem l
  WHERE l.l_shipdate >= DATE '1994-01-01' AND l.l_shipdate < DATE '1995-01-01'
  GROUP BY l.l_suppkey
), threshold AS (SELECT AVG(rev) * 1.2 AS cutoff FROM revenue)
SELECT s.s_suppkey, s.s_name, r.rev
FROM supplier s JOIN revenue r ON r.l_suppkey=s.s_suppkey CROSS JOIN threshold t
WHERE r.rev > t.cutoff;
