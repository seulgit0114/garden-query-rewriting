SELECT
    o.o_orderkey,
    o.o_clerk,
    o.o_totalprice,
    (SELECT l.l_extendedprice
     FROM lineitem l
     WHERE l.l_orderkey = o.o_orderkey
     ORDER BY l.l_extendedprice DESC
     LIMIT 1) AS max_line_price
FROM orders o
WHERE o.o_clerk = 'Clerk#000000001'
ORDER BY o.o_totalprice DESC;
