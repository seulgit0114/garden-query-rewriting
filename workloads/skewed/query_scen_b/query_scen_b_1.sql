-- Scenario B variant: EXISTS 패턴 (IN 대신 EXISTS)
-- IN → EXISTS 변환 효과를 반대 방향으로 측정
SELECT o.order_id, o.amount
FROM orders o
WHERE EXISTS (
    SELECT 1 FROM customers c
    WHERE c.id = o.customer_id
      AND c.tier = 'premium'
);
