SELECT o.order_id, o.amount
FROM orders o
WHERE o.customer_id IN (
    SELECT id FROM customers WHERE tier = 'premium'
);
