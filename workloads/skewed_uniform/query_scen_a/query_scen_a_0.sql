SELECT region, SUM(amount) AS total_amount, COUNT(*) AS cnt
FROM sales s
JOIN customers c ON s.customer_id = c.id
JOIN products p ON s.product_id = p.id
WHERE s.status = 'fraud'
GROUP BY region
HAVING COUNT(*) > 5;
