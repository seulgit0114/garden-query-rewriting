-- Scenario C variant: 특정 카테고리만 집계 (추가 필터 적용)
SELECT c.name, p.category, SUM(r.revenue) AS total_revenue, COUNT(*) AS sale_cnt
FROM sales r
JOIN products p ON r.product_id = p.id
JOIN customers c ON r.customer_id = c.id
WHERE p.category IN ('Electronics', 'Clothing', 'Food')
GROUP BY p.category, c.name
HAVING SUM(r.revenue) > 10000;
