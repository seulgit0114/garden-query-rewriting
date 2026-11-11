-- Scenario A variant: refunded 주문 분석 (selectivity 5%)
-- fraud(1%)보다 선택도가 낮지만 여전히 skewed 효과 측정
SELECT region, SUM(amount) AS total_amount, COUNT(*) AS cnt
FROM sales s
JOIN customers c ON s.customer_id = c.id
JOIN products p ON s.product_id = p.id
WHERE s.status = 'refunded'
GROUP BY region
HAVING COUNT(*) > 100;
