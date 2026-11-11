SELECT c.name, p.category, SUM(r.revenue) AS total_revenue
FROM sales r
JOIN products p ON r.product_id = p.id
JOIN customers c ON r.customer_id = c.id
GROUP BY p.category, c.name;
