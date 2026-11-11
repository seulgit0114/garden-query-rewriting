-- =============================================================
-- skewed dataset schema
-- Skewed Data Scenarios A / B / C
-- =============================================================

CREATE TABLE IF NOT EXISTS customers (
    id        SERIAL PRIMARY KEY,
    name      VARCHAR(100) NOT NULL,
    -- tier: 'premium' = 0.1% (1,000 / 1,000,000), 'standard' = 99.9%
    tier      VARCHAR(20)  NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id       SERIAL PRIMARY KEY,
    -- category: 10 distinct values (Electronics/Clothing/Food/Sports/Books/
    --           Toys/Health/Home/Beauty/Garden), each 1,000 rows out of 10,000
    category VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    id          BIGSERIAL PRIMARY KEY,
    customer_id INT           NOT NULL REFERENCES customers(id),
    product_id  INT           NOT NULL REFERENCES products(id),
    -- status Zipf distribution: 'completed'=94%, 'refunded'=5%, 'fraud'=1%
    status      VARCHAR(20)   NOT NULL,
    -- region: 5 values (North/South/East/West/Central), uniform distribution
    region      VARCHAR(20)   NOT NULL,
    amount      NUMERIC(12,2) NOT NULL,
    revenue     NUMERIC(12,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    BIGSERIAL PRIMARY KEY,
    customer_id INT           NOT NULL REFERENCES customers(id),
    amount      NUMERIC(12,2) NOT NULL
);
