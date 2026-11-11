-- PostgreSQL 16. Currency is USD; volume is cubic metres; lead time is in days.
-- This bootstrap runs on first creation of the Docker volume only.
CREATE TABLE IF NOT EXISTS dim_calendar (
    date DATE PRIMARY KEY,
    d INTEGER NOT NULL UNIQUE CHECK (d > 0),
    wm_yr_wk INTEGER NOT NULL,
    weekday TEXT NOT NULL,
    wday SMALLINT NOT NULL CHECK (wday BETWEEN 1 AND 7),
    month SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    year SMALLINT NOT NULL,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_ca BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_sku_economics (
    item_id TEXT PRIMARY KEY,
    dept_id TEXT NOT NULL,
    cat_id TEXT NOT NULL CHECK (cat_id IN ('FOODS', 'HOUSEHOLD', 'HOBBIES')),
    unit_purchase_cost NUMERIC(12, 4) NOT NULL CHECK (unit_purchase_cost > 0),
    daily_holding_cost NUMERIC(12, 6) NOT NULL CHECK (daily_holding_cost > 0),
    stockout_penalty NUMERIC(12, 6) NOT NULL,
    unit_volume_m3 NUMERIC(12, 6) NOT NULL CHECK (unit_volume_m3 > 0),
    lead_time_days SMALLINT NOT NULL CHECK (lead_time_days IN (2, 3, 5)),
    minimum_order_quantity INTEGER NOT NULL CHECK (minimum_order_quantity > 0),
    fixed_order_cost NUMERIC(12, 4) NOT NULL CHECK (fixed_order_cost >= 0),
    CHECK (stockout_penalty BETWEEN 3 * daily_holding_cost AND 8 * daily_holding_cost)
);

CREATE TABLE IF NOT EXISTS dim_prices (
    store_id TEXT NOT NULL CHECK (store_id IN ('CA_1', 'CA_2', 'CA_3')),
    item_id TEXT NOT NULL REFERENCES dim_sku_economics (item_id),
    wm_yr_wk INTEGER NOT NULL,
    sell_price NUMERIC(12, 4) NOT NULL CHECK (sell_price > 0),
    PRIMARY KEY (store_id, item_id, wm_yr_wk)
);

CREATE TABLE IF NOT EXISTS fact_daily_sales (
    store_id TEXT NOT NULL CHECK (store_id IN ('CA_1', 'CA_2', 'CA_3')),
    item_id TEXT NOT NULL REFERENCES dim_sku_economics (item_id),
    date DATE NOT NULL REFERENCES dim_calendar (date),
    sales INTEGER NOT NULL CHECK (sales >= 0),
    is_available BOOLEAN NOT NULL,
    PRIMARY KEY (store_id, item_id, date)
);

CREATE INDEX IF NOT EXISTS ix_fact_daily_sales_date ON fact_daily_sales (date);
CREATE INDEX IF NOT EXISTS ix_dim_calendar_week ON dim_calendar (wm_yr_wk);
CREATE INDEX IF NOT EXISTS ix_fact_daily_sales_item ON fact_daily_sales (item_id);
CREATE INDEX IF NOT EXISTS ix_dim_prices_item ON dim_prices (item_id);

-- Allow upgrading a database created by the initial scaffold. Historical rows
-- have unknown availability until ingestion reloads them from the source files.
ALTER TABLE fact_daily_sales ADD COLUMN IF NOT EXISTS is_available BOOLEAN;
