-- Direct multi-step features. Forecast origin is the last observed sales day.
-- Demand windows include the origin and never read later demand. Lag columns
-- are relative to the origin, not to the future target date. No target labels
-- are returned. Apply this script in a transaction after 01_schema.sql.

CREATE OR REPLACE VIEW feature_calendar AS
WITH flags AS (
    SELECT c.*,
        COALESCE(c.event_type_1 = 'National', FALSE)
            OR COALESCE(c.event_type_2 = 'National', FALSE) AS is_national_holiday,
        c.event_name_1 IS NOT NULL OR c.event_name_2 IS NOT NULL AS has_scheduled_event
    FROM dim_calendar c
), countdown AS (
    SELECT f.*,
        MIN(f.date) FILTER (WHERE f.is_national_holiday) OVER (
            ORDER BY f.date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        ) AS next_holiday
    FROM flags f
)
SELECT c.*, c.next_holiday - c.date AS days_until_next_holiday
FROM countdown c;

-- A dense calendar-week grid prevents missing prices from compressing a 52-week
-- window or making LAG refer to the previous listed week instead of last week.
CREATE OR REPLACE VIEW feature_weekly_prices AS
WITH weeks AS (
    SELECT c.wm_yr_wk, MIN(c.date) AS week_start
    FROM dim_calendar c GROUP BY c.wm_yr_wk
), series AS (
    SELECT DISTINCT f.store_id, f.item_id FROM fact_daily_sales f
), grid AS (
    SELECT s.store_id, s.item_id, w.wm_yr_wk, w.week_start, p.sell_price
    FROM series s CROSS JOIN weeks w
    LEFT JOIN dim_prices p USING (store_id, item_id, wm_yr_wk)
), history AS (
    SELECT g.*,
        AVG(g.sell_price) OVER w52 AS price_mean_52w,
        COUNT(g.sell_price) OVER w52 AS price_observed_weeks_52w,
        LAG(g.sell_price) OVER ws AS previous_week_price,
        MAX(g.week_start) FILTER (WHERE g.sell_price IS NOT NULL) OVER (
            PARTITION BY g.store_id, g.item_id ORDER BY g.week_start
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS last_price_week_start
    FROM grid g
    WINDOW ws AS (PARTITION BY g.store_id, g.item_id ORDER BY g.week_start),
        w52 AS (ws ROWS BETWEEN 51 PRECEDING AND CURRENT ROW)
)
SELECT h.store_id, h.item_id, h.wm_yr_wk, h.week_start,
    p.sell_price AS asof_price, h.price_mean_52w, h.price_observed_weeks_52w,
    p.sell_price / NULLIF(h.price_mean_52w, 0) AS price_relativity,
    h.sell_price / NULLIF(h.previous_week_price, 0) - 1 AS price_momentum,
    h.last_price_week_start,
    h.sell_price IS NULL AND p.sell_price IS NOT NULL AS price_carried_forward
FROM history h
LEFT JOIN weeks w ON w.week_start = h.last_price_week_start
LEFT JOIN dim_prices p ON p.store_id = h.store_id AND p.item_id = h.item_id
    AND p.wm_yr_wk = w.wm_yr_wk;

-- Complete the daily grid before applying ROWS windows. Missing observations
-- remain NULL, and observed unlisted days remain distinct from demand zeros.
CREATE OR REPLACE VIEW feature_daily_demand AS
WITH bounds AS (
    SELECT f.store_id, f.item_id, MIN(f.date) AS first_date, MAX(f.date) AS last_date
    FROM fact_daily_sales f GROUP BY f.store_id, f.item_id
), grid AS (
    SELECT b.store_id, b.item_id, c.date, f.sales, f.is_available,
        CASE WHEN f.is_available THEN f.sales END AS listed_sales
    FROM bounds b JOIN dim_calendar c ON c.date BETWEEN b.first_date AND b.last_date
    LEFT JOIN fact_daily_sales f USING (store_id, item_id, date)
), windows AS (
    SELECT g.*,
        LAG(g.listed_sales, 7) OVER ws AS origin_lag_7,
        LAG(g.listed_sales, 14) OVER ws AS origin_lag_14,
        LAG(g.listed_sales, 28) OVER ws AS origin_lag_28,
        COUNT(g.sales) OVER w7 AS observed_days_7,
        COUNT(g.sales) OVER w14 AS observed_days_14,
        COUNT(g.sales) OVER w28 AS observed_days_28,
        COUNT(g.sales) OVER (ws ROWS BETWEEN 28 PRECEDING AND CURRENT ROW) AS observed_days_29,
        COUNT(g.listed_sales) OVER w7 AS available_days_7,
        COUNT(g.listed_sales) OVER w14 AS available_days_14,
        COUNT(g.listed_sales) OVER w28 AS available_days_28,
        AVG(g.listed_sales) OVER w7 AS mean_7,
        AVG(g.listed_sales) OVER w14 AS mean_14,
        AVG(g.listed_sales) OVER w28 AS mean_28,
        STDDEV_SAMP(g.listed_sales) OVER w7 AS std_7,
        STDDEV_SAMP(g.listed_sales) OVER w14 AS std_14,
        STDDEV_SAMP(g.listed_sales) OVER w28 AS std_28,
        AVG(CASE WHEN g.is_available THEN (g.sales = 0)::INTEGER END) OVER w28 AS zero_share_28
    FROM grid g
    WINDOW ws AS (PARTITION BY g.store_id, g.item_id ORDER BY g.date),
        w7 AS (ws ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
        w14 AS (ws ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
        w28 AS (ws ROWS BETWEEN 27 PRECEDING AND CURRENT ROW)
)
SELECT w.store_id, w.item_id, w.date, w.is_available,
    w.origin_lag_7, w.origin_lag_14, w.origin_lag_28, w.observed_days_29,
    w.available_days_7, w.available_days_14, w.available_days_28,
    CASE WHEN w.observed_days_7 = 7 THEN w.mean_7 END AS rolling_mean_7,
    CASE WHEN w.observed_days_14 = 14 THEN w.mean_14 END AS rolling_mean_14,
    CASE WHEN w.observed_days_28 = 28 THEN w.mean_28 END AS rolling_mean_28,
    CASE WHEN w.observed_days_7 = 7 THEN w.std_7 END AS rolling_std_7,
    CASE WHEN w.observed_days_14 = 14 THEN w.std_14 END AS rolling_std_14,
    CASE WHEN w.observed_days_28 = 28 THEN w.std_28 END AS rolling_std_28,
    CASE WHEN w.observed_days_28 = 28 THEN w.zero_share_28 END AS zero_sales_share_28
FROM windows w;

CREATE OR REPLACE FUNCTION build_forecast_features(
    p_origin DATE, p_horizon INTEGER DEFAULT 28, p_known_future_prices BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    unique_id TEXT, store_id TEXT, item_id TEXT, dept_id TEXT, cat_id TEXT,
    forecast_origin DATE, ds DATE, horizon INTEGER,
    origin_lag_7 INTEGER, origin_lag_14 INTEGER, origin_lag_28 INTEGER,
    rolling_mean_7 DOUBLE PRECISION, rolling_mean_14 DOUBLE PRECISION,
    rolling_mean_28 DOUBLE PRECISION, rolling_std_7 DOUBLE PRECISION,
    rolling_std_14 DOUBLE PRECISION, rolling_std_28 DOUBLE PRECISION,
    zero_sales_share_28 DOUBLE PRECISION,
    available_days_7 BIGINT, available_days_14 BIGINT, available_days_28 BIGINT,
    origin_is_available BOOLEAN,
    sell_price DOUBLE PRECISION, price_mean_52w DOUBLE PRECISION,
    price_relativity DOUBLE PRECISION, price_momentum DOUBLE PRECISION,
    price_observed_weeks_52w BIGINT, price_age_days INTEGER,
    price_carried_forward BOOLEAN, known_future_prices BOOLEAN,
    snap_ca BOOLEAN, is_national_holiday BOOLEAN, has_scheduled_event BOOLEAN,
    day_of_week INTEGER, month INTEGER, year INTEGER, day_of_month INTEGER,
    week_of_year INTEGER, days_until_next_holiday INTEGER,
    event_name_1 TEXT, event_name_2 TEXT
) LANGUAGE plpgsql STABLE AS $features$
BEGIN
    IF p_origin IS NULL OR p_horizon IS NULL OR p_horizon NOT BETWEEN 1 AND 28
        OR p_known_future_prices IS NULL THEN
        RAISE EXCEPTION 'Require an origin, horizon between 1 and 28, and an explicit price policy';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM fact_daily_sales) THEN
        RAISE EXCEPTION 'Load daily sales before building features';
    END IF;
    IF (SELECT COUNT(*) FROM dim_calendar c
        WHERE c.date > p_origin AND c.date <= p_origin + p_horizon) <> p_horizon THEN
        RAISE EXCEPTION 'Calendar does not cover the requested forecast horizon';
    END IF;
    IF EXISTS (
        SELECT 1 FROM (SELECT DISTINCT f.store_id, f.item_id FROM fact_daily_sales f) s
        LEFT JOIN fact_daily_sales f ON f.store_id = s.store_id AND f.item_id = s.item_id
            AND f.date BETWEEN p_origin - 28 AND p_origin
        GROUP BY s.store_id, s.item_id
        HAVING COUNT(f.date) <> 29 OR COUNT(f.is_available) <> 29
    ) THEN
        RAISE EXCEPTION 'Every series requires 29 complete observed calendar days through the origin';
    END IF;
    RETURN QUERY
    -- Limit one-origin requests to 29 days instead of computing the historical
    -- view for every origin. FILTER aggregates have the same calendar boundaries
    -- as that view's windows. Materialize small snapshots before expanding days.
    WITH snapshot AS MATERIALIZED (
        SELECT f.store_id, f.item_id,
            MAX(f.sales) FILTER (WHERE f.is_available AND f.date = p_origin - 7) AS origin_lag_7,
            MAX(f.sales) FILTER (WHERE f.is_available AND f.date = p_origin - 14) AS origin_lag_14,
            MAX(f.sales) FILTER (WHERE f.is_available AND f.date = p_origin - 28) AS origin_lag_28,
            AVG(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 7) AS rolling_mean_7,
            AVG(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 14) AS rolling_mean_14,
            AVG(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 28) AS rolling_mean_28,
            STDDEV_SAMP(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 7) AS rolling_std_7,
            STDDEV_SAMP(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 14) AS rolling_std_14,
            STDDEV_SAMP(f.sales) FILTER (WHERE f.is_available AND f.date > p_origin - 28) AS rolling_std_28,
            AVG((f.sales = 0)::INTEGER) FILTER (
                WHERE f.is_available AND f.date > p_origin - 28
            ) AS zero_sales_share_28,
            COUNT(*) FILTER (WHERE f.is_available AND f.date > p_origin - 7) AS available_days_7,
            COUNT(*) FILTER (WHERE f.is_available AND f.date > p_origin - 14) AS available_days_14,
            COUNT(*) FILTER (WHERE f.is_available AND f.date > p_origin - 28) AS available_days_28,
            BOOL_OR(f.is_available) FILTER (WHERE f.date = p_origin) AS is_available
        FROM fact_daily_sales f WHERE f.date BETWEEN p_origin - 28 AND p_origin
        GROUP BY f.store_id, f.item_id
    ), targets AS MATERIALIZED (
        SELECT c.* FROM feature_calendar c
        WHERE c.date > p_origin AND c.date <= p_origin + p_horizon
    ), prices AS MATERIALIZED (
        SELECT p.* FROM feature_weekly_prices p
        WHERE p.wm_yr_wk IN (
            SELECT c.wm_yr_wk FROM dim_calendar c
            WHERE c.date BETWEEN p_origin AND
                CASE WHEN p_known_future_prices THEN p_origin + p_horizon ELSE p_origin END
        )
    )
    SELECT (s.store_id || '/' || s.item_id)::TEXT, s.store_id, s.item_id, e.dept_id, e.cat_id,
        p_origin, c.date, c.date - p_origin,
        s.origin_lag_7, s.origin_lag_14, s.origin_lag_28,
        s.rolling_mean_7::DOUBLE PRECISION, s.rolling_mean_14::DOUBLE PRECISION,
        s.rolling_mean_28::DOUBLE PRECISION, s.rolling_std_7::DOUBLE PRECISION,
        s.rolling_std_14::DOUBLE PRECISION, s.rolling_std_28::DOUBLE PRECISION,
        s.zero_sales_share_28::DOUBLE PRECISION,
        s.available_days_7, s.available_days_14, s.available_days_28, s.is_available,
        p.asof_price::DOUBLE PRECISION, p.price_mean_52w::DOUBLE PRECISION,
        p.price_relativity::DOUBLE PRECISION, p.price_momentum::DOUBLE PRECISION,
        p.price_observed_weeks_52w, c.date - p.last_price_week_start,
        p.price_carried_forward, p_known_future_prices,
        c.snap_ca, c.is_national_holiday, c.has_scheduled_event,
        EXTRACT(ISODOW FROM c.date)::INTEGER, c.month::INTEGER, c.year::INTEGER,
        EXTRACT(DAY FROM c.date)::INTEGER, EXTRACT(WEEK FROM c.date)::INTEGER,
        c.days_until_next_holiday, c.event_name_1, c.event_name_2
    FROM snapshot s
    JOIN dim_sku_economics e ON e.item_id = s.item_id
    CROSS JOIN targets c
    JOIN dim_calendar oc ON oc.date = p_origin
    LEFT JOIN prices p ON p.store_id = s.store_id AND p.item_id = s.item_id
        AND p.wm_yr_wk = CASE WHEN p_known_future_prices THEN c.wm_yr_wk ELSE oc.wm_yr_wk END
    WHERE c.date > p_origin AND c.date <= p_origin + p_horizon;
END;
$features$;

-- Convenience view for the live 28-day forecast. Backtests must call the
-- parameterized function with the fold's own origin instead of using this view.
CREATE OR REPLACE VIEW model_forecast_features AS
SELECT * FROM build_forecast_features((SELECT MAX(f.date) FROM fact_daily_sales f), 28, FALSE);
