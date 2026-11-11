**Project Title:** End-to-End Hierarchical Demand Forecasting & Prescriptive Inventory Optimization Engine

**Core Objective:** Build a production-grade analytics module that ingests retail transaction data from a relational database, generates coherent probabilistic forecasts across a product-store hierarchy, and solves a constrained Mathematical Optimization (MILP) problem to output daily warehouse replenishment orders.

### Business Scenario

A retail chain operates multiple stores across three product categories (**Foods, Household, Hobbies**). Every week, procurement managers must decide how many units of each SKU to order from suppliers.

* Ordering based on simple averages causes frequent stockouts on high-margin items.
* Forecasting SKUs independently creates numbers that conflict with store-level budget targets.
* Ordering too much violates warehouse storage capacity.

Your system bridges **Machine Learning** (predicting uncertain future demand) and **Operations Research** (computing the optimal order schedule under physical and financial constraints), wrapped in an interactive **Plotly Dash** simulator for business stakeholders.

---

## Repository & Tech Stack Blueprint

Set up a clean, modular Python repository so BCG X interviewers can inspect your engineering standards immediately:

```text
supply-chain-opt-engine/
├── data/                    # Raw M5 CSVs & synthetic cost/capacity metadata
├── sql/
│   ├── 01_schema.sql        # PostgreSQL table definitions & indexes
│   └── 02_feature_view.sql  # Window functions for lags, rolling stats, price ratios
├── src/
│   ├── ingestion.py         # Loads scoped M5 subset into PostgreSQL
│   ├── baseline.py          # Seasonal Naive, AutoETS, and Croston benchmarks
│   ├── forecasting.py       # Global LightGBM quantile models (mlforecast)
│   ├── reconciliation.py    # Bottom-Up & MinT reconciliation (hierarchicalforecast)
│   └── optimization.py      # Pyomo + HiGHS MILP inventory replenishment solver
├── app/
│   └── dashboard.py         # Plotly Dash executive KPI & What-If simulator
├── tests/                   # Pytest unit tests for leakage & solver constraints
└── pyproject.toml           # Dependencies managed via uv or poetry

```

---

## Step-by-Step Build Guide

1. **Scope the Dataset & Load into PostgreSQL:** Scope to 500-1,000 SKUs so local training runs under 3 minutes.
Download the **Walmart M5 Forecasting** dataset from Kaggle (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`). Instead of running all 30,490 series at once, filter a representative subset:

* **Scope:** 1 State (`CA`), 3 Stores (`CA_1`, `CA_2`, `CA_3`), all 3 Categories (`FOODS`, `HOBBIES`, `HOUSEHOLD`), and the top **600 SKUs** (mix of fast-moving and intermittent items) over the final **3 years** (~1,100 days).
* **Database Setup:** Spin up a local PostgreSQL instance (via Docker) and create three normalized tables: `fact_daily_sales`, `dim_calendar`, and `dim_prices`.
* **Synthetic Supply Chain Metadata:** Create a fourth table, `dim_sku_economics`, assigning realistic parameters to each SKU: unit purchase cost, daily holding cost ($c_h$), stockout penalty ($c_s$, set between 3x and 8x of $c_h$), item cubic volume ($v_i$), supplier lead time ($L_i \in \{2, 3, 5\}$ days), and Minimum Order Quantity ($MOQ$).


2. **Engineer Time-Series Features in SQL:** Shift all rolling windows by the forecast horizon to prevent data leakage.
Write a SQL script (`02_feature_view.sql`) using Common Table Expressions (CTEs) and window functions to build your modeling table directly inside PostgreSQL:

* **Lag & Rolling Demand Features:** Compute 7-day, 14-day, and 28-day rolling means and standard deviations (`AVG(sales) OVER (...)`, `STDDEV(sales) OVER (...)`).
* **Price Elasticity Signals:** Compute `price_relativity` (current week's price divided by the 52-week rolling average price of that SKU) and `price_momentum` (week-over-week price change percentage).
* **Calendar & Promo Flags:** Join SNAP benefit days, national holidays, and calculate a `days_until_next_holiday` countdown feature.
* **Intermittency Trackers:** Calculate the rolling percentage of zero-sales days over the past 28 days to help the tree model distinguish sparse SKUs from high-velocity SKUs.


3. **Build the Rolling-Origin Backtest & Statistical Baselines:** Never use random train-test splits on time-series data.
Before training any ML model, establish an evaluation harness in `src/baseline.py`:

* **Backtest Design:** Implement a **4-window rolling-origin evaluation** where each window predicts a **28-day horizon** ($h = 28$) across distinct seasonal periods (spring, summer, back-to-school, winter holidays).
* **Benchmark Models:** Use Nixtla's `statsforecast` library to run **Seasonal Naive** (lag 7), **AutoETS**, and **TSB (Teunter-Syntetos-Babai)** for intermittent demand.
* **Metrics Logged:** Compute **WMAPE** (Weighted Mean Absolute Percentage Error), **RMSSE** (Root Mean Squared Scaled Error), and **Pinball Loss** across the 4 folds. Save this table—it serves as your baseline benchmark in interviews.


4. **Train the Global LightGBM Probabilistic Forecaster:** Requires recursive or direct multi-step forecasting across the 28-day horizon.
In `src/forecasting.py`, use `mlforecast` and `lightgbm` to train a single **global model** across all 600 SKUs simultaneously:

* **Point & Quantile Models:** Train models targeting the conditional median ($\tau = 0.50$) as well as upper/lower quantiles ($\tau \in \{0.10, 0.50, 0.80, 0.90, 0.95\}$) using the `quantile` objective (`alpha` parameter in LightGBM).
* **Hyperparameter Tuning:** Use `Optuna` on the first 3 validation folds (keeping the 4th fold as a strict holdout) to tune `num_leaves`, `min_child_samples`, `learning_rate`, and `colsample_bytree`.
* **Feature Importance:** Extract SHAP values to verify which drivers (such as price drops, SNAP days, or 7-day lags) govern the predictions.


5. **Reconcile Forecasts Across the Hierarchy:** Ensures SKU forecasts sum exactly to Department and Store totals.
In `src/reconciliation.py`, aggregate your historical data and base forecasts into a 4-level hierarchy: **Total $\rightarrow$ Store (3) $\rightarrow$ Category (9) $\rightarrow$ SKU (600)**:

* **Apply Reconciliation:** Use `hierarchicalforecast` to run **Bottom-Up**, **Top-Down**, and **MinT (Minimum Trace with Shrinkage)** reconciliation on the base forecasts.
* **Verify Improvement:** Log the WMAPE at every level of the tree before and after reconciliation. You will typically observe a 2% to 5% accuracy gain at the middle and top levels while achieving 100% mathematical coherence across the hierarchy.


6. **Formulate the MILP Inventory Replenishment Solver:** Uses open-source HiGHS solver via Pyomo - no commercial license required.
In `src/optimization.py`, build the prescriptive engine using `pyomo` and the `highspy` solver. Pass the reconciled 28-day quantile forecast (for example, $\hat{D}_{i,t}^{(0.90)}$) as the demand input:

* **Decision Variables:** Daily order quantity $O_{i,t} \ge 0$, binary order trigger $z_{i,t} \in \{0,1\}$, projected ending inventory $I_{i,t} \ge 0$, and unmet demand/stockout units $U_{i,t} \ge 0$.
* **Objective Function:** Minimize total 28-day cost: fixed ordering fees + variable holding costs + stockout penalties.
* **Operational Constraints:**
1. *Flow Balance:* $I_{i,t} - U_{i,t} = I_{i,t-1} + O_{i, t - L_i} - \hat{D}_{i,t}^{(\tau)}$
2. *Store Backroom Capacity:* Total volume of stored items $\sum_i v_i I_{i,t} \le V_{\text{store}}$ on every day $t$.
3. *Minimum Order Quantity (MOQ):* $MOQ_i \cdot z_{i,t} \le O_{i,t} \le M \cdot z_{i,t}$.


* **Benchmark the Policy:** Compare your MILP policy against a naive "Order-Up-To Mean ($P_{50}$)" rule by simulating both schedules against the **actual realized demand** of the holdout window.


7. **Build the Interactive Plotly Dash Simulator:** Keep the UI focused on executive KPIs and What-If levers.
In `app/dashboard.py`, build a clean, two-tab `Plotly Dash` web application:

* **Tab 1 — Forecast & Hierarchy Explorer:** A drill-down view where users select Store $\rightarrow$ Category $\rightarrow$ SKU to inspect historical sales, the 28-day reconciled forecast fan chart ($P_{10}$ to $P_{95}$), and SHAP feature drivers.
* **Tab 2 — Replenishment & Scenario Simulator:** Interactive sliders for **Target Service Quantile ($\tau$)**, **Warehouse Capacity Limit (% of baseline)**, and **Supplier Lead Time Shock (+1 to +4 days)**. Clicking *"Run Optimizer"* triggers the Pyomo MILP solver in the background (solves in ~1–2 seconds) and updates four KPI cards: **Total Supply Chain Cost ($)**, **Fill Rate (%)**, **Working Capital in Inventory ($)**, and **Downloadable Purchase Order CSV**.


---

## Exact Evaluation Table to Include in Your README

When a BCG X Data Scientist opens your GitHub repository, this table at the top of your `README.md` proves you think like a consultant who connects ML metrics to client P&L:

| Pipeline Stage / Policy | Forecast WMAPE (%) | 90% Coverage Rate | Simulated Fill Rate (%) | 28-Day Holding Cost ($) | 28-Day Stockout Cost ($) | Total Supply Chain Cost ($) |
| --- | --- | --- | --- | --- | --- | --- |
| **1. Seasonal Naive + Order $P_{50}$** | 24.8% | N/A | 78.4% | $14,200 | $48,500 | **$62,700** |
| **2. AutoETS + Static Safety Stock** | 21.3% | 84.1% | 88.2% | $19,800 | $29,100 | **$48,900** |
| **3. LightGBM (Unreconciled) + $P_{90}$** | 18.1% | 89.4% | 94.1% | $24,600 | $14,800 | **$39,400** |
| **4. LightGBM (`MinT` Reconciled) + MILP Solver** | **16.9%** | **90.3%** | **96.2%** | **$21,100** | **$11,200** | **$32,300 (-18.0%)** |

*(Note: Populate this table with the actual numbers produced by your 4-fold backtest once you run Step 6.)*
