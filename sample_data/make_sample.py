"""Generate sample_data/retail_orders.csv with deliberate traps for the analyst checks.

Planted on purpose:
- Simpson's paradox: Store orders have a higher average value overall, but Online is
  higher inside every region (Store orders are concentrated in the high-value West).
- A handful of huge bulk orders that drag the mean far above the median.
- Mixed date formats (YYYY-MM-DD and MM/DD/YYYY) in order_date.
- Misspelled / inconsistent city names (Delhi, 'delhi ', Dehli, Mumbai, Mumbia).
- 25 fully duplicated rows.
- Refunds, so "revenue" is ambiguous (gross vs net of refunds) for the glossary demo.

Run:  python sample_data/make_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

regions = {
    # region: (base order value, share of orders through Store, cities)
    "West": (2000, 0.80, ["Mumbai", "Pune", "Ahmedabad"]),
    "North": (1200, 0.20, ["Delhi", "Jaipur", "Lucknow"]),
    "South": (900, 0.20, ["Bengaluru", "Chennai", "Hyderabad"]),
    "East": (700, 0.20, ["Kolkata", "Patna", "Bhubaneswar"]),
}
categories = ["Electronics", "Apparel", "Home", "Grocery", "Beauty"]

rows = []
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
for i in range(N):
    region = rng.choice(list(regions), p=[0.3, 0.25, 0.25, 0.2])
    base, store_share, cities = regions[region]
    channel = "Store" if rng.random() < store_share else "Online"
    value = rng.normal(base + (150 if channel == "Online" else 0), base * 0.25)
    value = max(value, 50)
    qty = int(rng.integers(1, 6))
    unit_price = round(value / qty, 2)
    discount = float(rng.choice([0, 0, 0, 5, 10, 15, 20]))
    order_value = round(qty * unit_price * (1 - discount / 100), 2)
    refund = round(order_value, 2) if rng.random() < 0.06 else 0.0
    d = dates[rng.integers(len(dates))]
    date_str = d.strftime("%m/%d/%Y") if rng.random() < 0.15 else d.strftime("%Y-%m-%d")
    city = str(rng.choice(cities))
    if city == "Delhi" and rng.random() < 0.15:
        city = str(rng.choice(["delhi ", "Dehli", "DELHI"]))
    if city == "Mumbai" and rng.random() < 0.1:
        city = "Mumbia"
    rows.append({
        "order_id": f"ORD{100000 + i}",
        "order_date": date_str,
        "region": region,
        "city": city,
        "channel": channel,
        "category": str(rng.choice(categories)),
        "quantity": qty,
        "unit_price": unit_price,
        "discount_pct": discount,
        "order_value": order_value,
        "refund_amount": refund,
        "customer_rating": int(np.clip(round(rng.normal(4.0, 0.8)), 1, 5)),
    })

df = pd.DataFrame(rows)

# a few bulk B2B orders (own channel, so the Online/Store paradox stays clean): mean >> median
bulk_idx = rng.choice(N, 8, replace=False)
df.loc[bulk_idx, "quantity"] = 400
df.loc[bulk_idx, "category"] = "Electronics"
df.loc[bulk_idx, "channel"] = "B2B"
df.loc[bulk_idx, "order_value"] = (df.loc[bulk_idx, "unit_price"] * 400).round(2)

# duplicated rows (e.g. a double export)
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)

out = Path(__file__).with_name("retail_orders.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
