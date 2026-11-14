"""Generate two more practice datasets, each with traps the agent should catch.

    python sample_data/make_more_samples.py

student_performance.csv (1,215 rows)
  - Simpson's paradox: tutored students score LOWER overall, but HIGHER inside every school
    (tutoring is concentrated in the school with the weakest intake).
  - City spelled several ways (Bengaluru / Bangalore / bengaluru, Mumbai / Bombay ...).
  - Mixed date formats in enroll_date, missing attendance values, 15 duplicated rows,
    and a few impossible study_hours_week values (typos like 120).

saas_subscriptions.xlsx (sheet "subscriptions", 900 rows; sheet "about" explains the columns)
  - Enterprise is only ~10 customers but holds over 40% of revenue (small group + outliers).
  - Support tickets correlate with churn (correlation, not proof of cause).
  - Many blank NPS scores; country names with inconsistent spelling.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)

# --------------------------------------------------------------------------- students
schools = {
    # school: (base score, share of students who get tutoring, cities)
    "Greenfield High": (78, 0.15, ["Bengaluru", "Mysuru"]),
    "Riverside Public": (66, 0.35, ["Mumbai", "Pune"]),
    "Hillcrest Academy": (52, 0.80, ["Delhi", "Jaipur"]),
}
city_variants = {"Bengaluru": ["Bangalore", "bengaluru", "Bengaluru "], "Mumbai": ["Bombay", "mumbai"],
                 "Delhi": ["New Delhi", "delhi"]}
parent_edu = ["High school", "Bachelor's", "Master's", "PhD"]
rows = []
dates = pd.date_range("2023-06-01", "2025-07-31", freq="D")
for i in range(1200):
    school = rng.choice(list(schools), p=[0.34, 0.33, 0.33])
    base, tutor_share, cities = schools[school]
    tutoring = rng.random() < tutor_share
    hours = float(np.clip(rng.normal(9 + (2 if tutoring else 0), 3), 0, 30))
    attendance = float(np.clip(rng.normal(88, 7), 50, 100))
    pe = rng.choice(parent_edu, p=[0.35, 0.4, 0.2, 0.05])
    lift = 6 if tutoring else 0                       # tutoring helps within a school
    math = np.clip(rng.normal(base + lift + 0.6 * (hours - 9) + 0.15 * (attendance - 88), 9), 0, 100)
    reading = np.clip(math + rng.normal(3, 7), 0, 100)
    city = str(rng.choice(cities))
    if city in city_variants and rng.random() < 0.2:
        city = str(rng.choice(city_variants[city]))
    d = dates[rng.integers(len(dates))]
    rows.append({
        "student_id": f"S{10000 + i}",
        "enroll_date": d.strftime("%d-%m-%Y") if rng.random() < 0.2 else d.strftime("%Y-%m-%d"),
        "school": school,
        "city": city,
        "gender": str(rng.choice(["Female", "Male", "Non-binary"], p=[0.49, 0.49, 0.02])),
        "parent_education": pe,
        "study_hours_week": round(hours, 1),
        "attendance_pct": round(attendance, 1) if rng.random() > 0.06 else None,
        "tutoring": "Yes" if tutoring else "No",
        "math_score": int(round(math)),
        "reading_score": int(round(reading)),
    })
students = pd.DataFrame(rows)
students["passed"] = np.where((students.math_score + students.reading_score) / 2 >= 60, "Yes", "No")
typo_idx = rng.choice(len(students), 5, replace=False)
students.loc[typo_idx, "study_hours_week"] = [120, 95, 140, 88, 110]   # data-entry typos
students = pd.concat([students, students.sample(15, random_state=3)], ignore_index=True)
students.to_csv(HERE / "student_performance.csv", index=False)

# --------------------------------------------------------------------------- SaaS
# plan: (price per seat per month, min seats, max seats)
plans = {"Basic": (29, 1, 1), "Pro": (19, 5, 25), "Enterprise": (45, 80, 400)}
countries = ["India", "United States", "United Kingdom", "Germany", "Singapore", "Brazil"]
country_variants = {"United States": ["USA", "U.S.", "united states"], "United Kingdom": ["UK", "U.K."],
                    "India": ["india"]}
rows = []
signup = pd.date_range("2024-01-01", "2025-10-31", freq="D")
for i in range(900):
    plan = rng.choice(list(plans), p=[0.5, 0.4867, 0.0133])
    price, smin, smax = plans[plan]
    seats = int(rng.integers(smin, smax + 1))
    revenue = price * seats * (1 - rng.choice([0, 0, 0.1, 0.2]))
    tickets = int(rng.poisson(1.5 + (2 if plan == "Basic" else 0)))
    churn_p = 1 / (1 + np.exp(-(-2.6 + 0.45 * tickets - (1.5 if plan == "Enterprise" else 0))))
    churned = rng.random() < churn_p
    nps = int(np.clip(round(rng.normal(8 - 0.6 * tickets - (2 if churned else 0), 1.8)), 0, 10))
    country = str(rng.choice(countries, p=[0.3, 0.25, 0.12, 0.1, 0.1, 0.13]))
    if country in country_variants and rng.random() < 0.18:
        country = str(rng.choice(country_variants[country]))
    rows.append({
        "customer_id": f"C{5000 + i}",
        "signup_date": signup[rng.integers(len(signup))].date(),
        "plan": plan,
        "country": country,
        "seats": seats,
        "monthly_revenue_usd": round(float(revenue), 2),
        "support_tickets_90d": tickets,
        "nps": nps if rng.random() > 0.35 else None,
        "churned": "Yes" if churned else "No",
    })
saas = pd.DataFrame(rows)
about = pd.DataFrame({
    "column": ["customer_id", "signup_date", "plan", "country", "seats", "monthly_revenue_usd",
               "support_tickets_90d", "nps", "churned"],
    "meaning": ["Customer ID", "Date the account was created", "Basic, Pro or Enterprise",
                "Billing country (spelling is inconsistent on purpose)", "Paid seats",
                "Monthly recurring revenue in USD", "Support tickets in the last 90 days",
                "Net Promoter Score 0-10 (blank = no survey answer)", "Cancelled? Yes/No"],
})
with pd.ExcelWriter(HERE / "saas_subscriptions.xlsx") as xl:
    saas.to_excel(xl, sheet_name="subscriptions", index=False)   # first sheet = the data the app loads
    about.to_excel(xl, sheet_name="about", index=False)

print(f"student_performance.csv: {len(students)} rows | saas_subscriptions.xlsx: {len(saas)} rows")
