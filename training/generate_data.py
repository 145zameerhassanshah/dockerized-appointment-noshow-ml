from pathlib import Path
import numpy as np
import pandas as pd

out = Path("data/appointments.csv")
out.parent.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 2500

age = rng.integers(18, 86, n)
days = rng.integers(0, 31, n)
prev = rng.integers(0, 15, n)
prev_ns = np.array([rng.integers(0, max(1, min(6, x+1))) for x in prev])
sms = rng.choice(["yes","no"], n, p=[0.72,0.28])
hour = rng.integers(8,18,n)
weekday = rng.choice(["Monday","Tuesday","Wednesday","Thursday","Friday"], n)
dept = rng.choice(["General Medicine","Dermatology","Orthopedics","Pediatrics"], n)

logit = -2.6 + 0.10*days + 0.55*prev_ns - 0.32*prev + 0.65*(sms=="no") + 0.20*(hour>=16)
p = 1/(1+np.exp(-logit))
target = rng.binomial(1, np.clip(p,0.02,0.95))

df = pd.DataFrame({
    "age": age,
    "days_until_appointment": days,
    "previous_appointments": prev,
    "previous_no_shows": prev_ns,
    "sms_reminder_sent": sms,
    "appointment_hour": hour,
    "weekday": weekday,
    "department": dept,
    "no_show": target,
})

for c in ["age","days_until_appointment","sms_reminder_sent","department"]:
    idx = rng.choice(df.index, int(0.02*n), replace=False)
    df.loc[idx, c] = np.nan

df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
