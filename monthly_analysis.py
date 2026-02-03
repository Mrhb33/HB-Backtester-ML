# Monthly returns analysis based on golden mode output

trades = [
    #1 [LONG] 2024-01-03 -> 2024-01-03, PnL=-0.7271%
    {"date": "2024-01-03", "pnl": -0.7271, "month": "2024-01"},
    #2 [LONG] 2024-01-11 -> 2024-01-12, PnL=-3.4819%
    {"date": "2024-01-12", "pnl": -3.4819, "month": "2024-01"},
    #3 [LONG] 2024-01-12 -> 2024-01-13, PnL=-3.3546%
    {"date": "2024-01-13", "pnl": -3.3546, "month": "2024-01"},
    #4 [LONG] 2024-03-05 -> 2024-03-05, PnL=-3.8831%
    {"date": "2024-03-05", "pnl": -3.8831, "month": "2024-03"},
    #5 [LONG] 2024-03-05 -> 2024-03-11, PnL=+14.8054%
    {"date": "2024-03-11", "pnl": 14.8054, "month": "2024-03"},
    #6 [LONG] 2024-03-15 -> 2024-03-15, PnL=-3.9258%
    {"date": "2024-03-15", "pnl": -3.9258, "month": "2024-03"},
    #7 [LONG] 2024-04-02 -> 2024-04-02, PnL=-2.5981%
    {"date": "2024-04-02", "pnl": -2.5981, "month": "2024-04"},
    #8 [LONG] 2024-04-12 -> 2024-04-13, PnL=-3.1473%
    {"date": "2024-04-13", "pnl": -3.1473, "month": "2024-04"},
    #9 [LONG] 2024-04-14 -> 2024-04-15, PnL=+3.2080%
    {"date": "2024-04-15", "pnl": 3.2080, "month": "2024-04"},
    #10 [LONG] 2024-04-19 -> 2024-04-22, PnL=+4.8718%
    {"date": "2024-04-22", "pnl": 4.8718, "month": "2024-04"},
]

# Calculate monthly returns
monthly_returns = {}
for trade in trades:
    month = trade["month"]
    if month not in monthly_returns:
        monthly_returns[month] = 1.0  # Start with 1.0 equity
    monthly_returns[month] *= (1 + trade["pnl"] / 100)

print("=" * 80)
print("MONTHLY RETURNS ANALYSIS (TEST WINDOW)")
print("=" * 80)
print()
print(f"{'Month':<12} | {'Return %':<10} | {'Trades':<7} | {'Equity Growth':<15}")
print("-" * 80)

prev_equity = 1.0
all_months = sorted(monthly_returns.keys())
for month in all_months:
    return_pct = (monthly_returns[month] - 1) * 100
    num_trades = sum(1 for t in trades if t["month"] == month)
    print(f"{month:<12} | {return_pct:>8.2f}% | {num_trades:>5}   | {monthly_returns[month]:>14.6f}x")
    prev_equity = monthly_returns[month]

total_return = (prev_equity - 1) * 100
print("-" * 80)
print(f"{'TOTAL':<12} | {total_return:>8.2f}% | {len(trades):>5}   | {prev_equity:>14.6f}x")
print()
print("NOTE: This is the TEST WINDOW (2024 data), different from VALIDATION window")
print("      The 11.98x return was from VALIDATION/OOS period, not this test period.")
