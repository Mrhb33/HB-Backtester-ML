#!/usr/bin/env python3
"""
Check if winners.jsonl contains monthly returns data
"""
import json

print("Testing monthly returns saving to winners.jsonl")
print("=" * 60)

# Read winners.jsonl
with open('winners.jsonl', 'r') as f:
    lines = f.readlines()

print(f"Total entries in winners.jsonl: {len(lines)}")

# Count entries with monthly returns
count = 0
has_monthly_returns = 0
sample_entry = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    count += 1

    try:
        strategy = json.loads(line)
    except:
        continue

    if 'oos_monthly_returns' in strategy:
        has_monthly_returns += 1
        if sample_entry is None:
            sample_entry = strategy

print(f"Entries with oos_monthly_returns: {has_monthly_returns}")

if has_monthly_returns > 0:
    print("\nSUCCESS: Monthly returns are being saved!")

    # Show a sample entry with monthly returns
    print("\nSample entry with monthly returns:")
    print(f"  Seed: {sample_entry.get('seed')}")
    print(f"  Fee BPS: {sample_entry.get('fee_bps')}")
    print(f"  OOS Final Equity: {sample_entry.get('oos_final_equity')}")

    monthly_returns = sample_entry.get('oos_monthly_returns', [])
    print(f"  Monthly Returns Count: {len(monthly_returns)}")

    if len(monthly_returns) > 0:
        print("  First 3 months:")
        for mr in monthly_returns[:3]:
            month = mr.get('month')
            ret = mr.get('return')
            trades = mr.get('trades')
            print(f"    Month {month}: {ret*100:.2f}% ({trades} trades)")
else:
    print("\nINFO: No entries with oos_monthly_returns found yet.")
    print("     New entries will include monthly returns after the code changes.")
    print("\nTo test:")
    print("  1. Run the search engine: go run . -wf")
    print("  2. Wait for strategies to pass walk-forward validation")
    print("  3. Run this test again to verify monthly returns are saved")
