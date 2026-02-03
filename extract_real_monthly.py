#!/usr/bin/env python3
import csv
from datetime import datetime
from collections import defaultdict

# Read the validation trace CSV
print("Reading val_trace.csv to extract REAL trades and monthly returns...")
print()

monthly_trades = defaultdict(list)
equity_curve = []
current_equity = 1.0
in_position = False
entry_price = None
entry_bar = None
entry_time = None

# ATR for stop loss/take profit calculations
atr14 = None

print("Processing trace data...")
with open('val_trace.csv', 'r') as f:
    reader = csv.DictReader(f)
    
    for i, row in enumerate(reader):
        # Skip header-like rows
        if row['BarIndex'].startswith('BarIndex'):
            continue
            
        bar_idx = int(row['BarIndex'])
        time_str = row['Time'].replace('Z', '')
        
        try:
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except:
            continue
            
        month_key = dt.strftime('%Y-%m')
        
        # Get ATR14 for position sizing
        if row['ATR14'] and row['ATR14'] != '0.0000':
            try:
                atr14 = float(row['ATR14'])
            except:
                pass
        
        state = row.get('State', 'FLAT')
        
        # Check for entry (FLAT -> LONG transition)
        if state == 'LONG' and not in_position:
            in_position = True
            entry_price = float(row['Close'])  # Assuming close price for entry
            entry_bar = bar_idx
            entry_time = time_str
            entry_month = month_key
            
        # Check for exit (LONG -> FLAT transition)
        elif state == 'FLAT' and in_position:
            exit_price = float(row['Close'])
            exit_time = time_str
            exit_month = month_key
            
            # Calculate P&L
            pnl = (exit_price - entry_price) / entry_price
            
            # Apply fees (10 bps = 0.10% per side, 5 bps slippage = 0.05% per side)
            # Total cost = 0.30% per round trip
            fee_cost = 0.0030  # 0.30% total
            
            actual_pnl = pnl - fee_cost
            
            # Update equity
            current_equity *= (1 + actual_pnl)
            
            # Store trade
            trade = {
                'entry_month': entry_month,
                'exit_month': exit_month,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': actual_pnl,
                'equity_after': current_equity
            }
            
            # Assign to exit month for monthly returns
            monthly_trades[exit_month].append(trade)
            
            # Reset
            in_position = False
            entry_price = None
        
        # Track equity at each bar for monthly aggregation
        equity_curve.append({
            'month': month_key,
            'equity': current_equity,
            'bar': bar_idx,
            'time': time_str
        })

# Now calculate monthly returns from equity curve
monthly_returns = {}
if equity_curve:
    # Get first equity of each month
    first_eq = equity_curve[0]['equity']
    for record in equity_curve:
        month = record['month']
        if month not in monthly_returns:
            monthly_returns[month] = {
                'start_equity': first_eq,
                'end_equity': first_eq,
                'trades': 0
            }
        monthly_returns[month]['end_equity'] = record['equity']
        monthly_returns[month]['trades'] += 1
        first_eq = record['equity']

# Calculate monthly returns
print("="*130)
print(" "*45 + "REAL VALIDATION WINDOW - MONTHLY RETURNS (Extracted from val_trace.csv)")
print("="*130)
print()

# Sort months
sorted_months = sorted(monthly_returns.keys())

if not sorted_months:
    print("ERROR: No monthly data found. The trace may not have captured position changes correctly.")
    print()
    print("Checking trace file structure...")
    with open('val_trace.csv', 'r') as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line.strip())
            else:
                break
else:
    print(f"{'Month':<10} | {'Return %':<10} | {'Start Eq':<12} | {'End Eq':<12} | {'$10k Start':<14} | {'$10k End':<14} | {'Trades':<8} | {'Status'}")
    print("-"*130)
    
    total_return = 1.0
    neg_months = 0
    
    for month in sorted_months:
        data = monthly_returns[month]
        start_eq = data['start_equity']
        end_eq = data['end_equity']
        
        monthly_return = (end_eq / start_eq) - 1
        total_return = end_eq
        
        start_10k = start_eq * 10000
        end_10k = end_eq * 10000
        
        num_trades = len(monthly_trades.get(month, []))
        
        status = "WIN" if monthly_return >= 0 else "LOSS"
        if monthly_return < 0:
            neg_months += 1
        
        print(f"{month:<10} | {monthly_return*100:>8.2f}% | {start_eq:>10.4f}x | {end_eq:>10.4f}x | ${start_10k:>12,.2f} | ${end_10k:>12,.2f} | {num_trades:>6} | {status:<6}")
    
    print("-"*130)
    print(f"{'TOTAL':<10} | {(total_return-1)*100:>8.2f}% | {1.0:>10.4f}x | {total_return:>10.4f}x | ${10000:>12,.2f} | ${total_return*10000:>12,.2f} | {len(sorted_months):>6} |")
    print()
    print(f"Negative months: {neg_months}/{len(sorted_months)}")
    
    # Save to file
    with open('real_monthly_returns.txt', 'w', encoding='utf-8') as f:
        f.write("="*150 + "\n")
        f.write(" "*50 + "REAL VALIDATION WINDOW - 68 MONTHLY RETURNS\n")
        f.write("="*150 + "\n")
        f.write(f"\nExtracted from actual backtest trace data (val_trace.csv)\n")
        f.write(f"Strategy ID: 1722829493838711123  |  Fee: 10 bps  |  Slippage: 5 bps\n")
        f.write("\n" + "-"*150 + "\n")
        f.write(f"{'Month':<10} | {'Return %':<12} | {'Start Eq':<12} | {'End Eq':<12} | {'$10k Start':<16} | {'$10k End':<16} | {'Trades':<8} | {'Status':<10}\n")
        f.write("-"*150 + "\n")
        
        for month in sorted_months:
            data = monthly_returns[month]
            start_eq = data['start_equity']
            end_eq = data['end_equity']
            monthly_return = (end_eq / start_eq) - 1
            
            start_10k = start_eq * 10000
            end_10k = end_eq * 10000
            num_trades = len(monthly_trades.get(month, []))
            
            status = "WIN" if monthly_return >= 0 else "LOSS"
            
            f.write(f"{month:<10} | {monthly_return*100:>10.2f}% | {start_eq:>10.4f}x | {end_eq:>10.4f}x | ${start_10k:>14,.2f} | ${end_10k:>14,.2f} | {num_trides:>6} | {status:<10}\n")
        
        f.write("-"*150 + "\n")
        total_return = sorted_months and monthly_returns[sorted_months[-1]]['end_equity'] or 1.0
        f.write(f"{'TOTAL':<10} | {(total_return-1)*100:>10.2f}% | {1.0:>10.4f}x | {total_return:>10.4f}x | ${10000:>14,.2f} | ${total_return*10000:>14,.2f} | {len(sorted_months):>6} |\n")
    
    print("\nFile saved: real_monthly_returns.txt")

