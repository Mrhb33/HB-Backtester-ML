package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"time"
)

// SimpleStrategy is a manually defined strategy for testing and verification
// It uses the same indicators and logic as the backtest engine
type SimpleStrategy struct {
	Name         string
	Direction    int // 1 = long, -1 = short
	FeeBps       float32
	SlippageBps  float32
	EntryRule    SimpleRule
	ExitRule     SimpleRule
	StopLossPct  float32
	TakeProfitPct float32
	MaxHoldBars  int
}

// SimpleRule represents a rule condition for entry/exit
type SimpleRule struct {
	Type string // "cross_up", "cross_down", "gt", "lt", "rising", "falling", "and", "or"
	A    string // Feature name A
	B    string // Feature name B (for cross rules)
	X    float32 // Threshold value (for gt/lt rules)
	N    int     // Lookback period (for rising/falling)
	LHS  *SimpleRule // Left-hand side (for and/or)
	RHS  *SimpleRule // Right-hand side (for and/or)
}

// SimpleTrade represents a completed trade
type SimpleTrade struct {
	EntryIdx   int
	EntryTime  time.Time
	EntryPrice float32
	ExitIdx    int
	ExitTime   time.Time
	ExitPrice  float32
	ExitReason string
	PnL        float32
}

// runSimpleStrategy executes a simple strategy and outputs CSV states
// This uses the same data and indicators as the full backtest
// Output format matches the trace mode: BarIndex,Time,State with detailed debug info
func runSimpleStrategy(s Series, f Features, strategy SimpleStrategy, csvPath string) ([]SimpleTrade, error) {
	const warmup = 200
	if len(s.Close) < warmup+100 {
		return nil, fmt.Errorf("not enough data (need %d, got %d)", warmup+100, len(s.Close))
	}

	file, err := os.Create(csvPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	var trades []SimpleTrade
	inPosition := false
	var entryPrice, stopLoss, takeProfit float32
	var entryIdx int
	var entryTime time.Time
	posDirection := 0 // 0=flat, 1=long, -1=short

	// Track consecutive states for signal generation (cross detection)
	// Entry signals fire on the bar AFTER the condition is met (next bar open)
	entrySignalPending := false
	pendingSignalDirection := 0

	for t := warmup; t < len(s.Close); t++ {
		state := "FLAT"
		details := ""

		// Check entry rule (only if not in position)
		if !inPosition && !entrySignalPending {
			signalDetected, signalDetails := evaluateSimpleRuleWithDetails(s, strategy.EntryRule, f, t)
			if signalDetected {
				// Signal generated - will enter on next bar
				entrySignalPending = true
				pendingSignalDirection = strategy.Direction
				state = "SIGNAL DETECT," + signalDetails
				// Write this row immediately (don't process entry on same bar)
				ts := time.Unix(s.CloseTimeMs[t]/1000, 0).Format(time.RFC3339)
				row := []string{fmt.Sprintf("%d", t), ts, state}
				if err := w.Write(row); err != nil {
					return nil, err
				}
				continue // Skip rest of loop for this bar
			}
		}

		// Process pending entry (enter at open of next bar after signal)
		if entrySignalPending && !inPosition {
			entryPrice = s.Open[t]
			entryIdx = t
			entryTime = time.Unix(s.OpenTimeMs[t]/1000, 0)
			posDirection = pendingSignalDirection
			inPosition = true

			// Calculate stop loss and take profit
			if strategy.StopLossPct > 0 {
				if posDirection == 1 {
					stopLoss = entryPrice * (1 - strategy.StopLossPct/100)
				} else {
					stopLoss = entryPrice * (1 + strategy.StopLossPct/100)
				}
			}
			if strategy.TakeProfitPct > 0 {
				if posDirection == 1 {
					takeProfit = entryPrice * (1 + strategy.TakeProfitPct/100)
				} else {
					takeProfit = entryPrice * (1 - strategy.TakeProfitPct/100)
				}
			}

			entrySignalPending = false
			state = "ENTRY EXEC"
			// Write ENTRY EXEC row immediately
			ts := time.Unix(s.CloseTimeMs[t]/1000, 0).Format(time.RFC3339)
			row := []string{fmt.Sprintf("%d", t), ts, state}
			if err := w.Write(row); err != nil {
				return nil, err
			}
			continue // Skip rest of loop for this bar
		}

		// Check exit conditions if in position
		exitReason := ""
		if inPosition {
			state = "HOLDING"

			// Check stop loss
			if posDirection == 1 && s.Low[t] <= stopLoss {
				exitReason = "H-SL"
			} else if posDirection == -1 && s.High[t] >= stopLoss {
				exitReason = "H-SL"
			}

			// Check take profit
			if exitReason == "" {
				if posDirection == 1 && s.High[t] >= takeProfit {
					exitReason = "H-TP"
				} else if posDirection == -1 && s.Low[t] <= takeProfit {
					exitReason = "H-TP"
				}
			}

			// Check exit rule
			if exitReason == "" && evaluateSimpleRule(strategy.ExitRule, f, t) {
				exitReason = "EXIT RULE"
			}

			// Check max hold
			if exitReason == "" && strategy.MaxHoldBars > 0 && (t-entryIdx) >= strategy.MaxHoldBars {
				exitReason = "H-MAX"
			}

			// Execute exit
			if exitReason != "" {
				exitPrice := s.Close[t]

				// Adjust exit price if SL/TP hit intrabar
				if exitReason == "H-SL" {
					exitPrice = stopLoss
				} else if exitReason == "H-TP" {
					exitPrice = takeProfit
				}

				// Calculate final PnL with fees and slippage
				totalCostBps := strategy.FeeBps + strategy.SlippageBps
				costPct := totalCostBps / 10000.0

				pnl := float32(0)
				if posDirection == 1 {
					pnl = (exitPrice - entryPrice) / entryPrice * 100
				} else {
					pnl = (entryPrice - exitPrice) / entryPrice * 100
				}
				pnl -= costPct * 2 // Entry and exit costs

				trades = append(trades, SimpleTrade{
					EntryIdx:   entryIdx,
					EntryTime:  entryTime,
					EntryPrice: entryPrice,
					ExitIdx:    t,
					ExitTime:   time.Unix(s.OpenTimeMs[t]/1000, 0),
					ExitPrice:  exitPrice,
					ExitReason: exitReason,
					PnL:        pnl,
				})

				state = exitReason
				inPosition = false
				entryPrice = 0
				stopLoss = 0
				takeProfit = 0
				posDirection = 0
			}
		}

		// Write CSV row
		ts := time.Unix(s.CloseTimeMs[t]/1000, 0).Format(time.RFC3339)
		row := []string{fmt.Sprintf("%d", t), ts, state + details}
		if err := w.Write(row); err != nil {
			return nil, err
		}
	}

	return trades, nil
}

// evaluateSimpleRuleWithDetails evaluates a rule and returns detailed debug info
func evaluateSimpleRuleWithDetails(s Series, rule SimpleRule, f Features, t int) (bool, string) {
	if t < 1 {
		return false, ""
	}

	switch rule.Type {
	case "cross_up":
		aIdx, okA := f.Index[rule.A]
		bIdx, okB := f.Index[rule.B]
		if !okA || !okB {
			return false, ""
		}
		prevA := f.F[aIdx][t-1]
		prevB := f.F[bIdx][t-1]
		currA := f.F[aIdx][t]
		currB := f.F[bIdx][t]

		prevCond := prevA <= prevB
		currCond := currA > currB
		result := prevCond && currCond

		details := fmt.Sprintf("Close=%f Open=%f %s[t-1]=%f %s[t-1]=%f %s[t]=%f %s[t]=%f CrossUp: prevA(%f)<=prevB(%f)=%v curA(%f)>curB(%f)=%v",
			s.Close[t], s.Open[t], rule.A, prevA, rule.B, prevB, rule.A, currA, rule.B, currB,
			prevA, prevB, prevCond, currA, currB, currCond)

		return result, details

	case "cross_down":
		aIdx, okA := f.Index[rule.A]
		bIdx, okB := f.Index[rule.B]
		if !okA || !okB {
			return false, ""
		}
		prevA := f.F[aIdx][t-1]
		prevB := f.F[bIdx][t-1]
		currA := f.F[aIdx][t]
		currB := f.F[bIdx][t]

		prevCond := prevA >= prevB
		currCond := currA < currB
		result := prevCond && currCond

		details := fmt.Sprintf("Close=%f Open=%f %s[t-1]=%f %s[t-1]=%f %s[t]=%f %s[t]=%f CrossDown: prevA(%f)>=prevB(%f)=%v curA(%f)<curB(%f)=%v",
			s.Close[t], s.Open[t], rule.A, prevA, rule.B, prevB, rule.A, currA, rule.B, currB,
			prevA, prevB, prevCond, currA, currB, currCond)

		return result, details

	case "gt":
		aIdx, ok := f.Index[rule.A]
		if !ok {
			return false, ""
		}
		result := f.F[aIdx][t] > rule.X
		details := fmt.Sprintf("Close=%f %s[t]=%f > %f = %v", s.Close[t], rule.A, f.F[aIdx][t], rule.X, result)
		return result, details

	case "lt":
		aIdx, ok := f.Index[rule.A]
		if !ok {
			return false, ""
		}
		result := f.F[aIdx][t] < rule.X
		details := fmt.Sprintf("Close=%f %s[t]=%f < %f = %v", s.Close[t], rule.A, f.F[aIdx][t], rule.X, result)
		return result, details

	case "rising":
		aIdx, ok := f.Index[rule.A]
		if !ok || t < rule.N {
			return false, ""
		}
		result := f.F[aIdx][t] > f.F[aIdx][t-rule.N]
		details := fmt.Sprintf("Close=%f %s[t]=%f %s[t-%d]=%f Rising: %v",
			s.Close[t], rule.A, f.F[aIdx][t], rule.A, rule.N, f.F[aIdx][t-rule.N], result)
		return result, details

	case "falling":
		aIdx, ok := f.Index[rule.A]
		if !ok || t < rule.N {
			return false, ""
		}
		result := f.F[aIdx][t] < f.F[aIdx][t-rule.N]
		details := fmt.Sprintf("Close=%f %s[t]=%f %s[t-%d]=%f Falling: %v",
			s.Close[t], rule.A, f.F[aIdx][t], rule.A, rule.N, f.F[aIdx][t-rule.N], result)
		return result, details

	case "and":
		if rule.LHS == nil || rule.RHS == nil {
			return false, ""
		}
		leftRes, leftDetails := evaluateSimpleRuleWithDetails(s, *rule.LHS, f, t)
		rightRes, rightDetails := evaluateSimpleRuleWithDetails(s, *rule.RHS, f, t)
		result := leftRes && rightRes
		details := fmt.Sprintf("AND: (%v) && (%v) = %v", leftDetails, rightDetails, result)
		return result, details

	case "or":
		if rule.LHS == nil || rule.RHS == nil {
			return false, ""
		}
		leftRes, leftDetails := evaluateSimpleRuleWithDetails(s, *rule.LHS, f, t)
		rightRes, rightDetails := evaluateSimpleRuleWithDetails(s, *rule.RHS, f, t)
		result := leftRes || rightRes
		details := fmt.Sprintf("OR: (%v) || (%v) = %v", leftDetails, rightDetails, result)
		return result, details

	default:
		return false, ""
	}
}

// evaluateSimpleRule evaluates a simple rule condition at time t
func evaluateSimpleRule(rule SimpleRule, f Features, t int) bool {
	if t < 1 {
		return false
	}

	switch rule.Type {
	case "cross_up":
		aIdx, okA := f.Index[rule.A]
		bIdx, okB := f.Index[rule.B]
		if !okA || !okB {
			return false
		}
		if t < 1 {
			return false
		}
		prevA := f.F[aIdx][t-1]
		prevB := f.F[bIdx][t-1]
		currA := f.F[aIdx][t]
		currB := f.F[bIdx][t]
		return prevA <= prevB && currA > currB

	case "cross_down":
		aIdx, okA := f.Index[rule.A]
		bIdx, okB := f.Index[rule.B]
		if !okA || !okB {
			return false
		}
		if t < 1 {
			return false
		}
		prevA := f.F[aIdx][t-1]
		prevB := f.F[bIdx][t-1]
		currA := f.F[aIdx][t]
		currB := f.F[bIdx][t]
		return prevA >= prevB && currA < currB

	case "gt":
		aIdx, ok := f.Index[rule.A]
		if !ok {
			return false
		}
		return f.F[aIdx][t] > rule.X

	case "lt":
		aIdx, ok := f.Index[rule.A]
		if !ok {
			return false
		}
		return f.F[aIdx][t] < rule.X

	case "rising":
		aIdx, ok := f.Index[rule.A]
		if !ok || t < rule.N {
			return false
		}
		return f.F[aIdx][t] > f.F[aIdx][t-rule.N]

	case "falling":
		aIdx, ok := f.Index[rule.A]
		if !ok || t < rule.N {
			return false
		}
		return f.F[aIdx][t] < f.F[aIdx][t-rule.N]

	case "and":
		if rule.LHS == nil || rule.RHS == nil {
			return false
		}
		return evaluateSimpleRule(*rule.LHS, f, t) && evaluateSimpleRule(*rule.RHS, f, t)

	case "or":
		if rule.LHS == nil || rule.RHS == nil {
			return false
		}
		return evaluateSimpleRule(*rule.LHS, f, t) || evaluateSimpleRule(*rule.RHS, f, t)

	default:
		return false
	}
}

// Helper functions to create common strategies

// createEMACrossStrategy creates a simple EMA crossover strategy
func createEMACrossStrategy(fastPeriod, slowPeriod int, feeBps, slipBps float32) SimpleStrategy {
	fastName := fmt.Sprintf("EMA%d", fastPeriod)
	slowName := fmt.Sprintf("EMA%d", slowPeriod)

	return SimpleStrategy{
		Name:         fmt.Sprintf("EMA%dx%d_Cross", fastPeriod, slowPeriod),
		Direction:    1, // Long
		FeeBps:       feeBps,
		SlippageBps:  slipBps,
		StopLossPct:  2.0,  // 2% stop loss
		TakeProfitPct: 4.0, // 4% take profit (2:1 reward:risk)
		MaxHoldBars:  150,
		EntryRule: SimpleRule{
			Type: "cross_up",
			A:    fastName,
			B:    slowName,
		},
		ExitRule: SimpleRule{
			Type: "cross_down",
			A:    fastName,
			B:    slowName,
		},
	}
}

// createRSIEMAStrategy creates an RSI + EMA combo strategy
func createRSIEMAStrategy(rsiPeriod int, rsiOversold, emaPeriod int, feeBps, slipBps float32) SimpleStrategy {
	rsiName := fmt.Sprintf("RSI%d", rsiPeriod)
	emaName := fmt.Sprintf("EMA%d", emaPeriod)

	return SimpleStrategy{
		Name:         fmt.Sprintf("RSI%d_EMA%d", rsiPeriod, emaPeriod),
		Direction:    1, // Long
		FeeBps:       feeBps,
		SlippageBps:  slipBps,
		StopLossPct:  1.5,
		TakeProfitPct: 3.0,
		MaxHoldBars:  100,
		EntryRule: SimpleRule{
			Type: "and",
			LHS: &SimpleRule{
				Type: "lt",
				A:    rsiName,
				X:    float32(rsiOversold),
			},
			RHS: &SimpleRule{
				Type: "rising",
				A:    emaName,
				N:    5,
			},
		},
		ExitRule: SimpleRule{
			Type: "gt",
			A:    rsiName,
			X:    70, // Overbought
		},
	}
}

// createMACDCrossStrategy creates a MACD crossover strategy
func createMACDCrossStrategy(feeBps, slipBps float32) SimpleStrategy {
	return SimpleStrategy{
		Name:         "MACD_Cross",
		Direction:    1, // Long
		FeeBps:       feeBps,
		SlippageBps:  slipBps,
		StopLossPct:  2.5,
		TakeProfitPct: 5.0,
		MaxHoldBars:  120,
		EntryRule: SimpleRule{
			Type: "cross_up",
			A:    "MACD",
			B:    "MACD_Signal",
		},
		ExitRule: SimpleRule{
			Type: "cross_down",
			A:    "MACD",
			B:    "MACD_Signal",
		},
	}
}

// printSimpleStrategyResults prints summary statistics for a simple strategy backtest
func printSimpleStrategyResults(trades []SimpleTrade, strategy SimpleStrategy) {
	fmt.Println("\n=== Simple Strategy Results ===")
	fmt.Printf("Strategy: %s\n", strategy.Name)
	fmt.Printf("Direction: %s\n", map[int]string{1: "LONG", -1: "SHORT"}[strategy.Direction])
	fmt.Printf("Stop Loss: %.2f%%, Take Profit: %.2f%%\n", strategy.StopLossPct, strategy.TakeProfitPct)
	fmt.Printf("Fees: %.2f bps, Slippage: %.2f bps\n", strategy.FeeBps, strategy.SlippageBps)
	fmt.Println()

	if len(trades) == 0 {
		fmt.Println("No trades executed!")
		return
	}

	totalReturn := float32(0)
	wins := 0
	losses := 0
	grossWin := float32(0)
	grossLoss := float32(0)
	totalHoldBars := 0

	exitReasons := make(map[string]int)

	for _, t := range trades {
		totalReturn += t.PnL
		totalHoldBars += t.ExitIdx - t.EntryIdx
		exitReasons[t.ExitReason]++

		if t.PnL > 0 {
			wins++
			grossWin += t.PnL
		} else {
			losses++
			grossLoss += -t.PnL
		}
	}

	winRate := float32(wins) / float32(len(trades)) * 100
	expectancy := totalReturn / float32(len(trades))
	avgHoldBars := float32(totalHoldBars) / float32(len(trades))

	profitFactor := float32(0)
	if grossLoss > 0 {
		profitFactor = grossWin / grossLoss
	}

	fmt.Printf("Total Trades: %d\n", len(trades))
	fmt.Printf("Total Return: %.2f%%\n", totalReturn)
	fmt.Printf("Win Rate: %.2f%%\n", winRate)
	fmt.Printf("Expectancy: %.4f%%\n", expectancy)
	fmt.Printf("Profit Factor: %.2f\n", profitFactor)
	fmt.Printf("Avg Hold Bars: %.1f\n", avgHoldBars)
	fmt.Println()
	fmt.Println("Exit Reasons:")
	for reason, count := range exitReasons {
		fmt.Printf("  %s: %d\n", reason, count)
	}
	fmt.Println()

	// Find best and worst trades
	if len(trades) > 0 {
		bestTrade := trades[0]
		worstTrade := trades[0]

		for _, t := range trades {
			if t.PnL > bestTrade.PnL {
				bestTrade = t
			}
			if t.PnL < worstTrade.PnL {
				worstTrade = t
			}
		}

		fmt.Printf("Best Trade: %.2f%% (Entry: %s, Exit: %s)\n",
			bestTrade.PnL, bestTrade.EntryTime.Format("2006-01-02 15:04"),
			bestTrade.ExitTime.Format("2006-01-02 15:04"))
		fmt.Printf("Worst Trade: %.2f%% (Entry: %s, Exit: %s)\n",
			worstTrade.PnL, worstTrade.EntryTime.Format("2006-01-02 15:04"),
			worstTrade.ExitTime.Format("2006-01-02 15:04"))
	}
}
