package main

import (
	"fmt"
	"os"

	"hb_bactest_checker/logx"
)

// Single-trade debugging mode
// Set this to true to stop after first trade for faster debugging
const debugSingleTrade = false

// EnableSingleTradeMode enables single-trade debugging
// When enabled, the backtest will stop after the first trade closes
// and dump detailed information about that trade
func EnableSingleTradeMode(enabled bool) {
	// This would be set via a flag or environment variable in practice
	// For now, we use the const above
}

// dumpTradeWindow dumps a full window around a trade for debugging
// Includes bars from EntryIdx-1 to ExitIdx+1
func dumpTradeWindow(s Series, f Features, tr Trade) {
	startIdx := tr.EntryIdx - 1
	if startIdx < 0 {
		startIdx = 0
	}
	endIdx := tr.ExitIdx + 2
	if endIdx > len(s.Close) {
		endIdx = len(s.Close)
	}

	logx.PrintTradeWindowHeader()
	fmt.Printf("Trade %d: EntryIdx=%d, ExitIdx=%d, EntryPrice=%.2f, ExitPrice=%.2f, PnL=%.4f\n",
		0, tr.EntryIdx, tr.ExitIdx, tr.EntryPrice, tr.ExitPrice, tr.PnL)
	fmt.Printf("Direction: %s, Reason: %s, HoldBars: %d\n",
		map[int]string{1: "LONG", -1: "SHORT"}[tr.Direction],
		tr.Reason, tr.HoldBars)
	fmt.Println("\n--- OHLC and Feature Values ---")

	for i := startIdx; i < endIdx; i++ {
		isBeforeEntry := (i == tr.EntryIdx-1)
		isEntry := (i == tr.EntryIdx)
		isExit := (i == tr.ExitIdx)
		isAfterExit := (i == tr.ExitIdx+1)

		logx.PrintTradeWindowBar(i, isBeforeEntry, isEntry, isExit, isAfterExit)
		logx.PrintTradeWindowOHLC(s.Open[i], s.High[i], s.Low[i], s.Close[i])

		// Print all feature values at this bar
		fmt.Printf("  Features at t=%d:\n", i)
		for name, idx := range f.Index {
			if idx < len(f.F) && i < len(f.F[idx]) {
				fmt.Printf("    %s[t]=%.4f", name, f.F[idx][i])
				if (idx+1)%5 == 0 {
					fmt.Printf("\n    ")
				}
			}
		}
		fmt.Printf("\n")

		// Print trade-specific information
		if i == tr.EntryIdx-1 {
			fmt.Printf("  >> NEXT BAR: Entry signal should be evaluated here\n")
		} else if i == tr.EntryIdx {
			fmt.Printf("  >> ENTRY EXECUTED: EntryPrice=%.2f (Open with slippage)\n", tr.EntryPrice)
			fmt.Printf("  >> StopLoss=%.2f, TakeProfit=%.2f\n", tr.StopPrice, tr.TPPrice)
		} else if i == tr.ExitIdx {
			fmt.Printf("  >> EXIT EXECUTED: ExitPrice=%.2f, Reason=%s\n", tr.ExitPrice, tr.Reason)
		}
	}

	fmt.Println("\n=== END TRADE WINDOW DUMP ===")
}

// dumpFirstTradeOnly checks if we're in single-trade mode and dumps the first trade
// This is called from coreBacktest when debugSingleTrade is enabled
func handleSingleTradeMode(trades []Trade, s Series, f Features) {
	if !debugSingleTrade || len(trades) == 0 {
		return
	}

	tr := trades[0]

	fmt.Println("\n=== SINGLE TRADE MODE: STOPPING AFTER FIRST TRADE ===")
	fmt.Printf("Trade 0: EntryIdx=%d, ExitIdx=%d, EntryPrice=%.2f, ExitPrice=%.2f, PnL=%.4f\n",
		tr.EntryIdx, tr.ExitIdx, tr.EntryPrice, tr.ExitPrice, tr.PnL)
	fmt.Printf("Direction: %s, Reason: %s, HoldBars: %d\n",
		map[int]string{1: "LONG", -1: "SHORT"}[tr.Direction],
		tr.Reason, tr.HoldBars)

	// Dump full 3-bar window for this trade
	dumpTradeWindow(s, f, tr)

	// Write detailed trace to file if path is set
	if globalTraceConfig.OutputPath != "" {
		writeSingleTradeTrace(tr, s, f, globalTraceConfig.OutputPath)
	}

	fmt.Println("\n=== SINGLE TRADE MODE: EXITING ===")
	os.Exit(0)
}

// writeSingleTradeTrace writes detailed trace information for a single trade
func writeSingleTradeTrace(tr Trade, s Series, features Features, outputPath string) {
	file, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("[ERROR] Failed to open trace file: %v\n", err)
		return
	}
	defer file.Close()

	fmt.Fprintf(file, "\n=== SINGLE TRADE TRACE ===\n")
	fmt.Fprintf(file, "EntryIdx=%d, ExitIdx=%d\n", tr.EntryIdx, tr.ExitIdx)
	fmt.Fprintf(file, "EntryPrice=%.2f, ExitPrice=%.2f\n", tr.EntryPrice, tr.ExitPrice)
	fmt.Fprintf(file, "Direction=%d, Reason=%s\n", tr.Direction, tr.Reason)
	fmt.Fprintf(file, "PnL=%.4f, HoldBars=%d\n", tr.PnL, tr.HoldBars)
	fmt.Fprintf(file, "SignalIndex=%d, SignalTime=%s\n", tr.SignalIndex, tr.SignalTime.Format("2006-01-02 15:04:05"))

	// Dump bars around the trade
	startIdx := tr.EntryIdx - 1
	if startIdx < 0 {
		startIdx = 0
	}
	endIdx := tr.ExitIdx + 1
	if endIdx > len(s.Close) {
		endIdx = len(s.Close)
	}

	for i := startIdx; i < endIdx; i++ {
		fmt.Fprintf(file, "\n--- Bar i=%d ---\n", i)
		fmt.Fprintf(file, "OHLC: O=%.2f H=%.2f L=%.2f C=%.2f\n",
			s.Open[i], s.High[i], s.Low[i], s.Close[i])

		for name, idx := range features.Index {
			if idx < len(features.F) && i < len(features.F[idx]) {
				fmt.Fprintf(file, "%s[t]=%.4f ", name, features.F[idx][i])
			}
		}
		fmt.Fprintf(file, "\n")
	}

	fmt.Fprintf(file, "\n=== END SINGLE TRADE TRACE ===\n")
}

// checkSingleTradeMode should be called after each trade in coreBacktest
// If in single-trade mode and we have at least one trade, it will handle and exit
func checkSingleTradeMode(trades []Trade, s Series, f Features) {
	if debugSingleTrade && len(trades) >= 1 {
		handleSingleTradeMode(trades, s, f)
	}
}

// SingleTradeStats provides detailed statistics about a single trade
type SingleTradeStats struct {
	EntryIndex       int
	ExitIndex        int
	EntryPrice       float32
	ExitPrice        float32
	PnL              float32
	HoldBars         int
	Direction        int
	ExitReason       string
	SignalIndex      int
	SignalToEntry    int  // Bars between signal and entry execution
	EntryToExit      int  // Bars held (should equal HoldBars)
	MaxFavorableMove float32 // Best price during trade
	MaxAdverseMove   float32 // Worst price during trade
}

// analyzeSingleTrade performs detailed analysis of a single trade
func analyzeSingleTrade(tr Trade, s Series, f Features) SingleTradeStats {
	stats := SingleTradeStats{
		EntryIndex:     tr.EntryIdx,
		ExitIndex:      tr.ExitIdx,
		EntryPrice:     tr.EntryPrice,
		ExitPrice:      tr.ExitPrice,
		PnL:            tr.PnL,
		HoldBars:       tr.HoldBars,
		Direction:      tr.Direction,
		ExitReason:     tr.Reason,
		SignalIndex:    tr.SignalIndex,
		SignalToEntry:  tr.EntryIdx - tr.SignalIndex,
		EntryToExit:    tr.ExitIdx - tr.EntryIdx,
	}

	// Calculate max favorable and adverse moves
	if tr.Direction == 1 {
		// Long position
		maxHigh := tr.EntryPrice
		minLow := tr.EntryPrice
		for i := tr.EntryIdx; i <= tr.ExitIdx && i < len(s.High); i++ {
			if s.High[i] > maxHigh {
				maxHigh = s.High[i]
			}
			if s.Low[i] < minLow {
				minLow = s.Low[i]
			}
		}
		stats.MaxFavorableMove = maxHigh - tr.EntryPrice
		stats.MaxAdverseMove = tr.EntryPrice - minLow
	} else {
		// Short position
		maxHigh := tr.EntryPrice
		minLow := tr.EntryPrice
		for i := tr.EntryIdx; i <= tr.ExitIdx && i < len(s.High); i++ {
			if s.High[i] > maxHigh {
				maxHigh = s.High[i]
			}
			if s.Low[i] < minLow {
				minLow = s.Low[i]
			}
		}
		stats.MaxFavorableMove = tr.EntryPrice - minLow
		stats.MaxAdverseMove = maxHigh - tr.EntryPrice
	}

	return stats
}

// printSingleTradeStats prints detailed statistics for a single trade
func printSingleTradeStats(stats SingleTradeStats) {
	fmt.Println("\n=== SINGLE TRADE ANALYSIS ===")
	fmt.Printf("EntryIdx=%d, ExitIdx=%d, HoldBars=%d\n", stats.EntryIndex, stats.ExitIndex, stats.HoldBars)
	fmt.Printf("EntryPrice=%.2f, ExitPrice=%.2f\n", stats.EntryPrice, stats.ExitPrice)
	fmt.Printf("Direction: %s\n", map[int]string{1: "LONG", -1: "SHORT"}[stats.Direction])
	fmt.Printf("ExitReason: %s\n", stats.ExitReason)
	fmt.Printf("PnL: %.4f%%\n", stats.PnL)
	fmt.Printf("\nTiming:\n")
	fmt.Printf("  SignalIndex=%d → EntryIndex=%d (bars: %d)\n", stats.SignalIndex, stats.EntryIndex, stats.SignalToEntry)
	fmt.Printf("  EntryIndex=%d → ExitIndex=%d (bars: %d)\n", stats.EntryIndex, stats.ExitIndex, stats.EntryToExit)
	fmt.Printf("\nPrice Movement:\n")
	fmt.Printf("  Max Favorable Move: %.2f (%.2f%%)\n", stats.MaxFavorableMove,
		(stats.MaxFavorableMove/stats.EntryPrice)*100)
	fmt.Printf("  Max Adverse Move: %.2f (%.2f%%)\n", stats.MaxAdverseMove,
		(stats.MaxAdverseMove/stats.EntryPrice)*100)
	fmt.Println("=== END SINGLE TRADE ANALYSIS ===")
}
