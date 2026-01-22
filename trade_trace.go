package main

import (
	"fmt"
	"os"
)

// IndexMap tracks the relationship between window-local and global indices
type IndexMap struct {
	GlobalStart int  // Global index that maps to local 0
	LocalWarmup int  // Local warmup bars
}

// LogIndex returns a formatted string showing both local and global indices
func (m IndexMap) LogIndex(localIdx int) string {
	globalIdx := m.GlobalStart + localIdx
	return fmt.Sprintf("tLocal=%d, windowStart=%d, tGlobal=%d", localIdx, m.GlobalStart, globalIdx)
}

// TradeTraceConfig holds configuration for trade tracing
type TradeTraceConfig struct {
	OutputPath string      // Path to CSV output file
	IndexMap   IndexMap    // Index mapping
	Series     Series      // OHLCV data
	Features   Features    // Feature data
	Strategy   Strategy    // Strategy being traced
}

// TraceEntryEvent logs a 3-bar window for an entry event
// This proves/disproves lookahead in one screenshot
func TraceEntryEvent(cfg TradeTraceConfig, tLocal, signalIdx, entryIdx int, dir int, entryPrice float32) {
	if cfg.OutputPath == "" {
		return
	}

	f, err := os.OpenFile(cfg.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("[ERROR] Failed to open trace file: %v\n", err)
		return
	}
	defer f.Close()

	warmup := cfg.IndexMap.LocalWarmup

	// ENTRY EVENT HEADER
	_ = cfg.IndexMap.GlobalStart + tLocal // globalIdx calculated for reference
	fmt.Fprintf(f, "\n=== ENTRY EVENT ===\n")
	fmt.Fprintf(f, "%s\n", cfg.IndexMap.LogIndex(tLocal))
	fmt.Fprintf(f, "Signal detected at: tEval=%d (global=%d)\n", signalIdx, cfg.IndexMap.GlobalStart+signalIdx)
	fmt.Fprintf(f, "Entry executed at:   tExec=%d (global=%d)\n", entryIdx, cfg.IndexMap.GlobalStart+entryIdx)
	fmt.Fprintf(f, "Direction: %s, EntryPrice: %.2f\n", map[int]string{1: "LONG", -1: "SHORT"}[dir], entryPrice)

	// Dump tEval-1 bar (the bar before signal detection)
	tEvalMinusOne := signalIdx - 1
	if tEvalMinusOne >= warmup {
		dumpBarWithFeatures(f, cfg, tEvalMinusOne, "Bar tEval-1", "SIGNAL DETECTED AT NEXT BAR")
	}

	// Dump tEval bar (signal detection bar)
	dumpBarWithFeatures(f, cfg, signalIdx, "Bar tEval (SIGNAL DETECTED)", "Action: Schedule entry for t+1")

	// Dump tExec bar (entry execution bar)
	dumpBarWithFeatures(f, cfg, entryIdx, "Bar tExec (ENTRY EXECUTED)", fmt.Sprintf("EntryPrice: %.2f (Open with slippage)", entryPrice))
}

// TraceExitEvent logs a 2-bar window for an exit event
// This proves/disproves lookahead in one screenshot
func TraceExitEvent(cfg TradeTraceConfig, tLocal, evalIdx, exitIdx int, exitRule bool, exitPrice float32, reason string) {
	if cfg.OutputPath == "" {
		return
	}

	f, err := os.OpenFile(cfg.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("[ERROR] Failed to open trace file: %v\n", err)
		return
	}
	defer f.Close()

	// EXIT EVENT HEADER
	fmt.Fprintf(f, "\n=== EXIT EVENT ===\n")
	fmt.Fprintf(f, "%s\n", cfg.IndexMap.LogIndex(tLocal))
	fmt.Fprintf(f, "Exit rule evaluated at: tEval=%d (global=%d)\n", evalIdx, cfg.IndexMap.GlobalStart+evalIdx)
	fmt.Fprintf(f, "Exit executed at:       tExec=%d (global=%d)\n", exitIdx, cfg.IndexMap.GlobalStart+exitIdx)
	fmt.Fprintf(f, "ExitRule result: %v, Reason: %s\n", exitRule, reason)

	// Dump tEval bar (exit rule evaluation bar)
	dumpBarWithFeatures(f, cfg, evalIdx, "Bar tEval (EXIT RULE EVALUATION)",
		fmt.Sprintf("ExitRule evaluated: %v → Will exit at open of t+1", exitRule))

	// Dump tExec bar (exit execution bar)
	dumpBarWithFeatures(f, cfg, exitIdx, "Bar tExec (EXIT EXECUTED)",
		fmt.Sprintf("ExitPrice: %.2f (Open with slippage)", exitPrice))
}

// dumpBarWithFeatures dumps a single bar with all feature values
func dumpBarWithFeatures(f *os.File, cfg TradeTraceConfig, t int, label, action string) {
	if t < 0 || t >= len(cfg.Series.Close) {
		return
	}

	fmt.Fprintf(f, "\n--- %s (local=%d, global=%d) ---\n",
		label, t, cfg.IndexMap.GlobalStart+t)
	fmt.Fprintf(f, "  OHLC: O=%.2f H=%.2f L=%.2f C=%.2f\n",
		cfg.Series.Open[t], cfg.Series.High[t], cfg.Series.Low[t], cfg.Series.Close[t])

	// Print feature values at this bar
	for i, name := range cfg.Features.Names {
		if t < len(cfg.Features.F[i]) {
			fmt.Fprintf(f, "  %s[t]=%.4f", name, cfg.Features.F[i][t])
			if (i+1)%5 == 0 {
				fmt.Fprintf(f, "\n")
			}
		}
	}
	if len(cfg.Features.Names)%5 != 0 {
		fmt.Fprintf(f, "\n")
	}

	// Print action
	fmt.Fprintf(f, "  Action: %s\n", action)
}

// InitTraceFile initializes the trace file with header
func InitTraceFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintf(f, "=== TRADE TRACE OUTPUT ===\n")
	fmt.Fprintf(f, "Format: 3-bar window for each entry/exit event\n")
	fmt.Fprintf(f, "         Shows tEval (signal detection) and tExec (execution) indices separately\n")
	fmt.Fprintf(f, "==============================================================================\n")

	return nil
}

// TraceSignalDetection logs when an entry signal is detected
func TraceSignalDetection(cfg TradeTraceConfig, tLocal int, entryRuleResult bool, strategyName string) {
	if cfg.OutputPath == "" {
		return
	}

	f, err := os.OpenFile(cfg.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "\n[SIGNAL DETECTION at %s]\n", cfg.IndexMap.LogIndex(tLocal))
	fmt.Fprintf(f, "  Strategy: %s\n", strategyName)
	fmt.Fprintf(f, "  EntryRule result: %v\n", entryRuleResult)
	if entryRuleResult {
		fmt.Fprintf(f, "  Action: Scheduling entry for t+1 (local=%d, global=%d)\n",
			tLocal+1, cfg.IndexMap.GlobalStart+tLocal+1)
	}
}

// TracePendingEntry logs when a pending entry is executed
func TracePendingEntry(cfg TradeTraceConfig, tLocal, signalIdx int, executed bool) {
	if cfg.OutputPath == "" {
		return
	}

	f, err := os.OpenFile(cfg.OutputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	fmt.Fprintf(f, "\n[PENDING ENTRY CHECK at %s]\n", cfg.IndexMap.LogIndex(tLocal))
	fmt.Fprintf(f, "  Signal was at t=%d (global=%d)\n", signalIdx, cfg.IndexMap.GlobalStart+signalIdx)
	fmt.Fprintf(f, "  Executed: %v\n", executed)
}

// Global tracer instance
var globalTraceConfig TradeTraceConfig

// EnableTracing enables global trade tracing
func EnableTracing(outputPath string, globalStart, warmup int, s Series, f Features, st Strategy) {
	globalTraceConfig = TradeTraceConfig{
		OutputPath: outputPath,
		IndexMap: IndexMap{
			GlobalStart: globalStart,
			LocalWarmup: warmup,
		},
		Series:   s,
		Features: f,
		Strategy: st,
	}
	InitTraceFile(outputPath)
}

// DisableTracing disables global trade tracing
func DisableTracing() {
	globalTraceConfig = TradeTraceConfig{}
}

// GetTraceConfig returns the current trace config
func GetTraceConfig() TradeTraceConfig {
	return globalTraceConfig
}
