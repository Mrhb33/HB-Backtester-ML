package main

import (
	"fmt"
	"math"
	"strings"
	"sync/atomic"
)

// min helper for integer minimum
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Global rejection counters (track why strategies fail)
var (
	screenFailScore         int64 // Failed screen: score too low
	screenFailTrades        int64 // Failed screen: not enough trades
	screenFailTooManyTrades int64 // Failed screen: too many trades (spammy)
	screenFailDD            int64 // Failed screen: drawdown too high
	screenFailEntryRateLow  int64 // Failed screen: entry rate too low (dead strategy)
	screenFailEntryRateHigh int64 // Failed screen: entry rate too high (spam strategy)
	trainFailScore          int64 // Failed train: score too low
	trainFailTrades         int64 // Failed train: not enough trades
	trainFailTooManyTrades  int64 // Failed train: too many trades (spammy)
	trainFailDD             int64 // Failed train: drawdown too high
	trainZeroTrades         int64 // Passed all gates but had 0 trades
	strategiesPassed        int64 // Strategies that passed validation
)

// Global screen relax level (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
// Accessed atomically for thread-safe reads from workers
var globalScreenRelaxLevel int32 = 3 // Very relaxed mode - TEMP warm-start: freeze at 3 until elites > 0

// WARNING: allowZeroTrades is a DEBUG flag that bypasses critical validation gates
// When enabled, strategies with zero trades can pass validation, which should NEVER happen in production
// This is ONLY for debugging specific issues with the backtest engine
// If you enable this flag, you MUST disable it before running any production search
// Set to true ONLY for controlled debugging experiments, NEVER for live trading or production searches
var allowZeroTrades bool = false

// printDebugModeWarning prints a warning if debug mode is enabled
func printDebugModeWarning() {
	if allowZeroTrades {
		fmt.Println("\n" + strings.Repeat("!", 70))
		fmt.Println("!!! WARNING: DEBUG MODE ENABLED - allowZeroTrades = true")
		fmt.Println("!!! This bypasses critical validation gates and allows zero-trade strategies")
		fmt.Println("!!! DO NOT use this mode for production searches or live trading")
		fmt.Println("!!! This is ONLY for debugging specific backtest engine issues")
		fmt.Println(strings.Repeat("!", 70) + "\n")
	}
}

func init() {
	printDebugModeWarning()
}

// getScreenRelaxLevel returns the current screen relax level (thread-safe)
func getScreenRelaxLevel() int {
	return int(atomic.LoadInt32(&globalScreenRelaxLevel))
}

// setScreenRelaxLevel sets the screen relax level (thread-safe)
func setScreenRelaxLevel(level int) {
	atomic.StoreInt32(&globalScreenRelaxLevel, int32(level))
}

// checkEntryRate quickly scans the window to count entry signals
// Returns (entryCount, tooLow, tooHigh) - rejects strategies that are dead or spam
// This is much cheaper than full backtest and filters out invalid strategies early
func checkEntryRate(full Series, fullF Features, st Strategy, w Window) (int, bool, bool) {
	i0 := w.Start - w.Warmup
	if i0 < 0 {
		i0 = 0
	}
	i1 := w.End
	if i1 > full.T {
		i1 = full.T
	}

	// Check if entry rule compiles
	if st.EntryCompiled.Code == nil {
		return 0, true, false // No entry rule = dead
	}

	entryEdgeCount := 0
	const minSampleHits = 3    // Minimum entry edges to not be "dead"
	const maxSampleHits = 120  // Maximum entry edges to not be "spam" - filter always-on rules

	// Quick scan: count entry EDGES (when entry becomes true), not "true bars"
	// This is because you can only take one trade while in-position
	var entryPrev bool
	for t := i0; t < i1; t++ {
		// Skip warmup period for entry evaluation
		if t < w.Start {
			// Still track entry state during warmup for edge detection
			entryPrev = evaluateCompiled(st.EntryCompiled.Code, fullF.F, t)
			continue
		}

		entryNow := evaluateCompiled(st.EntryCompiled.Code, fullF.F, t)

		// Count only when entry transitions from false to true (rising edge)
		if entryNow && !entryPrev {
			entryEdgeCount++
			// Early exit if we already exceeded max
			if entryEdgeCount > maxSampleHits {
				return entryEdgeCount, false, true
			}
		}

		entryPrev = entryNow
	}

	tooLow := entryEdgeCount < minSampleHits
	tooHigh := entryEdgeCount > maxSampleHits
	return entryEdgeCount, tooLow, tooHigh
}

// FidelityLevel represents evaluation depth
type FidelityLevel int

const (
	FidelityScreen FidelityLevel = iota // Fast screen (3-6 months)
	FidelityFull                        // Full train window
	FidelityVal                         // Full validation
)

// evaluateMultiFidelity runs strategy through multi-fidelity pipeline
// Returns: (passedScreen, passedFull, passedValidation, trainResult, valResult, rejectionReason)
// rejectionReason: "" = passed, "screen_score", "screen_trades", "screen_dd", "train_score", "train_trades", "train_dd"
func evaluateMultiFidelity(full Series, fullF Features, st Strategy, screenW, trainW, valW Window, testedCount int64) (bool, bool, bool, Result, Result, string) {
	// Get relax level from meta (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
	// Default to very relaxed (3) to unblock candidate flow
	relaxLevel := getScreenRelaxLevel()

	// TEMPORARY: Allow zero trades for debugging generator issues
	// If allowZeroTrades is true, bypass trade count checks
	minTradesOverride := -1 // -1 means "no override" (use normal gates)
	if allowZeroTrades {
		minTradesOverride = 1 // Allow 1 trade minimum instead of normal gates
	}

	// Set gates based on relax level
	var minScreenTrades int
	var maxScreenDD float32
	var minTrainTrades int
	var maxTrainDD float32
	// TEMP warm-start: score gates disabled until elites exist
	// var minScreenScore float32
	// var minTrainScore float32

	switch relaxLevel {
	case 0: // Strict
		// minScreenScore = -0.10
		minScreenTrades = 30
		maxScreenDD = 0.70
		// minTrainScore = -0.20
		minTrainTrades = 80
		maxTrainDD = 0.50
	case 1: // Normal
		// minScreenScore = -0.20
		minScreenTrades = 20
		maxScreenDD = 0.80
		// minTrainScore = -0.20
		minTrainTrades = 60
		maxTrainDD = 0.55
	case 2: // Relaxed
		// minScreenScore = -0.40
		minScreenTrades = 15
		maxScreenDD = 0.85
		// minTrainScore = -0.20
		minTrainTrades = 40
		maxTrainDD = 0.60
	case 3: // Very Relaxed (unblock mode) - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5  // TEMP warm-start: allow sparse entries (was 10)
		maxScreenDD = 0.90
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	default: // Default to very relaxed - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5  // TEMP warm-start: allow sparse entries (was 10)
		maxScreenDD = 0.90
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	}

	// Apply min trades override if set (for debugging)
	if minTradesOverride >= 0 {
		minScreenTrades = minTradesOverride
		minTrainTrades = minTradesOverride
	}

	// Stage 0: Entry-rate precheck (fast rejection of dead/spam strategies)
	// This is much cheaper than full backtest and filters before expensive evaluation
	_, entryTooLow, entryTooHigh := checkEntryRate(full, fullF, st, screenW)
	if entryTooLow {
		atomic.AddInt64(&screenFailEntryRateLow, 1)
		return false, false, false, Result{}, Result{}, "screen_entry_rate_low"
	}
	if entryTooHigh {
		atomic.AddInt64(&screenFailEntryRateHigh, 1)
		return false, false, false, Result{}, Result{}, "screen_entry_rate_high"
	}

	// Stage 1: Fast screen (quick filter)
	screenResult := evaluateStrategyWindow(full, fullF, st, screenW)

	// TEMP warm-start: disable score gates until elites exist
	// Score can be negative early on, so we skip these computations during warm-start
	// Compute DSR-lite score for screen with smoothness
	// screenScore := computeScoreWithSmoothness(
	// 	screenResult.Return,
	// 	screenResult.MaxDD,
	// 	screenResult.Expectancy,
	// 	screenResult.SmoothVol,
	// 	screenResult.DownsideVol,
	// 	screenResult.Trades,
	// 	0, // No deflation penalty for screen
	// )

	// Quick filter: screen must pass basic criteria
	// Less strict than full criteria (we just want to filter obvious junk)
	// TEMP: Allow zero-trade strategies through for debugging
	if !allowZeroTrades {
		// TEMP warm-start: disable score gates until elites exist
		// Score can be negative early on, so this gate kills everything before we have data
		// if screenScore < minScreenScore {
		// 	atomic.AddInt64(&screenFailScore, 1)
		// 	return false, false, false, screenResult, Result{}, "screen_score"
		// }
		if screenResult.Trades < minScreenTrades {
			atomic.AddInt64(&screenFailTrades, 1)
			return false, false, false, screenResult, Result{}, "screen_trades"
		}
		// MAX trades gate: reject spammy scalpers (too many trades)
		if screenResult.Trades > 2000 {
			atomic.AddInt64(&screenFailTooManyTrades, 1)
			return false, false, false, screenResult, Result{}, "screen_too_many_trades"
		}
		if screenResult.MaxDD >= maxScreenDD {
			atomic.AddInt64(&screenFailDD, 1)
			return false, false, false, screenResult, Result{}, "screen_dd"
		}
	}

	// Stage 2: Full train window evaluation
	trainResult := evaluateStrategyWindow(full, fullF, st, trainW)

	// TEMP warm-start: disable score gates until elites exist
	// Compute DSR-lite score for train with smoothness (no penalty)
	// trainScore := computeScoreWithSmoothness(
	// 	trainResult.Return,
	// 	trainResult.MaxDD,
	// 	trainResult.Expectancy,
	// 	trainResult.SmoothVol,
	// 	trainResult.DownsideVol,
	// 	trainResult.Trades,
	// 	0, // No deflation penalty for train phase
	// )

	// Basic train filter (relaxed to get candidates flowing to validation)
	// TEMP: Allow zero-trade strategies through for debugging
	if !allowZeroTrades {
		// TEMP warm-start: disable score gates until elites exist
		// Score can be negative early on, so this gate kills everything before we have data
		// if trainScore < minTrainScore {
		// 	atomic.AddInt64(&trainFailScore, 1)
		// 	return true, false, false, trainResult, Result{}, "train_score"
		// }
		if trainResult.Trades < minTrainTrades {
			atomic.AddInt64(&trainFailTrades, 1)
			return true, false, false, trainResult, Result{}, "train_trades"
		}
		// MAX trades gate: reject spammy scalpers (too many trades)
		if trainResult.Trades > 5000 {
			atomic.AddInt64(&trainFailTooManyTrades, 1)
			return true, false, false, trainResult, Result{}, "train_too_many_trades"
		}
		if trainResult.MaxDD >= maxTrainDD {
			atomic.AddInt64(&trainFailDD, 1)
			return true, false, false, trainResult, Result{}, "train_dd"
		}
	}

	// Track zero-trade strategies
	if trainResult.Trades == 0 {
		atomic.AddInt64(&trainZeroTrades, 1)
	}

	// Stage 3: Validation (only if passed train)
	valResult := evaluateStrategyWindow(full, fullF, st, valW)

	// Compute DSR-lite score for validation with smoothness (WITH deflation penalty)
	valScore := computeScoreWithSmoothness(
		valResult.Return,
		valResult.MaxDD,
		valResult.Expectancy,
		valResult.SmoothVol,
		valResult.DownsideVol,
		valResult.Trades,
		testedCount, // Apply DSR-lite penalty here!
	)

	// Update result with DSR-lite score
	valResult.Score = valScore

	// NOTE: strategiesPassed counter is now incremented in main.go where candidates
	// truly pass all validation gates (QuickTest, CPCV, profit sanity, etc.)
	// This counter should only reflect strategies that actually become elites.

	return true, true, true, trainResult, valResult, ""
}

// printRejectionStats prints the rejection reason statistics
func printRejectionStats() {
	screenFailScore := atomic.LoadInt64(&screenFailScore)
	screenFailTrades := atomic.LoadInt64(&screenFailTrades)
	screenFailTooManyTrades := atomic.LoadInt64(&screenFailTooManyTrades)
	screenFailDD := atomic.LoadInt64(&screenFailDD)
	screenFailEntryRateLow := atomic.LoadInt64(&screenFailEntryRateLow)
	screenFailEntryRateHigh := atomic.LoadInt64(&screenFailEntryRateHigh)
	trainFailScore := atomic.LoadInt64(&trainFailScore)
	trainFailTrades := atomic.LoadInt64(&trainFailTrades)
	trainFailTooManyTrades := atomic.LoadInt64(&trainFailTooManyTrades)
	trainFailDD := atomic.LoadInt64(&trainFailDD)
	trainZeroTrades := atomic.LoadInt64(&trainZeroTrades)
	strategiesPassed := atomic.LoadInt64(&strategiesPassed)

	totalRejected := screenFailScore + screenFailTrades + screenFailTooManyTrades + screenFailDD +
		screenFailEntryRateLow + screenFailEntryRateHigh +
		trainFailScore + trainFailTrades + trainFailTooManyTrades + trainFailDD
	totalEval := totalRejected + strategiesPassed

	fmt.Printf("\n=== REJECTION STATISTICS ===\n")
	fmt.Printf("Total Evaluated: %d\n", totalEval)
	fmt.Printf("Strategies Passed: %d\n", strategiesPassed)
	fmt.Printf("Total Rejected: %d\n", totalRejected)
	if totalRejected > 0 {
		fmt.Printf("\nScreen Stage Failures:\n")
		fmt.Printf("  Entry rate too low (<3 edges): %d (%.1f%%)\n", screenFailEntryRateLow, float64(screenFailEntryRateLow)*100/float64(totalRejected))
		fmt.Printf("  Entry rate too high (>120 edges): %d (%.1f%%)\n", screenFailEntryRateHigh, float64(screenFailEntryRateHigh)*100/float64(totalRejected))
		fmt.Printf("  Not enough trades: %d (%.1f%%)\n", screenFailTrades, float64(screenFailTrades)*100/float64(totalRejected))
		fmt.Printf("  Too many trades (>2000): %d (%.1f%%)\n", screenFailTooManyTrades, float64(screenFailTooManyTrades)*100/float64(totalRejected))
		fmt.Printf("  Drawdown too high: %d (%.1f%%)\n", screenFailDD, float64(screenFailDD)*100/float64(totalRejected))
		fmt.Printf("\nTrain Stage Failures:\n")
		fmt.Printf("  Not enough trades: %d (%.1f%%)\n", trainFailTrades, float64(trainFailTrades)*100/float64(totalRejected))
		fmt.Printf("  Too many trades (>5000): %d (%.1f%%)\n", trainFailTooManyTrades, float64(trainFailTooManyTrades)*100/float64(totalRejected))
		fmt.Printf("  Drawdown too high: %d (%.1f%%)\n", trainFailDD, float64(trainFailDD)*100/float64(totalRejected))
	}
	fmt.Printf("\nZero-Trade Strategies: %d\n", trainZeroTrades)
	fmt.Printf("===========================\n\n")
}

// computeDeflatedScore applies DSR-lite deflation penalty
// Formula: deflated = baseScore - k * log(1 + testedCount / 10000)
func computeDeflatedScore(baseScore float32, testedCount int64) float32 {
	if testedCount <= 0 {
		return baseScore
	}

	// k = 0.5 gives gentle penalty that scales with testing
	deflationPenalty := float32(0.5 * math.Log(1.0 + float64(testedCount)/10000.0))
	return baseScore - deflationPenalty
}

// getScreenWindow creates a smaller screening window (3-6 months)
// Use last 6 months of train window for screening
func getScreenWindow(trainW Window) Window {
	// Assume 5min candles, 6 months ≈ 6 * 30 * 24 * 12 = 51,840 candles
	// Use ~6 months to reduce "VAL magic tricks" and overfitting to local quirks
	screenCandles := 51840 // ~6 months (6 * 30 * 24 * 12 = 51,840 candles)

	screenStart := trainW.End - screenCandles
	if screenStart < trainW.Start {
		screenStart = trainW.Start
	}

	return Window{
		Start:  screenStart,
		End:    trainW.End,
		Warmup: trainW.Warmup,
	}
}
