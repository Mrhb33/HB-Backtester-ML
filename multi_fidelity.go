package main

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"

	"hb_bactest_checker/logx"
)

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
	// Cross sanity check rejection counters
	rejectedCrossSanityMutation int64 // Rejected during mutation sanity check
	rejectedCrossSanitySurrogate int64 // Rejected during surrogate generation
	rejectedCrossSanityLoad     int64 // Rejected when loading winners from disk
	// Total evaluations for sampling
	totalEvaluated int64 // Total number of candidates evaluated
	// Sampled rejection logging (log 1 out of every 1000 rejections)
	rejectionLogSampleInterval int64 = 1000
)

// Global screen relax level (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
// Accessed atomically for thread-safe reads from workers
var globalScreenRelaxLevel int32 = 1 // CRITICAL FIX #6: Default to normal screening (was 3)
// The complexity rule in strategy.go prevents volume-only junk strategies,
// so we can keep screening strict without needing desperate relaxation

// Global edge minimum multiplier (default 3, can be lowered to 2 during recovery)
var globalEdgeMinMult int32 = 3

func getEdgeMinMultiplier() int {
	return int(atomic.LoadInt32(&globalEdgeMinMult))
}

func setEdgeMinMultiplier(val int) {
	atomic.StoreInt32(&globalEdgeMinMult, int32(val))
}

// Global flag to track if DD thresholds have been logged
var ddThresholdsLogged bool = false
var ddThresholdsLoggedMu sync.Mutex

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

// logDDThresholdsOnce logs the configured DD thresholds (only once)
func logDDThresholdsOnce(maxScreenDD, maxTrainDD float32, minScreenTrades, minTrainTrades int, relaxLevel int) {
	ddThresholdsLoggedMu.Lock()
	defer ddThresholdsLoggedMu.Unlock()

	if ddThresholdsLogged {
		return
	}
	ddThresholdsLogged = true

	relaxNames := []string{"Strict", "Normal", "Relaxed", "Very_Relaxed"}
	relaxName := "Unknown"
	if relaxLevel >= 0 && relaxLevel < len(relaxNames) {
		relaxName = relaxNames[relaxLevel]
	}

	fmt.Printf("\n=== STAGED FILTER GATES (relax_level=%d: %s) ===\n", relaxLevel, relaxName)
	fmt.Printf("Stage A (screen): DD <= %.1f%%, trades >= %d\n", maxScreenDD*100, minScreenTrades)
	fmt.Printf("Stage B (train):  DD <= %.1f%%, trades >= %d\n", maxTrainDD*100, minTrainTrades)
	fmt.Printf("Stage C (val):    Full strict scoring with DSR-lite penalty\n")
	fmt.Printf("=============================================\n\n")
}

// checkEntryRateResult contains the result of entry rate checking with soft scoring
type checkEntryRateResult struct {
	EntryCount    int     // Number of entry edges detected
	TooLow        bool    // Entry rate too low (dead strategy)
	TooHigh       bool    // Entry rate too high (spam strategy)
	PenaltyFactor float32 // Soft penalty factor: 1.0 = no penalty, <1.0 = penalized
}

// checkEntryRate quickly scans the window to count entry signals
// Returns adaptive limits based on window size + soft penalty instead of hard reject
// This is much cheaper than full backtest and filters out invalid strategies early
func checkEntryRate(full Series, fullF Features, st Strategy, w Window) checkEntryRateResult {
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
		return checkEntryRateResult{EntryCount: 0, TooLow: true, TooHigh: false, PenaltyFactor: 0.0}
	}

	// ADAPTIVE ENTRY-RATE LIMITS: Scale by window size
	// Base: 3-120 for ~6 month window (51840 candles)
	// Scale linearly with actual window size
	windowCandles := i1 - i0
	baseWindowSize := 51840 // ~6 months at 5min

	// Use global multiplier (can be lowered during recovery)
	multiplier := getEdgeMinMultiplier()
	minSampleHits := multiplier * windowCandles / baseWindowSize
	if minSampleHits < 2 {
		minSampleHits = 2 // Absolute minimum (allow sniper systems)
	}
	if minSampleHits > 8 {
		minSampleHits = 8 // Cap low end for statistical confidence
	}

	maxSampleHits := 120 * windowCandles / baseWindowSize
	if maxSampleHits < 80 {
		maxSampleHits = 80 // Minimum high bound (allow reasonable activity)
	}
	if maxSampleHits > 200 {
		maxSampleHits = 200 // Cap high bound (reduce spam for 5min BTC)
	}

	entryEdgeCount := 0

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
		}

		entryPrev = entryNow
	}

	// SOFT SCORING: Instead of hard reject, apply penalty factor
	// This preserves exploration and reduces wasted generation
	tooLow := entryEdgeCount < minSampleHits
	tooHigh := entryEdgeCount > maxSampleHits

	var penaltyFactor float32 = 1.0 // No penalty by default

	if tooLow {
		// Penalize low entry rates (lower confidence)
		// Penalty scales linearly: 0 trades = 0.0 factor, min trades = 1.0 factor
		if entryEdgeCount == 0 {
			penaltyFactor = 0.0 // Dead strategy - no signal at all
		} else {
			ratio := float32(entryEdgeCount) / float32(minSampleHits)
			penaltyFactor = ratio // 0.33 at 1/3 of min, 0.67 at 2/3 of min
		}
	} else if tooHigh {
		// Penalize high entry rates (fee/overtrade penalty)
		// Penalty scales: max+20% = 1.0, 2x max = 0.5, 3x max = 0.0
		excessRatio := float32(entryEdgeCount) / float32(maxSampleHits)
		if excessRatio >= 3.0 {
			penaltyFactor = 0.0 // Extreme spam
		} else if excessRatio >= 2.0 {
			penaltyFactor = 0.3 // Heavy spam
		} else {
			// Linear from 1.0 at max to 0.5 at 2x max
			penaltyFactor = 1.0 - 0.5*(excessRatio-1.0)
			if penaltyFactor < 0.3 {
				penaltyFactor = 0.3
			}
		}
	}

	return checkEntryRateResult{
		EntryCount:    entryEdgeCount,
		TooLow:        tooLow,
		TooHigh:       tooHigh,
		PenaltyFactor: penaltyFactor,
	}
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

	// STAGED GATES: Stage A (screen) allows high DD, Stage B (train) has tight DD
	// Screen is a cheap pre-filter: only check trade count, allow any DD up to 95%
	// Train is the real filter: check both trade count AND DD
	maxScreenDD = 0.95 // Allow up to 95% DD in screen stage (cheap filter)

	switch relaxLevel {
	case 0: // Strict
		// minScreenScore = -0.10
		minScreenTrades = 30
		// minTrainScore = -0.20
		minTrainTrades = 80
		maxTrainDD = 0.50
	case 1: // Normal
		// minScreenScore = -0.20
		minScreenTrades = 5  // Lowered from 20 to 5 to allow more strategies through warm-start
		// minTrainScore = -0.20
		minTrainTrades = 20  // Lowered from 60 to 20 for warm-start
		maxTrainDD = 0.55
	case 2: // Relaxed
		// minScreenScore = -0.40
		minScreenTrades = 15
		// minTrainScore = -0.20
		minTrainTrades = 40
		maxTrainDD = 0.60
	case 3: // Very Relaxed (unblock mode) - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5  // TEMP warm-start: allow sparse entries (was 10)
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	default: // Default to very relaxed - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5  // TEMP warm-start: allow sparse entries (was 10)
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	}

	// Apply min trades override if set (for debugging)
	if minTradesOverride >= 0 {
		minScreenTrades = minTradesOverride
		minTrainTrades = minTradesOverride
	}

	// Log DD thresholds once (for debugging)
	logDDThresholdsOnce(maxScreenDD, maxTrainDD, minScreenTrades, minTrainTrades, relaxLevel)

	// Stage 0: Entry-rate precheck with SOFT SCORING (not hard rejection)
	// This is much cheaper than full backtest and filters before expensive evaluation
	entryRateResult := checkEntryRate(full, fullF, st, screenW)

	// Only hard-reject completely dead strategies (0 entry edges)
	if entryRateResult.EntryCount == 0 {
		atomic.AddInt64(&screenFailEntryRateLow, 1)
		maybeLogSampledRejection("screen_entry_rate_dead", Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, "screen_entry_rate_dead"
	}

	// Stage 1: Fast screen (quick filter)
	screenResult := evaluateStrategyWindow(full, fullF, st, screenW)

	// Apply soft penalty to screen score based on entry rate
	// This penalizes rather than rejects, preserving exploration
	if entryRateResult.PenaltyFactor < 1.0 {
		screenResult.Score *= entryRateResult.PenaltyFactor
	}

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
			maybeLogSampledRejection("screen_trades", screenResult, st.Seed)
			return false, false, false, screenResult, Result{}, "screen_trades"
		}
		// MAX trades gate: reject spammy scalpers (too many trades)
		if screenResult.Trades > 2000 {
			atomic.AddInt64(&screenFailTooManyTrades, 1)
			maybeLogSampledRejection("screen_too_many_trades", screenResult, st.Seed)
			return false, false, false, screenResult, Result{}, "screen_too_many_trades"
		}
		if screenResult.MaxDD >= maxScreenDD {
			atomic.AddInt64(&screenFailDD, 1)
			maybeLogSampledRejection("screen_dd", screenResult, st.Seed, maxScreenDD)
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
			maybeLogSampledRejection("train_trades", trainResult, st.Seed)
			return true, false, false, trainResult, Result{}, "train_trades"
		}
		// MAX trades gate: reject spammy scalpers (too many trades)
		if trainResult.Trades > 5000 {
			atomic.AddInt64(&trainFailTooManyTrades, 1)
			maybeLogSampledRejection("train_too_many_trades", trainResult, st.Seed)
			return true, false, false, trainResult, Result{}, "train_too_many_trades"
		}
		if trainResult.MaxDD >= maxTrainDD {
			atomic.AddInt64(&trainFailDD, 1)
			maybeLogSampledRejection("train_dd", trainResult, st.Seed, maxTrainDD)
			return true, false, false, trainResult, Result{}, "train_dd"
		}
	}

	// Track zero-trade strategies
	if trainResult.Trades == 0 {
		atomic.AddInt64(&trainZeroTrades, 1)
		maybeLogSampledRejection("train_zero_trades", trainResult, st.Seed)
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

	// Part A1: Score sanity check for validation result
	if !IsValidScore(valScore) {
		return false, false, false, trainResult, Result{}, "val_invalid_score"
	}

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
	rejectedCrossSanityMutation := atomic.LoadInt64(&rejectedCrossSanityMutation)
	rejectedCrossSanitySurrogate := atomic.LoadInt64(&rejectedCrossSanitySurrogate)
	rejectedCrossSanityLoad := atomic.LoadInt64(&rejectedCrossSanityLoad)

	totalRejected := screenFailScore + screenFailTrades + screenFailTooManyTrades + screenFailDD +
		screenFailEntryRateLow + screenFailEntryRateHigh +
		trainFailScore + trainFailTrades + trainFailTooManyTrades + trainFailDD
	totalEval := totalRejected + strategiesPassed

	// Use structured logging for rejection statistics
	logx.LogRejectionStatsHeader()
	logx.LogRejectionStatsSummary(totalEval, totalRejected, strategiesPassed)
	logx.LogRejectionStatsScreen(screenFailEntryRateLow, screenFailEntryRateHigh, screenFailTrades, screenFailTooManyTrades, screenFailDD, totalRejected)
	logx.LogRejectionStatsTrain(trainFailTrades, trainFailTooManyTrades, trainFailDD, totalRejected)
	logx.LogRejectionStatsFooter(trainZeroTrades, rejectedCrossSanityMutation, rejectedCrossSanitySurrogate, rejectedCrossSanityLoad)
}

// maybeLogSampledRejection logs a rejection reason for sampled candidates (1 in N)
// This helps diagnose why candidates are being rejected without spamming logs
// Shows detailed metrics: trades, return, profit factor, expectancy to identify specific failures
func maybeLogSampledRejection(reason string, result Result, seed int64, ddThreshold ...float32) {
	// Increment evaluation counter
	count := atomic.AddInt64(&totalEvaluated, 1)

	// Log 1 out of every N rejections (default 1000)
	if count%rejectionLogSampleInterval == 1 {
		ddInfo := ""
		if len(ddThreshold) > 0 {
			ddInfo = fmt.Sprintf(", dd_threshold=%.2f%%", ddThreshold[0]*100)
		}
		// Include detailed metrics: trades, return, profit factor, expectancy
		// This helps identify whether rejection was due to ret vs trades vs pf vs dd
		fmt.Printf("[SAMPLED REJECTION #%d] reason=%s, seed=%d, trades=%d, ret=%.2f%%, pf=%.2f, exp=%.5f, dd=%.2f%%%s\n",
			count, reason, seed, result.Trades, result.Return*100, result.ProfitFactor, result.Expectancy, result.MaxDD*100, ddInfo)
	}
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
