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
	screenZeroTrades        int64 // Screen stage: strategies with 0 trades
	trainFailScore          int64 // Failed train: score too low
	trainFailTrades         int64 // Failed train: not enough trades
	trainFailTooManyTrades  int64 // Failed train: too many trades (spammy)
	trainFailDD             int64 // Failed train: drawdown too high
	trainZeroTrades         int64 // Train stage: strategies with 0 trades
	strategiesPassed        int64 // Strategies that passed validation
	// Cross sanity check rejection counters
	rejectedCrossSanityMutation  int64 // Rejected during mutation sanity check
	rejectedCrossSanitySurrogate int64 // Rejected during surrogate generation
	rejectedCrossSanityLoad      int64 // Rejected when loading winners from disk
	// Total evaluations for sampling
	totalEvaluated       int64 // Total number of candidates evaluated (incremented only for OOS rejects)
	debugSampleEvalCount int64 // Counter for debug sampling (incremented for all candidates at gate check)
	// Sampled rejection logging (log 1 out of every 10000 rejections)
	rejectionLogSampleInterval int64 = 10000
	// Separate counter for logging (to avoid double-counting with totalEvaluated)
	rejectionLogCounter int64 // Only for sampling/printing (1 in N rejections logged)
	// OOS rejection counters (walk-forward)
	oosTradesTooLow     int64 // Rejected: insufficient OOS trades
	oosMaxDDTooHigh     int64 // Rejected: OOS max DD too high
	oosWorstMonthTooBad int64 // Rejected: worst month too bad
	oosOtherReject      int64 // Rejected: other OOS constraint
	oosZeroTrades       int64 // OOS stage: strategies with 0 trades
	// Granular OOS rejection counters for detailed reporting
	oosRejectEntryRateLow      int64 // Rejected: entry rate too low
	oosRejectTooSparseMonths   int64 // Rejected: sparse months ratio too high
	oosRejectActiveMonthsLow   int64 // Rejected: active months ratio too low
	oosRejectMinMonthFail      int64 // Rejected: min month return too low
	oosRejectGeoMonthlyFail    int64 // Rejected: geo avg monthly return too low
	oosRejectMedianMonthlyFail int64 // Rejected: median monthly return too low
	oosRejectTooComplex        int64 // Rejected: complexity too high
)

// Global screen relax level (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
// Accessed atomically for thread-safe reads from workers
var globalScreenRelaxLevel int32 = 1 // CRITICAL FIX #6: Default to normal screening (was 3)
// The complexity rule in strategy.go prevents volume-only junk strategies,
// so we can keep screening strict without needing desperate relaxation

// FIX #6: Global quiet mode flag (suppresses sampled rejection messages)
// Use atomic int32 for thread-safe access from multiple goroutines (0=false, 1=true)
var globalQuiet int32 = 0

// Global edge minimum multiplier (default 2, can be lowered to 1 during recovery)
var globalEdgeMinMult int32 = 2

// PROBLEM A FIX: Relaxed entry rate threshold for discovery phase (was 120)
// During discovery: use 30-60 edges/year to avoid killing promising edges early
// Can be raised back to 120+ after finding good candidates
var globalMinEdgesPerYear int32 = 40      // Default: 40 edges/year (relaxed for discovery)
var wfDiscoveryEdgesPerYear float64 = 0.5 // EMERGENCY: Allow 6 edges/year at 1H (0.5 × 12)

// DEBUG: Disable entry rate gate entirely for diagnosis (set to true to allow all strategies through)
var debugDisableEntryRateGate bool = true

// setDebugDisableEntryRateGate sets the debug flag for disabling entry rate gate
func setDebugDisableEntryRateGate(disable bool) {
	debugDisableEntryRateGate = disable
}

func getEdgeMinMultiplier() int {
	return int(atomic.LoadInt32(&globalEdgeMinMult))
}

func setEdgeMinMultiplier(val int) {
	atomic.StoreInt32(&globalEdgeMinMult, int32(val))
}

// getMinEdgesPerYear returns the minimum edges per year threshold
func getMinEdgesPerYear() int {
	return int(atomic.LoadInt32(&globalMinEdgesPerYear))
}

// setMinEdgesPerYear sets the minimum edges per year threshold
// Use 120 for swing style, 240 for active strategies
func setMinEdgesPerYear(val int) {
	atomic.StoreInt32(&globalMinEdgesPerYear, int32(val))
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
	// CRITICAL FILTER: Reject empty/nil strategies BEFORE any expensive evaluation
	// Check for nil rule roots OR nil/empty compiled code (compilation may fail silently)
	if st.EntryRule.Root == nil || st.ExitRule.Root == nil ||
		st.EntryCompiled.Code == nil || len(st.EntryCompiled.Code) == 0 ||
		st.ExitCompiled.Code == nil || len(st.ExitCompiled.Code) == 0 {
		atomic.AddInt64(&screenFailEntryRateLow, 1)
		maybeLogSampledRejection("wf_empty_strategy", Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, "wf_empty_strategy"
	}

	// CRITICAL FILTER: Validate cross operations BEFORE any expensive evaluation
	// This rejects strategies with invalid CrossUp/CrossDown (A=B) that are always false
	// Doing this first avoids wasting compute on invalid/empty strategies
	_, entryInvalid := validateCrossSanity(st.EntryRule.Root, fullF)
	_, exitInvalid := validateCrossSanity(st.ExitRule.Root, fullF)
	_, regimeInvalid := validateCrossSanity(st.RegimeFilter.Root, fullF)

	totalInvalid := entryInvalid + exitInvalid + regimeInvalid
	if totalInvalid > 0 {
		atomic.AddInt64(&screenFailEntryRateLow, 1) // Reuse counter for tracking
		maybeLogSampledRejection("wf_invalid_cross_operations", Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, "wf_invalid_cross_operations"
	}

	// Get relax level from meta (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
	// Default to very relaxed (3) to unblock candidate flow
	relaxLevel := getScreenRelaxLevel()

	// FIX #5: allowZeroTrades bypasses ALL validation gates (DD, max trades, etc.)
	// This is intentional for debugging - we want to allow zero-trade strategies through
	// The warning is printed at startup to remind users this mode is active

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
		minScreenTrades = 5 // Lowered from 20 to 5 to allow more strategies through warm-start
		// minTrainScore = -0.20
		minTrainTrades = 20 // Lowered from 60 to 20 for warm-start
		maxTrainDD = 0.75   // EDIT #1a: Raised from 0.70 to 0.75 - strategies were hitting 70.9%
	case 2: // Relaxed
		// minScreenScore = -0.40
		minScreenTrades = 15
		// minTrainScore = -0.20
		minTrainTrades = 40
		maxTrainDD = 0.60
	case 3: // Very Relaxed (unblock mode) - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5 // TEMP warm-start: allow sparse entries (was 10)
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	default: // Default to very relaxed - TEMP warm-start: ultra-low min trades
		// minScreenScore = -0.60
		minScreenTrades = 5 // TEMP warm-start: allow sparse entries (was 10)
		// minTrainScore = -0.20
		minTrainTrades = 15 // TEMP warm-start: allow sparse strategies (was 30)
		maxTrainDD = 0.65
	}

	// Override gates for 1H timeframe (user-specified)
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	if tfMinutes >= 60 {
		// 1H or higher: require minimum trades for statistical significance
		minScreenTrades = 8 // Screen: at least 8 trades
		minTrainTrades = 20 // EDIT: Train: at least 20 trades (lowered from 25 to reduce dead-entry spam)
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

	// For 1H: reject if entry rate is too low (< 10 trades/year for discovery)
	// PROBLEM B FIX: Reduced from 20 to 10 to allow more sniper strategies through
	// FIXED: Use safer formula - compute per bar first, then annualize
	if tfMinutes >= 60 && entryRateResult.EntryCount > 0 {
		barsInWindow := int64(screenW.End - screenW.Start)
		if barsInWindow > 0 {
			entriesPerBar := float64(entryRateResult.EntryCount) / float64(barsInWindow)
			// FIX #8: Use 365.25 days to account for leap years (365.25 * 24 * 60 = 525960)
			const minutesPerYear = 365.25 * 24 * 60
			barsPerYear := minutesPerYear / float64(tfMinutes)
			entriesPerYear := entriesPerBar * barsPerYear

			if entriesPerYear < 10 { // Less than 10 trades/year expected (reduced from 20 for discovery)
				atomic.AddInt64(&screenFailEntryRateLow, 1)
				maybeLogSampledRejection("screen_entry_rate_too_low_1h", Result{Trades: int(entryRateResult.EntryCount)}, st.Seed)
				return false, false, false, Result{}, Result{}, "screen_entry_rate_too_low_1h"
			}
		}
	}

	// Stage 1: Fast screen (quick filter)
	screenResult := evaluateStrategyWindow(full, fullF, st, screenW)

	// FIX #4: Apply entry-rate penalty to Return (affects downstream) instead of Score
	// Score gates are disabled during warm-start, so Score penalty has no effect.
	// Return penalty affects DSR-lite calculation in validation phase.
	if entryRateResult.PenaltyFactor < 1.0 {
		screenResult.Return *= entryRateResult.PenaltyFactor
		screenResult.Score *= entryRateResult.PenaltyFactor // Also apply for consistency
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
	// NOTE: allowZeroTrades bypasses ALL validation gates (trades, DD, spam checks)
	// This is DEBUG mode only - see warning printed at startup
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

		// CRITICAL FIX: Catastrophic DD gate - reject strategies with extreme DD
		// This prevents 96%+ DD strategies from proceeding to expensive OOS validation
		if screenResult.MaxDD >= 0.80 {
			atomic.AddInt64(&screenFailDD, 1)
			maybeLogSampledRejection("screen_dd_catastrophic", screenResult, st.Seed)
			return false, false, false, screenResult, Result{}, "screen_dd_catastrophic"
		}
	}

	// Track zero-trade strategies at screen stage (after all gates pass)
	if screenResult.Trades == 0 {
		atomic.AddInt64(&screenZeroTrades, 1)
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

		// CRITICAL FIX: Train DD explosion check - reject strategies with catastrophic training DD
		// This prevents risk explosion from reaching OOS validation
		if trainResult.MaxDD >= 0.50 {
			atomic.AddInt64(&trainFailDD, 1)
			maybeLogSampledRejection("train_dd_explosion", trainResult, st.Seed)
			return true, false, false, trainResult, Result{}, "train_dd_explosion"
		}

		// CRITICAL FIX: Train return floor - reject strategies with very negative training performance
		// This prevents strategies that don't work in training from passing to OOS
		if trainResult.Return < -0.10 {
			atomic.AddInt64(&trainFailScore, 1)
			maybeLogSampledRejection("train_return_negative", trainResult, st.Seed)
			return true, false, false, trainResult, Result{}, "train_return_negative"
		}
	}

	// Track zero-trade strategies
	if trainResult.Trades == 0 {
		atomic.AddInt64(&trainZeroTrades, 1)
		maybeLogSampledRejection("train_zero_trades", trainResult, st.Seed)
		// Bug warning: Zero-trade strategy accepted
		logx.LogBugWarning(fmt.Sprintf("Zero-trade strategy in train stage (seed=%d)", st.Seed))
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
		// Log bug warning for NaN scores
		if math.IsNaN(float64(valScore)) {
			logx.LogBugWarning(fmt.Sprintf("NaN score in validation (seed=%d)", st.Seed))
		}
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
// FIX #9: Use local variable names that don't shadow globals (add 'local' prefix)
func printRejectionStats() {
	localScreenFailScore := atomic.LoadInt64(&screenFailScore)
	localScreenFailTrades := atomic.LoadInt64(&screenFailTrades)
	localScreenFailTooManyTrades := atomic.LoadInt64(&screenFailTooManyTrades)
	localScreenFailDD := atomic.LoadInt64(&screenFailDD)
	localScreenFailEntryRateLow := atomic.LoadInt64(&screenFailEntryRateLow)
	localScreenFailEntryRateHigh := atomic.LoadInt64(&screenFailEntryRateHigh)
	localScreenZeroTrades := atomic.LoadInt64(&screenZeroTrades)
	localTrainFailScore := atomic.LoadInt64(&trainFailScore)
	localTrainFailTrades := atomic.LoadInt64(&trainFailTrades)
	localTrainFailTooManyTrades := atomic.LoadInt64(&trainFailTooManyTrades)
	localTrainFailDD := atomic.LoadInt64(&trainFailDD)
	localTrainZeroTrades := atomic.LoadInt64(&trainZeroTrades)
	localStrategiesPassed := atomic.LoadInt64(&strategiesPassed)

	// OOS rejection counters (walk-forward)
	localOOSTradesTooLow := atomic.LoadInt64(&oosTradesTooLow)
	localOOSMaxDDTooHigh := atomic.LoadInt64(&oosMaxDDTooHigh)
	localOOSWorstMonthTooBad := atomic.LoadInt64(&oosWorstMonthTooBad)
	localOOSOtherReject := atomic.LoadInt64(&oosOtherReject)
	localOOSZeroTrades := atomic.LoadInt64(&oosZeroTrades)
	// Granular OOS rejection counters
	localOOSRejectEntryRateLow := atomic.LoadInt64(&oosRejectEntryRateLow)
	localOOSRejectTooSparseMonths := atomic.LoadInt64(&oosRejectTooSparseMonths)
	localOOSRejectActiveMonthsLow := atomic.LoadInt64(&oosRejectActiveMonthsLow)
	localOOSRejectMinMonthFail := atomic.LoadInt64(&oosRejectMinMonthFail)
	localOOSRejectGeoMonthlyFail := atomic.LoadInt64(&oosRejectGeoMonthlyFail)
	localOOSRejectMedianMonthlyFail := atomic.LoadInt64(&oosRejectMedianMonthlyFail)
	localOOSRejectTooComplex := atomic.LoadInt64(&oosRejectTooComplex)

	totalRejected := localScreenFailScore + localScreenFailTrades + localScreenFailTooManyTrades + localScreenFailDD +
		localScreenFailEntryRateLow + localScreenFailEntryRateHigh +
		localTrainFailScore + localTrainFailTrades + localTrainFailTooManyTrades + localTrainFailDD +
		localOOSTradesTooLow + localOOSMaxDDTooHigh + localOOSWorstMonthTooBad + localOOSOtherReject +
		localOOSRejectEntryRateLow + localOOSRejectTooSparseMonths + localOOSRejectActiveMonthsLow +
		localOOSRejectMinMonthFail + localOOSRejectGeoMonthlyFail + localOOSRejectMedianMonthlyFail +
		localOOSRejectTooComplex
	totalEval := totalRejected + localStrategiesPassed

	// Use simplified rejection statistics
	logx.LogRejectionStatsSimple(totalEval, totalRejected, localStrategiesPassed,
		localScreenFailEntryRateLow+localOOSRejectEntryRateLow,
		localOOSRejectTooSparseMonths,
		localScreenFailDD+localTrainFailDD+localOOSMaxDDTooHigh)

	// Print zero-trade stage breakdown (simplified)
	if localScreenZeroTrades > 0 || localTrainZeroTrades > 0 || localOOSZeroTrades > 0 {
		fmt.Printf("Zero-trades: screen=%s train=%s oos=%s\n",
			logx.FormatNumberSimple(int(localScreenZeroTrades)),
			logx.FormatNumberSimple(int(localTrainZeroTrades)),
			logx.FormatNumberSimple(int(localOOSZeroTrades)))
	}
}

// maybeLogSampledRejection logs a rejection reason for sampled candidates (1 in N)
// This helps diagnose why candidates are being rejected without spamming logs
// Shows detailed metrics: trades, return, profit factor, expectancy to identify specific failures
func maybeLogSampledRejection(reason string, result Result, seed int64, ddThreshold ...float32) {
	// Use separate counter for logging (not totalEvaluated)
	count := atomic.AddInt64(&rejectionLogCounter, 1)

	// Log 1 out of every N rejections (default 10000), unless quiet mode
	quiet := atomic.LoadInt32(&globalQuiet) == 1
	if !quiet && count%rejectionLogSampleInterval == 1 {
		ddInfo := ""
		if len(ddThreshold) > 0 {
			ddInfo = fmt.Sprintf(", dd_threshold=%.2f%%", ddThreshold[0]*100)
		}

		// DETAILED DEBUG: Trigger by seed OR by condition (high-trade OOS rejection with PF=0)
		// Replace 123456789 with an actual seed from your logs
		triggerSeed := int64(123456789) // TODO: Set this from command line or config
		triggerCondition := (reason == "wf_oos_rejected: " && result.Trades > 10000 && result.ProfitFactor == 0)

		if seed == triggerSeed || triggerCondition {
			fmt.Printf("\n=== DETAILED REJECTION DEBUG ===\n")
			fmt.Printf("seed=%d, reason=%s\n", seed, reason)
			fmt.Printf("trades=%d, ret=%.2f%%, pf=%.2f, exp=%.5f, dd=%.2f%%\n",
				result.Trades, result.Return*100, result.ProfitFactor, result.Expectancy, result.MaxDD*100)
			fmt.Printf("=== END DEBUG ===\n\n")
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
	deflationPenalty := float32(0.5 * math.Log(1.0+float64(testedCount)/10000.0))
	return baseScore - deflationPenalty
}

// evaluateWithWalkForward evaluates a strategy using walk-forward validation if enabled
// Returns: (passedScreen, passedFull, passedValidation, trainResult, valResult, rejectionReason)
// If WF is disabled, falls back to standard evaluateMultiFidelity
func evaluateWithWalkForward(full Series, fullF Features, st Strategy, screenW, trainW, valW Window, testedCount int64, wfConfig WFConfig) (bool, bool, bool, Result, Result, string) {
	if !wfConfig.Enable {
		// Fall back to standard evaluation
		return evaluateMultiFidelity(full, fullF, st, screenW, trainW, valW, testedCount)
	}

	// Walk-forward validation enabled
	// Build folds using combined train+val window as data source
	combinedStart := trainW.Start
	combinedEnd := valW.End

	// CRITICAL FIX: Check entry rate on WF window BEFORE running expensive WF evaluation
	// This prevents wasting ~70% of evaluations on strategies that never trade in OOS
	// Check if entry rule compiles
	if st.EntryCompiled.Code == nil {
		return false, false, false, Result{}, Result{}, "wf_entry_rule_not_compiled"
	}

	// Look up active filter index from features (same way as backtest.go)
	activeIdx, okActive := fullF.Index["Active"]

	// FIX A1: In WF mode, DISABLE the OLD combined-window gate entirely.
	// The fold-based gate (below) is more accurate because it counts edges in actual test windows.
	// Skip all the combined window edge counting and reject logic.
	// Only keep the minimal "has entry rule" check at line 631.

	// OLD combined-window gate code (DISABLED in WF mode):
	// This gate was rejecting strategies before fold gates even mattered.
	// The fold-based gate (lines 844-964) is the proper check for WF validation.
	if false { // Disabled - fold gate handles this properly
		// Count entry edges on the combined WF window (what WF will actually run on)
		// CRITICAL FIX: Count RISING EDGES of all trading conditions, not bars true
		entryEdgeCount := 0
		regimeOKCount := 0
		entryAndRegimeOKCount := 0
		entryTrueCount := 0 // Total bars where entry is true (not just edges)
		volOkCount := 0
		allConditionsOKCount := 0     // All conditions that would allow trading (bars true)
		allConditionsRisingEdges := 0 // CRITICAL FIX: Count rising edges of (isActive && regimeOK && volOk && entryOK)
		var entryPrev bool
		var allConditionsPrev bool

		for t := combinedStart; t < combinedEnd; t++ {
			if t >= full.T {
				break
			}

			// Step 1: Check isActive (active feature filter) - same as backtest.go
			isActive := true
			if okActive && activeIdx >= 0 && activeIdx < len(fullF.F) && t < len(fullF.F[activeIdx]) {
				isActive = fullF.F[activeIdx][t] > 0
			}

			// Step 2: Check regime filter
			regimeOK := st.RegimeFilter.Root == nil
			if !regimeOK && st.RegimeCompiled.Code != nil {
				regimeOK = evaluateCompiled(st.RegimeCompiled.Code, fullF.F, t)
			}
			if regimeOK {
				regimeOKCount++
			}

			// Step 3: Check volume filter (simplified version for gate)
			volOk := true // Assume OK for gate (full check requires more context)

			// Step 4: Check entry signal
			entryNow := false
			if st.EntryCompiled.Code != nil {
				entryNow = evaluateCompiled(st.EntryCompiled.Code, fullF.F, t)
			}
			// Count total entry true bars (for debugging)
			if entryNow {
				entryTrueCount++
			}
			// Count only when entry transitions from false to true (rising edge)
			if entryNow && !entryPrev {
				entryEdgeCount++
			}
			entryPrev = entryNow

			// Count bars where both entry AND regime are true
			if entryNow && regimeOK {
				entryAndRegimeOKCount++
			}

			// Count volume OK
			if volOk {
				volOkCount++
			}

			// Check if ALL trading conditions are met (same logic as backtest.go)
			allConditionsNow := isActive && regimeOK && volOk && entryNow
			if allConditionsNow {
				allConditionsOKCount++ // Count bars where conditions are true (for debug)
			}
			// CRITICAL FIX: Count rising edges of all conditions (what actually triggers trades)
			if allConditionsNow && !allConditionsPrev {
				allConditionsRisingEdges++
			}
			allConditionsPrev = allConditionsNow
		}

		// DEBUG: Sample detailed edge counting for 20 random strategies (once per 10000 evaluations)
		// CRITICAL FIX #2: Use separate counter for evaluation sampling to avoid double-counting
		// We use a local sampling counter that doesn't interfere with totalEvaluated
		debugSampleCounter := atomic.AddInt64(&debugSampleEvalCount, 1)
		doDetailedEdgeLog := false
		if debugSampleCounter%10000 == 1 {
			doDetailedEdgeLog = true
		}
		if doDetailedEdgeLog && entryEdgeCount >= 0 {
			// Compute stats per year
			tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
			if tfMinutes <= 0 {
				tfMinutes = 5
			}
			const minutesPerYear = 365.25 * 24 * 60
			barsPerYear := float64(minutesPerYear) / float64(tfMinutes)
			barsInWindow := float64(combinedEnd - combinedStart)
			regimeOKPct := float64(regimeOKCount) / barsInWindow * 100

			// Analyze regime filter effectiveness
			hasRegimeFilter := st.RegimeCompiled.Code != nil
			regimeStatus := "DISABLED (no filter)"
			if hasRegimeFilter {
				regimeStatus = fmt.Sprintf("ENABLED (%.1f%% of bars pass)", regimeOKPct)
				if regimeOKPct < 10 {
					regimeStatus += " - ULTRA-RARE!"
				} else if regimeOKPct < 30 {
					regimeStatus += " - RARE"
				}
			}

			fmt.Printf("\n=== EDGE COUNTING DEBUG (seed=%d) ===\n", st.Seed)
			fmt.Printf("Window: bars[%d:%d] (%.0f bars, %.1f years)\n", combinedStart, combinedEnd, barsInWindow, barsInWindow/barsPerYear)
			fmt.Printf("Entry edges (rising):         %d (%.1f/year, %.1f%% of bars)\n", entryEdgeCount, float64(entryEdgeCount)/barsInWindow*barsPerYear, float64(entryEdgeCount)/barsInWindow*100)
			fmt.Printf("Entry true bars:              %d (%.1f/year, %.1f%% of bars)\n", entryTrueCount, float64(entryTrueCount)/barsInWindow*barsPerYear, float64(entryTrueCount)/barsInWindow*100)
			fmt.Printf("Regime OK bars:               %d (%.1f/year, %.1f%% of bars)\n", regimeOKCount, float64(regimeOKCount)/barsInWindow*barsPerYear, float64(regimeOKCount)/barsInWindow*100)
			fmt.Printf("Volume OK bars:               %d (%.1f/year, %.1f%% of bars)\n", volOkCount, float64(volOkCount)/barsInWindow*barsPerYear, float64(volOkCount)/barsInWindow*100)
			fmt.Printf("Entry AND Regime OK:          %d (%.1f/year)\n", entryAndRegimeOKCount, float64(entryAndRegimeOKCount)/barsInWindow*barsPerYear)
			fmt.Printf("ALL CONDITIONS OK (bars):       %d (%.1f bars true/year, %.1f%%)\n", allConditionsOKCount, float64(allConditionsOKCount)/barsInWindow*barsPerYear, float64(allConditionsOKCount)/barsInWindow*100)
			fmt.Printf("ALL CONDITIONS RISING EDGES:    %d (%.1f/year, %.1f%% of bars) *** COMBINED WINDOW (OLD - DISABLED) ***\n", allConditionsRisingEdges, float64(allConditionsRisingEdges)/barsInWindow*barsPerYear, float64(allConditionsRisingEdges)/barsInWindow*100)
			fmt.Printf("Regime filter:                 %s\n", regimeStatus)
			fmt.Printf("Entry rule: %s\n", ruleTreeToString(st.EntryRule.Root))
			if hasRegimeFilter {
				fmt.Printf("Regime filter: %s\n", ruleTreeToString(st.RegimeFilter.Root))
			} else {
				fmt.Printf("Regime filter: (none)\n")
			}
			fmt.Printf("Active filter idx:             %d (isActive check)\n", activeIdx)
			fmt.Printf("=====================================\n\n")
		}

		// Compute edgesPerYear and hard reject if too low (dead strategy detector)
		// This catches strategies that would produce 0 OOS trades
		tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
		if tfMinutes <= 0 {
			tfMinutes = 5 // Default to 5min if not set
		}
		const minutesPerYear = 365.25 * 24 * 60
		barsPerYear := float64(minutesPerYear) / float64(tfMinutes)
		barsInWindow := float64(combinedEnd - combinedStart)
		if barsInWindow > 0 {
			// CRITICAL FIX: Use allConditionsRisingEdges (rising edges of all conditions) instead of entryEdgeCount
			// The gate must match what actually triggers trades: rising edge of (isActive && regimeOk && volOk && entryOK)
			tradableEdgesPerYear := float64(allConditionsRisingEdges) / barsInWindow * barsPerYear

			// DEBUG: Skip entry rate gate if debug flag is set (for diagnosis)
			if !debugDisableEntryRateGate {
				// UNIFIED SEMANTICS: -1=auto (fallback), 0=disabled, >0=explicit
				// This matches the fold gate behavior for consistency
				minEdgesPerYear := wfConfig.MinEdgesPerYear
				if minEdgesPerYear < 0 {
					minEdgesPerYear = wfDiscoveryEdgesPerYear // Fallback to global default (-1 = auto)
				}

				// CRITICAL: Only apply edge rate gate when threshold is > 0
				// When 0 is passed, the gate is completely disabled (no fallback, no check)
				if minEdgesPerYear > 0 {
					if tfMinutes < 60 {
						// For 5min/15min: scale threshold up proportionally
						// At 5min: 12x more bars than 1H, so expect 12x more edges
						scaleFactor := 60.0 / float64(tfMinutes)
						minEdgesPerYear = minEdgesPerYear * scaleFactor
					}

					if tradableEdgesPerYear < minEdgesPerYear {
						atomic.AddInt64(&screenFailEntryRateLow, 1)
						reason := fmt.Sprintf("wf_entry_rate_too_low: %.1f tradable_edges/year < %.1f (entry_edges=%d, all_conditions_rising=%d, all_conditions_bars=%d, window_bars=%d)",
							tradableEdgesPerYear, minEdgesPerYear, entryEdgeCount, allConditionsRisingEdges, allConditionsOKCount, int(barsInWindow))
						maybeLogSampledRejection(reason, Result{Trades: allConditionsRisingEdges}, st.Seed)
						return false, false, false, Result{}, Result{}, reason
					}
				}
			} else {
				// DEBUG MODE: Log that entry rate gate is disabled
				if debugSampleCounter%1000 == 1 {
					fmt.Printf("[DEBUG] Entry rate gate DISABLED - allowing all strategies through (all_conditions_rising=%d, %.1f/year)\n",
						allConditionsRisingEdges, tradableEdgesPerYear)
				}
			}
		}
	} // END of disabled OLD combined-window gate

	// FIX #1: Compute warmup from the strategy's actual indicator lookback
	// instead of using the generic trainW.Warmup window value.
	// This ensures each fold has sufficient history for its specific indicators.
	warmup := ComputeWarmupForStrategy(st, fullF)
	i0 := combinedStart - warmup
	if i0 < 0 {
		i0 = 0
	}
	s := SliceSeries(full, i0, combinedEnd)
	f := SliceFeatures(fullF, i0, combinedEnd)

	// FIX #7: Validate minimum required history before building folds
	// Minimum required: (TrainDays + TestDays) for first fold, plus MinFolds-1 additional steps
	// Approximate check: ensure we have at least MinFolds worth of data
	barsPerDay := computeBarsPerDay(s)
	minRequiredBars := int(float64(wfConfig.TrainDays+wfConfig.TestDays) * barsPerDay)
	if len(s.Close) < minRequiredBars {
		reason := fmt.Sprintf("wf_insufficient_history: need %d bars for 1 fold, have %d bars",
			minRequiredBars, len(s.Close))
		// Log this rejection so it appears in statistics
		maybeLogSampledRejection(reason, Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, reason
	}

	// Build walk-forward folds
	folds, err := BuildWalkForwardFolds(s, wfConfig.TrainDays, wfConfig.TestDays, wfConfig.StepDays, wfConfig.MinFolds)
	if err != nil {
		reason := "wf_fold_build_failed: " + err.Error()
		// DEBUG: Log data size info
		reason += fmt.Sprintf(" [series_bars=%d, series_days=%.0f]", len(s.Close), float64(len(s.Close))/barsPerDay)
		maybeLogSampledRejection(reason, Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, reason
	}

	// FIX #B: Fold-based entry viability gate (count edges in TEST windows, not combined window)
	// This catches strategies that are dead in OOS test windows specifically
	// Count edges PER FOLD in test windows + warmup buffer

	// Compute timeframe constants for edge rate calculations
	const minutesPerYear = 365.25 * 24 * 60
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	if tfMinutes <= 0 {
		tfMinutes = 5 // Default to 5min if not set
	}
	barsPerYear := float64(minutesPerYear) / float64(tfMinutes)

	// FIX #B: Use fixed warmup bars for edge detection (sets prev state)
	// Warmup allows proper edge detection at test start, but edges are only counted in test
	warmupBars := 20 // Fixed warmup for edge state initialization
	sumTestEdges := 0
	zeroTradeFolds := 0
	totalTestBars := 0
	perFoldEdgeCounts := make([]int, len(folds))

	// FIX #B: Use sliced features (f) instead of full features (fullF)
	// The fold indices are relative to the sliced series s/f, NOT the full series
	// So we need to get activeIdx from f.Index, not fullF.Index
	activeIdxSliced, okActiveSliced := f.Index["Active"]

	for i, fold := range folds {
		// FIX #B: Start warmup bars before test window for state initialization
		// Use max(0, ...) to handle fold indices that might be close to slice start
		// DO NOT constrain to fold.TrainEnd - this prevents warmup from working
		start := fold.TestStart - warmupBars
		if start < 0 {
			start = 0
		}
		end := fold.TestEnd
		if end > len(s.Close) {
			end = len(s.Close)
		}

		// Count edges within this fold's TEST window only (warmup only sets prev state)
		foldEdges := 0
		var allConditionsPrev bool

		// FIX #B: Use sliced features (f) with fold indices directly
		// fold indices are relative to the sliced series (s/f), so use t as-is
		// NO need to map to full series indices - we're working with the sliced view
		for t := start; t < end; t++ {
			// Check isActive (active feature filter) using SLICED features
			isActive := true
			if okActiveSliced && activeIdxSliced >= 0 && activeIdxSliced < len(f.F) && t < len(f.F[activeIdxSliced]) {
				isActive = f.F[activeIdxSliced][t] > 0
			}

			// Check regime filter using SLICED features
			regimeOK := (st.RegimeFilter.Root == nil)
			if !regimeOK && st.RegimeCompiled.Code != nil {
				regimeOK = evaluateCompiled(st.RegimeCompiled.Code, f.F, t)
			}

			// Check volatility filter to match backtest behavior
			volOk := true
			if st.VolatilityFilter.IsActive() {
				// Get ATR14 index
				if atr14Idx, ok14 := f.Index["ATR14"]; ok14 && atr14Idx >= 0 && atr14Idx < len(f.F) && t < len(f.F[atr14Idx]) {
					// Get ATR14_SMA50 index
					if smaIdx, okSMA := f.Index["ATR14_SMA50"]; okSMA && smaIdx >= 0 && smaIdx < len(f.F) && t < len(f.F[smaIdx]) {
						atr14 := f.F[atr14Idx][t]
						atrSMA := f.F[smaIdx][t]
						if atrSMA > 0 {
							// Check if ATR14 is above threshold x SMA (same as backtest)
							volOk = atr14 >= (atrSMA * st.VolatilityFilter.Threshold)
						}
					}
				}
			}

			// Check entry signal using SLICED features
			entryNow := false
			if st.EntryCompiled.Code != nil {
				entryNow = evaluateCompiled(st.EntryCompiled.Code, f.F, t)
			}

			// Check if ALL trading conditions are met (same logic as backtest.go)
			allConditionsNow := isActive && regimeOK && volOk && entryNow

			// FIX #B: Only count edges inside test window (t >= fold.TestStart)
			// Warmup period (t < fold.TestStart) only sets prev state, doesn't count edges
			if t >= fold.TestStart && allConditionsNow && !allConditionsPrev {
				foldEdges++
			}
			allConditionsPrev = allConditionsNow
		}

		perFoldEdgeCounts[i] = foldEdges
		sumTestEdges += foldEdges
		// FIX #B: Use test bars only for rate calculation (exclude warmup from denominator)
		// Warmup is used for edge detection but shouldn't count as test exposure
		totalTestBars += (fold.TestEnd - fold.TestStart)
		if foldEdges == 0 {
			zeroTradeFolds++
		}
	}

	// Compute annualized rate from test windows only
	tfMinutes = atomic.LoadInt32(&globalTimeframeMinutes)
	if tfMinutes <= 0 {
		tfMinutes = 5 // Default to 5min if not set
	}
	// minutesPerYear is already a const declared earlier (line 767)
	barsPerYear = float64(minutesPerYear) / float64(tfMinutes)

	var tradableEdgesPerYear float64
	if totalTestBars > 0 {
		tradableEdgesPerYear = float64(sumTestEdges) / float64(totalTestBars) * barsPerYear
	}

	// REJECT conditions:
	// 1. Total test edges == 0 (completely dead in OOS)
	// 2. >= 50% of folds have zero edges (adaptive threshold - half or more folds dead)
	// 3. Edges per year below minimum (only when threshold > 0)
	zeroFoldThreshold := (len(folds) + 1) / 2 // ceil(N/2) = >= 50% folds dead

	if !debugDisableEntryRateGate && totalTestBars > 0 {
		// FIX #A: Treat 0 as disabled, -1 as auto (fallback to default), >0 as explicit
		// This aligns with CLI flag: -1=auto, 0=disabled, >0=explicit
		minEdgesPerYear := wfConfig.MinEdgesPerYear
		if minEdgesPerYear < 0 {
			minEdgesPerYear = wfDiscoveryEdgesPerYear // Fallback to global default (-1 = auto)
		}

		// CRITICAL FIX #A: Only apply edge rate gate when threshold is > 0
		// When 0 is passed, the gate is completely disabled (no fallback, no check)
		if minEdgesPerYear > 0 {
			if tfMinutes < 60 {
				// For 5min/15min: scale threshold up proportionally
				// At 5min: 12x more bars than 1H, so expect 12x more edges
				scaleFactor := 60.0 / float64(tfMinutes)
				minEdgesPerYear = minEdgesPerYear * scaleFactor
			}

			if sumTestEdges == 0 || zeroTradeFolds >= zeroFoldThreshold || tradableEdgesPerYear < minEdgesPerYear {
				atomic.AddInt64(&screenFailEntryRateLow, 1)
				reason := fmt.Sprintf("wf_fold_entry_rate_too_low: edges=%d, zero_folds=%d/%d, edge_rate=%.1f/year < %.1f",
					sumTestEdges, zeroTradeFolds, len(folds), tradableEdgesPerYear, minEdgesPerYear)
				maybeLogSampledRejection(reason, Result{Trades: sumTestEdges}, st.Seed)
				return false, false, false, Result{}, Result{}, reason
			}

			// Per-fold edge rate gate: check each individual fold meets minimum edges per year
			foldMinEdgesPerYear := wfConfig.FoldMinEdgesPerYear
			if foldMinEdgesPerYear < 0 {
				foldMinEdgesPerYear = wfDiscoveryEdgesPerYear // Fallback to global default (-1 = auto)
			}

			if foldMinEdgesPerYear > 0 {
				// Scale threshold for timeframes < 60min (same logic as overall gate)
				if tfMinutes < 60 {
					scaleFactor := 60.0 / float64(tfMinutes)
					foldMinEdgesPerYear = foldMinEdgesPerYear * scaleFactor
				}

				// Check each fold individually
				for i, fold := range folds {
					foldTestBars := fold.TestEnd - fold.TestStart
					if foldTestBars > 0 {
						foldEdgesPerYear := float64(perFoldEdgeCounts[i]) / float64(foldTestBars) * barsPerYear
						if foldEdgesPerYear < foldMinEdgesPerYear {
							atomic.AddInt64(&screenFailEntryRateLow, 1)
							reason := fmt.Sprintf("wf_per_fold_entry_rate_too_low: fold=%d/%d, fold_edges=%d, fold_bars=%d, fold_edge_rate=%.1f/year < %.1f",
								i, len(folds), perFoldEdgeCounts[i], foldTestBars, foldEdgesPerYear, foldMinEdgesPerYear)
							maybeLogSampledRejection(reason, Result{Trades: sumTestEdges}, st.Seed)
							return false, false, false, Result{}, Result{}, reason
						}
					}
				}
			}
		}
	}

	// DIAGNOSTIC: Log detailed edge analysis for first 5 strategies
	if count := atomic.LoadInt64(&totalEvaluated); count < 5 {
		// Compute threshold for display (same logic as gate)
		displayThreshold := wfConfig.MinEdgesPerYear
		if displayThreshold < 0 {
			displayThreshold = wfDiscoveryEdgesPerYear
		}
		if tfMinutes < 60 && displayThreshold > 0 {
			scaleFactor := 60.0 / float64(tfMinutes)
			displayThreshold = displayThreshold * scaleFactor
		}

		fmt.Printf("\n[FOLD EDGE DIAGNOSTIC #%d seed=%d]\n", count+1, st.Seed)
		fmt.Printf("  Folds: %d, SumTestEdges: %d, ZeroTradeFolds: %d/%d\n",
			len(folds), sumTestEdges, zeroTradeFolds, len(folds))
		fmt.Printf("  Tradable edges/year: %.2f, Threshold: %.2f\n",
			tradableEdgesPerYear, displayThreshold)
		fmt.Printf("  Per-fold edge counts: %v\n", perFoldEdgeCounts)

		// Sample first fold to diagnose WHY edges are 0
		if len(folds) > 0 && len(perFoldEdgeCounts) > 0 {
			sampleFold := folds[0]
			entryTrueCount := 0
			regimeOKCount := 0
			activeCount := 0
			foldBars := sampleFold.TestEnd - sampleFold.TestStart

			for t := sampleFold.TestStart; t < sampleFold.TestEnd && t < len(s.Close); t++ {
				if st.EntryCompiled.Code != nil && evaluateCompiled(st.EntryCompiled.Code, f.F, t) {
					entryTrueCount++
				}
				if st.RegimeCompiled.Code != nil && evaluateCompiled(st.RegimeCompiled.Code, f.F, t) {
					regimeOKCount++
				}
				if okActiveSliced && activeIdxSliced >= 0 && activeIdxSliced < len(f.F) && t < len(f.F[activeIdxSliced]) {
					if f.F[activeIdxSliced][t] > 0 {
						activeCount++
					}
				}
			}

			fmt.Printf("  Fold 0 diagnosis (bars %d-%d, total %d):\n",
				sampleFold.TestStart, sampleFold.TestEnd, foldBars)
			fmt.Printf("    Entry true: %d/%d bars (%.1f%%)\n",
				entryTrueCount, foldBars, 100.0*float64(entryTrueCount)/float64(foldBars))
			fmt.Printf("    Regime OK: %d/%d bars (%.1f%%)\n",
				regimeOKCount, foldBars, 100.0*float64(regimeOKCount)/float64(foldBars))
			fmt.Printf("    Active: %d/%d bars (%.1f%%)\n",
				activeCount, foldBars, 100.0*float64(activeCount)/float64(foldBars))
		}
		fmt.Printf("[END DIAGNOSTIC]\n\n")
	}

	// DEBUG: Log fold info for first few strategies - always log first time, then every 500
	count := atomic.LoadInt64(&totalEvaluated)
	if (count == 1 || count%500 == 1) && len(folds) < 10 {
		fmt.Printf("[DEBUG FOLDS] Generated %d folds from %d bars (%.0f days)\n", len(folds), len(s.Close), float64(len(s.Close))/barsPerDay)
		for i, fold := range folds {
			trainDays := float64(fold.TrainEnd-fold.TrainStart) / barsPerDay
			testDays := float64(fold.TestEnd-fold.TestStart) / barsPerDay
			fmt.Printf("  Fold %d: train=%.1fd, test=%.1fd, edges=%d (test+warmup)\n", i, trainDays, testDays, perFoldEdgeCounts[i])
		}
		fmt.Printf("  Total: sumTestEdges=%d, zeroTradeFolds=%d/%d, edge_rate=%.1f/year\n",
			sumTestEdges, zeroTradeFolds, len(folds), tradableEdgesPerYear)
	}

	// Evaluate strategy across all folds
	oosStats, _, _ := EvaluateWalkForward(s, f, st, folds, wfConfig)

	// CRITICAL DEBUG: If we have many folds but get few OOS months, there's a stitching bug
	if len(folds) >= 10 && oosStats.TotalMonths < 3 {
		fmt.Printf("[BUG] Many folds (%d) but few OOS months (%d)!\n", len(folds), oosStats.TotalMonths)
		fmt.Printf("  This indicates a bug in StitchOOSEquityCurve or ComputeMonthlyReturnsFromEquity\n")
	}

	// Track OOS zero-trade strategies
	if oosStats.TotalTrades == 0 {
		atomic.AddInt64(&oosZeroTrades, 1)

		// CRITICAL FIX #3: Hard reject strategies with zero OOS trades
		// These strategies cannot be validated and should not proceed
		reason := "wf_oos_zero_trades: strategy produced 0 trades in OOS validation"
		maybeLogSampledRejection(reason, Result{
			Trades:       0,
			Return:       float32(oosStats.GeoAvgMonthly),
			MaxDD:        float32(oosStats.MaxDD),
			ProfitFactor: float32(oosStats.OOSProfitFactor),
			Expectancy:   float32(oosStats.OOSExpectancy),
		}, st.Seed)
		return false, false, false, Result{}, Result{}, reason
	}

	// Check OOS hard constraints
	if oosStats.Rejected {
		// Increment the base counter (totalEvaluated is used for fold debugging, so we increment it here)
		atomic.AddInt64(&totalEvaluated, 1)

		// Increment specific counter based on rejection reason (granular tracking)
		reason := oosStats.RejectReason
		switch {
		case strings.Contains(reason, "Insufficient OOS trades") || strings.Contains(reason, "Insufficient trades"):
			atomic.AddInt64(&oosTradesTooLow, 1)
		case strings.Contains(reason, "Max DD exceeded"):
			atomic.AddInt64(&oosMaxDDTooHigh, 1)
		case strings.Contains(reason, "Worst month too bad"):
			atomic.AddInt64(&oosWorstMonthTooBad, 1)
		case strings.Contains(reason, "Sparse months ratio too high"):
			atomic.AddInt64(&oosRejectTooSparseMonths, 1)
		case strings.Contains(reason, "Active months ratio too low"):
			atomic.AddInt64(&oosRejectActiveMonthsLow, 1)
		case strings.Contains(reason, "Median monthly return too low"):
			atomic.AddInt64(&oosRejectMedianMonthlyFail, 1)
		case strings.Contains(reason, "Geo avg monthly return too low"):
			atomic.AddInt64(&oosRejectGeoMonthlyFail, 1)
		default:
			atomic.AddInt64(&oosOtherReject, 1)
		}

		// Log rejection reason for debugging - include OOS stats in Result for visibility
		reasonWithPrefix := "wf_oos_rejected: " + reason
		maybeLogSampledRejection(reasonWithPrefix, Result{
			Trades:       oosStats.TotalTrades,
			Return:       float32(oosStats.GeoAvgMonthly),
			MaxDD:        float32(oosStats.MaxDD),
			ProfitFactor: float32(oosStats.OOSProfitFactor),
			Expectancy:   float32(oosStats.OOSExpectancy),
		}, st.Seed)
		return false, false, false, Result{}, Result{}, reason
	}

	// Compute combined complexity (entry + exit)
	complexity := ComputeCombinedComplexity(st.EntryRule.Root, st.ExitRule.Root)

	// Check complexity caps
	if complexity.NodeCount > wfConfig.MaxNodes || complexity.UniqueFeatureCount() > wfConfig.MaxFeatures {
		atomic.AddInt64(&totalEvaluated, 1)
		atomic.AddInt64(&oosRejectTooComplex, 1)
		reason := "wf_too_complex"
		maybeLogSampledRejection(reason, Result{Trades: 0}, st.Seed)
		return false, false, false, Result{}, Result{}, reason
	}

	// Create synthetic results for compatibility
	// Use train window for basic result structure
	trainR := evaluateStrategyWindow(full, fullF, st, trainW)

	// Compute train performance in the same units as OOS (geo avg monthly) for overfit detection
	trainMonths := countMonthsInWindow(full.CloseTimeMs, trainW.Start, trainW.End)
	trainGeoMonthly := geoMonthlyFromTotalReturn(float64(trainR.Return), trainMonths)

	// Compute final fitness using aligned train/OOS units
	fitness := ComputeFinalFitness(oosStats, trainGeoMonthly, complexity, wfConfig)
	trainR.Score = float32(fitness) // Override with WF fitness

	// Use REAL OOS PF/Expectancy/WinRate from computed trades (not sentinel placeholders)
	// These are now computed from actual OOS trades in EvaluateWalkForward
	valR := Result{
		Strategy:     st,
		Score:        float32(fitness),
		Return:       float32(oosStats.GeoAvgMonthly),
		MaxDD:        float32(oosStats.MaxDD),
		WinRate:      float32(oosStats.OOSWinRate), // Real win rate from OOS trades
		Trades:       oosStats.TotalTrades,
		Expectancy:   float32(oosStats.OOSExpectancy),   // Real expectancy from OOS trades
		ProfitFactor: float32(oosStats.OOSProfitFactor), // Real PF from OOS trades

		// OOS Stats for EliteLog output - these contain the real OOS metrics
		OOSGeoAvgMonthly:  oosStats.GeoAvgMonthly,
		OOSMedianMonthly:  oosStats.MedianMonthly,
		OOSMinMonth:       oosStats.MinMonth,
		OOSStdMonth:       oosStats.StdMonth,
		OOSMaxDD:          oosStats.MaxDD,
		OOSTotalMonths:    oosStats.TotalMonths,
		OOSTotalTrades:    oosStats.TotalTrades,
		OOSMonthlyReturns: convertMonthlyReturnsToJSON(oosStats.MonthlyReturns),
	}

	// NOTE: LogOOSResults disabled for walk-forward to reduce terminal spam.
	// Monthly returns are already displayed in the terminal for accepted elites.
	// if Verbose {
	// 	LogOOSResults(oosStats, nil) // foldResults no longer captured
	// }

	// Return results with passedScreen=true, passedFull=true, passedValidation=true
	return true, true, true, trainR, valR, ""
}

// getScreenWindow creates a smaller screening window (3-6 months)
// Use last 6 months of train window for screening
func getScreenWindow(trainW Window) Window {
	// FIX #3: Make screen window timeframe-aware
	// Original code assumed 5min candles (~51,840 candles for 6 months)
	// Now adapts to actual timeframe (5m, 15m, 1h, etc.)
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	barsPerDay := int(1440.0 / tfMinutes) // Bars per day = 1440 min / minutes per bar
	screenCandles := barsPerDay * 30 * 6  // ~6 months (30 days * 6 months)

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
