package main

import (
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"
)

// Package-level variables for rate-limited debug warnings
var (
	printedStitchBug  bool
	stitchWarnCount   atomic.Int32
	maxStitchWarnings = int32(5)
)

func warnStitch(format string, args ...any) {
	// Use existing DebugWalkForward flag (defined in main.go)
	if !DebugWalkForward {
		return
	}
	current := stitchWarnCount.Load()
	if current >= maxStitchWarnings {
		return
	}
	if stitchWarnCount.CompareAndSwap(current, current+1) {
		fmt.Printf(format, args...)
		if current+1 >= maxStitchWarnings {
			fmt.Printf("[STITCH] Further warnings suppressed (max %d reached)\n", maxStitchWarnings)
		}
	}
}

// MonthlyReturn represents return statistics for a single month
type MonthlyReturn struct {
	Month   int     // Month index (0 = first OOS month)
	StartEq float64 // Equity at start of month
	EndEq   float64 // Equity at end of month
	Return  float64 // Monthly return: (EndEq - StartEq) / StartEq
	Trades  int     // Trades in this month
	DD      float64 // Max DD in this month
}

// StitchedOOSData combines equity with its matching timestamps/indices
type StitchedOOSData struct {
	Equity     []float64 // Stitched equity values
	BarIndices []int     // Matching absolute bar indices in series
	Timestamps []int64   // Matching timestamps from series
}

// MonthlyReturnJSON is the JSON-serializable version of MonthlyReturn for EliteLog output
type MonthlyReturnJSON struct {
	Month  int     `json:"month"`
	Return float64 `json:"return"`
	DD     float64 `json:"dd"`
	Trades int     `json:"trades"`
}

// OOSRejectCode identifies the reason for OOS rejection
type OOSRejectCode int

const (
	RejectNone OOSRejectCode = iota
	RejectInsufficientMonths
	RejectInsufficientTrades
	RejectMaxDD
	RejectMinMonth
	RejectGeoMonthly
	RejectActiveMonths  // NEW: Sparse months
	RejectMedianMonthly // NEW: Median too low
)

// OOSStats holds out-of-sample statistics
type OOSStats struct {
	// Primary metrics (use float64 for precision in geo mean calculation)
	GeoAvgMonthly      float64 // Geometric average monthly return (MAIN METRIC)
	MedianMonthly      float64 // Kept for backward compatibility = MedianActiveMonths
	MedianAllMonths    float64 // NEW: Median across ALL months (including sparse)
	MedianActiveMonths float64 // NEW: Median across only active months (trades > 0)
	MinMonth           float64 // Worst single month return
	StdMonth           float64 // Std dev of monthly returns (volatility)

	// Breakdown
	MonthlyReturns []MonthlyReturn // Per-month breakdown

	// Aggregates
	TotalMonths int
	TotalTrades int
	MaxDD       float64 // Worst drawdown across all OOS

	// Per-month trade stats (FIX #10)
	MinTradesPerMonth int // Minimum trades across all months

	// Active months ratio (for stability assessment)
	// FIX A: Active month = (Trades > 0) OR (abs(Return) > epsilon)
	// This counts months with carried positions as active, not just months with new trades
	ActiveMonthsCount int     // Number of months with trades OR meaningful returns
	ActiveMonthsRatio float64 // Active months / Total months

	// Sparse months ratio (for quality assessment)
	// Sparse month = month with 0 trades (regardless of carried positions)
	// High sparse ratio indicates strategy is not consistently active
	SparseMonthsCount int     // Number of months with 0 trades
	SparseMonthsRatio float64 // Sparse months / Total months

	// OOS Trade Metrics (computed from actual OOS trades)
	OOSProfitFactor float64 // Profit factor from OOS trades
	OOSExpectancy   float64 // Expectancy from OOS trades (avg PnL per trade)
	OOSWinRate      float64 // Win rate from OOS trades (wins / total trades)

	// Rejection
	Rejected     bool
	RejectReason string
	RejectCode   OOSRejectCode // NEW: Machine-readable reject code
}

// convertMonthlyReturnsToJSON converts MonthlyReturn array to MonthlyReturnJSON format for serialization
func convertMonthlyReturnsToJSON(monthlyReturns []MonthlyReturn) []MonthlyReturnJSON {
	if monthlyReturns == nil {
		return nil
	}
	result := make([]MonthlyReturnJSON, len(monthlyReturns))
	for i, mr := range monthlyReturns {
		result[i] = MonthlyReturnJSON{
			Month:  mr.Month,
			Return: mr.Return,
			DD:     mr.DD,
			Trades: mr.Trades,
		}
	}
	return result
}

// StitchOOSEquityCurve stitches together equity curves from NON-OVERLAPPING test folds only
// Returns both equity AND matching timestamps/indices
func StitchOOSEquityCurve(foldResults []FoldResult, series Series) StitchedOOSData {
	if len(foldResults) == 0 {
		return StitchedOOSData{}
	}

	var stitched []float64
	var stitchedIndices []int
	var stitchedTimestamps []int64
	currentEquity := 1.0

	// Fix C: Sort folds by TestStart before stitching
	sort.Slice(foldResults, func(i, j int) bool {
		return foldResults[i].TestStart < foldResults[j].TestStart
	})

	// Step 2B: Validate TestStart/TestEnd are global indices
	for _, fr := range foldResults {
		if fr.TestStart < 0 || fr.TestEnd > len(series.CloseTimeMs) || fr.TestStart >= fr.TestEnd {
			warnStitch("[BAD-FOLD-IDX] start=%d end=%d seriesLen=%d\n", fr.TestStart, fr.TestEnd, len(series.CloseTimeMs))
			break
		}
	}

	// Fix B: Use timestamp-based duplicate detection (not barIndex)
	lastTS := int64(-1)

	for _, fr := range foldResults {
		foldEquity := fr.EquityCurve

		for i, eq := range foldEquity {
			barIndex := fr.TestStart + i

			// bounds safety
			if barIndex < 0 || barIndex >= len(series.CloseTimeMs) {
				break
			}

			ts := series.CloseTimeMs[barIndex]

			// skip ONLY true duplicate boundary (same timestamp)
			if lastTS != -1 && ts == lastTS {
				continue
			}

			stitched = append(stitched, currentEquity*eq)
			stitchedIndices = append(stitchedIndices, barIndex)
			stitchedTimestamps = append(stitchedTimestamps, ts)
			lastTS = ts
		}

		if len(stitched) > 0 {
			currentEquity = stitched[len(stitched)-1]
		}
	}

	// Step 2C: Hard assertion - check stitching is not collapsing
	if len(stitchedTimestamps) != len(stitched) {
		panic("stitched timestamps and equity length mismatch")
	}

	// Fix 1: Wrap debug prints with rate-limited warning function
	if DebugWalkForward {
		expectedBars := foldResults[len(foldResults)-1].TestEnd - foldResults[0].TestStart
		warnStitch("[STITCH-TOO-SHORT] stitched=%d expected~=%d\n",
			len(stitchedTimestamps), expectedBars)

		// Step 1: Diagnostic print (only when DebugWalkForward is true)
		if len(stitchedTimestamps) > 0 {
			warnStitch("[STITCH-DBG] folds=%d stitchedBars=%d ts0=%d (%s) tsN=%d (%s)\n",
				len(foldResults), len(stitchedTimestamps),
				stitchedTimestamps[0], time.UnixMilli(stitchedTimestamps[0]).UTC().Format(time.RFC3339),
				stitchedTimestamps[len(stitchedTimestamps)-1], time.UnixMilli(stitchedTimestamps[len(stitchedTimestamps)-1]).UTC().Format(time.RFC3339))
		}
	}

	return StitchedOOSData{
		Equity:     stitched,
		BarIndices: stitchedIndices,
		Timestamps: stitchedTimestamps,
	}
}

// collectTradeEntryIndices collects all trade entry indices across all folds
func collectTradeEntryIndices(foldResults []FoldResult) []int {
	var indices []int
	for _, fr := range foldResults {
		indices = append(indices, fr.TradeEntryIdxs...)
	}
	return indices
}

// computeDrawdownFromEquitySlice computes drawdown from an equity slice
func computeDrawdownFromEquitySlice(equity []float64, barIndices []int, series Series) float64 {
	if len(equity) == 0 {
		return 0
	}

	peak := equity[0]
	maxDD := 0.0

	for _, eq := range equity {
		if eq > peak {
			peak = eq
		}
		dd := (peak - eq) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}

	return maxDD
}

// extractYearMonthUTC extracts (year, month) from timestamp in UTC
func extractYearMonthUTC(ms int64) (int, int) {
	t := time.UnixMilli(ms).UTC()
	return t.Year(), int(t.Month())
}

// countMonthsInWindow counts distinct UTC months in [start, end) using series timestamps.
// Returns 0 if the range is empty or indices are invalid.
func countMonthsInWindow(times []int64, start, end int) int {
	if len(times) == 0 {
		return 0
	}
	if start < 0 {
		start = 0
	}
	if end > len(times) {
		end = len(times)
	}
	if end <= start {
		return 0
	}

	year, month := extractYearMonthUTC(times[start])
	months := 1
	for i := start + 1; i < end; i++ {
		y, m := extractYearMonthUTC(times[i])
		if y != year || m != month {
			months++
			year, month = y, m
		}
	}
	return months
}

// geoMonthlyFromTotalReturn converts a total compounded return into a per-month geometric average.
// Returns 0 if months <= 0, and -1 if return <= -100%.
func geoMonthlyFromTotalReturn(totalReturn float64, months int) float64 {
	if months <= 0 {
		return 0
	}
	if totalReturn <= -0.999999 {
		return -1.0
	}
	return math.Pow(1.0+totalReturn, 1.0/float64(months)) - 1.0
}

// ComputeMonthlyReturnsFromEquity computes monthly returns using UTC boundaries
func ComputeMonthlyReturnsFromEquity(stitched StitchedOOSData,
	foldResults []FoldResult,
	series Series) []MonthlyReturn {
	var monthlyReturns []MonthlyReturn

	if len(stitched.Equity) == 0 || len(stitched.Timestamps) == 0 {
		return monthlyReturns
	}

	// First, collect all trade entry indices across all folds
	allTradeEntryIndices := collectTradeEntryIndices(foldResults)

	currentMonthStart := 0
	var monthIndex int
	currentYear, currentMonth, _ := time.UnixMilli(stitched.Timestamps[0]).UTC().Date()

	for i := 1; i < len(stitched.Equity); i++ {
		// Use UTC for month boundary detection
		thisYear, thisMonth, _ := time.UnixMilli(stitched.Timestamps[i]).UTC().Date()

		if thisYear != currentYear || thisMonth != currentMonth {
			// Month ended - compute return for bars [currentMonthStart, i)
			// endEq is at i-1 (last bar of current month), NOT i (first bar of next month)
			startEq := stitched.Equity[currentMonthStart]
			endEq := stitched.Equity[i-1]

			// Count trades in this month by checking entry bar indices
			// Use half-open interval [monthStartMs, nextMonthStartMs)
			tradesInMonth := 0
			monthStartMs := stitched.Timestamps[currentMonthStart]
			nextMonthStartMs := stitched.Timestamps[i]

			for _, entryIdx := range allTradeEntryIndices {
				tradeMs := series.CloseTimeMs[entryIdx]
				if tradeMs >= monthStartMs && tradeMs < nextMonthStartMs {
					tradesInMonth++
				}
			}

			// Compute per-month drawdown from equity curve
			// Slice is [currentMonthStart:i) to exclude first bar of next month
			monthDD := computeDrawdownFromEquitySlice(
				stitched.Equity[currentMonthStart:i],
				stitched.BarIndices[currentMonthStart:i],
				series,
			)

			mr := MonthlyReturn{
				Month:   monthIndex,
				StartEq: startEq,
				EndEq:   endEq,
				Return:  (endEq - startEq) / startEq,
				Trades:  tradesInMonth,
				DD:      monthDD,
			}
			monthlyReturns = append(monthlyReturns, mr)

			monthIndex++
			currentYear, currentMonth = thisYear, thisMonth
			currentMonthStart = i
		}
	}

	// After loop ends, finalize last month
	if currentMonthStart < len(stitched.Equity) {
		lastMonthStart := currentMonthStart
		lastMonthEnd := len(stitched.Equity) - 1

		startEq := stitched.Equity[lastMonthStart]
		endEq := stitched.Equity[lastMonthEnd]

		// Count trades in last month (no upper bound)
		monthStartMs := stitched.Timestamps[lastMonthStart]
		tradesInMonth := 0
		for _, entryIdx := range allTradeEntryIndices {
			tradeMs := series.CloseTimeMs[entryIdx]
			if tradeMs >= monthStartMs {
				tradesInMonth++
			}
		}

		monthDD := computeDrawdownFromEquitySlice(
			stitched.Equity[lastMonthStart:],
			stitched.BarIndices[lastMonthStart:],
			series,
		)

		mr := MonthlyReturn{
			Month:   monthIndex,
			StartEq: startEq,
			EndEq:   endEq,
			Return:  (endEq - startEq) / startEq,
			Trades:  tradesInMonth,
			DD:      monthDD,
		}
		monthlyReturns = append(monthlyReturns, mr)
	}

	return monthlyReturns
}

// ComputeGeoAvgMonthly computes geometric average monthly return
// Formula: geoAvg = exp((1/N) * sum(ln(1 + r_m))) - 1
func ComputeGeoAvgMonthly(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	sumLog := 0.0
	for _, r := range returns {
		if r <= -1.0 {
			// Total loss in a month - invalid for geo mean
			return math.Inf(-1)
		}
		sumLog += math.Log(1.0 + r)
	}

	meanLog := sumLog / float64(len(returns))
	return math.Exp(meanLog) - 1.0
}

// sumTradesFromMonthlyReturns sums trades across all months
func sumTradesFromMonthlyReturns(monthlyReturns []MonthlyReturn) int {
	total := 0
	for _, mr := range monthlyReturns {
		total += mr.Trades
	}
	return total
}

// median computes median of a float64 slice
func median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// std computes standard deviation of a float64 slice
func std(data []float64) float64 {
	if len(data) < 2 {
		return 0
	}
	meanVal := mean(data)
	sumSq := 0.0
	for _, v := range data {
		diff := v - meanVal
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(data)-1))
}

// mean computes mean of a float64 slice
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// ValidateOOSConstraints validates OOS constraints and sets rejection status
func ValidateOOSConstraints(stats *OOSStats, config WFConfig) {
	// Constraint 1: Min months (default: 24 = 2 years OOS)
	if stats.TotalMonths < config.MinMonths {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientMonths
		stats.RejectReason = fmt.Sprintf("Insufficient months: %d < %d (need 2+ years OOS)",
			stats.TotalMonths, config.MinMonths)
		return
	}

	// Constraint 2: Min trades (default: 200, or 100 for lower-frequency strategies)
	if stats.TotalTrades < config.MinTotalTradesOOS {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientTrades
		stats.RejectReason = fmt.Sprintf("Insufficient OOS trades: %d < %d",
			stats.TotalTrades, config.MinTotalTradesOOS)
		return
	}

	// Constraint 2b: Min trades per month (prevents rare-trade winners)
	if config.MinTradesPerMonth > 0 && stats.MinTradesPerMonth < config.MinTradesPerMonth {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientTrades
		stats.RejectReason = fmt.Sprintf("Insufficient trades per month: %d < %d (some months too sparse)",
			stats.MinTradesPerMonth, config.MinTradesPerMonth)
		return
	}

	// Constraint 3: Max drawdown (default: 0.25 = 25%)
	if stats.MaxDD > config.MaxDrawdown {
		stats.Rejected = true
		stats.RejectCode = RejectMaxDD
		stats.RejectReason = fmt.Sprintf("Max DD exceeded: %.2f%% > %.2f%%",
			stats.MaxDD*100, config.MaxDrawdown*100)
		return
	}

	// PROBLEM B FIX: Constraint 4 (optional but strong): Worst month cap - made trade-count dependent
	// A single bad month with 0-2 trades shouldn't reject an otherwise good strategy
	// Only reject worst month if: (a) it's very bad OR (b) it had meaningful trades
	if config.EnableMinMonthConstraint {
		// Find the worst month and check its trade count
		worstMonthTrades := 0
		for _, mr := range stats.MonthlyReturns {
			if mr.Return == stats.MinMonth {
				worstMonthTrades = mr.Trades
				break
			}
		}

		// FIX #1: Add absolute floor at -15% (disaster prevention)
		// No month can be worse than -15% regardless of trade count
		// RESPECT CLI: floor also respects MinMonthReturn (more lenient if configured lower)
		absoluteFloor := math.Min(-0.15, config.MinMonthReturn)
		// if config.MinMonthReturn is -0.20, floor becomes -0.20 (more lenient for Phase A)

		// Dynamic threshold: stricter for months with more trades, lenient for sparse months
		// If worst month had < 3 trades, allow double the normal drawdown
		// If worst month had >= 10 trades, use standard threshold
		dynamicThreshold := config.MinMonthReturn
		if worstMonthTrades < 3 {
			dynamicThreshold = math.Max(absoluteFloor, config.MinMonthReturn*2.0)
		} else if worstMonthTrades < 10 {
			dynamicThreshold = math.Max(absoluteFloor, config.MinMonthReturn*1.5)
		} else {
			dynamicThreshold = math.Max(absoluteFloor, config.MinMonthReturn)
		}

		// NEW: Also reject if worst month is beyond absolute floor (catastrophic month)
		if stats.MinMonth < absoluteFloor {
			// DEBUG: Print rejection details
			fmt.Printf("[MIN-MONTH-REJECT] absolute_floor: MinMonth=%.2f%% < Floor=%.2f%% (trds=%d, config_min=%.2f%%)\n",
				stats.MinMonth*100, absoluteFloor*100, worstMonthTrades, config.MinMonthReturn*100)
			stats.Rejected = true
			stats.RejectCode = RejectMinMonth
			stats.RejectReason = fmt.Sprintf("Worst month catastrophic: %.2f%% < %.2f%% (absolute floor, trds=%d)",
				stats.MinMonth*100, absoluteFloor*100, worstMonthTrades)
			return
		}

		if stats.MinMonth < dynamicThreshold {
			// DEBUG: Print rejection details
			fmt.Printf("[MIN-MONTH-REJECT] dynamic_threshold: MinMonth=%.2f%% < Threshold=%.2f%% (trds=%d, config_min=%.2f%%, floor=%.2f%%)\n",
				stats.MinMonth*100, dynamicThreshold*100, worstMonthTrades, config.MinMonthReturn*100, absoluteFloor*100)
			stats.Rejected = true
			stats.RejectCode = RejectMinMonth
			stats.RejectReason = fmt.Sprintf("Worst month too bad: %.2f%% < %.2f%% (threshold=%.2f%%, trds=%d)",
				stats.MinMonth*100, config.MinMonthReturn*100, dynamicThreshold*100, worstMonthTrades)
			return
		}
	}

	// Constraint 5: Minimum geometric average monthly return
	// Prevents strategies with very low returns from being added to Hall of Fame
	if config.MinGeoMonthlyReturn > 0 && stats.GeoAvgMonthly < config.MinGeoMonthlyReturn {
		stats.Rejected = true
		stats.RejectCode = RejectGeoMonthly
		stats.RejectReason = fmt.Sprintf("Geo avg monthly return too low: %.4f%% < %.4f%%",
			stats.GeoAvgMonthly*100, config.MinGeoMonthlyReturn*100)
		return
	}

	// FIX A: Constraint 6: Active months ratio gate
	// Prevents strategies that only trade in a few months from passing
	// Active month = (Trades > 0) OR (abs(Return) > epsilon) - includes carried positions
	if config.MinActiveMonthsRatio > 0 && stats.ActiveMonthsRatio < config.MinActiveMonthsRatio {
		stats.Rejected = true
		stats.RejectCode = RejectActiveMonths
		stats.RejectReason = fmt.Sprintf("Active months ratio too low: %.2f%% < %.2f%% (%d/%d months active with trades OR returns)",
			stats.ActiveMonthsRatio*100, config.MinActiveMonthsRatio*100,
			stats.ActiveMonthsCount, stats.TotalMonths)
		return
	}

	// Constraint 6b: Sparse months ratio gate (NEW)
	// Rejects strategies with too many months having 0 trades
	// This eliminates "Median=0.00%" strategies that are not consistently active
	// Sparse month = month with 0 trades (regardless of carried positions)
	if config.MaxSparseMonthsRatio > 0 && stats.SparseMonthsRatio > config.MaxSparseMonthsRatio {
		stats.Rejected = true
		stats.RejectCode = RejectActiveMonths // Reuse reject code for now
		stats.RejectReason = fmt.Sprintf("Sparse months ratio too high: %.2f%% > %.2f%% (%d/%d months with 0 trades)",
			stats.SparseMonthsRatio*100, config.MaxSparseMonthsRatio*100,
			stats.SparseMonthsCount, stats.TotalMonths)
		return
	}

	// Constraint 7: Minimum median monthly return
	// Ensures consistent positive performance across active months
	if config.MinMedianMonthly > 0 && stats.MedianMonthly < config.MinMedianMonthly {
		stats.Rejected = true
		stats.RejectCode = RejectMedianMonthly
		stats.RejectReason = fmt.Sprintf("Median active monthly return too low: %.2f%% < %.2f%% (all=%.2f%%, active=%.2f%%)",
			stats.MedianActiveMonths*100, config.MinMedianMonthly*100,
			stats.MedianAllMonths*100, stats.MedianActiveMonths*100)
		return
	}

	// Constraint 8: Maximum monthly volatility (hard stability gate)
	// Rejects strategies with extreme monthly volatility (configurable via -wf_max_std_month)
	// This prevents unstable strategies from passing validation
	if config.MaxStdMonth > 0 && stats.StdMonth > config.MaxStdMonth {
		stats.Rejected = true
		stats.RejectCode = RejectActiveMonths // Reuse reject code
		stats.RejectReason = fmt.Sprintf("Monthly volatility too high: %.2f%% > %.2f%% (unstable strategy)",
			stats.StdMonth*100, config.MaxStdMonth*100)
		return
	}
}

// ComputeOOSStats computes OOS statistics from stitched equity and monthly returns
func ComputeOOSStats(stitched StitchedOOSData, monthlyReturns []MonthlyReturn,
	foldResults []FoldResult, series Series, config WFConfig) OOSStats {
	stats := OOSStats{
		MonthlyReturns: monthlyReturns,
		TotalMonths:    len(monthlyReturns),
		TotalTrades:    sumTradesFromMonthlyReturns(monthlyReturns),
	}

	// CRITICAL DEBUG: If we have many folds but get few OOS months/bars, there's a stitching bug
	// Fix 2: Print only once per run using package-level guard
	if DebugWalkForward && !printedStitchBug {
		expectedBars := 0
		for _, fr := range foldResults {
			expectedBars += len(fr.EquityCurve)
		}
		if len(foldResults) >= 10 && len(monthlyReturns) < 3 && len(stitched.Equity) < expectedBars/2 {
			printedStitchBug = true
			warnStitch("[BUG] Many folds (%d) but few OOS months (%d) and few stitched bars (%d vs %d expected)!\n",
				len(foldResults), len(monthlyReturns), len(stitched.Equity), expectedBars)
			fmt.Printf("  This indicates a bug in StitchOOSEquityCurve - not all folds being stitched!\n")
		}
	}

	// Extract return series for geo mean
	// FIX A: Active month = (Trades > 0) OR (abs(Return) > epsilon)
	// A month with 0 trades but non-zero return means a position was carried across months
	// This is STILL an active month with market exposure
	const tinyEps = 1e-6
	var activeReturns []float64
	var activeMonthCount int
	// FIX: Initialize MinMonth to +Inf so all-positive strategies get correct worst month
	stats.MinMonth = math.Inf(1)
	for _, mr := range monthlyReturns {
		// FIX A: Month is active if it has trades OR meaningful return (position carried over)
		isActive := mr.Trades > 0 || mr.Return > tinyEps || mr.Return < -tinyEps
		if isActive {
			activeReturns = append(activeReturns, mr.Return)
			activeMonthCount++
		}
		// Still track MinMonth across ALL months (for worst-month constraint)
		if mr.Return < stats.MinMonth {
			stats.MinMonth = mr.Return
		}
	}

	// If no active months (shouldn't happen with the new logic), use all returns
	if len(activeReturns) == 0 {
		for _, mr := range monthlyReturns {
			activeReturns = append(activeReturns, mr.Return)
			activeMonthCount++
		}
	}

	// Compute median from active months only (better stability representation)
	stats.MedianActiveMonths = median(activeReturns)

	// Compute median across ALL months (including sparse with 0 trades)
	var allMonthlyReturns []float64
	for _, mr := range monthlyReturns {
		allMonthlyReturns = append(allMonthlyReturns, mr.Return)
	}
	stats.MedianAllMonths = median(allMonthlyReturns)

	// Maintain backward compatibility
	stats.MedianMonthly = stats.MedianActiveMonths

	// Compute overall OOS MaxDD from stitched equity curve
	stats.MaxDD = computeDrawdownFromEquitySlice(stitched.Equity, stitched.BarIndices, series)

	// Track minimum trades per month (FIX #10)
	// Initialize to large value instead of count of months
	stats.MinTradesPerMonth = math.MaxInt32
	for _, mr := range monthlyReturns {
		if mr.Trades < stats.MinTradesPerMonth {
			stats.MinTradesPerMonth = mr.Trades
		}
	}
	// Handle edge case of empty monthlyReturns
	if len(monthlyReturns) == 0 {
		stats.MinTradesPerMonth = 0
	}

	// Compute monthly volatility (std dev) from active months only
	// This prevents months with 0 trades from inflating volatility
	stats.StdMonth = std(activeReturns)

	// FIX A: Compute active months ratio for stability gate
	// Active month = (Trades > 0) OR (abs(Return) > epsilon)
	stats.ActiveMonthsCount = activeMonthCount
	if stats.TotalMonths > 0 {
		stats.ActiveMonthsRatio = float64(activeMonthCount) / float64(stats.TotalMonths)
	}

	// Compute sparse months ratio for quality gate
	// Sparse month = month with 0 trades (regardless of carried positions)
	// This catches strategies that are not consistently active
	var sparseMonthCount int
	for _, mr := range monthlyReturns {
		if mr.Trades == 0 {
			sparseMonthCount++
		}
	}
	stats.SparseMonthsCount = sparseMonthCount
	if stats.TotalMonths > 0 {
		stats.SparseMonthsRatio = float64(sparseMonthCount) / float64(stats.TotalMonths)
	}

	// Geometric average: exp(mean(log(1+r))) - 1 from active months only
	// This focuses on actual trading performance, not periods of inactivity
	stats.GeoAvgMonthly = ComputeGeoAvgMonthly(activeReturns)

	// Validate constraints
	ValidateOOSConstraints(&stats, config)

	return stats
}

// LogOOSResults logs OOS results with fold-by-fold and monthly breakdown
func LogOOSResults(oos OOSStats, foldResults []FoldResult) {
	fmt.Printf("\n=== Out-of-Sample Results ===\n")
	fmt.Printf("Months: %d | Total Trades: %d | Max DD: %.2f%%\n",
		oos.TotalMonths, oos.TotalTrades, oos.MaxDD*100)
	fmt.Printf("Geo Avg Monthly: %.2f%% | Worst Month: %.2f%%\n",
		oos.GeoAvgMonthly*100, oos.MinMonth*100)

	fmt.Printf("\n--- Monthly Breakdown ---\n")
	for _, mr := range oos.MonthlyReturns {
		fmt.Printf("Month %2d: Ret=%+6.2f%% | DD=%5.2f%% | Trades=%2d\n",
			mr.Month, mr.Return*100, mr.DD*100, mr.Trades)
	}

	fmt.Printf("\n--- Fold Details ---\n")
	for _, fr := range foldResults {
		mtmInfo := ""
		if fr.MTMPositions > 0 {
			mtmInfo = fmt.Sprintf(" [MTM: %d pos]", fr.MTMPositions)
		}
		fmt.Printf("Fold %d: Score=%.4f | Ret=%+.2f%% | DD=%.2f%% | Trades=%d%s [%s -> %s]\n",
			fr.FoldNumber, fr.TestScore, fr.TestReturn*100, fr.TestDD*100,
			fr.TestTrades, mtmInfo, fr.TrainDate, fr.TestDate)
	}
}

// LogBestStrategy logs best strategy metrics with UniqueFeatureCount fix
func LogBestStrategy(stats OOSStats, complexity Complexity, fitness float64) {
	fmt.Printf("bestFit=%.4f ", fitness)
	fmt.Printf("geoMo=%.1f%% ", stats.GeoAvgMonthly*100)
	fmt.Printf("medMo=%.1f%% ", stats.MedianMonthly*100)
	fmt.Printf("minMo=%.1f%% ", stats.MinMonth*100)
	fmt.Printf("months=%d ", stats.TotalMonths)
	fmt.Printf("trades=%d ", stats.TotalTrades)
	fmt.Printf("dd=%.1f%% ", stats.MaxDD*100)
	fmt.Printf("nodes=%d ", complexity.NodeCount)
	fmt.Printf("feats=%d\n", complexity.UniqueFeatureCount()) // FIX #8: Added ()
}
