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
	GeoAvgMonthly       float64 // Geometric average monthly return across ALL months (MAIN METRIC)
	ActiveGeoAvgMonthly float64 // Geometric average across active months only (diagnostic)
	MedianMonthly       float64 // Kept for backward compatibility = MedianActiveMonths
	MedianAllMonths     float64 // NEW: Median across ALL months (including sparse)
	MedianActiveMonths  float64 // NEW: Median across only active months (trades > 0)
	MinMonth            float64 // Worst single month return
	StdMonth            float64 // Std dev of monthly returns (volatility)

	// Breakdown
	MonthlyReturns []MonthlyReturn // Per-month breakdown

	// Aggregates
	TotalMonths int
	TotalTrades int
	MaxDD       float64 // Worst drawdown across all OOS

	// Per-month trade stats (FIX #10)
	MinTradesPerMonth int // Minimum trades across all months

	// Active months ratio (for stability assessment)
	ActiveMonthsCount int     // Number of months with trades OR meaningful returns
	ActiveMonthsRatio float64 // Active months / Total months

	// Sparse months ratio (for quality assessment)
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
			// CRITICAL FIX: Ensure equity doesn't become negative or NaN
			lastEq := stitched[len(stitched)-1]
			if math.IsNaN(lastEq) || math.IsInf(lastEq, 0) {
				lastEq = 0
			}
			if lastEq < 0 {
				lastEq = 0
			}
			currentEquity = lastEq
		}
	}

	return StitchedOOSData{
		Equity:     stitched,
		BarIndices: stitchedIndices,
		Timestamps: stitchedTimestamps,
	}
}

// collectTradeEntryTimesMs collects all trade entry timestamps across all folds
func collectTradeEntryTimesMs(foldResults []FoldResult) []int64 {
	var times []int64
	for _, fr := range foldResults {
		for _, tr := range fr.Trades {
			times = append(times, tr.EntryTime.Unix()*1000)
		}
	}
	return times
}

// computeDrawdownFromEquitySlice computes drawdown from an equity slice
func computeDrawdownFromEquitySlice(equity []float64, barIndices []int, series Series) float64 {
	if len(equity) == 0 {
		return 0
	}

	peak := equity[0]
	maxDD := 0.0

	for _, eq := range equity {
		// FIX: Handle bad data points
		if math.IsNaN(eq) || math.IsInf(eq, 0) {
			continue
		}
		if eq > peak {
			peak = eq
		}
		// Prevent division by zero if peak is 0
		if peak > 1e-9 {
			dd := (peak - eq) / peak
			if dd > maxDD {
				maxDD = dd
			}
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

	// CRITICAL FIX: Use trade timestamps instead of indices for robust counting
	allTradeEntryTimes := collectTradeEntryTimesMs(foldResults)

	currentMonthStart := 0
	var monthIndex int
	currentYear, currentMonth, _ := time.UnixMilli(stitched.Timestamps[0]).UTC().Date()

	for i := 1; i < len(stitched.Equity); i++ {
		thisYear, thisMonth, _ := time.UnixMilli(stitched.Timestamps[i]).UTC().Date()

		if thisYear != currentYear || thisMonth != currentMonth {
			// Month ended - compute return for bars [currentMonthStart, i)
			startEq := stitched.Equity[currentMonthStart]
			endEq := stitched.Equity[i-1]

			tradesInMonth := 0
			monthStartMs := stitched.Timestamps[currentMonthStart]
			nextMonthStartMs := stitched.Timestamps[i]

			for _, tradeMs := range allTradeEntryTimes {
				if tradeMs >= monthStartMs && tradeMs < nextMonthStartMs {
					tradesInMonth++
				}
			}

			monthDD := computeDrawdownFromEquitySlice(
				stitched.Equity[currentMonthStart:i],
				stitched.BarIndices[currentMonthStart:i],
				series,
			)

			// FIX: Safety check for division by zero (StartEq=0) and NaN
			var monthRet float64
			if startEq <= 1e-9 {
				monthRet = 0.0
			} else {
				monthRet = (endEq - startEq) / startEq
			}

			if math.IsNaN(monthRet) || math.IsInf(monthRet, 0) {
				monthRet = 0.0
			}

			mr := MonthlyReturn{
				Month:   monthIndex,
				StartEq: startEq,
				EndEq:   endEq,
				Return:  monthRet,
				Trades:  tradesInMonth,
				DD:      monthDD,
			}
			monthlyReturns = append(monthlyReturns, mr)

			monthIndex++
			currentYear, currentMonth = thisYear, thisMonth
			currentMonthStart = i
		}
	}

	// Finalize last month
	if currentMonthStart < len(stitched.Equity) {
		lastMonthStart := currentMonthStart
		lastMonthEnd := len(stitched.Equity) - 1

		startEq := stitched.Equity[lastMonthStart]
		endEq := stitched.Equity[lastMonthEnd]

		monthStartMs := stitched.Timestamps[lastMonthStart]
		tradesInMonth := 0
		for _, tradeMs := range allTradeEntryTimes {
			if tradeMs >= monthStartMs {
				tradesInMonth++
			}
		}

		monthDD := computeDrawdownFromEquitySlice(
			stitched.Equity[lastMonthStart:],
			stitched.BarIndices[lastMonthStart:],
			series,
		)

		var monthRet float64
		if startEq <= 1e-9 {
			monthRet = 0.0
		} else {
			monthRet = (endEq - startEq) / startEq
		}

		if math.IsNaN(monthRet) || math.IsInf(monthRet, 0) {
			monthRet = 0.0
		}

		mr := MonthlyReturn{
			Month:   monthIndex,
			StartEq: startEq,
			EndEq:   endEq,
			Return:  monthRet,
			Trades:  tradesInMonth,
			DD:      monthDD,
		}
		monthlyReturns = append(monthlyReturns, mr)
	}

	return monthlyReturns
}

// ComputeGeoAvgMonthly computes geometric average monthly return
func ComputeGeoAvgMonthly(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}

	sumLog := 0.0
	validCount := 0
	for _, r := range returns {
		// FIX: Handle NaN and Inf explicitly
		if math.IsNaN(r) || math.IsInf(r, 0) {
			continue // Skip invalid data points
		}
		if r <= -1.0 {
			return -1.0
		}
		sumLog += math.Log(1.0 + r)
		validCount++
	}

	if validCount == 0 {
		return 0
	}

	meanLog := sumLog / float64(validCount)
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

	// FIX: Filter out NaNs before sorting
	validData := make([]float64, 0, len(data))
	for _, v := range data {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			validData = append(validData, v)
		}
	}

	if len(validData) == 0 {
		return 0
	}

	sorted := make([]float64, len(validData))
	copy(sorted, validData)
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
	count := 0
	for _, v := range data {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			diff := v - meanVal
			sumSq += diff * diff
			count++
		}
	}
	if count < 2 {
		return 0
	}
	return math.Sqrt(sumSq / float64(count-1))
}

// mean computes mean of a float64 slice
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	count := 0
	for _, v := range data {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			sum += v
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// ValidateOOSConstraints validates OOS constraints and sets rejection status
func ValidateOOSConstraints(stats *OOSStats, config WFConfig) {
	// Constraint 1: Min months
	if stats.TotalMonths < config.MinMonths {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientMonths
		stats.RejectReason = fmt.Sprintf("Insufficient months: %d < %d", stats.TotalMonths, config.MinMonths)
		return
	}

	// Constraint 2: Min trades
	if stats.TotalTrades < config.MinTotalTradesOOS {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientTrades
		stats.RejectReason = fmt.Sprintf("Insufficient OOS trades: %d < %d", stats.TotalTrades, config.MinTotalTradesOOS)
		return
	}

	// Constraint 2b: Min trades per month
	if config.MinTradesPerMonth > 0 && stats.MinTradesPerMonth < config.MinTradesPerMonth {
		stats.Rejected = true
		stats.RejectCode = RejectInsufficientTrades
		stats.RejectReason = fmt.Sprintf("Insufficient trades per month: %d < %d", stats.MinTradesPerMonth, config.MinTradesPerMonth)
		return
	}

	// Constraint 3: Max drawdown
	if stats.MaxDD > config.MaxDrawdown {
		stats.Rejected = true
		stats.RejectCode = RejectMaxDD
		stats.RejectReason = fmt.Sprintf("Max DD too high: %.2f%% > %.2f%%", stats.MaxDD*100, config.MaxDrawdown*100)
		return
	}

	// Constraint 4: Min monthly return (worst month)
	if config.EnableMinMonthConstraint && stats.MinMonth < config.MinMonthReturn {
		stats.Rejected = true
		stats.RejectCode = RejectMinMonth
		stats.RejectReason = fmt.Sprintf("Worst month too bad: %.2f%% < %.2f%%", stats.MinMonth*100, config.MinMonthReturn*100)
		return
	}

	// Constraint 5: Positive geometric expectancy
	if stats.GeoAvgMonthly < config.MinGeoMonthlyReturn {
		stats.Rejected = true
		stats.RejectCode = RejectGeoMonthly
		stats.RejectReason = fmt.Sprintf("Geo avg monthly too low: %.2f%% < %.2f%%", stats.GeoAvgMonthly*100, config.MinGeoMonthlyReturn*100)
		return
	}

	// Constraint 6: Active months ratio
	if stats.ActiveMonthsRatio < config.MinActiveMonthsRatio {
		stats.Rejected = true
		stats.RejectCode = RejectActiveMonths
		stats.RejectReason = fmt.Sprintf("Active months ratio low: %.1f%% < %.1f%%", stats.ActiveMonthsRatio*100, config.MinActiveMonthsRatio*100)
		return
	}

	// Constraint 7: Sparse months ratio
	if config.MaxSparseMonthsRatio > 0 && stats.SparseMonthsRatio > config.MaxSparseMonthsRatio {
		stats.Rejected = true
		stats.RejectCode = RejectActiveMonths
		stats.RejectReason = fmt.Sprintf("Too many sparse months: %.1f%% > %.1f%%", stats.SparseMonthsRatio*100, config.MaxSparseMonthsRatio*100)
		return
	}

	// Constraint 8: Median monthly return
	if config.MinMedianMonthly > 0 && stats.MedianAllMonths < config.MinMedianMonthly {
		stats.Rejected = true
		stats.RejectCode = RejectMedianMonthly
		stats.RejectReason = fmt.Sprintf("Median monthly too low: %.2f%% < %.2f%%", stats.MedianAllMonths*100, config.MinMedianMonthly*100)
		return
	}

	// Constraint 9: Monthly volatility
	if config.MaxStdMonth > 0 && stats.StdMonth > config.MaxStdMonth {
		stats.Rejected = true
		stats.RejectCode = RejectGeoMonthly
		stats.RejectReason = fmt.Sprintf("Monthly volatility too high: %.2f%% > %.2f%%", stats.StdMonth*100, config.MaxStdMonth*100)
		return
	}
}

// CalculateOOSStats computes full statistics from fold results
func CalculateOOSStats(foldResults []FoldResult, series Series, config WFConfig) OOSStats {
	stats := OOSStats{}

	// 1. Stitch equity curve
	stitched := StitchOOSEquityCurve(foldResults, series)
	if len(stitched.Equity) == 0 {
		stats.Rejected = true
		stats.RejectReason = "No OOS equity curve generated (empty folds?)"
		stats.RejectCode = RejectInsufficientTrades
		return stats
	}

	// 2. Compute monthly returns
	monthlyReturns := ComputeMonthlyReturnsFromEquity(stitched, foldResults, series)
	stats.MonthlyReturns = monthlyReturns
	stats.TotalMonths = len(monthlyReturns)

	if stats.TotalMonths == 0 {
		stats.Rejected = true
		stats.RejectReason = "No monthly returns generated (OOS period too short?)"
		stats.RejectCode = RejectInsufficientMonths
		return stats
	}

	// 3. Compute aggregates
	stats.TotalTrades = sumTradesFromMonthlyReturns(monthlyReturns)

	// FIX: Robust MaxDD calculation
	stats.MaxDD = computeDrawdownFromEquitySlice(stitched.Equity, stitched.BarIndices, series)

	// 4. Compute distribution stats
	var allReturns []float64
	var activeReturns []float64
	minMonth := 1.0
	minTrades := 999999

	activeMonths := 0
	sparseMonths := 0

	for _, mr := range monthlyReturns {
		allReturns = append(allReturns, mr.Return)
		if mr.Return < minMonth {
			minMonth = mr.Return
		}
		if mr.Trades < minTrades {
			minTrades = mr.Trades
		}

		// Active month: has trades OR significant return
		isActive := mr.Trades > 0 || math.Abs(mr.Return) > 0.001
		if isActive {
			activeMonths++
			activeReturns = append(activeReturns, mr.Return)
		}

		// Sparse month: 0 trades
		if mr.Trades == 0 {
			sparseMonths++
		}
	}

	stats.MinMonth = minMonth
	stats.MinTradesPerMonth = minTrades
	if minTrades == 999999 {
		stats.MinTradesPerMonth = 0
	}

	stats.ActiveMonthsCount = activeMonths
	if stats.TotalMonths > 0 {
		stats.ActiveMonthsRatio = float64(activeMonths) / float64(stats.TotalMonths)
		stats.SparseMonthsCount = sparseMonths
		stats.SparseMonthsRatio = float64(sparseMonths) / float64(stats.TotalMonths)
	}

	// 5. Compute averages using robust methods (NaN-safe)
	stats.GeoAvgMonthly = ComputeGeoAvgMonthly(allReturns)
	stats.ActiveGeoAvgMonthly = ComputeGeoAvgMonthly(activeReturns)
	stats.MedianMonthly = median(activeReturns)
	stats.MedianAllMonths = median(allReturns)
	stats.MedianActiveMonths = median(activeReturns)
	stats.StdMonth = std(allReturns)

	// 6. Compute OOS trade metrics
	var oosTrades []Trade
	for _, fr := range foldResults {
		oosTrades = append(oosTrades, fr.Trades...)
	}

	stats.OOSProfitFactor = computeProfitFactor(oosTrades)
	stats.OOSExpectancy = computeExpectancy(oosTrades)
	stats.OOSWinRate = computeWinRate(oosTrades)

	// 7. Validate constraints
	ValidateOOSConstraints(&stats, config)

	return stats
}

func computeProfitFactor(trades []Trade) float64 {
	grossWin := 0.0
	grossLoss := 0.0
	for _, t := range trades {
		if t.PnL > 0 {
			grossWin += float64(t.PnL)
		} else {
			grossLoss -= float64(t.PnL)
		}
	}
	if grossLoss == 0 {
		if grossWin > 0 {
			return 999.0
		}
		return 0.0
	}
	return grossWin / grossLoss
}

func computeExpectancy(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}
	totalPnL := 0.0
	for _, t := range trades {
		totalPnL += float64(t.PnL)
	}
	return totalPnL / float64(len(trades))
}

func computeWinRate(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}
	wins := 0
	for _, t := range trades {
		if t.PnL > 0 {
			wins++
		}
	}
	return float64(wins) / float64(len(trades))
}
