package main

import (
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"
)

// Rate-limit fold debug logging across strategies
var wfFoldLogCounter atomic.Int64

// WFConfig holds walk-forward validation configuration
type WFConfig struct {
	// Fold generation
	Enable           bool
	TrainDays        int
	TestDays         int
	StepDays         int
	MinFolds         int
	MinTradesPerFold int

	// OOS constraints (hard rejections)
	MinMonths                int
	MinTotalTradesOOS        int
	MinTradesPerMonth        int
	MaxDrawdown              float64
	MinMonthReturn           float64
	EnableMinMonthConstraint bool
	MinGeoMonthlyReturn      float64 // Minimum geometric average monthly return
	MinActiveMonthsRatio     float64 // FIX #1: Minimum ratio of active months (trades >= 5) to total months
	MaxSparseMonthsRatio     float64 // FIX: Maximum ratio of sparse months (0 trades) to total months (reject if > 0.35)
	MinMedianMonthly         float64 // NEW: Minimum median monthly return
	MaxStdMonth              float64 // Maximum monthly volatility (std dev) - rejects unstable strategies

	// Penalty weights
	MonthlyVolLambda float64
	DDPenaltyLambda  float64

	// Simplicity pressure
	LambdaNodes float64
	LambdaFeats float64
	LambdaDepth float64
	MaxNodes    int
	MaxFeatures int

	// Entry rate gate
	MinEdgesPerYear     float64 // Minimum edges per year threshold (-1=auto, 0=disabled, >0=explicit)
	FoldMinEdgesPerYear float64 // Minimum edges per year threshold per-fold (-1=auto, 0=disabled, >0=explicit)
}

// Fold represents a single walk-forward validation fold
type Fold struct {
	TrainStart int // Training window start index (inclusive)
	TrainEnd   int // Training window end index (exclusive)
	TestStart  int // Test window start index (inclusive)
	TestEnd    int // Test window end index (exclusive)
	Warmup     int // Warmup bars before train start (computed dynamically)
}

// FoldResult holds results for a single fold
type FoldResult struct {
	TestScore       float64
	TestReturn      float64
	TestDD          float64
	TestTrades      int
	TestWinRate     float64
	FoldNumber      int
	TrainDate       string
	TestDate        string
	TestStart       int       // Required for stitching
	TestEnd         int       // Required for stitching
	EquityCurve     []float64 // Test-period equity (normalized to start=1.0)
	TradeEntryIdxs  []int     // Entry bar indices for each trade (for monthly assignment)
	ClosedPositions int       // Trades fully closed in test
	MTMPositions    int       // Positions marked-to-market at TestEnd
	Trades          []Trade   // OOS trades for this fold (for PF/Expectancy calculation)
}

// lowerBoundClose returns the first index i where CloseTimeMs[i] >= target
// This is the "lower_bound" in C++ std terminology.
// Used for inclusive start boundaries and exclusive end boundaries in half-open intervals [start, end).
func lowerBoundClose(series Series, target int64) int {
	low, high := 0, len(series.CloseTimeMs)
	for low < high {
		mid := (low + high) / 2
		if series.CloseTimeMs[mid] < target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// upperBoundClose returns the first index i where CloseTimeMs[i] > target
// This is the "upper_bound" in C++ std terminology.
// Used to find the first bar strictly after a timestamp.
// Ensures no overlap between training and test windows.
func upperBoundClose(series Series, target int64) int {
	low, high := 0, len(series.CloseTimeMs)
	for low < high {
		mid := (low + high) / 2
		if series.CloseTimeMs[mid] <= target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// validateTimestampOrdering checks that timestamps are non-decreasing
// Allows equal timestamps (for multi-instrument data or same-tick bars)
func validateTimestampOrdering(series Series) bool {
	for i := 1; i < len(series.CloseTimeMs); i++ {
		if series.CloseTimeMs[i] < series.CloseTimeMs[i-1] {
			return false
		}
	}
	return true
}

// validateTimestampOrderingDetailed returns detailed diagnostics (used only for error messages)
// Returns (isValid, badIndex, prevTs, curTs)
func validateTimestampOrderingDetailed(series Series) (bool, int, int64, int64) {
	for i := 1; i < len(series.CloseTimeMs); i++ {
		if series.CloseTimeMs[i] < series.CloseTimeMs[i-1] {
			return false, i, series.CloseTimeMs[i-1], series.CloseTimeMs[i]
		}
	}
	return true, -1, 0, 0
}

// computeBarsPerDay robustly computes bars per day from actual data
func computeBarsPerDay(series Series) float64 {
	if len(series.CloseTimeMs) < 2 {
		return 0
	}

	// Sample median delta (avoid gaps)
	deltas := make([]int64, 0, 100)
	sample := len(series.CloseTimeMs) / 100 // Sample every 100th bar
	if sample < 1 {
		sample = 1
	}

	for i := sample; i < len(series.CloseTimeMs); i += sample {
		delta := series.CloseTimeMs[i] - series.CloseTimeMs[i-sample]
		deltas = append(deltas, delta)
	}

	// Find median
	sort.Slice(deltas, func(i, j int) bool { return deltas[i] < deltas[j] })
	medianSampledDelta := deltas[len(deltas)/2]

	// CRITICAL FIX: Divide by sample to get per-bar delta
	// The medianDelta is the total time across 'sample' bars, so we divide to get per-bar time
	medianPerBarDelta := medianSampledDelta / int64(sample)

	// Convert to bars per day
	msPerDay := 24 * 60 * 60 * 1000
	return float64(msPerDay) / float64(medianPerBarDelta)
}

// BuildWalkForwardFolds generates walk-forward folds using timestamp-based boundaries
func BuildWalkForwardFolds(series Series, trainDays, testDays, stepDays int, minFolds int) ([]Fold, error) {
	// VALIDATION 1: Check timestamp ordering
	if !validateTimestampOrdering(series) {
		// Get diagnostics for better error message
		_, badIdx, prevTs, curTs := validateTimestampOrderingDetailed(series)
		csvRow := "unknown"
		if badIdx >= 0 && badIdx < len(series.CSVRowIndex) {
			csvRow = fmt.Sprintf("%d", series.CSVRowIndex[badIdx])
		}
		return nil, fmt.Errorf("timestamp ordering validation failed at index %d (CSV row %s): %d < %d",
			badIdx, csvRow, curTs, prevTs)
	}

	// Compute bars per day robustly from actual data
	barsPerDay := computeBarsPerDay(series)
	if barsPerDay <= 0 {
		return nil, fmt.Errorf("invalid bars per day: %.2f", barsPerDay)
	}

	// Use timestamp-based boundaries (convert ms to time.Time first)
	firstMs := series.CloseTimeMs[0]
	lastMs := series.CloseTimeMs[len(series.CloseTimeMs)-1]
	firstTime := time.UnixMilli(firstMs).UTC()
	lastTime := time.UnixMilli(lastMs).UTC()

	var folds []Fold
	trainStartTime := firstTime

	foldNum := 0
	for {
		trainEndTime := trainStartTime.AddDate(0, 0, trainDays)
		testEndTime := trainEndTime.AddDate(0, 0, testDays)

		// Convert time.Time to milliseconds for binary search
		trainStartMs := trainStartTime.UnixMilli()
		trainEndMs := trainEndTime.UnixMilli()
		testEndMs := testEndTime.UnixMilli()

		// VALIDATION 2: Check we haven't exceeded data (use time.Time comparison)
		if testEndTime.After(lastTime) {
			break
		}

		// Convert to indices using half-open interval semantics
		// trainStartIdx: first bar with CloseTimeMs >= trainStartMs (inclusive start)
		// trainEndIdx: first bar with CloseTimeMs >= trainEndMs (exclusive end)
		// testStartIdx: equal to trainEndIdx - no gap, test starts where train ends
		// testEndIdx: first bar with CloseTimeMs >= testEndMs (exclusive end)
		trainStartIdx := lowerBoundClose(series, trainStartMs)
		trainEndIdx := lowerBoundClose(series, trainEndMs)
		testStartIdx := trainEndIdx // No gap: test starts exactly where train ends
		testEndIdx := lowerBoundClose(series, testEndMs)

		// VALIDATION 3: Ensure non-empty windows
		if trainEndIdx <= trainStartIdx {
			return nil, fmt.Errorf("fold %d: empty train window (trainEndIdx=%d <= trainStartIdx=%d)", foldNum, trainEndIdx, trainStartIdx)
		}
		if testEndIdx <= testStartIdx {
			return nil, fmt.Errorf("fold %d: empty test window (testEndIdx=%d <= testStartIdx=%d)", foldNum, testEndIdx, testStartIdx)
		}

		fold := Fold{
			TrainStart: trainStartIdx,
			TrainEnd:   trainEndIdx,
			TestStart:  testStartIdx, // No gap: adjacent half-open intervals [start,end) [end,nextEnd)
			TestEnd:    testEndIdx,
		}

		folds = append(folds, fold)

		// Roll forward
		trainStartTime = trainStartTime.AddDate(0, 0, stepDays)
		foldNum++
	}

	// VALIDATION 4: Check minimum fold count
	if len(folds) < minFolds {
		return nil, fmt.Errorf("insufficient folds: generated %d, need at least %d",
			len(folds), minFolds)
	}

	return folds, nil
}

// extractLookbackFromFeatureName parses the feature name to extract period
// Pattern: extract trailing number from feature name
// EMA200 -> 200, BB_Lower50 -> 50, RSI7 -> 7
// ATR14 -> 14, MACD (no period) -> 0
func extractLookbackFromFeatureName(name string) int {
	// Find all digits at end of string
	var digits []rune
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] >= '0' && name[i] <= '9' {
			digits = append([]rune{rune(name[i])}, digits...)
		} else if len(digits) > 0 {
			// Stop at first non-digit from right
			break
		}
	}

	if len(digits) == 0 {
		// No period found (e.g., MACD, ADX)
		return 0
	}

	// Parse digits to int
	period := 0
	for _, d := range digits {
		period = period*10 + int(d-'0')
	}

	return period
}

// getUsedFeatureIndices walks the AST and collects unique feature indices used
func getUsedFeatureIndices(entryRoot, exitRoot *RuleNode) []int {
	// Walk AST recursively and collect unique feature indices
	used := make(map[int]bool)
	collectFeatureIndices(entryRoot, used)
	collectFeatureIndices(exitRoot, used)

	// Convert to slice
	indices := make([]int, 0, len(used))
	for idx := range used {
		indices = append(indices, idx)
	}
	return indices
}

// collectFeatureIndices recursively walks the AST and collects feature indices
func collectFeatureIndices(node *RuleNode, used map[int]bool) {
	if node == nil {
		return
	}
	if node.Op == OpLeaf {
		used[node.Leaf.A] = true
		if node.Leaf.Kind == LeafCrossUp || node.Leaf.Kind == LeafCrossDown ||
			node.Leaf.Kind == LeafBetween || node.Leaf.Kind == LeafSlopeGT ||
			node.Leaf.Kind == LeafSlopeLT {
			used[node.Leaf.B] = true
		}
	}
	collectFeatureIndices(node.L, used)
	collectFeatureIndices(node.R, used)
}

// ComputeWarmupForStrategy dynamically computes max indicator lookback from features used in strategy
func ComputeWarmupForStrategy(strategy Strategy, features Features) int {
	// Method 1: Traverse AST and collect feature indices used
	usedFeatures := getUsedFeatureIndices(strategy.EntryRule.Root, strategy.ExitRule.Root)

	// Method 2: Find max lookback among used features
	maxLookback := 0
	for _, featIdx := range usedFeatures {
		if featIdx >= 0 && featIdx < len(features.Names) {
			lookback := extractLookbackFromFeatureName(features.Names[featIdx])
			if lookback > maxLookback {
				maxLookback = lookback
			}
		}
	}

	// Method 3: Add safety margin (minimum 200 bars for smooth indicators)
	safetyMargin := 200
	warmup := maxLookback + safetyMargin

	// Also check regime filter
	if strategy.RegimeFilter.Root != nil {
		regimeFeatures := getUsedFeatureIndices(strategy.RegimeFilter.Root, nil)
		for _, featIdx := range regimeFeatures {
			if featIdx >= 0 && featIdx < len(features.Names) {
				lookback := extractLookbackFromFeatureName(features.Names[featIdx])
				if lookback > maxLookback {
					maxLookback = lookback
				}
			}
		}
		warmup = max(warmup, maxLookback+safetyMargin)
	}

	return warmup
}

// GetGlobalMaxLookback scans all feature names and returns max period (fallback)
func GetGlobalMaxLookback(features Features) int {
	// Scan all feature names and return max period
	maxLookback := 0
	for _, name := range features.Names {
		lookback := extractLookbackFromFeatureName(name)
		if lookback > maxLookback {
			maxLookback = lookback
		}
	}
	// Add safety margin (not hardcoded 200 - only if no periods found)
	safetyMargin := 200
	return max(maxLookback, safetyMargin)
}

// ExtractTestPeriodMetrics extracts test-period metrics from a full backtest result
// Handles mark-to-market for positions crossing TestEnd with exit costs applied
// Step 3B: globalTestStart/globalTestEnd are the GLOBAL indices (for stitching), not local
func ExtractTestPeriodMetrics(fullBacktest coreBacktestResult, testStart, testEnd, actualStart int, series Series,
	globalTestStart, globalTestEnd int, feeBps, slipBps float64) FoldResult {
	var testTrades []Trade
	var mtmTrades []Trade
	var tradeEntryIdxs []int

	for _, trade := range fullBacktest.trades {
		if trade.EntryIdx >= testStart && trade.EntryIdx < testEnd {
			// Convert local index to global index for proper month assignment
			globalEntryIdx := actualStart + trade.EntryIdx
			tradeEntryIdxs = append(tradeEntryIdxs, globalEntryIdx)

			if trade.ExitIdx < testEnd {
				// Trade fully closed in test window
				testTrades = append(testTrades, trade)
			} else {
				// Trade enters in test but exits after TestEnd
				// Mark-to-market with exit costs applied

				entryPrice := float64(trade.EntryPrice)
				mtmPrice := float64(series.Close[testEnd-1]) // Last bar of test window
				direction := float64(trade.Direction)

				// Compute raw P&L
				var rawPnL float64
				if direction > 0 { // Long
					rawPnL = (mtmPrice - entryPrice) / entryPrice
				} else { // Short
					rawPnL = (entryPrice - mtmPrice) / entryPrice
				}

				// Apply costs on MTM close (align with coreBacktest fee model)
				// coreBacktest applies entry+exit fees on close; entry slippage is already in EntryPrice
				// so we add 2*fee + exit slippage here for consistency.
				totalCost := (2.0*feeBps + slipBps) / 10000.0 // Convert bps to decimal
				rawPnL -= totalCost                           // Cost reduces profit for both long and short

				// FIX #C (Problem C): Removed dead code 'closedMTM' variable
				// MTM PnL is already captured in mtmTrades and used in allTrades for metrics

				// Create synthetic MTM trade record
				mtmTrade := trade
				mtmTrade.ExitIdx = testEnd - 1
				mtmTrade.ExitPrice = float32(mtmPrice)
				mtmTrade.PnL = float32(rawPnL)
				mtmTrades = append(mtmTrades, mtmTrade)
			}
		}
	}

	// Calculate metrics using testTrades + mtmTrades
	allTrades := append(testTrades, mtmTrades...)

	// Build equity curve for this fold using MTM equity (normalized to start=1.0)
	equityCurve := extractMTMEquityForTestPeriod(fullBacktest, testStart, testEnd)

	// Return FoldResult with all required fields
	// Step 3B: Use GLOBAL indices for TestStart/TestEnd (not local ones)
	return FoldResult{
		TestScore:       computeScoreFromTrades(allTrades),
		TestReturn:      computeReturnFromTrades(allTrades),
		TestDD:          computeDDFromTrades(allTrades),
		TestTrades:      len(allTrades),
		TestWinRate:     computeWinRate(allTrades),
		FoldNumber:      0,               // Set by caller
		TrainDate:       "",              // Set by caller
		TestDate:        "",              // Set by caller
		TestStart:       globalTestStart, // GLOBAL index for stitching
		TestEnd:         globalTestEnd,   // GLOBAL index for stitching
		EquityCurve:     equityCurve,
		TradeEntryIdxs:  tradeEntryIdxs,
		ClosedPositions: len(testTrades),
		MTMPositions:    len(mtmTrades),
		Trades:          allTrades, // Include trades for OOS PF/Expectancy calculation
	}
}

// extractMTMEquityForTestPeriod extracts test-period MTM equity with compatibility handling
// Uses RAW MTM equity (RiskPct=1.0) for consistency with scoring/returns.
// Handles both old (test-window-only) and new (full-range) recording formats.
func extractMTMEquityForTestPeriod(full coreBacktestResult, testStartLocal, testEndLocal int) []float64 {
	want := testEndLocal - testStartLocal
	if want <= 0 {
		return []float64{1.0}
	}

	// Prefer RAW MTM equity (RiskPct=1.0) so OOS stats align with trade PnL/score units.
	// Fall back to risk-adjusted MTM if raw is unavailable.
	src := full.RawMTMEquity
	if len(src) == 0 {
		src = full.MTMEquity
	}
	if len(src) == 0 {
		// No MTM data - return flat curve
		out := make([]float64, want)
		for i := range out {
			out[i] = 1.0
		}
		return out
	}

	var raw []float64

	// Handle different recording scenarios
	if len(src) == testEndLocal || len(src) == testEndLocal-1 {
		// Recorded for entire local range - slice test window
		// src[0] corresponds to bar 1 (because loop starts t=1)
		// Map barIdx -> srcIdx = barIdx - 1
		startIdx := testStartLocal - 1
		endIdx := testEndLocal - 1 // exclusive

		// Handle edge case: if test starts at bar 0, prepend initial equity
		prependBar0 := false
		if startIdx < 0 {
			startIdx = 0
			prependBar0 = true // testStartLocal == 0, missing bar0 in src
		}

		// Boundary checks
		if endIdx < 0 {
			endIdx = 0
		}
		if endIdx > len(src) {
			endIdx = len(src)
		}
		if startIdx > endIdx {
			startIdx = endIdx
		}

		// Extract slice
		raw = append([]float64{}, src[startIdx:endIdx]...)

		// If test started at bar 0, inject bar0 equity = 1.0 so normalization is sane
		if prependBar0 {
			raw = append([]float64{1.0}, raw...)
		}

		// Force exact length = want (testEndLocal - testStartLocal)
		if len(raw) > want {
			raw = raw[:want]
		} else if len(raw) < want {
			last := 1.0
			if len(raw) > 0 {
				last = raw[len(raw)-1]
			}
			for len(raw) < want {
				raw = append(raw, last)
			}
		}
	} else if len(src) >= want {
		// Recorded only for test window - use as-is
		raw = src
	} else {
		// Recorded but shorter - pad to expected length
		raw = make([]float64, want)
		copy(raw, src)
		if len(src) > 0 {
			last := src[len(src)-1]
			for i := len(src); i < want; i++ {
				raw[i] = last
			}
		}
	}

	if len(raw) == 0 {
		return []float64{1.0}
	}

	// Normalize to start=1.0
	start := raw[0]
	if start == 0 {
		start = 1.0
	}
	norm := make([]float64, len(raw))
	for i, v := range raw {
		norm[i] = v / start
	}
	return norm
}

// computeEquityCurveFromTrades builds equity curve from trades for a fold
func computeEquityCurveFromTrades(trades []Trade, testStart, testEnd int, series Series) []float64 {
	// Initialize equity curve starting at 1.0
	equity := make([]float64, testEnd-testStart)
	for i := range equity {
		equity[i] = 1.0
	}

	// Apply trade P&L to equity curve
	for _, trade := range trades {
		entryRel := trade.EntryIdx - testStart
		exitRel := trade.ExitIdx - testStart

		if exitRel >= len(equity) {
			exitRel = len(equity) - 1
		}

		for i := entryRel; i < len(equity); i++ {
			if i >= exitRel {
				equity[i] *= (1.0 + float64(trade.PnL))
			}
		}
	}

	return equity
}

// computeScoreFromTrades computes a score from a list of trades
func computeScoreFromTrades(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}

	totalReturn := 0.0
	wins := 0
	losses := 0
	totalWinPnL := 0.0
	totalLossPnL := 0.0

	for _, t := range trades {
		totalReturn += float64(t.PnL)
		if t.PnL > 0 {
			wins++
			totalWinPnL += float64(t.PnL)
		} else {
			losses++
			totalLossPnL += -float64(t.PnL)
		}
	}

	winRate := float64(wins) / float64(len(trades))

	var expectancy float64
	if wins > 0 && losses > 0 {
		expectancy = (totalWinPnL/float64(wins))*winRate - (totalLossPnL/float64(losses))*(1-winRate)
	} else if wins > 0 {
		expectancy = totalWinPnL / float64(wins)
	} else if losses > 0 {
		expectancy = -totalLossPnL / float64(losses)
	}

	// Score formula: balances return, drawdown penalty, and expectancy
	// Components are normalized to similar scales:
	// - totalReturn: raw PnL sum
	// - dd: drawdown as ratio (0.0-1.0)
	// - expectancy: avg PnL per trade
	dd := computeDDFromTrades(trades)

	// Rebalanced weights: expectancy weight reduced from 10 to 1 for better balance
	return totalReturn - 0.5*dd + expectancy
}

// computeReturnFromTrades computes total return from trades
// FIX #B (Problem B): Compounded return instead of sum for consistency with equity/DD
func computeReturnFromTrades(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}
	// FIX #B: Use compounding to match equity curve behavior (same as computeDDFromTrades)
	equity := 1.0
	for _, t := range trades {
		equity *= (1.0 + float64(t.PnL))
	}
	return equity - 1.0 // Return as percentage change from 1.0
}

// computeDDFromTrades computes max drawdown from trades
func computeDDFromTrades(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}

	// Build equity curve and find DD
	equity := 1.0
	peak := 1.0
	maxDD := 0.0

	for _, t := range trades {
		equity *= (1.0 + float64(t.PnL))
		if equity > peak {
			peak = equity
		}
		dd := (peak - equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}

	return maxDD
}

// EvaluateWalkForward evaluates a strategy across all walk-forward folds
func EvaluateWalkForward(series Series, features Features, strategy Strategy, folds []Fold, config WFConfig) (OOSStats, StitchedOOSData, []FoldResult) {
	var foldResults []FoldResult

	// Debug fold logging is extremely noisy; rate-limit across strategies.
	logFolds := false
	if DebugWalkForward {
		n := wfFoldLogCounter.Add(1)
		if n == 1 || n%500 == 0 {
			logFolds = true
		}
	}

	for i, fold := range folds {
		// Compute warmup for this strategy
		warmup := ComputeWarmupForStrategy(strategy, features)

		// Set actual start with warmup
		actualStart := fold.TrainStart - warmup
		if actualStart < 0 {
			actualStart = 0
		}

		// Slice series and features for this fold
		s := SliceSeries(series, actualStart, fold.TestEnd)
		f := SliceFeatures(features, actualStart, fold.TestEnd)

		// Compute local indices
		testStartLocal := fold.TestStart - actualStart
		testEndLocal := fold.TestEnd - actualStart

		// Run backtest starting at test window (warmup history included in slice, but trading starts at test)
		// This prevents trades from opening during training and carrying into test
		core := coreBacktest(s, f, strategy, testStartLocal, testEndLocal, false, false)

		// Extract test period metrics (convert float32 to float64)
		// Step 3B: Pass GLOBAL indices (fold.TestStart/TestEnd) not local ones
		fr := ExtractTestPeriodMetrics(core, testStartLocal, testEndLocal, actualStart, s,
			fold.TestStart, fold.TestEnd, // GLOBAL indices for stitching
			float64(strategy.FeeBps), float64(strategy.SlippageBps))

		// Set fold metadata
		fr.FoldNumber = i

		// Format dates
		if fold.TrainStart > 0 && fold.TrainStart < len(series.CloseTimeMs) {
			fr.TrainDate = time.UnixMilli(series.CloseTimeMs[fold.TrainStart]).UTC().Format("2006-01")
		}
		if fold.TestStart > 0 && fold.TestStart < len(series.CloseTimeMs) {
			fr.TestDate = time.UnixMilli(series.CloseTimeMs[fold.TestStart]).UTC().Format("2006-01")
		}

		// DEBUG: Print fold execution path diagnostics (rate-limited to avoid spam)
		// If signals>0 but trades=0, your execution path is broken
		if logFolds {
			// Log only first 3 folds and the last fold to keep output readable
			if i < 3 || i == len(folds)-1 {
				fmt.Printf("[FOLD %d] bars=%d signals=%d entries=%d exits=%d test_trades=%d %s\n",
					i, testEndLocal-testStartLocal, core.signalCount, core.entryCount, core.exitCount, fr.TestTrades,
					fr.TrainDate+" -> "+fr.TestDate)
			}
		}

		foldResults = append(foldResults, fr)
	}

	// Compute OOS stats (includes stitching and monthly returns internally)
	oosStats := CalculateOOSStats(foldResults, series, config)

	// Stitch OOS equity curve for return value
	stitched := StitchOOSEquityCurve(foldResults, series)

	return oosStats, stitched, foldResults
}

// ComputeFinalFitness computes the final fitness score from OOS stats and complexity.
// fitness = geoAvg - lambda*stdMonth - mu*maxDD - simplicityPenalty - overfitPenalty
func ComputeFinalFitness(stats OOSStats, trainGeoMonthly float64, complexity Complexity, config WFConfig) float64 {
	if stats.Rejected {
		return math.Inf(-1)
	}

	// Base: geometric average monthly return
	fitness := stats.GeoAvgMonthly

	// ANTI-OVERFITTING: Penalize ANY large train/OOS divergence (both directions)
	// This detects overfitting AND score inversion (train negative, val positive)
	trainOOSGap := trainGeoMonthly - stats.GeoAvgMonthly
	absGap := math.Abs(trainOOSGap)

	// CRITICAL FIX: Penalize score inversion (train negative, val positive)
	// This is a major red flag - strategy doesn't work in training data
	if trainGeoMonthly < 0 && stats.GeoAvgMonthly > 0 {
		// Apply massive penalty for train/OOS score inversion
		fitness -= 5.0
	} else if absGap > 0.05 { // 5% gap in monthly returns
		// Penalize any significant gap (train > OOS OR OOS > train)
		overfitPenalty := 0.2 * absGap // Increased from 0.1 for stronger penalty
		fitness -= overfitPenalty
	}

	// Stability penalty: tiered monthly volatility penalty
	// Heavy penalty for extreme volatility, moderate for unstable
	volatility := stats.StdMonth
	if volatility > 0.15 { // >15% monthly std dev is VERY unstable
		// Heavy penalty for extreme volatility
		fitness -= 5.0 * (volatility - 0.15)
	} else if volatility > 0.10 { // >10% is unstable
		// Moderate penalty
		fitness -= 2.0 * (volatility - 0.10)
	}
	// Apply base lambda penalty on top of tiered penalty
	fitness -= config.MonthlyVolLambda * volatility

	// Drawdown penalty
	fitness -= config.DDPenaltyLambda * stats.MaxDD

	// Simplicity penalty (all float64 for consistency)
	complexityPenalty :=
		config.LambdaNodes*float64(complexity.NodeCount) +
			config.LambdaFeats*float64(complexity.UniqueFeatureCount()) +
			config.LambdaDepth*float64(complexity.MaxDepth)
	fitness -= complexityPenalty

	return fitness
}
