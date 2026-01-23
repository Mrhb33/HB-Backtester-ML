package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"hb_bactest_checker/logx"
)

var pfDebugCount uint64 // package-level counter for throttled PF debug logging

// slippageRef returns the last fully-known bar index at the time of a fill.
// For fills at bar t open or intrabar during bar t, use t-1.
// For force-close at final close, use finalIdx-1 (if possible).
func slippageRef(t int) int {
	if t <= 0 {
		return 0
	}
	return t - 1
}

// LeafKey is a composite key for uniquely identifying leaf nodes in OR-branch tracing
type LeafKey struct {
	Kind     LeafKind
	A        int
	B        int
	Lookback int
}

// shouldLogPFDebug implements throttling for PF debug output
// Returns true for first 20 calls, then every 5000th call thereafter
func shouldLogPFDebug() bool {
	n := atomic.AddUint64(&pfDebugCount, 1)
	return n <= 20 || (n%5000 == 0)
}

// Trade represents a single completed trade with detailed information
type Trade struct {
	Direction   int       // 1 for Long, -1 for Short
	EntryIdx    int       // Candle index when entry was executed
	EntryTime   time.Time // Timestamp when entry was executed
	EntryPrice  float32   // Price at entry
	ExitIdx     int       // Candle index when exit was executed
	ExitTime    time.Time // Timestamp when exit was executed
	ExitPrice   float32   // Price at exit
	Reason      string    // Exit reason: "TP", "SL", "TRAIL", "EXIT_RULE", "MAX_HOLD", "tp_gap_open", "sl_gap_open", "tp_hit", "sl_hit", "trail_hit"
	PnL         float32   // Profit/Loss in percent (after fees)
	HoldBars    int       // Number of bars held
	StopPrice   float32   // Stop loss price at entry
	TPPrice     float32   // Take profit price at entry
	TPPct       float32   // Take profit percentage (absolute, e.g., 0.02 = 2%)
	TrailActive bool      // Whether trailing stop was active
	ExitOpen    float32   // Open price of exit candle (for reference)
	ExitHigh    float32   // High price of exit candle
	ExitLow     float32   // Low price of exit candle
	ExitClose   float32   // Close price of exit candle
	SignalIndex int       // Index of the bar that generated the entry signal
	SignalTime  time.Time // Time when the entry signal was generated
	StopBefore  float32   // Stop loss price at start of exit bar (before any trailing update)
	TPBefore    float32   // Take profit price at start of exit bar (before any update)
	Proofs      []LeafProof // Mathematical proofs of no lookahead bias for signal detection
}

// PendingEntry represents a scheduled entry for the next bar
type PendingEntry struct {
	signalIdx     int       // Bar index where signal was generated
	entryIdx      int       // Bar index where entry will execute
	signalTime    time.Time // Time when signal was generated
	dir           int       // Direction: +1 long, -1 short
	closePrice    float64   // Close price at signal detection
	regimeResult  bool      // Regime filter result at signal time
	entryResult   bool      // Entry rule result at signal time
	subConditions string    // Sub-condition T/F results (for OR/AND trees)
	proofs        []LeafProof // Mathematical proofs for first 3 executed trades
}

// ActiveTrade represents a currently open position
type ActiveTrade struct {
	tradeIdx    int       // Index in trades array (will be filled on exit)
	entryIdx    int       // Entry bar index
	entryTime   time.Time // Entry time
	dir         int       // Direction: +1 long, -1 short
	entryPrice  float32   // Entry price
	tp          float32   // Take profit price
	sl          float32   // Stop loss price (current, includes trailing if active)
	trailActive bool      // Whether trailing stop is active
}

// StateRecord tracks the state per bar for debugging
type StateRecord struct {
	BarIndex int
	Time     time.Time
	Statuses string // Comma-separated state flags
}

// TradeRecord is the unified export format (matches reference implementation)
type TradeRecord struct {
	SignalIndex int
	EntryIndex  int
	ExitIndex   int
	EntryPrice  float32
	ExitPrice   float32
	ExitReason  string // "stop_hit", "stop_gap_open", "tp_hit", "tp_gap_open", "exit_rule", "max_hold", "trail_hit"
	PnL         float32
	DurationBars int
}

// GoldenResult is the result of a backtest with full trade logging
type GoldenResult struct {
	Trades        []Trade                 // List of all trades
	TotalTrades   int                     // Total number of trades
	TotalHoldBars int                     // Sum of all hold bars
	ReturnPct     float32                 // Total return in percent (risk-adjusted, uses RiskPct)
	RawReturnPct  float32                 // Total return in percent (raw, RiskPct=1.0)
	MaxDDPct      float32                 // Maximum drawdown in percent (mark-to-market)
	WinRate       float32                 // Win rate (0-1)
	Expectancy    float32                 // Average expectancy per trade
	ProfitFactor  float32                 // Profit factor (gross wins/gross losses)
	ExitReasons   map[string]int          // Count of trades by exit reason
	States       []StateRecord           // Per-bar state tracking
	WindowOffset  int                     // Offset of window start from global index (i0)
	SignalProofs map[int][]LeafProof      // Proofs for each signal bar (local index)
}

type Result struct {
	Strategy     Strategy
	Score        float32
	Return       float32
	MaxDD        float32
	MaxDDRaw     float32 // Raw max drawdown (not normalized)
	WinRate      float32
	Expectancy   float32
	ProfitFactor float32
	Trades       int
	SmoothVol    float32 // EMA volatility of equity changes (lower = smoother)
	DownsideVol  float32 // Downside volatility for Sortino ratio (negative returns only)
	ValResult    *Result // Optional validation result from multi-fidelity
	ExitReasons  map[string]int // Count of trades by exit reason
}

type PositionState int

const (
	Flat PositionState = iota
	Long
	Short
)

type Position struct {
	State      PositionState
	Direction  int
	EntryPrice float32
	Size       float32
	StopPrice  float32
	TPPrice    float32
	EntryTime  int
	TrailHigh  float32
	TrailLow   float32
}

func evaluateStrategy(s Series, f Features, st Strategy) Result {
	const warmup = 200
	minData := warmup + 50 // Reduced from 100 to 50 for flexibility

	if len(s.Close) < minData {
		return Result{Strategy: st, Score: -1e30, Return: 0, MaxDD: 1, MaxDDRaw: 1, WinRate: 0, Trades: 0, ExitReasons: make(map[string]int)}
	}

	return evaluateStrategyRange(s, f, st, warmup, len(s.Close))
}

func evaluateStrategyWindow(full Series, fullF Features, st Strategy, w Window) Result {
	// Include warmup history so indicators are "real"
	i0 := w.Start - w.Warmup
	if i0 < 0 {
		i0 = 0
	}
	i1 := w.End
	if i1 > full.T {
		i1 = full.T
	}

	s := SliceSeries(full, i0, i1)
	f := SliceFeatures(fullF, i0, i1)

	tradeStartLocal := w.Start - i0
	tradeEndLocal := w.End - i0

	return evaluateStrategyRange(s, f, st, tradeStartLocal, tradeEndLocal)
}

func evaluateStrategyRange(s Series, f Features, st Strategy, tradeStartLocal, endLocal int) Result {
	// Use unified core backtest engine to ensure consistency
	// computeSmoothness=true enables EMA volatility and Sortino metrics for scoring
	core := coreBacktest(s, f, st, tradeStartLocal, endLocal, true, false)

	rawReturn := core.rawReturnPct

	// Validate results - but still return actual trade count (not 0)
	// This fixes "Trades=0 but Ret=-90%" bug
	if rawReturn > 10 || rawReturn < -0.9 || math.IsNaN(float64(rawReturn)) || math.IsInf(float64(rawReturn), 0) {
		return Result{
			Strategy:     st,
			Score:        -1e30,
			Return:       rawReturn,
			MaxDD:        core.rawMaxDD,
			MaxDDRaw:     core.rawMaxDD,
			WinRate:      core.winRate,
			Trades:       core.totalTrades, // FIXED: Return actual trade count, not 0
			Expectancy:   core.expectancy,
			ProfitFactor: core.profitFactor,
			SmoothVol:    0,
			DownsideVol:  0,
			ExitReasons:  core.exitReasons,
		}
	}

	// Score using raw metrics (RiskPct=1.0) to prevent score inflation
	score := computeScoreWithSmoothness(rawReturn, core.rawMaxDD, core.expectancy, core.smoothVol, core.downsideVol, core.totalTrades, 0)

	// If no trades, set score to very low number
	if core.totalTrades == 0 {
		return Result{
			Strategy:     st,
			Score:        -1e30,
			Return:       0,
			MaxDD:        core.maxDD,
			MaxDDRaw:     core.rawMaxDD,
			WinRate:      0,
			Expectancy:   0,
			ProfitFactor: 0,
			Trades:       0,
			SmoothVol:    0,
			DownsideVol:  0,
			ExitReasons:  make(map[string]int),
		}
	}

	// Return raw metrics (RiskPct=1.0) for scoring to prevent score inflation
	return Result{
		Strategy:     st,
		Score:        score,
		Return:       rawReturn,       // Raw return (RiskPct=1.0)
		MaxDD:        core.rawMaxDD,   // Raw max DD (RiskPct=1.0)
		MaxDDRaw:     core.rawMaxDD,   // Keep raw maxDDRaw as is
		WinRate:      core.winRate,
		Expectancy:   core.expectancy,
		ProfitFactor: core.profitFactor,
		Trades:       core.totalTrades,
		SmoothVol:    core.smoothVol,
		DownsideVol:  core.downsideVol,
		ExitReasons:  core.exitReasons,
	}
}

func computeScore(ret, dd, expectancy float32, trades int, testedCount int64) float32 {
	// Hard-reject broken strategies
	// Tightened DD threshold from 0.80 to 0.45 to prevent near-bankruptcy strategies from ever being "good"
	if trades == 0 || dd >= 0.45 {
		return -1e30
	}

	// CRITICAL FIX: Log if inputs are already broken before scoring
	if math.IsInf(float64(ret), 0) || math.IsNaN(float64(ret)) {
		fmt.Printf("[SCORE-BUG] ret=Inf/NaN trades=%d dd=%.4f exp=%.4f\n", trades, dd, expectancy)
		return -1e30
	}
	if math.IsInf(float64(dd), 0) || math.IsNaN(float64(dd)) {
		fmt.Printf("[SCORE-BUG] dd=Inf/NaN trades=%d ret=%.4f exp=%.4f\n", trades, ret, expectancy)
		return -1e30
	}
	if math.IsInf(float64(expectancy), 0) || math.IsNaN(float64(expectancy)) {
		fmt.Printf("[SCORE-BUG] exp=Inf/NaN trades=%d ret=%.4f dd=%.4f\n", trades, ret, dd)
		return -1e30
	}

	// Penalty system instead of hard rejection
	tradePenalty := float32(0)
	if trades < 30 {
		tradePenalty = -2.0 // align with validation crit trds>=30
	}

	retPenalty := float32(0)
	if ret < 0.0 {
		retPenalty = -1.0 // losing strategies score lower but still exist
	}

	// Hold duration constraint: reject strategies with tiny hold times
	holdPenalty := float32(0)
	if trades > 0 {
		// Only applies if we have avg hold data (from golden mode, otherwise 0)
		// Main evaluator doesn't track individual trades, so we skip this constraint there
	}

	// Calmar-ish: return per drawdown
	calmar := ret / (dd + 1e-4)
	if calmar > 10 {
		calmar = 10
	}

	// Expectancy should matter more than trade count
	expReward := expectancy * 200.0
	if expReward > 5 {
		expReward = 5
	}
	if expReward < -5 {
		expReward = -5
	}

	// Very small trades reward (reliability only)
	tradesReward := float32(math.Log(float64(trades))) * 0.01

	// Strong DD penalty
	ddPenalty := 6.0 * dd

	// Return is important (log to reduce outliers)
	logReturn := float32(math.Log(float64(1+ret))) * 6.0

	baseScore := logReturn + 2.0*calmar + expReward + tradesReward - ddPenalty + tradePenalty + retPenalty + holdPenalty

	// DSR-lite: apply deflation penalty as tested count grows
	// This forces strategies to have higher significance as we test more
	// Formula: deflated = baseScore - k * log(1 + testedCount / 10000)
	// k = 0.5 means after 10000 strategies, bar increases by ~0.5 points
	// CRITICAL FIX: Don't modify sentinel values
	if testedCount > 0 && baseScore > -1e29 {
		deflationPenalty := float32(0.5 * math.Log(1.0 + float64(testedCount)/10000.0))
		baseScore -= deflationPenalty
	}

	return baseScore
}

// IsValidScore checks if a score value is numerically valid
// Exported (capital I) so it can be called from other packages
// Returns false for NaN, Inf, or absurdly large magnitudes
func IsValidScore(score float32) bool {
	if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
		return false
	}
	// Check for absurd magnitudes (e.g., sentinel arithmetic leak)
	if math.Abs(float64(score)) > 1e9 {
		return false
	}
	return true
}

// computeScoreWithSmoothness applies a smoothness penalty to strategies with volatile equity curves
// and includes Sortino ratio for downside risk focus
func computeScoreWithSmoothness(ret, dd, expectancy, smoothVol, downsideVol float32, trades int, testedCount int64) float32 {
	baseScore := computeScore(ret, dd, expectancy, trades, testedCount)

	// CRITICAL FIX: Don't operate on sentinel values
	// If computeScore returned -1e30 (hard rejection), return it directly
	if baseScore <= -1e29 {
		return baseScore
	}

	// ---- Guardrails to prevent score explosion ----
	// Floor downsideVol so Sortino can't go infinite / absurd.
	// (Tune 0.002-0.01 depending on your bar timeframe.)
	const downsideFloor = float32(0.005)
	if downsideVol < downsideFloor {
		downsideVol = downsideFloor
	}

	sortino := ret / (downsideVol + 1e-4)

	// Cap Sortino contribution so it can't dominate everything
	if sortino > 20 {
		sortino = 20
	}
	if sortino < -10 {
		sortino = -10
	}

	// Smoothness penalty (keep as-is)
	smoothPenalty := float32(0)
	if trades >= 50 {
		denom := float32(math.Abs(float64(ret))) + 1e-6
		normalized := smoothVol / denom
		smoothPenalty = normalized * 0.5
	}

	return baseScore + 0.5*sortino - smoothPenalty
}

// evaluateStrategyWithTrades runs a backtest and returns detailed trade information
func evaluateStrategyWithTrades(full Series, fullF Features, st Strategy, w Window, debugSignals bool) GoldenResult {
	// Include warmup history so indicators are "real"
	i0 := w.Start - w.Warmup
	if i0 < 0 {
		i0 = 0
	}
	i1 := w.End
	if i1 > full.T {
		i1 = full.T
	}

	s := SliceSeries(full, i0, i1)
	f := SliceFeatures(fullF, i0, i1)

	tradeStartLocal := w.Start - i0
	tradeEndLocal := w.End - i0

	result := evaluateStrategyRangeWithTrades(s, f, st, tradeStartLocal, tradeEndLocal, debugSignals)
	result.WindowOffset = i0 // Store offset for CSV export
	return result
}

// evaluateStrategyRangeWithTrades runs a backtest with full trade logging
func evaluateStrategyRangeWithTrades(s Series, f Features, st Strategy, tradeStartLocal, endLocal int, debugSignals bool) GoldenResult {
	// Use unified core backtest engine to ensure consistency
	// computeSmoothness=false since we don't need EMA metrics for golden results
	core := coreBacktest(s, f, st, tradeStartLocal, endLocal, false, debugSignals)

	return GoldenResult{
		Trades:        core.trades,
		TotalTrades:   core.totalTrades,
		TotalHoldBars: core.totalHoldBars,
		ReturnPct:     core.returnPct,
		RawReturnPct:  core.rawReturnPct,
		MaxDDPct:      core.rawMaxDD, // Use raw DD for consistency with reports
		WinRate:       core.winRate,
		Expectancy:    core.expectancy,
		ProfitFactor:  core.profitFactor,
		ExitReasons:   core.exitReasons,
		States:        core.states,
		SignalProofs:  core.signalProofs,
	}
}

// coreBacktestResult is the unified return type for the core backtest engine
// Contains all data needed for both Result and GoldenResult
type coreBacktestResult struct {
	trades        []Trade
	totalTrades   int
	totalHoldBars int
	returnPct     float32 // Risk-adjusted return (uses st.RiskPct)
	rawReturnPct  float32 // Raw return (RiskPct=1.0)
	maxDD         float32 // Risk-adjusted max DD
	rawMaxDD      float32 // Raw max DD (RiskPct=1.0)
	winRate       float32
	expectancy    float32
	profitFactor  float32
	exitReasons   map[string]int
	smoothVol     float32 // EMA volatility of equity changes
	downsideVol   float32 // Downside volatility for Sortino ratio
	states        []StateRecord // Per-bar state tracking for debugging
	signalProofs  map[int][]LeafProof // Proofs for each signal bar (local index)
}

// WriteStatesToCSV writes the per-bar state tracking to a CSV file
func WriteStatesToCSV(states []StateRecord, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write CSV header
	_, err = file.WriteString("BarIndex,Time,Statuses\n")
	if err != nil {
		return err
	}

	// Write state rows
	for _, state := range states {
		// Format: BarIndex,Time,Statuses
		timeStr := state.Time.Format("2006-01-02 15:04:05.000000000") // RFC3339 format
		_, err = file.WriteString(fmt.Sprintf("%d,%s,%s\n", state.BarIndex, timeStr, state.Statuses))
		if err != nil {
			return err
		}
	}

	return nil
}

// StateWithProof extends StateRecord with proof information for signal bars
type StateWithProof struct {
	BarIndex     int
	Time         time.Time
	States       string
	RegimeOK     bool
	EntryOK      bool
	ProofSummary string // Only populated for signal bars
}

// WriteStatesWithProofsToCSV writes per-bar states with proof information for signal bars
func WriteStatesWithProofsToCSV(states []StateRecord, series Series, st Strategy, f Features, signalProofs map[int][]LeafProof, windowOffset int, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write CSV header with proof columns
	_, err = file.WriteString("CSVRow,BarIndex,Time,States,RegimeOK,EntryOK,ProofSummary\n")
	if err != nil {
		return err
	}

	// Write state rows
	for _, state := range states {
		// CRITICAL FIX: f.F contains the FULL dataset (not sliced), so we need to use windowOffset
		// state.BarIndex is the index in the window (starting from warmup)
		// To get the correct global index for the full feature array, we add windowOffset
		globalIdx := windowOffset + state.BarIndex

		// Get original CSV row number using GLOBAL index
		csvRow := 0
		if globalIdx >= 0 && globalIdx < len(series.CSVRowIndex) {
			csvRow = series.CSVRowIndex[globalIdx]
		}

		timeStr := state.Time.Format("2006-01-02 15:04:05")

		// Check if this bar has proofs (it was a signal bar)
		proofs, hasProofs := signalProofs[state.BarIndex]

		var regimeOK, entryOK bool
		var proofSummary string

		if hasProofs {
			// Evaluate regime and entry using bytecode with GLOBAL index
			regimeOK = st.RegimeFilter.Root == nil || (len(st.RegimeCompiled.Code) > 0 && evaluateCompiled(st.RegimeCompiled.Code, f.F, globalIdx))
			entryOK = len(st.EntryCompiled.Code) > 0 && evaluateCompiled(st.EntryCompiled.Code, f.F, globalIdx)

			// Build proof summary with proofs and indicators
			proofSummary = buildProofSummary(st, f, proofs, globalIdx)
		} else {
			// Build proof summary with indicators only (no regime/entry evaluation)
			proofSummary = buildProofSummary(st, f, []LeafProof{}, globalIdx)
			regimeOK = false
			entryOK = false
		}

		// Format: BarIndex,Time,States,RegimeOK,EntryOK,ProofSummary
		regimeStr := "false"
		if regimeOK {
			regimeStr = "true"
		}
		entryStr := "false"
		if entryOK {
			entryStr = "true"
		}

		_, err = file.WriteString(fmt.Sprintf("%d,%d,%s,%s,%s,%s,\"%s\"\n",
			csvRow, state.BarIndex, timeStr, state.Statuses, regimeStr, entryStr, proofSummary))
		if err != nil {
			return err
		}
	}

	return nil
}

// TradeCSVRow represents a single row in the trades CSV export
type TradeCSVRow struct {
	TradeID      int
	SignalIndex  int
	EntryIndex   int
	ExitIndex    int
	SignalTime   string
	EntryTime    string
	ExitTime     string
	Direction    string
	EntryPrice   float64
	ExitPrice    float64
	PnL          float64
	HoldBars     int
	ExitReason   string
	StopPrice    float64
	TPPrice      float64
	TrailActive  bool
	// Proof fields (as JSON string to handle variable structure)
	ProofSummary string // Compressed summary of all proofs
}

// WriteTradesToCSV writes trades with proofs to a CSV file
func WriteTradesToCSV(trades []Trade, st Strategy, f Features, windowOffset int, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write CSV header
	header := "TradeID,SignalIndex,EntryIndex,ExitIndex,SignalTime,EntryTime,ExitTime,Direction,EntryPrice,ExitPrice,PnL,HoldBars,ExitReason,StopPrice,TPPrice,TrailActive,ProofSummary\n"
	_, err = file.WriteString(header)
	if err != nil {
		return err
	}

	// Write trade rows
	for i, trade := range trades {
		// CRITICAL FIX: Use global index for consistency with WriteStatesWithProofsToCSV
		// f.F contains the FULL dataset (not sliced), so we need to use windowOffset
		// trade.SignalIndex is local index (within window), convert to global
		globalSignalIdx := windowOffset + trade.SignalIndex

		// Build proof summary with GLOBAL index
		proofSummary := buildProofSummary(st, f, trade.Proofs, globalSignalIdx)

		// Format times
		signalTime := trade.SignalTime.Format("2006-01-02 15:04:05")
		entryTime := trade.EntryTime.Format("2006-01-02 15:04:05")
		exitTime := trade.ExitTime.Format("2006-01-02 15:04:05")

		direction := "LONG"
		if trade.Direction == -1 {
			direction = "SHORT"
		}

		trailActive := "false"
		if trade.TrailActive {
			trailActive = "true"
		}

		// Format: TradeID,SignalIndex,EntryIndex,ExitIndex,SignalTime,EntryTime,ExitTime,Direction,EntryPrice,ExitPrice,PnL,HoldBars,ExitReason,StopPrice,TPPrice,TrailActive,ProofSummary
		row := fmt.Sprintf("%d,%d,%d,%d,%s,%s,%s,%s,%.2f,%.2f,%.4f,%d,%s,%.2f,%.2f,%s,\"%s\"\n",
			i+1, // 1-based trade ID
			trade.SignalIndex,
			trade.EntryIdx,
			trade.ExitIdx,
			signalTime,
			entryTime,
			exitTime,
			direction,
			trade.EntryPrice,
			trade.ExitPrice,
			trade.PnL,
			trade.HoldBars,
			trade.Reason,
			trade.StopPrice,
			trade.TPPrice,
			trailActive,
			proofSummary,
		)
		_, err = file.WriteString(row)
		if err != nil {
			return err
		}
	}

	return nil
}

// buildAllIndicatorsSummary creates a summary of all 43 indicator values
func buildAllIndicatorsSummary(f Features, idx int) string {
	var indicators []string
	n := len(f.Names)
	if len(f.F) < n {
		n = len(f.F)
	}
	for i := 0; i < n; i++ {
		name := f.Names[i]
		// CRITICAL FIX: Add bounds checking to prevent panic on out-of-bounds access
		if idx < 0 || idx >= len(f.F[i]) {
			indicators = append(indicators, fmt.Sprintf("%s=OUT_OF_BOUNDS", name))
			continue
		}
		value := f.F[i][idx]
		// Format: EMA20=43921.95 (compact format)
		indicators = append(indicators, fmt.Sprintf("%s=%.2f", name, value))
	}
	return "indicators={" + strings.Join(indicators, ", ") + "}"
}

// buildProofSummary creates a compressed summary of all proofs for CSV export
// Uses BYTECODE evaluation to match actual execution path
// Format: "regime_ok=true/false, entry_ok=true/false, path=branch_N, conditions..."
func buildProofSummary(st Strategy, f Features, proofs []LeafProof, signalIdx int) string {
	var parts []string

	// Handle case when there are no proofs (non-signal bars)
	if len(proofs) == 0 {
		// Just build indicator values without regime/entry evaluation
		indicatorsPart := buildAllIndicatorsSummary(f, signalIdx)
		return indicatorsPart
	}

	// Evaluate regime and entry rules using bytecode (actual execution path)
	regimeOK := st.RegimeFilter.Root == nil || (len(st.RegimeCompiled.Code) > 0 && evaluateCompiled(st.RegimeCompiled.Code, f.F, signalIdx))
	entryOK := len(st.EntryCompiled.Code) > 0 && evaluateCompiled(st.EntryCompiled.Code, f.F, signalIdx)

	// Build condition results using bytecode evaluation for accuracy
	parts = append(parts, fmt.Sprintf("regime_ok=%v", regimeOK))
	parts = append(parts, fmt.Sprintf("entry_ok=%v", entryOK))

	// Find which OR branch matched using the already-computed bytecode leaf results
	branchPath := findMatchingORBranchFromProofs(st.EntryRule.Root, proofs)
	parts = append(parts, fmt.Sprintf("path=%s", branchPath))

	// Add individual condition results
	for _, p := range proofs {
		part := p.Kind
		if p.FeatureA != "" {
			part += "(" + p.FeatureA
			if p.FeatureB != "" {
				part += "," + p.FeatureB
			}
			part += ")"
		}

		// Use the bytecode result stored in proof.Result
		result := "false"
		if p.Result {
			result = "true"
		}
		part += ":" + result

		// Add key values for most important operators with clear indicator labels
		if p.Kind == "CrossUp" || p.Kind == "CrossDown" {
			if len(p.Values) >= 4 {
				// Format: FeatureA[t-1]=value | FeatureB[t-1]=value | FeatureA[t]=value | FeatureB[t]=value
				part += fmt.Sprintf("[%s[t-1]=%.2f | %s[t-1]=%.2f | %s[t]=%.2f | %s[t]=%.2f]",
					p.FeatureA, p.Values[0], p.FeatureB, p.Values[1], p.FeatureA, p.Values[2], p.FeatureB, p.Values[3])
			}
		} else if p.Kind == "Rising" || p.Kind == "Falling" {
			if len(p.Values) >= 2 {
				lookback := 0
				if p.LeafNode.Kind >= 0 { // Use the leaf node's lookback
					lookback = p.LeafNode.Lookback
				}
				part += fmt.Sprintf("[%s[t]=%.2f | %s[t-%d]=%.2f]",
					p.FeatureA, p.Values[len(p.Values)-1], p.FeatureA, lookback, p.Values[0])
			}
		} else if p.Kind == "SlopeGT" || p.Kind == "SlopeLT" {
			part += fmt.Sprintf("[%s[slope]=%.4f | threshold=%.2f]", p.FeatureA, p.ComputedSlope, p.Threshold)
		} else if p.Kind == "GT" || p.Kind == "LT" || p.Kind == "AbsGT" || p.Kind == "AbsLT" {
			if len(p.Values) >= 1 {
				part += fmt.Sprintf("[%s[t]=%.2f | threshold=%.2f]", p.FeatureA, p.Values[0], p.Threshold)
			}
		}

		parts = append(parts, part)
	}

	// Add all 43 indicator values at the end
	indicatorsPart := buildAllIndicatorsSummary(f, signalIdx)
	parts = append(parts, indicatorsPart)

	// Join with " | " separator and escape quotes
	summary := strings.Join(parts, " | ")
	summary = strings.ReplaceAll(summary, "\"", "'") // Escape quotes for CSV

	return summary
}

// findMatchingORBranchFromProofs finds which OR branch matched using the already-computed proof results
func findMatchingORBranchFromProofs(node *RuleNode, proofs []LeafProof) string {
	if node == nil {
		return "no_rule"
	}

	// CRITICAL FIX: Use composite key for uniqueness (Kind, A, B, Lookback)
	// Multiple leaves with same feature index (.A) but different operators/lookbacks were overwriting each other
	leafResults := make(map[LeafKey]bool)
	for _, p := range proofs {
		key := LeafKey{
			Kind:     p.LeafNode.Kind,
			A:        p.LeafNode.A,
			B:        p.LeafNode.B,
			Lookback: p.LeafNode.Lookback,
		}
		leafResults[key] = p.Result
	}

	return traceORBranch(node, leafResults)
}

// traceORBranch traces which branch of an OR tree was true based on leaf results
func traceORBranch(node *RuleNode, leafResults map[LeafKey]bool) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		key := LeafKey{
			Kind:     node.Leaf.Kind,
			A:        node.Leaf.A,
			B:        node.Leaf.B,
			Lookback: node.Leaf.Lookback,
		}
		if leafResults[key] {
			return leafKindToString(node.Leaf.Kind)
		}
		return ""
	}

	if node.Op == OpOr {
		// Check left branch
		leftPath := traceORBranch(node.L, leafResults)
		if leftPath != "" {
			return "OR_left:" + leftPath
		}

		// Check right branch
		rightPath := traceORBranch(node.R, leafResults)
		if rightPath != "" {
			return "OR_right:" + rightPath
		}

		return "OR:both_false"
	}

	if node.Op == OpAnd {
		leftPath := traceORBranch(node.L, leafResults)
		rightPath := traceORBranch(node.R, leafResults)

		if leftPath != "" && rightPath != "" {
			return fmt.Sprintf("AND(%s && %s)", leftPath, rightPath)
		}
		return ""
	}

	if node.Op == OpNot {
		leftPath := traceORBranch(node.L, leafResults)
		if leftPath != "" {
			return "NOT(" + leftPath + ")"
		}
		return "NOT(false)"
	}

	return ""
}

// leafKindToString converts LeafKind to string
func leafKindToString(kind LeafKind) string {
	switch kind {
	case LeafGT:
		return "GT"
	case LeafLT:
		return "LT"
	case LeafCrossUp:
		return "CrossUp"
	case LeafCrossDown:
		return "CrossDown"
	case LeafRising:
		return "Rising"
	case LeafFalling:
		return "Falling"
	case LeafBetween:
		return "Between"
	case LeafAbsGT:
		return "AbsGT"
	case LeafAbsLT:
		return "AbsLT"
	case LeafSlopeGT:
		return "SlopeGT"
	case LeafSlopeLT:
		return "SlopeLT"
	default:
		return "?"
	}
}

// coreBacktest is the unified backtest engine that both evaluateStrategyRange and
// evaluateStrategyRangeWithTrades now call. This ensures consistency and eliminates drift.
func coreBacktest(s Series, f Features, st Strategy, tradeStartLocal, endLocal int, computeSmoothness bool, debugSignals bool) coreBacktestResult {
	// Validate strategy parameters
	// FeeBps should be reasonable (typical crypto fees are 5-50 bps)
	// A fee > 1000 bps (10%) is likely misconfigured
	const maxFeeBps = float32(1000.0)
	if st.FeeBps < 0 || st.FeeBps > maxFeeBps {
		return coreBacktestResult{
			trades:        []Trade{},
			totalTrades:   0,
			totalHoldBars: 0,
			returnPct:     0,
			rawReturnPct:  0,
			maxDD:         1.0, // Indicate error with high DD
			rawMaxDD:      1.0,
			winRate:       0,
			expectancy:    0,
			profitFactor:  0,
			exitReasons:   make(map[string]int),
			smoothVol:     0,
			downsideVol:   0,
			states:        []StateRecord{},
			signalProofs:  make(map[int][]LeafProof),
		}
	}

	// Early exit for invalid inputs
	if len(s.Close) < 100 || tradeStartLocal >= len(s.Close) || endLocal > len(s.Close) {
		return coreBacktestResult{
			trades:        []Trade{},
			totalTrades:   0,
			totalHoldBars: 0,
			returnPct:     0,
			rawReturnPct:  0,
			maxDD:         0,
			rawMaxDD:      0,
			winRate:       0,
			expectancy:    0,
			profitFactor:  0,
			exitReasons:   make(map[string]int),
			smoothVol:     0,
			downsideVol:   0,
			states:        []StateRecord{},
			signalProofs:  make(map[int][]LeafProof),
		}
	}

	position := Position{State: Flat, Direction: 1}
	equity := float32(1.0)    // Risk-adjusted equity (uses st.RiskPct)
	rawEquity := float32(1.0) // Raw equity (RiskPct=1.0, for comparison with reports)
	peakEquity := float32(1.0)
	rawPeakEquity := float32(1.0)
	maxDD := float32(0.0)
	rawMaxDD := float32(0.0)

	trades := []Trade{}
	exitReasons := make(map[string]int)
	states := []StateRecord{} // Per-bar state tracking
	signalProofs := make(map[int][]LeafProof) // Track proofs for signal bars

	// Proof throttling DISABLED for full proof logging (set to high number)
	// Original value was 3, which limited proofs to first 3 signals per leaf kind
	const maxProofsPerLeafKind = 999999  // Essentially unlimited - log all proofs
	proofCounts := make(map[string]int) // Track how many times we've built proofs for each leaf kind

	// Smoothness metric: EMA volatility of equity changes (only if computeSmoothness is true)
	var prevMarkEquity float32
	var ema, emaVar float32
	var downsideSumSq float32
	var downsideCount int
	alpha := float32(2.0 / (50.0 + 1.0)) // EMA period 50 bars
	if computeSmoothness {
		prevMarkEquity = float32(1.0)
	}

	// New tracking: pending entry and active trade
	var pending *PendingEntry
	var activeTrade *ActiveTrade

	// Track executed trades for first-3 proof logging (not signals, but actual executed trades)
	executedTradeCount := 0

	// Verification mode: track bytecode vs AST discrepancies
	var verificationEntryDiscrepancies int
	var verificationExitDiscrepancies int
	var verificationRegimeDiscrepancies int
	var verificationDetails []string

	// Cooldown / bust protection: track consecutive losses and cooldown state
	consecLosses := 0
	cooldownUntil := -1

	// Warmup bars: use the MAX of tradeStartLocal and computed strategy warmup
	// This ensures we have enough history for all indicators (entry, exit, regime)
	// CRITICAL FIX: Add indicator stabilization buffer (200 bars)
	// EMAs need ~200 bars to stabilize from initial seed values
	const stabilizationBars = 200
	minEval := computeWarmup(st) + stabilizationBars
	if minEval < 1 {
		minEval = 1
	}

	atr7Idx, ok7 := f.Index["ATR7"]
	atr14Idx, ok14 := f.Index["ATR14"]
	activeIdx, okActive := f.Index["Active"]
	volZIdx, okVolZ := f.Index["VolZ20"]

	// Debug: track first N signals for detailed print (set to 0 to disable)
	const debugMaxSignals = 0 // Print first 0 signals (disabled)
	debugSignalCount := 0

	getATR := func(t int) float32 {
		atrIdx := -1
		if ok14 {
			atrIdx = atr14Idx
		} else if ok7 {
			atrIdx = atr7Idx
		}
		if atrIdx >= 0 && atrIdx < len(f.F) && t >= 0 && t < len(f.F[atrIdx]) {
			return f.F[atrIdx][t]
		}
		return 0
	}

	// closeTrade is the "never lose track" helper - always resets position and sets cooldown in one place
	closeTrade := func(exitIdx int, exitPrice float32, reason string, stopBefore, tpBefore float32, triggers logx.ExitTriggerStatus) {
		if activeTrade == nil {
			if Verbose {
				fmt.Printf("[DEBUG] ERROR: closeTrade called with activeTrade=nil! exitIdx=%d, price=%.2f, reason=%s\n", exitIdx, exitPrice, reason)
			}
			return
		}
		tr := &trades[activeTrade.tradeIdx]
		tr.ExitIdx = exitIdx
		tr.ExitPrice = exitPrice
		tr.ExitTime = time.Unix(int64(s.OpenTimeMs[exitIdx])/1000, 0)
		tr.ExitOpen = s.Open[exitIdx]
		tr.ExitHigh = s.High[exitIdx]
		tr.ExitLow = s.Low[exitIdx]
		tr.ExitClose = s.Close[exitIdx]
		tr.Reason = reason
		tr.StopBefore = stopBefore
		tr.TPBefore = tpBefore

		if Verbose {
			fmt.Printf("[DEBUG] closeTrade: tradeIdx=%d, entryIdx=%d, exitIdx=%d, reason=%s, price=%.2f\n",
				activeTrade.tradeIdx, activeTrade.entryIdx, exitIdx, reason, exitPrice)
		}

		rawPnL := (exitPrice - activeTrade.entryPrice) / activeTrade.entryPrice
		if activeTrade.dir == -1 {
			rawPnL = -rawPnL
		}
		feeRate := st.FeeBps / 10000
		feePnL := feeRate * 2
		pnl := rawPnL - feePnL
		tr.PnL = pnl
		tr.HoldBars = exitIdx - activeTrade.entryIdx + 1 // +1 to include both entry and exit bars

		// Update equity
		equity *= (1 + pnl*st.RiskPct)
		rawEquity *= (1 + pnl)

		// Update realized equity tracking
		if equity > peakEquity {
			peakEquity = equity
		}
		if rawEquity > rawPeakEquity {
			rawPeakEquity = rawEquity
		}

		exitReasons[reason]++

		// Track consecutive losses for bust/cooldown protection
		if pnl < 0 {
			consecLosses++
			if st.MaxConsecLosses > 0 && consecLosses >= st.MaxConsecLosses && st.CooldownBars > 0 {
				cooldownUntil = exitIdx + st.CooldownBars
			}
		} else {
			consecLosses = 0
		}

		// Log EXIT event with trigger status for ALL trades
		if LogTrades {
			// Determine exit evaluation bar based on reason
			var evalBar int
			var evalTime time.Time

			switch reason {
			case "exit_rule", "max_hold":
				// For exit_rule and max_hold, exit was evaluated at t-1
				evalBar = exitIdx - 1
				if evalBar < 0 {
					evalBar = 0
				}
				evalTime = time.Unix(int64(s.OpenTimeMs[evalBar])/1000, 0)
			default:
				// For SL/TP hits, exit was evaluated at current bar t
				evalBar = exitIdx
				evalTime = time.Unix(int64(s.OpenTimeMs[evalBar])/1000, 0)
			}

			pnlStr := fmt.Sprintf("%.2f%%", pnl*100)
			if pnl > 0 {
				pnlStr = fmt.Sprintf("+%.2f%%", pnl*100)
			}

			// Use detailed logging with triggers for ALL trades
			if triggers.EntryPrice > 0 {
				triggers.ActualReason = reason
				triggers.ExitPrice = float64(exitPrice)
				triggers.HoldBars = tr.HoldBars
				triggers.MaxHoldBars = st.MaxHoldBars
				logx.LogExitBlockWithTriggers(evalBar, exitIdx, evalTime, tr.ExitTime, float64(exitPrice), reason, pnlStr, triggers)
			} else {
				logx.LogExitBlock(evalBar, exitIdx, evalTime, tr.ExitTime, float64(exitPrice), reason, pnlStr)
			}
		}

		// Always reset position and set cooldown in one place
		position.State = Flat
		activeTrade = nil
		if st.CooldownBars > 0 {
			// Apply cooldown after any exit (matches reference behavior)
			if cooldownUntil < exitIdx + st.CooldownBars {
				cooldownUntil = exitIdx + st.CooldownBars
			}
		}
	}

	// Mark-to-market equity tracking
	peakMarkEquity := float32(1.0)
	markTroughEquity := float32(1.0)
	rawPeakMarkEquity := float32(1.0)

	// Helper function to join strings
	stringJoin := func(strs []string, sep string) string {
		if len(strs) == 0 {
			return ""
		}
		result := strs[0]
		for i := 1; i < len(strs); i++ {
			result += sep + strs[i]
		}
		return result
	}

	// Always loop from 1 so t-1 exists for indicator access
	for t := 1; t < endLocal; t++ {
		closePrice := s.Close[t]
		highPrice := s.High[t]
		lowPrice := s.Low[t]
		openPrice := s.Open[t]
		barTime := time.Unix(int64(s.OpenTimeMs[t])/1000, 0)

		// State tracking for debugging - build status string
		statuses := []string{}

		// Declare all variables at the top to avoid goto issues
		var isActive bool
		var regimeOk bool
		var markEquity float32
		var rawMarkEquity float32
		var rawDD float32
		var dd float32

		// ============================================================
		// STEP A: Execute pending entry first (at bar open)
		// ============================================================
		if pending != nil && pending.entryIdx == t {
			// CRITICAL FIX: Use t-1 for all indicator access when executing at bar t open
			// Bar t's close/high/low/volume are not known at bar t open
			ref := t - 1
			if ref < 0 {
				ref = 0
			}

			// Check if still in regime and active (nil-safe guards)
			regimeOk = st.RegimeFilter.Root == nil || (len(st.RegimeCompiled.Code) > 0 && evaluateCompiled(st.RegimeCompiled.Code, f.F, ref))
			isActive = true
			if okActive && activeIdx >= 0 && activeIdx < len(f.F) && ref < len(f.F[activeIdx]) {
				isActive = f.F[activeIdx][ref] > 0
			}

			if regimeOk && isActive {
				// Execute entry at open price with slippage
				entryPrice := openPrice
				slip := st.SlippageBps / 10000
				if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
					volZ := f.F[volZIdx][ref]
					if volZ < -2.0 {
						slip *= 4.0
					} else if volZ < -1.0 {
						slip *= 2.0
					}
				}
				if pending.dir == 1 {
					entryPrice *= (1 + slip)
				} else {
					entryPrice *= (1 - slip)
				}

				// Set stop loss
				var sl float32
				switch st.StopLoss.Kind {
				case "fixed":
					if pending.dir == 1 {
						sl = entryPrice * (1 - st.StopLoss.Value/100)
					} else {
						sl = entryPrice * (1 + st.StopLoss.Value/100)
					}
				case "atr":
					atr := getATR(ref)
					if atr > 0 {
						if pending.dir == 1 {
							sl = entryPrice - atr*st.StopLoss.ATRMult
						} else {
							sl = entryPrice + atr*st.StopLoss.ATRMult
						}
					} else {
						if pending.dir == 1 {
							sl = entryPrice * 0.98
						} else {
							sl = entryPrice * 1.02
						}
					}
				case "swing":
					swingLow := s.Close[ref]
					swingHigh := s.Close[ref]
					for i := 1; i <= st.StopLoss.SwingIdx && ref-i >= 0; i++ {
						if s.Low[ref-i] < swingLow {
							swingLow = s.Low[ref-i]
						}
						if s.High[ref-i] > swingHigh {
							swingHigh = s.High[ref-i]
						}
					}
					if pending.dir == 1 {
						sl = swingLow
					} else {
						sl = swingHigh
					}
				}

				// Set take profit
				var tp float32
				switch st.TakeProfit.Kind {
				case "fixed":
					if pending.dir == 1 {
						tp = entryPrice * (1 + st.TakeProfit.Value/100)
					} else {
						tp = entryPrice * (1 - st.TakeProfit.Value/100)
					}
				case "atr":
					atr := getATR(ref)
					if atr > 0 {
						if pending.dir == 1 {
							tp = entryPrice + atr*st.TakeProfit.ATRMult
						} else {
							tp = entryPrice - atr*st.TakeProfit.ATRMult
						}
					} else {
						if pending.dir == 1 {
							tp = entryPrice * 1.04
						} else {
							tp = entryPrice * 0.96
						}
					}
				}

				// Create active trade
				activeTrade = &ActiveTrade{
					tradeIdx:    len(trades),
					entryIdx:    t,
					entryTime:   barTime,
					dir:         pending.dir,
					entryPrice:  entryPrice,
					tp:          tp,
					sl:          sl,
					trailActive: st.Trail.Active,
				}

				// Track executed trade count (for first-3 proof logging)
				executedTradeCount++

				// Also update position.State to keep them in sync (critical bug fix!)
				if pending.dir == 1 {
					position.State = Long
				} else {
					position.State = Short
				}
				position.EntryPrice = entryPrice
				position.EntryTime = t
				position.StopPrice = sl
				position.TPPrice = tp

				// Create placeholder trade record (will be filled on exit)
				trade := Trade{
					Direction:    pending.dir,
					EntryIdx:     t,
					EntryTime:    barTime,
					EntryPrice:   entryPrice,
					SignalIndex:  pending.signalIdx,
					SignalTime:   pending.signalTime,
					StopPrice:    sl,
					TPPrice:      tp,
					TrailActive:  st.Trail.Active,
					Proofs:       pending.proofs, // Copy mathematical proofs from signal detection
				}
				trades = append(trades, trade)

				statuses = append(statuses, "enter")
				if Verbose {
					fmt.Printf("[DEBUG] Created activeTrade: tradeIdx=%d, entryIdx=%d, dir=%d, position.State=%d\n", activeTrade.tradeIdx, activeTrade.entryIdx, activeTrade.dir, position.State)
				}

				// Log ENTRY EXEC event
				if LogTrades {
					logx.LogEntryBlock(t, barTime, float64(entryPrice), dirString(pending.dir))
				}
			}

			// Clear pending regardless of whether entry executed
			pending = nil

			// NOTE: Allow exit evaluation on same bar as entry for strategies with tight TP/SL
			// However, gap-open exits are NOT allowed on same bar as entry since we just entered at open
		}

		// ============================================================
		// STEP B: If in position, check exits BEFORE anything else
		// ============================================================
		if activeTrade != nil {
			// Snapshot stop/TP before any updates (gap-open logic uses these)
			stopBefore := activeTrade.sl
			tpBefore := activeTrade.tp

			if Verbose {
				fmt.Printf("[DEBUG] Bar %d: position exists, entryIdx=%d, dir=%d, sl=%.2f, tp=%.2f\n",
					t, activeTrade.entryIdx, activeTrade.dir, stopBefore, tpBefore)
			}

			hitSL := false
			hitTP := false

			// Check if this is the same bar as entry (for gap-open guard)
			sameBarEntry := activeTrade.entryIdx == t

			// B1: Gap-open TP/SL checks (open jumped past your level)
			// IMPORTANT: Skip gap-open checks on same bar as entry since we just entered at open
			if !sameBarEntry {
				if activeTrade.dir == 1 {
					// LONG: gap-open TP if open >= tp, gap-open SL if open <= sl
					if openPrice >= tpBefore {
						// Apply slippage for consistency
						ref := slippageRef(t)
						slip := st.SlippageBps / 10000.0
						if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
							volZ := f.F[volZIdx][ref]
							if volZ < -2.0 {
								slip *= 4.0
							} else if volZ < -1.0 {
								slip *= 2.0
							}
						}
						fillPrice := openPrice * (1.0 - slip) // long exit sells -> worse
						if Verbose {
							fmt.Printf("[DEBUG] Bar %d: gap-open TP hit (open=%.2f >= tp=%.2f)\n", t, openPrice, tpBefore)
						}
						closeTrade(t, fillPrice, "tp_gap_open", stopBefore, tpBefore, logx.ExitTriggerStatus{})
						statuses = append(statuses, "exit_tp_gap_open")
						goto END_BAR
					}
					if openPrice <= stopBefore {
						ref := slippageRef(t)
						slip := st.SlippageBps / 10000.0
						if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
							volZ := f.F[volZIdx][ref]
							if volZ < -2.0 {
								slip *= 4.0
							} else if volZ < -1.0 {
								slip *= 2.0
							}
						}
						fillPrice := openPrice * (1.0 - slip) // long exit sells -> worse
						if Verbose {
							fmt.Printf("[DEBUG] Bar %d: gap-open SL hit (open=%.2f <= sl=%.2f)\n", t, openPrice, stopBefore)
						}
						closeTrade(t, fillPrice, "sl_gap_open", stopBefore, tpBefore, logx.ExitTriggerStatus{})
						statuses = append(statuses, "exit_sl_gap_open")
						goto END_BAR
					}
				} else {
					// SHORT: gap-open TP if open <= tp, gap-open SL if open >= sl
					if openPrice <= tpBefore {
						ref := slippageRef(t)
						slip := st.SlippageBps / 10000.0
						if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
							volZ := f.F[volZIdx][ref]
							if volZ < -2.0 {
								slip *= 4.0
							} else if volZ < -1.0 {
								slip *= 2.0
							}
						}
						fillPrice := openPrice * (1.0 + slip) // short exit buys -> worse
						closeTrade(t, fillPrice, "tp_gap_open", stopBefore, tpBefore, logx.ExitTriggerStatus{})
						statuses = append(statuses, "exit_tp_gap_open")
						goto END_BAR
					}
					if openPrice >= stopBefore {
						ref := slippageRef(t)
						slip := st.SlippageBps / 10000.0
						if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
							volZ := f.F[volZIdx][ref]
							if volZ < -2.0 {
								slip *= 4.0
							} else if volZ < -1.0 {
								slip *= 2.0
							}
						}
						fillPrice := openPrice * (1.0 + slip) // short exit buys -> worse
						closeTrade(t, fillPrice, "sl_gap_open", stopBefore, tpBefore, logx.ExitTriggerStatus{})
						statuses = append(statuses, "exit_sl_gap_open")
						goto END_BAR
					}
				}
			}

			// B2: Intrabar TP/SL checks
			// INTRABAR NOTE: Using VolZ20[t] is correct here since:
			// 1. Gap-open checks (B1) already happened at bar t open (using t-1 data)
			// 2. Intrabar SL/TP means price hit the level DURING bar t
			// 3. By the time we check intrabar, bar t's price action is known
			// 4. This differs from entry/exit at open where bar t is unknown
			if activeTrade.dir == 1 {
				// LONG: SL triggers if low <= sl, TP triggers if high >= tp
				hitSL = lowPrice <= stopBefore
				hitTP = highPrice >= tpBefore
			} else {
				// SHORT: SL triggers if high >= sl, TP triggers if low <= tp
				hitSL = highPrice >= stopBefore
				hitTP = lowPrice <= tpBefore
			}

			// Conservative: prioritize SL over TP when both hit
			if hitSL {
				// Apply slippage
				exitPrice := stopBefore
				slip := (st.SlippageBps * 2.0) / 10000
				ref := slippageRef(t)
				if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
					volZ := f.F[volZIdx][ref]
					if volZ < -2.0 {
						slip *= 4.0
					} else if volZ < -1.0 {
						slip *= 2.0
					}
				}
				if activeTrade.dir == 1 {
					exitPrice *= (1 - slip)
				} else {
					exitPrice *= (1 + slip)
				}

				// Determine if it's a trailing stop exit
				reason := "sl_hit"
				if st.Trail.Active {
					if activeTrade.dir == 1 && stopBefore >= activeTrade.entryPrice {
						reason = "trail_hit"
					} else if activeTrade.dir == -1 && stopBefore <= activeTrade.entryPrice {
						reason = "trail_hit"
					}
				}
				closeTrade(t, exitPrice, reason, stopBefore, tpBefore, logx.ExitTriggerStatus{})
				statuses = append(statuses, "exit_"+reason)
				goto END_BAR
			} else if hitTP {
				exitPrice := tpBefore
				slip := st.SlippageBps / 10000
				ref := slippageRef(t)
				if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
					volZ := f.F[volZIdx][ref]
					if volZ < -2.0 {
						slip *= 4.0
					} else if volZ < -1.0 {
						slip *= 2.0
					}
				}
				if activeTrade.dir == 1 {
					exitPrice *= (1 - slip)
				} else {
					exitPrice *= (1 + slip)
				}
				closeTrade(t, exitPrice, "tp_hit", stopBefore, tpBefore, logx.ExitTriggerStatus{})
				statuses = append(statuses, "exit_tp_hit")
				goto END_BAR
			}

			// B3: Check exit rule and max hold (nil-safe guard for ExitCompiled.Code)
			// IMPORTANT: Evaluate exit rule on t-1, exit at open of t to avoid lookahead bias
			// If features include close/high/low of bar t, we can't know them at the open
			exitRule := false
			if t > activeTrade.entryIdx && len(st.ExitCompiled.Code) > 0 {
				exitRule = evaluateCompiled(st.ExitCompiled.Code, f.F, t-1)
			}
			maxHold := false
			if st.MaxHoldBars > 0 && (t-activeTrade.entryIdx) >= st.MaxHoldBars {
				maxHold = true
			}

			if exitRule || maxHold {
				// CRITICAL FIX: Use t-1 for slippage when exiting at bar t open
				// Bar t's volatility is not known at bar t open
				ref := t - 1
				if ref < 0 {
					ref = 0
				}

				exitPrice := openPrice
				slip := st.SlippageBps / 10000
				if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
					volZ := f.F[volZIdx][ref]
					if volZ < -2.0 {
						slip *= 4.0
					} else if volZ < -1.0 {
						slip *= 2.0
					}
				}
				if activeTrade.dir == 1 {
					exitPrice *= (1 - slip)
				} else {
					exitPrice *= (1 + slip)
				}
				reason := "exit_rule"
				if maxHold {
					reason = "max_hold"
				}
				closeTrade(t, exitPrice, reason, stopBefore, tpBefore, logx.ExitTriggerStatus{})
				statuses = append(statuses, "exit_"+reason)
				goto END_BAR
			}

			// B4: Still in position - update trailing stop for NEXT bar
			if st.Trail.Active {
				switch st.Trail.Kind {
				case "atr":
					atr := getATR(t)
					if atr > 0 {
						if activeTrade.dir == 1 {
							trailStop := closePrice - atr*st.Trail.ATRMult
							if trailStop > activeTrade.sl {
								activeTrade.sl = trailStop
							}
						} else {
							trailStop := closePrice + atr*st.Trail.ATRMult
							if trailStop < activeTrade.sl {
								activeTrade.sl = trailStop
							}
						}
					}
				case "swing":
					lookback := 50
					if t > lookback {
						if activeTrade.dir == 1 {
							best := activeTrade.sl
							for i := t - lookback; i <= t; i++ {
								if s.Low[i] > best {
									best = s.Low[i]
								}
							}
							if best > activeTrade.sl {
								activeTrade.sl = best
							}
						} else {
							best := activeTrade.sl
							for i := t - lookback; i <= t; i++ {
								if s.High[i] < best {
									best = s.High[i]
								}
							}
							if best < activeTrade.sl {
								activeTrade.sl = best
							}
						}
					}
				}
			}
		}

		// ============================================================
		// HOLDING HEARTBEAT: Log current position status every N bars
		// ============================================================
		if LogTrades && HoldingHeartbeat > 0 && activeTrade != nil {
			holdBars := t - activeTrade.entryIdx + 1
			if holdBars > 0 && holdBars%HoldingHeartbeat == 0 {
				// Calculate current PnL
				var currentPnL float64
				if activeTrade.dir == 1 {
					currentPnL = float64((closePrice - activeTrade.entryPrice) / activeTrade.entryPrice * 100)
				} else {
					currentPnL = float64((activeTrade.entryPrice - closePrice) / activeTrade.entryPrice * 100)
				}

				// Get trail price (if active)
				trailPrice := float64(0.0)
				if activeTrade.trailActive {
					trailPrice = float64(activeTrade.sl)
				}

				logx.LogHoldingBlock(t, barTime, currentPnL, float64(activeTrade.sl), float64(activeTrade.tp), trailPrice, holdBars)
			}
		}

		// ============================================================
		// STEP C: Apply cooldown filter (warmup removed - signals now evaluated during warmup)
		// ============================================================
		if t <= cooldownUntil {
			statuses = append(statuses, "cooldown")
			goto END_BAR
		}

		// ============================================================
		// STEP D: If pending exists, don't schedule another
		// ============================================================
		if pending != nil {
			statuses = append(statuses, "pending")
			goto END_BAR
		}

		// ============================================================
		// STEP E: Only now evaluate signal and schedule entry
		// ============================================================
		isActive = true
		if okActive && activeIdx >= 0 && activeIdx < len(f.F) && t < len(f.F[activeIdx]) {
			isActive = f.F[activeIdx][t] > 0
		}

		regimeOk = st.RegimeFilter.Root == nil || (st.RegimeCompiled.Code != nil && evaluateCompiled(st.RegimeCompiled.Code, f.F, t))

		// IMPORTANT: Check if we're already in a position (critical bug fix)
		if position.State != Flat {
			goto END_BAR
		}

		if !isActive || !regimeOk {
			goto END_BAR
		}

		// Check for entry signal (nil-safe guard for EntryCompiled.Code)
		if len(st.EntryCompiled.Code) > 0 && evaluateCompiled(st.EntryCompiled.Code, f.F, t) {
			// Check bust protection
			busted := st.MaxConsecLosses > 0 && consecLosses >= st.MaxConsecLosses
			if busted {
				statuses = append(statuses, "busted")
				if debugSignals && debugSignalCount < debugMaxSignals {
					fmt.Printf("  [BLOCKED] Busted: %d consecutive losses (MaxConsecLosses=%d)\n", consecLosses, st.MaxConsecLosses)
				}
			} else if t < tradeStartLocal {
				statuses = append(statuses, "pre_trade_window")
			} else if t < minEval {
				statuses = append(statuses, "signal_warmup")
			} else if t+1 >= endLocal {
				// End-of-window guard: don't schedule entry past the end of data
				statuses = append(statuses, "signal_skipped_end_of_data")
			} else {
				// Build sub-conditions string for logging
				subConds := buildSubConditionString(st, f, t)

				// Collect proofs for ALL trades (mathematical proof of no lookahead bias)
				// THROTTLED: Only build proofs for first N occurrences per leaf kind
				var proofs []LeafProof
				if st.EntryRule.Root != nil {
					proofs = buildSubConditionProofsThrottled(st.EntryRule.Root, f, t, proofCounts, maxProofsPerLeafKind)
				}

				// Store proofs for this signal bar (for state export)
				if len(proofs) > 0 {
					signalProofs[t] = proofs
				}

				// Schedule entry for next bar
				pending = &PendingEntry{
					signalIdx:     t,
					entryIdx:      t + 1,
					signalTime:    barTime,
					dir:           st.Direction,
					closePrice:    float64(closePrice),
					regimeResult:  regimeOk,
					entryResult:   true,
					subConditions: subConds,
					proofs:        proofs,
				}
				statuses = append(statuses, "signal")
				statuses = append(statuses, "pending") // Show that this bar has a pending entry
				if debugSignals && debugSignalCount < debugMaxSignals {
					debugSignalCount++
					fmt.Printf("\n[SIGNAL #%d at t=%d]\n", debugSignalCount, t)
					fmt.Printf("  Close[t]=%.6f Open[t]=%.6f\n", closePrice, openPrice)
				}

				// Log SIGNAL DETECT event (with mathematical proofs for ALL trades)
				if LogTrades {
					if len(proofs) > 0 {
						// Convert LeafProof to logx.LeafProof (same struct but different package)
						logxProofs := make([]logx.LeafProof, len(proofs))
						for i, p := range proofs {
							logxProofs[i] = logx.LeafProof{
								Kind:          p.Kind,
								Operator:      p.Operator,
								FeatureA:      p.FeatureA,
								FeatureB:      p.FeatureB,
								BarIndex:      p.BarIndex,
								Values:        p.Values,
								Comparisons:   p.Comparisons,
								GuardChecks:   p.GuardChecks,
								ComputedSlope: p.ComputedSlope,
								Threshold:     p.Threshold,
								Result:        p.Result,
							}
						}
						logx.LogSignalBlockWithProof(t, barTime, float64(closePrice), regimeOk, true, subConds, logxProofs)
					} else {
						logx.LogSignalBlock(t, barTime, float64(closePrice), regimeOk, true, subConds)
					}
				}
			}
		}

		// ============================================================
		// Mark-to-market: compute AFTER all entry/exit logic for consistency
		// ============================================================
		if activeTrade != nil {
			markPnL := float32(0.0)
			if activeTrade.dir == 1 {
				markPnL = (closePrice - activeTrade.entryPrice) / activeTrade.entryPrice
			} else {
				markPnL = (activeTrade.entryPrice - closePrice) / activeTrade.entryPrice
			}
			markEquity = equity * (1.0 + markPnL*st.RiskPct)
			rawMarkEquity = rawEquity * (1.0 + markPnL)
			statuses = append(statuses, "holding")
		} else {
			markEquity = equity
			rawMarkEquity = rawEquity
			statuses = append(statuses, "flat")
		}

		// Update peaks and DD EVERY BAR
		if rawMarkEquity > rawPeakMarkEquity {
			rawPeakMarkEquity = rawMarkEquity
		}
		rawDD = float32(0.0)
		if rawPeakMarkEquity > 0 {
			rawDD = (rawPeakMarkEquity - rawMarkEquity) / rawPeakMarkEquity
		}
		if rawDD > rawMaxDD {
			rawMaxDD = rawDD
		}

		// Update risk-adjusted peak and DD
		if markEquity > peakMarkEquity {
			peakMarkEquity = markEquity
		}
		if markEquity < markTroughEquity {
			markTroughEquity = markEquity
		}
		dd = float32(0.0)
		if peakMarkEquity > 0 {
			dd = (peakMarkEquity - markEquity) / peakMarkEquity
		}
		if dd > maxDD {
			maxDD = dd
		}

		// Compute smoothness metrics if requested (with safety guard)
		if computeSmoothness && t > 0 && prevMarkEquity > 0 {
			r := (rawMarkEquity - prevMarkEquity) / prevMarkEquity
			diff := r - ema
			ema += alpha * diff
			emaVar += alpha * (diff*diff - emaVar)
			if r < 0 {
				downsideSumSq += r * r
				downsideCount++
			}
			prevMarkEquity = rawMarkEquity
		}

		END_BAR:
		// Finalize state for this bar - ALWAYS runs, no matter what path we took
		states = append(states, StateRecord{BarIndex: t, Time: barTime, Statuses: stringJoin(statuses, ",")})
	}

	// CRITICAL: Force-close any remaining open positions at end of backtest
	// This prevents trades from having zero exit values and exit reason
	// Also prevents the "Trades=0 but Ret=-90%" bug where positions are left open
	finalIdx := endLocal - 1
	if finalIdx < 0 {
		finalIdx = 0
	}
	finalPrice := s.Close[finalIdx]

	// Check both activeTrade and position.State for safety
	if activeTrade != nil || position.State != Flat {
		// If activeTrade is nil but position.State is not Flat, we have a bug
		// Create a synthetic trade record to avoid losing track
		if activeTrade == nil && position.State != Flat {
			// This should never happen, but if it does, create a minimal trade record
			if Verbose {
				fmt.Printf("[DEBUG] WARNING: position.State=%d but activeTrade=nil at end of backtest. Creating synthetic trade record.\n", position.State)
			}
			// Convert PositionState to Trade Direction (Flat=0, Long=1, Short=2 -> Direction: 1, -1)
			dir := 1
			if position.State == Short {
				dir = -1
			}
			// Create a synthetic trade to track the position
			trades = append(trades, Trade{
				Direction:   dir,
				EntryIdx:    finalIdx - 100, // Approximate entry (unknown)
				EntryPrice:  finalPrice,     // Approximate (will show 0 PnL)
				ExitIdx:     finalIdx,
				ExitPrice:   finalPrice,
				PnL:         0,
				Reason:      "force_close_synthetic",
			})
		} else if activeTrade != nil {
			// Normal force-close path
			// Apply slippage for consistency with other exits
			ref := finalIdx
			if ref > 0 {
				ref = finalIdx - 1
			}
			slip := st.SlippageBps / 10000
			if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) && ref < len(f.F[volZIdx]) {
				volZ := f.F[volZIdx][ref]
				if volZ < -2.0 {
					slip *= 4.0
				} else if volZ < -1.0 {
					slip *= 2.0
				}
			}
			forceClosePrice := finalPrice
			if activeTrade.dir == 1 {
				forceClosePrice *= (1 - slip) // Long exit: we sell, so price goes down
			} else {
				forceClosePrice *= (1 + slip) // Short exit: we buy, so price goes up
			}

			finalStop := activeTrade.sl
			finalTP := activeTrade.tp

			// DEBUG: Log end-of-data close
			if Verbose {
				fmt.Printf("[DEBUG] Closing open position at end of backtest: entryIdx=%d, finalIdx=%d, finalPrice=%.2f (with slippage)\n",
					activeTrade.entryIdx, finalIdx, forceClosePrice)
			}

			closeTrade(finalIdx, forceClosePrice, "end_of_data", finalStop, finalTP, logx.ExitTriggerStatus{})
		}

		// Reset position state to Flat
		position.State = Flat
	}

	// DEBUG: Check for trades with zero exit values
	zeroExitCount := 0
	for i, tr := range trades {
		if tr.ExitIdx == 0 || tr.ExitPrice == 0 {
			zeroExitCount++
			if Verbose {
				fmt.Printf("[DEBUG] Trade %d has zero exit: EntryIdx=%d, ExitIdx=%d, EntryPrice=%.2f, ExitPrice=%.2f, Reason='%s'\n",
					i, tr.EntryIdx, tr.ExitIdx, tr.EntryPrice, tr.ExitPrice, tr.Reason)
			}
		}
	}
	if zeroExitCount > 0 && Verbose {
		fmt.Printf("[DEBUG] Total trades with zero exit: %d out of %d\n", zeroExitCount, len(trades))
	}

	// Compute final statistics
	totalTrades := len(trades)
	wins := 0
	losses := 0
	totalWinPnL := float32(0)
	totalLossPnL := float32(0)
	totalHoldBars := 0

	for _, tr := range trades {
		totalHoldBars += tr.HoldBars
		if tr.PnL > 0 {
			wins++
			totalWinPnL += tr.PnL
		} else {
			losses++
			totalLossPnL += -tr.PnL
		}
	}

	winRate := float32(0)
	expectancy := float32(0)
	profitFactor := float32(0)

	if totalTrades > 0 {
		winRate = float32(wins) / float32(totalTrades)

		// CRITICAL FIX: Handle expectancy edge cases for consistency with PF
		if wins > 0 && losses > 0 {
			// Normal case: both wins and losses
			expectancy = (totalWinPnL/float32(wins))*winRate - (totalLossPnL/float32(losses))*(1-winRate)
		} else if wins > 0 && losses == 0 {
			// All wins - expectancy is avg win (winRate is 1.0)
			expectancy = totalWinPnL / float32(wins)
		} else if wins == 0 && losses > 0 {
			// All losses - expectancy is -avg loss
			expectancy = -totalLossPnL / float32(losses)
		}
		// else: no trades (handled by totalTrades > 0 check)

		// CRITICAL FIX: Handle all edge cases for profit factor
		if totalLossPnL > 0 && totalWinPnL > 0 {
			// Normal case: both wins and losses
			profitFactor = totalWinPnL / totalLossPnL
		} else if totalLossPnL == 0 && totalWinPnL > 0 {
			// All wins, no losses - PF is effectively infinite, cap at 999
			profitFactor = 999.0
		} else if totalLossPnL > 0 && totalWinPnL == 0 {
			// All losses, no wins - PF = 0 (already initialized)
			profitFactor = 0
		} else {
			// No profit/loss (no trades or flat)
			profitFactor = 1.0
		}

		// Debug: log only inconsistent / broken states (NOT normal PF==0 losers)
		if DebugPF && Verbose && totalTrades > 0 && shouldLogPFDebug() {
			inconsistent :=
				(wins == 0 && totalWinPnL != 0) ||
				(losses == 0 && totalLossPnL != 0) ||
				(wins == 0 && losses > 0 && profitFactor != 0) ||
				(wins > 0 && losses == 0 && profitFactor < 100) // should be 999 in your logic

			if inconsistent || math.IsNaN(float64(expectancy)) || math.IsInf(float64(expectancy), 0) ||
				math.IsNaN(float64(profitFactor)) || math.IsInf(float64(profitFactor), 0) {
				fmt.Printf("[PF-DEBUG] trades=%d wins=%d losses=%d winPnL=%.4f lossPnL=%.4f PF=%.4f exp=%.6f\n",
					totalTrades, wins, losses, totalWinPnL, totalLossPnL, profitFactor, expectancy)
			}
		}
	}

	returnPct := equity - 1.0
	rawReturnPct := rawEquity - 1.0

	// Compute smoothness metrics if requested
	var smoothVol float32
	var downsideVol float32
	if computeSmoothness {
		smoothVol = float32(math.Sqrt(float64(emaVar)))
		if downsideCount > 0 {
			downsideVol = float32(math.Sqrt(float64(downsideSumSq / float32(downsideCount))))
		}
	}

	// Verification mode: sample bars and compare bytecode vs AST evaluation
	if VerifyMode {
		fmt.Println("\n=== VERIFICATION MODE: Bytecode vs AST Evaluation ===")
		sampleSize := 100 // Number of bars to sample
		sampleInterval := 1
		if endLocal > sampleSize {
			sampleInterval = endLocal / sampleSize
		}

		for t := tradeStartLocal; t < endLocal; t += sampleInterval {
			if t >= endLocal {
				break
			}
			summary := VerifyStrategyAtBar(st, f, t)
			if summary.TotalChecked > 0 {
				verificationEntryDiscrepancies += summary.EntryDiscrepancies
				verificationExitDiscrepancies += summary.ExitDiscrepancies
				verificationRegimeDiscrepancies += summary.RegimeDiscrepancies
				for _, detail := range summary.DiscrepancyDetails {
					verificationDetails = append(verificationDetails, detail)
				}
			}
		}

		totalDiscrepancies := verificationEntryDiscrepancies + verificationExitDiscrepancies + verificationRegimeDiscrepancies
		if totalDiscrepancies == 0 {
			fmt.Printf("  OK: No discrepancies found in sampled bars (entry=0, exit=0, regime=0)\n")
		} else {
			fmt.Printf("  WARNING: Found %d discrepancies in sampled bars:\n", totalDiscrepancies)
			fmt.Printf("    Entry rule discrepancies: %d\n", verificationEntryDiscrepancies)
			fmt.Printf("    Exit rule discrepancies: %d\n", verificationExitDiscrepancies)
			fmt.Printf("    Regime filter discrepancies: %d\n", verificationRegimeDiscrepancies)
			// Show first 5 details
			maxDetails := 5
			if len(verificationDetails) < maxDetails {
				maxDetails = len(verificationDetails)
			}
			fmt.Printf("  Details (first %d):\n", maxDetails)
			for i := 0; i < maxDetails; i++ {
				fmt.Printf("    - %s\n", verificationDetails[i])
			}
		}
		fmt.Println("========================================================")
	}

	return coreBacktestResult{
		trades:        trades,
		totalTrades:   totalTrades,
		totalHoldBars: totalHoldBars,
		returnPct:     returnPct,
		rawReturnPct:  rawReturnPct,
		maxDD:         maxDD,
		rawMaxDD:      rawMaxDD,
		winRate:       winRate,
		expectancy:    expectancy,
		profitFactor:  profitFactor,
		exitReasons:   exitReasons,
		smoothVol:     smoothVol,
		downsideVol:   downsideVol,
		states:        states,
		signalProofs:  signalProofs,
	}
}
