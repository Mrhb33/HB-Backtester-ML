package main

import (
	"fmt"
	"math"
	"time"
)

// Trade represents a single completed trade with detailed information
type Trade struct {
	Direction   int       // 1 for Long, -1 for Short
	EntryIdx    int       // Candle index when entry was executed
	EntryTime   time.Time // Timestamp when entry was executed
	EntryPrice  float32   // Price at entry
	ExitIdx     int       // Candle index when exit was executed
	ExitTime    time.Time // Timestamp when exit was executed
	ExitPrice   float32   // Price at exit
	Reason      string    // Exit reason: "TP", "SL", "TRAIL", "EXIT_RULE", "MAX_HOLD"
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

	if len(s.Close) < warmup+100 {
		return Result{Strategy: st, Score: -1e30, Return: 0, MaxDD: 1, MaxDDRaw: 1, WinRate: 0, Trades: 0}
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
	core := coreBacktest(s, f, st, tradeStartLocal, endLocal, true)

	rawReturn := core.rawReturnPct

	// Validate results
	if rawReturn > 10 || rawReturn < -0.9 || math.IsNaN(float64(rawReturn)) || math.IsInf(float64(rawReturn), 0) {
		return Result{Strategy: st, Score: -1e30, Return: rawReturn, MaxDD: core.rawMaxDD, MaxDDRaw: core.rawMaxDD, WinRate: 0, Trades: 0, SmoothVol: 0, DownsideVol: 0}
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
	}
}

func computeScore(ret, dd, expectancy float32, trades int, testedCount int64) float32 {
	// Only hard-reject truly broken strategies
	if trades == 0 || dd >= 0.80 {
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
	if testedCount > 0 {
		deflationPenalty := float32(0.5 * math.Log(float64(1.0+testedCount)/10000.0))
		baseScore -= deflationPenalty
	}

	return baseScore
}

// computeScoreWithSmoothness applies a smoothness penalty to strategies with volatile equity curves
// and includes Sortino ratio for downside risk focus
func computeScoreWithSmoothness(ret, dd, expectancy, smoothVol, downsideVol float32, trades int, testedCount int64) float32 {
	baseScore := computeScore(ret, dd, expectancy, trades, testedCount)

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
func evaluateStrategyWithTrades(full Series, fullF Features, st Strategy, w Window) GoldenResult {
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

	return evaluateStrategyRangeWithTrades(s, f, st, tradeStartLocal, tradeEndLocal)
}

// evaluateStrategyRangeWithTrades runs a backtest with full trade logging
func evaluateStrategyRangeWithTrades(s Series, f Features, st Strategy, tradeStartLocal, endLocal int) GoldenResult {
	// Use unified core backtest engine to ensure consistency
	// computeSmoothness=false since we don't need EMA metrics for golden results
	core := coreBacktest(s, f, st, tradeStartLocal, endLocal, false)

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
}

// coreBacktest is the unified backtest engine that both evaluateStrategyRange and
// evaluateStrategyRangeWithTrades now call. This ensures consistency and eliminates drift.
func coreBacktest(s Series, f Features, st Strategy, tradeStartLocal, endLocal int, computeSmoothness bool) coreBacktestResult {
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

	// Smoothness metric: EMA volatility of equity changes (only if computeSmoothness is true)
	var prevMarkEquity float32
	var ema, emaVar float32
	var downsideSumSq float32
	var downsideCount int
	alpha := float32(2.0 / (50.0 + 1.0)) // EMA period 50 bars
	if computeSmoothness {
		prevMarkEquity = float32(1.0)
	}

	// Execution delay: pending entry/exit signals
	pendingEntry := false
	pendingExit := false
	pendingExitExec := false
	pendingDirection := 0

	atr7Idx, ok7 := f.Index["ATR7"]
	atr14Idx, ok14 := f.Index["ATR14"]
	activeIdx, okActive := f.Index["Active"]
	volZIdx, okVolZ := f.Index["VolZ20"]

	// Debug: track first N signals for detailed print (set to 0 to disable)
	const debugMaxSignals = 3 // Print first 3 signals
	debugSignalCount := 0
	ema20Idx, ok20 := f.Index["EMA20"]
	ema50Idx, ok50 := f.Index["EMA50"]

	getATR := func(t int) float32 {
		atrIdx := -1
		if ok14 {
			atrIdx = atr14Idx
		} else if ok7 {
			atrIdx = atr7Idx
		}
		if atrIdx >= 0 && atrIdx < len(f.F) {
			return f.F[atrIdx][t]
		}
		return 0
	}

	// Mark-to-market equity tracking
	peakMarkEquity := float32(1.0)
	markTroughEquity := float32(1.0)
	rawPeakMarkEquity := float32(1.0)

	for t := 0; t < endLocal; t++ {
		closePrice := s.Close[t]
		highPrice := s.High[t]
		lowPrice := s.Low[t]
		openPrice := s.Open[t]

		// Mark-to-market: compute every bar for realistic DD tracking
		var markEquity float32
		var rawMarkEquity float32
		if position.State != Flat {
			markPnL := float32(0.0)
			if position.State == Long {
				markPnL = (closePrice - position.EntryPrice) / position.EntryPrice
			} else { // Short
				markPnL = (position.EntryPrice - closePrice) / position.EntryPrice
			}
			markEquity = equity * (1.0 + markPnL*st.RiskPct)
			rawMarkEquity = rawEquity * (1.0 + markPnL) // Raw: RiskPct=1.0
		} else {
			// When flat, mark-to-market equity equals realized equity
			markEquity = equity
			rawMarkEquity = rawEquity
		}

		// Update raw equity peak and DD EVERY BAR (matches main evaluator)
		if rawMarkEquity > rawPeakMarkEquity {
			rawPeakMarkEquity = rawMarkEquity
		}
		rawDD := (rawPeakMarkEquity - rawMarkEquity) / rawPeakMarkEquity
		if rawDD > rawMaxDD {
			rawMaxDD = rawDD
		}

		// Update risk-adjusted peak and DD EVERY BAR
		if markEquity > peakMarkEquity {
			peakMarkEquity = markEquity
		}
		if markEquity < markTroughEquity {
			markTroughEquity = markEquity
		}
		dd := (peakMarkEquity - markEquity) / peakMarkEquity
		if dd > maxDD {
			maxDD = dd
		}

		// Compute smoothness metrics if requested (for scoring)
		if computeSmoothness && t > 0 {
			// Bar return (percent change) - use raw equity to match raw scoring
			r := (rawMarkEquity - prevMarkEquity) / prevMarkEquity

			// EMA mean
			diff := r - ema
			ema += alpha * diff

			// EMA variance (approx)
			emaVar += alpha * (diff*diff - emaVar)

			// Track downside returns for Sortino ratio (negative returns only)
			if r < 0 {
				downsideSumSq += r * r
				downsideCount++
			}

			prevMarkEquity = rawMarkEquity
		}

		// Carry forward pending exit from previous bar
		pendingExitExec = pendingExit
		pendingExit = false

		// Safety: prevent stale pending entries
		if position.State != Flat {
			pendingEntry = false
		}

		isActive := true
		if okActive && activeIdx >= 0 && activeIdx < len(f.F) {
			isActive = f.F[activeIdx][t] > 0
		}

		if position.State == Flat {
			// Don't allow entries before trade start window
			if t < tradeStartLocal {
				continue
			}

			regimeOk := st.RegimeFilter.Root == nil || evaluateCompiled(st.RegimeCompiled.Code, f.F, t)

			if !isActive {
				pendingEntry = false
				continue
			}

			// Process pending entry from previous candle
			if pendingEntry {
				if !regimeOk {
					pendingEntry = false
					continue
				}
				position.Direction = pendingDirection
				entryPrice := openPrice

				// Apply variable slippage based on volume conditions
				slip := st.SlippageBps / 10000
				if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) {
					volZ := f.F[volZIdx][t]
					// Low volume = higher slippage
					if volZ < -2.0 {
						slip *= 4.0 // Very low volume: 4x slippage
					} else if volZ < -1.0 {
						slip *= 2.0 // Low volume: 2x slippage
					}
				}

				if position.Direction == 1 {
					entryPrice *= (1 + slip)
				} else {
					entryPrice *= (1 - slip)
				}

				// Set stop loss
				switch st.StopLoss.Kind {
				case "fixed":
					if position.Direction == 1 {
						position.StopPrice = entryPrice * (1 - st.StopLoss.Value/100)
					} else {
						position.StopPrice = entryPrice * (1 + st.StopLoss.Value/100)
					}
				case "atr":
					atr := getATR(t)
					if atr > 0 {
						if position.Direction == 1 {
							position.StopPrice = entryPrice - atr*st.StopLoss.ATRMult
						} else {
							position.StopPrice = entryPrice + atr*st.StopLoss.ATRMult
						}
					} else {
						if position.Direction == 1 {
							position.StopPrice = entryPrice * 0.98
						} else {
							position.StopPrice = entryPrice * 1.02
						}
					}
				case "swing":
					swingLow := closePrice
					swingHigh := closePrice
					for i := 1; i <= st.StopLoss.SwingIdx && t-i >= 0; i++ {
						if s.Low[t-i] < swingLow {
							swingLow = s.Low[t-i]
						}
						if s.High[t-i] > swingHigh {
							swingHigh = s.High[t-i]
						}
					}
					if position.Direction == 1 {
						position.StopPrice = swingLow
					} else {
						position.StopPrice = swingHigh
					}
				}

				// Set take profit
				switch st.TakeProfit.Kind {
				case "fixed":
					if position.Direction == 1 {
						position.TPPrice = entryPrice * (1 + st.TakeProfit.Value/100)
					} else {
						position.TPPrice = entryPrice * (1 - st.TakeProfit.Value/100)
					}
				case "atr":
					atr := getATR(t)
					if atr > 0 {
						if position.Direction == 1 {
							position.TPPrice = entryPrice + atr*st.TakeProfit.ATRMult
						} else {
							position.TPPrice = entryPrice - atr*st.TakeProfit.ATRMult
						}
					} else {
						if position.Direction == 1 {
							position.TPPrice = entryPrice * 1.04
						} else {
							position.TPPrice = entryPrice * 0.96
						}
					}
				}

				if position.Direction == 1 {
					position.State = Long
				} else {
					position.State = Short
				}
				position.EntryPrice = entryPrice
				position.EntryTime = t
				pendingEntry = false

				// IMPORTANT: Skip intrabar TP/SL checks on entry bar for realism
				// Without tick data, we can't know if SL or TP was hit first
				continue
			}

			// Check for entry signal
			if regimeOk && evaluateCompiled(st.EntryCompiled.Code, f.F, t) {
				pendingEntry = true
				pendingDirection = st.Direction

				// Debug: print signal details for first N signals
				if debugSignalCount < debugMaxSignals {
					fmt.Printf("\n[SIGNAL #%d at t=%d]\n", debugSignalCount+1, t)
					fmt.Printf("  Close[t]=%.6f Open[t]=%.6f\n", closePrice, openPrice)

					// Print EMA20/EMA50 values if available
					if ok20 && ema20Idx >= 0 && ema20Idx < len(f.F) && t > 0 {
						ema20 := f.F[ema20Idx]
						if ok50 && ema50Idx >= 0 && ema50Idx < len(f.F) {
							ema50 := f.F[ema50Idx]
							fmt.Printf("  EMA20[t-1]=%.6f EMA50[t-1]=%.6f\n", ema20[t-1], ema50[t-1])
							fmt.Printf("  EMA20[t]=%.6f   EMA50[t]=%.6f\n", ema20[t], ema50[t])
						}
					}

					// Print cross debug info for CrossUp/CrossDown leaves
					crossInfo := evaluateCrossDebug(st.EntryCompiled.Code, f.F, f.Names, t)
					for _, ci := range crossInfo {
						if ci.Result {
							if ci.Kind == "CrossUp" {
								prevLePrevB := ci.PrevA <= ci.PrevB
								curAGtCurB := ci.CurA > ci.CurB
								fmt.Printf("  CrossUp[%s vs %s]: prevA(%.6f) <= prevB(%.6f) = %v, curA(%.6f) > curB(%.6f) = %v\n",
									ci.FeatAName, ci.FeatBName, ci.PrevA, ci.PrevB, prevLePrevB, ci.CurA, ci.CurB, curAGtCurB)
							} else if ci.Kind == "CrossDown" {
								prevAGePrevB := ci.PrevA >= ci.PrevB
								curALtCurB := ci.CurA < ci.CurB
								fmt.Printf("  CrossDown[%s vs %s]: prevA(%.6f) >= prevB(%.6f) = %v, curA(%.6f) < curB(%.6f) = %v\n",
									ci.FeatAName, ci.FeatBName, ci.PrevA, ci.PrevB, prevAGePrevB, ci.CurA, ci.CurB, curALtCurB)
							}
						}
					}
					debugSignalCount++
				}
			}
		} else {
			// In position - check for exit
			// Skip exit rule evaluation on entry bar (t == position.EntryTime)
			// This prevents unrealistic same-bar entry/exit via exit rule
			if t > position.EntryTime {
				exitSignal := evaluateCompiled(st.ExitCompiled.Code, f.F, t)
				if exitSignal {
					pendingExit = true
				}
			}

			// Time-based exit
			if st.MaxHoldBars > 0 && (t-position.EntryTime) >= st.MaxHoldBars {
				pendingExit = true
			}

			// Trailing stop
			if st.Trail.Active {
				switch st.Trail.Kind {
				case "atr":
					atr := getATR(t)
					if atr > 0 {
						if position.Direction == 1 {
							trailStop := closePrice - atr*st.Trail.ATRMult
							if trailStop > position.StopPrice {
								position.StopPrice = trailStop
							}
						} else {
							trailStop := closePrice + atr*st.Trail.ATRMult
							if trailStop < position.StopPrice {
								position.StopPrice = trailStop
							}
						}
					}
				case "swing":
					lookback := 10
					if t > lookback {
						if position.Direction == 1 {
							best := position.StopPrice
							for i := t - lookback; i <= t; i++ {
								if s.Low[i] > best {
									best = s.Low[i]
								}
							}
							if best > position.StopPrice {
								position.StopPrice = best
							}
						} else {
							best := position.StopPrice
							for i := t - lookback; i <= t; i++ {
								if s.High[i] < best {
									best = s.High[i]
								}
							}
							if best < position.StopPrice {
								position.StopPrice = best
							}
						}
					}
				}
			}

			hitSL := false
			hitTP := false

			if position.Direction == 1 {
				// LONG: SL triggers if low <= slPrice, TP triggers if high >= tpPrice
				hitSL = lowPrice <= position.StopPrice
				hitTP = highPrice >= position.TPPrice
			} else {
				// SHORT: SL triggers if high >= slPrice, TP triggers if low <= tpPrice
				hitSL = highPrice >= position.StopPrice
				hitTP = lowPrice <= position.TPPrice
			}

			// IMPORTANT: Conservative approach when both TP and SL hit in same candle
			// Without tick data, we assume worst-case: SL was hit first
			// This prevents inflated performance from assuming best-case path
			hitTakeProfit := false
			hitStopLoss := false

			// Always prioritize SL over TP when both hit (conservative/worst-case)
			if hitSL {
				hitStopLoss = true
			} else if hitTP {
				hitTakeProfit = true
			}

			// Check for exit conditions
			// Intrabar TP/SL checks happen on subsequent bars only (entry bar is skipped via continue)
			// Exit rule (pendingExitExec) only triggers on subsequent bars (t > EntryTime)
			shouldExit := hitStopLoss || hitTakeProfit || (pendingExitExec && t > position.EntryTime)

			if shouldExit {
				exitPrice := closePrice
				reason := "EXIT_RULE"

				if hitStopLoss {
					exitPrice = position.StopPrice
					// Distinguish SL vs TRAIL: if stop is on profitable side, it's a trailing stop
					if position.Direction == 1 {
						// LONG: stop moved ABOVE entry = profitable = trailing
						if position.StopPrice >= position.EntryPrice {
							reason = "TRAIL"
						} else {
							reason = "SL"
						}
					} else {
						// SHORT: stop moved BELOW entry = profitable = trailing
						if position.StopPrice <= position.EntryPrice {
							reason = "TRAIL"
						} else {
							reason = "SL"
						}
					}
					slip := (st.SlippageBps * 2.0) / 10000
					if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) {
						volZ := f.F[volZIdx][t]
						if volZ < -2.0 {
							slip *= 4.0
						} else if volZ < -1.0 {
							slip *= 2.0
						}
					}
					if position.Direction == 1 {
						exitPrice *= (1 - slip)
					} else {
						exitPrice *= (1 + slip)
					}
				} else if hitTakeProfit {
					exitPrice = position.TPPrice
					reason = "TP"
					slip := st.SlippageBps / 10000
					if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) {
						volZ := f.F[volZIdx][t]
						if volZ < -2.0 {
							slip *= 4.0
						} else if volZ < -1.0 {
							slip *= 2.0
						}
					}
					if position.Direction == 1 {
						exitPrice *= (1 - slip)
					} else {
						exitPrice *= (1 + slip)
					}
				} else if pendingExitExec && t > position.EntryTime {
					if st.MaxHoldBars > 0 && (t-position.EntryTime) >= st.MaxHoldBars {
						reason = "MAX_HOLD"
					}
					exitPrice = openPrice
					slip := st.SlippageBps / 10000
					if okVolZ && volZIdx >= 0 && volZIdx < len(f.F) {
						volZ := f.F[volZIdx][t]
						if volZ < -2.0 {
							slip *= 4.0
						} else if volZ < -1.0 {
							slip *= 2.0
						}
					}
					if position.Direction == 1 {
						exitPrice *= (1 - slip)
					} else {
						exitPrice *= (1 + slip)
					}
				}

				rawPnL := (exitPrice - position.EntryPrice) / position.EntryPrice
				if position.Direction == -1 {
					rawPnL = -rawPnL
				}

				feeRate := st.FeeBps / 10000
				feePnL := feeRate * 2
				pnl := rawPnL - feePnL

				// Update both risk-adjusted and raw equity (matches main evaluator)
				equity *= (1 + pnl*st.RiskPct)
				rawEquity *= (1 + pnl) // Raw: RiskPct=1.0

				// Update realized equity tracking (risk-adjusted)
				if equity > peakEquity {
					peakEquity = equity
				}

				// Update realized raw equity tracking
				if rawEquity > rawPeakEquity {
					rawPeakEquity = rawEquity
				}

				// Record trade
				trade := Trade{
					Direction:    position.Direction,
					EntryIdx:     position.EntryTime,
					EntryTime:    time.Unix(int64(s.OpenTimeMs[position.EntryTime])/1000, 0),
					EntryPrice:   position.EntryPrice,
					ExitIdx:      t,
					ExitTime:     time.Unix(int64(s.OpenTimeMs[t])/1000, 0),
					ExitPrice:    exitPrice,
					Reason:       reason,
					PnL:          pnl,
					HoldBars:     t - position.EntryTime,
					StopPrice:    position.StopPrice,
					TPPrice:      position.TPPrice,
					TrailActive:  st.Trail.Active,
					ExitOpen:     openPrice,
					ExitHigh:     highPrice,
					ExitLow:      lowPrice,
					ExitClose:    closePrice,
				}
				trades = append(trades, trade)
				exitReasons[reason]++

				position.State = Flat
				pendingExit = false
			}
		}
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
		if wins > 0 && losses > 0 {
			expectancy = (totalWinPnL/float32(wins))*winRate - (totalLossPnL/float32(losses))*(1-winRate)
		}
		if totalLossPnL > 0 {
			profitFactor = totalWinPnL / totalLossPnL
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
	}
}
