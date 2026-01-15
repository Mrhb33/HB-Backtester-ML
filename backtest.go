package main

import (
	"math"
)

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
	ValResult    *Result  // Optional validation result from multi-fidelity
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
	if len(s.Close) < 100 || tradeStartLocal >= len(s.Close) || endLocal > len(s.Close) {
		return Result{Strategy: st, Score: -1e30, Return: 0, MaxDD: 1, WinRate: 0, Trades: 0}
	}

	position := Position{State: Flat, Direction: 1}
	equity := float32(1.0)
	peakEquity := float32(1.0)
	troughEquity := float32(1.0) // Track minimum equity for maxDDRaw
	maxDD := float32(0.0)
	maxDDRaw := float32(0.0) // Raw max drawdown (absolute % from peak to trough)

	// Raw equity (RiskPct=1.0) for scoring - prevents score inflation from tiny returns
	rawEquity := float32(1.0)
	rawPeakEquity := float32(1.0)
	rawMaxDD := float32(0.0)

	// Mark-to-market equity for intratrade drawdown tracking
	markEquity := float32(1.0)
	peakMarkEquity := float32(1.0)
	markTroughEquity := float32(1.0) // Track minimum mark-to-market equity for raw DD

	wins := 0
	losses := 0
	totalWinPnL := float32(0)
	totalLossPnL := float32(0)

	// Execution delay: pending entry/exit signals
	pendingEntry := false
	pendingExit := false
	pendingExitExec := false // Track if pending exit was set on previous bar (for realistic delay)
	pendingDirection := 0

	atr7Idx, ok7 := f.Index["ATR7"]
	atr14Idx, ok14 := f.Index["ATR14"]
	activeIdx, okActive := f.Index["Active"]
	volZIdx, okVolZ := f.Index["VolZ20"] // For variable slippage

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

	for t := 0; t < endLocal; t++ {
		closePrice := s.Close[t]
		highPrice := s.High[t]
		lowPrice := s.Low[t]
		openPrice := s.Open[t]

		// Mark-to-market: compute every bar, not just when in position
		// This ensures drawdown is tracked accurately even after closing trades
		var rawMarkEquity float32

		if position.State != Flat {
			markPnL := float32(0.0)
			if position.State == Long {
				markPnL = (closePrice - position.EntryPrice) / position.EntryPrice
			} else { // Short
				markPnL = (position.EntryPrice - closePrice) / position.EntryPrice
			}
			// Track both risk-adjusted and raw equity
			markEquity = equity * (1.0 + markPnL*st.RiskPct)
			rawMarkEquity = rawEquity * (1.0 + markPnL) // Raw: RiskPct=1.0
		} else {
			// When flat, mark-to-market equity equals realized equity
			markEquity = equity
			rawMarkEquity = rawEquity
		}

		// Update raw equity peak and DD EVERY BAR (not just when in position)
		// This fixes the bug where drawdown wasn't tracked after closing trades
		if rawMarkEquity > rawPeakEquity {
			rawPeakEquity = rawMarkEquity
		}
		rawDD := (rawPeakEquity - rawMarkEquity) / rawPeakEquity
		if rawDD > rawMaxDD {
			rawMaxDD = rawDD
		}

		// Update peak and DD every bar (not just on exit)
		if markEquity > peakMarkEquity {
			peakMarkEquity = markEquity
		}
		// Track mark-to-market trough for consistent raw DD
		if markEquity < markTroughEquity {
			markTroughEquity = markEquity
		}
		dd := (peakMarkEquity - markEquity) / peakMarkEquity
		if dd > maxDD {
			maxDD = dd
		}

		// Carry forward pending exit from previous bar (for realistic execution delay)
		pendingExitExec = pendingExit
		pendingExit = false // Reset for new signals on this bar

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

			// Check isActive BEFORE executing pending entry (strict regime compliance)
			if !isActive {
				// Cancel pending entry if regime becomes inactive
				pendingEntry = false
				continue
			}

			// Process pending entry from previous candle
			if pendingEntry {
				// Cancel if regime fails on execution bar (not just if !isActive)
				if !regimeOk {
					pendingEntry = false
					continue
				}
				// Enter at THIS candle's open (realistic execution delay)
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
					// Long entry: buy with slippage
					entryPrice *= (1 + slip)
				} else {
					// Short entry: sell with slippage
					entryPrice *= (1 - slip)
				}

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
			}

			// Check for entry signal - but don't execute yet, set pending
			if regimeOk && evaluateCompiled(st.EntryCompiled.Code, f.F, t) {
				pendingEntry = true
				pendingDirection = st.Direction
			}
		} else {
			// Check for exit signal - set pending, don't execute yet
			exitSignal := evaluateCompiled(st.ExitCompiled.Code, f.F, t)
			if exitSignal {
				pendingExit = true
			}

			// Only apply ATR-based trailing (swing is not implemented)
			if st.Trail.Active && st.Trail.Kind == "atr" {
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
			}

			hitSL := false
			hitTP := false

			if position.Direction == 1 {
				hitSL = lowPrice <= position.StopPrice
				hitTP = highPrice >= position.TPPrice
			} else {
				hitSL = highPrice >= position.StopPrice
				hitTP = lowPrice <= position.TPPrice
			}

		if hitSL || hitTP || pendingExitExec {
			// Determine exit price based on how we're exiting
			exitPrice := closePrice
			exitOnOpen := false

			if hitSL {
				exitPrice = position.StopPrice
				// Apply slippage to SL (2x worse for stop fills - realistic)
				slip := (st.SlippageBps * 2.0) / 10000 // 2x slippage for stops
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
					exitPrice *= (1 - slip) // Long SL: worse exit price
				} else {
					exitPrice *= (1 + slip) // Short SL: worse exit price
				}
			} else if hitTP {
				exitPrice = position.TPPrice
				// TP also gets slippage (1x normal, not 2x like stops)
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
			} else if pendingExitExec {
				// Rule-based exit: execute at THIS bar's open (signaled on previous bar)
				// This makes entry and exit symmetric - both execute at next bar's open
				exitPrice = openPrice
				exitOnOpen = true
			}

			// Apply slippage for rule-based exits (executed at open)
			if exitOnOpen {
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
					exitPrice *= (1 - slip)
				} else {
					exitPrice *= (1 + slip)
				}
			}

				// Fixed: Flip PnL for short trades (profit when price goes down)
				rawPnL := (exitPrice - position.EntryPrice) / position.EntryPrice
				if position.Direction == -1 {
					rawPnL = -rawPnL
				}

			feeRate := st.FeeBps / 10000
			feePnL := feeRate * 2
			pnl := rawPnL - feePnL

			// Apply position sizing (RiskPct) for realized equity
			equity *= (1 + pnl*st.RiskPct)

			// Raw equity: no RiskPct scaling (RiskPct=1.0) - for scoring only
			rawEquity *= (1 + pnl)

			// Update realized equity tracking
			if equity > peakEquity {
				peakEquity = equity
			}
			if equity < troughEquity {
				troughEquity = equity
			}
			// Update raw equity peak
			if rawEquity > rawPeakEquity {
				rawPeakEquity = rawEquity
			}

			// Use mark-to-market for DD tracking (captures intratrade dips)
			dd := (peakMarkEquity - markEquity) / peakMarkEquity
			if dd > maxDD {
				maxDD = dd
			}
			// Raw drawdown: worst peak-to-trough percentage using consistent mark-to-market trough
			ddRaw := (peakMarkEquity - markTroughEquity) / peakMarkEquity
			if ddRaw > maxDDRaw {
				maxDDRaw = ddRaw
			}

				// Track risk-adjusted PnL for expectancy calculation (scaled by RiskPct)
				riskAdjPnL := pnl * st.RiskPct
				if pnl > 0 {
					wins++
					totalWinPnL += riskAdjPnL
				} else {
					losses++
					totalLossPnL += -riskAdjPnL
				}

				position.State = Flat
				pendingExit = false
			}
		}
	}

	totalTrades := wins + losses
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

	rawReturn := rawEquity - 1.0 // Raw return for scoring (RiskPct=1.0)

	// Debug: print DD calculation for non-empty results
	if totalTrades > 0 {
		// Uncomment for debugging:
		// fmt.Printf("[DD DEBUG] FinalEquity=%.6f PeakEquity=%.6f TroughEquity=%.6f MaxDD=%.6f MaxDDRaw=%.6f\n",
		// 	equity, peakEquity, troughEquity, maxDD, maxDDRaw)
	}

	if rawReturn > 10 || rawReturn < -0.9 || math.IsNaN(float64(rawReturn)) || math.IsInf(float64(rawReturn), 0) {
		return Result{Strategy: st, Score: -1e30, Return: rawReturn, MaxDD: rawMaxDD, MaxDDRaw: maxDDRaw, WinRate: 0, Trades: 0}
	}

	// Use DSR-lite (no penalty for train phase - only apply in validation)
	// Score using raw metrics (RiskPct=1.0) to prevent score inflation from tiny returns
	score := computeScore(rawReturn, rawMaxDD, expectancy, totalTrades, 0)

	// If no trades, set score to very low number
	if totalTrades == 0 {
		return Result{
			Strategy:     st,
			Score:        -1e30,
			Return:       0,
			MaxDD:        maxDD,
			MaxDDRaw:     maxDDRaw,
			WinRate:      0,
			Expectancy:   0,
			ProfitFactor: 0,
			Trades:       0,
		}
	}

	// Return raw metrics (RiskPct=1.0) for scoring to prevent score inflation
	// Note: This means Return and MaxDD will be larger than before (more realistic)
	return Result{
		Strategy:     st,
		Score:        score,
		Return:       rawReturn,    // Raw return (RiskPct=1.0)
		MaxDD:        rawMaxDD,    // Raw max DD (RiskPct=1.0)
		MaxDDRaw:     maxDDRaw,    // Keep raw maxDDRaw as is
		WinRate:      winRate,
		Expectancy:   expectancy,
		ProfitFactor: profitFactor,
		Trades:       totalTrades,
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
	logReturn := float32(math.Log(float64(1 + ret))) * 6.0

	baseScore := logReturn + 2.0*calmar + expReward + tradesReward - ddPenalty + tradePenalty + retPenalty

	// DSR-lite: apply deflation penalty as tested count grows
	// This forces strategies to have higher significance as we test more
	// Formula: deflated = baseScore - k * log(1 + testedCount / 10000)
	// k = 0.5 means after 10000 strategies, bar increases by ~0.5 points
	if testedCount > 0 {
		deflationPenalty := float32(0.5 * math.Log(float64(1.0+testedCount)/10000.0))
		return baseScore - deflationPenalty
	}

	return baseScore
}
