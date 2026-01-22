package main

import (
	"fmt"
	"log"
)

// Invariants holds configuration for runtime assertion checking
type Invariants struct {
	Enabled                bool  // Enable/disable all invariant checks
	CheckEntryIndices      bool  // Verify entry indices are valid
	CheckExitIndices       bool  // Verify exit indices are valid
	CheckPrices            bool  // Verify entry/exit prices are valid
	CheckExecutionModel    bool  // Verify execution model (eval at t, exec at t+1)
	CheckExitRuleConsistency bool // Verify exit rule matches exit reason
}

// DefaultInvariants returns the default invariant checking configuration
func DefaultInvariants() Invariants {
	return Invariants{
		Enabled:                  true,
		CheckEntryIndices:        true,
		CheckExitIndices:         true,
		CheckPrices:              true,
		CheckExecutionModel:      true,
		CheckExitRuleConsistency: true,
	}
}

// assertEntryInvariants checks invariants at trade entry
func assertEntryInvariants(inv Invariants, activeTrade ActiveTrade, s Series, st Strategy, t int) {
	if !inv.Enabled {
		return
	}

	// Check 1: Entry index is valid
	if inv.CheckEntryIndices {
		assert(activeTrade.entryIdx < len(s.Close),
			fmt.Sprintf("entryIdx=%d out of bounds (len=%d)", activeTrade.entryIdx, len(s.Close)))
		assert(activeTrade.entryIdx >= 0,
			fmt.Sprintf("entryIdx=%d is negative", activeTrade.entryIdx))
	}

	// Check 2: Entry price matches expected Open with slippage
	if inv.CheckPrices {
		slip := st.SlippageBps / 10000
		expectedEntryPrice := s.Open[activeTrade.entryIdx]
		if activeTrade.dir == 1 {
			expectedEntryPrice *= (1 + slip)
		} else {
			expectedEntryPrice *= (1 - slip)
		}

		priceDiff := absInv(activeTrade.entryPrice - expectedEntryPrice)
		tolerance := expectedEntryPrice * 0.0001 // 0.01% tolerance for floating point

		assert(priceDiff < tolerance,
			fmt.Sprintf("entryPrice=%.4f doesn't match expected=%.4f (diff=%.4f, tolerance=%.4f)",
				activeTrade.entryPrice, expectedEntryPrice, priceDiff, tolerance))
	}

	// Check 3: Direction is valid
	assert(activeTrade.dir == 1 || activeTrade.dir == -1,
		fmt.Sprintf("invalid direction=%d (must be 1 or -1)", activeTrade.dir))
}

// assertExitInvariants checks invariants at trade exit
func assertExitInvariants(inv Invariants, tr Trade, s Series, f Features, st Strategy, t int) {
	if !inv.Enabled {
		return
	}

	// Check 1: Exit index is valid
	if inv.CheckExitIndices {
		assert(tr.ExitIdx < len(s.Close),
			fmt.Sprintf("exitIdx=%d out of bounds (len=%d)", tr.ExitIdx, len(s.Close)))
		assert(tr.ExitIdx >= 0,
			fmt.Sprintf("exitIdx=%d is negative", tr.ExitIdx))
	}

	// Check 2: Exit index > Entry index
	assert(tr.ExitIdx > tr.EntryIdx,
		fmt.Sprintf("exitIdx=%d must be > entryIdx=%d", tr.ExitIdx, tr.EntryIdx))

	// Check 3: Exit price is positive
	assert(tr.ExitPrice > 0,
		fmt.Sprintf("exitPrice=%.4f must be positive", tr.ExitPrice))

	// Check 4: Entry price is positive
	assert(tr.EntryPrice > 0,
		fmt.Sprintf("entryPrice=%.4f must be positive", tr.EntryPrice))

	// Check 5: Direction is valid
	assert(tr.Direction == 1 || tr.Direction == -1,
		fmt.Sprintf("invalid direction=%d (must be 1 or -1)", tr.Direction))

	// Check 6: Exit reason is not empty
	assert(tr.Reason != "",
		"exit reason cannot be empty")

	// Check 7: Exit rule consistency (if exit was due to rule)
	if inv.CheckExitRuleConsistency && tr.Reason == "exit_rule" {
		// The exit rule should have been true at the evaluation index
		if st.ExitCompiled.Code != nil && t > tr.EntryIdx+1 {
			// Exit rule is evaluated at t-1, executed at t
			exitRuleResult := evaluateCompiled(st.ExitCompiled.Code, f.F, t-1)
			assert(exitRuleResult == true,
				fmt.Sprintf("exit rule must be true at evaluated index t-1=%d", t-1))
		}
	}

	// Check 8: PnL is within reasonable bounds
	assert(tr.PnL > -100 && tr.PnL < 1000,
		fmt.Sprintf("PnL=%.4f%% is outside reasonable bounds [-100, 1000]", tr.PnL))

	// Check 9: Hold bars is positive
	assert(tr.HoldBars > 0,
		fmt.Sprintf("holdBars=%d must be positive", tr.HoldBars))
}

// assertExecutionModel checks the execution model for a trade
func assertExecutionModel(inv Invariants, tr Trade) {
	if !inv.Enabled || !inv.CheckExecutionModel {
		return
	}

	// Signal index should be before entry index
	signalToEntry := tr.EntryIdx - tr.SignalIndex
	assert(signalToEntry == 1,
		fmt.Sprintf("execution model violation: EntryIdx (%d) - SignalIndex (%d) = %d, expected 1",
			tr.EntryIdx, tr.SignalIndex, signalToEntry))
}

// assertStopLossInvariants checks stop loss related invariants
func assertStopLossInvariants(inv Invariants, activeTrade ActiveTrade, s Series, st Strategy, t int) {
	if !inv.Enabled {
		return
	}

	// Check 1: Stop loss is set
	assert(activeTrade.sl > 0,
		fmt.Sprintf("stop loss must be positive, got %.4f", activeTrade.sl))

	// Check 2: Stop loss is on correct side of entry price
	if activeTrade.dir == 1 {
		// Long: stop loss should be below entry
		assert(activeTrade.sl < activeTrade.entryPrice,
			fmt.Sprintf("LONG stop loss=%.4f should be < entryPrice=%.4f",
				activeTrade.sl, activeTrade.entryPrice))
	} else {
		// Short: stop loss should be above entry
		assert(activeTrade.sl > activeTrade.entryPrice,
			fmt.Sprintf("SHORT stop loss=%.4f should be > entryPrice=%.4f",
				activeTrade.sl, activeTrade.entryPrice))
	}

	// Check 3: Take profit is set (if strategy has TP)
	if st.TakeProfit.Kind != "none" {
		assert(activeTrade.tp > 0,
			fmt.Sprintf("take profit must be positive, got %.4f", activeTrade.tp))

		// TP should be on correct side of entry
		if activeTrade.dir == 1 {
			// Long: TP should be above entry
			assert(activeTrade.tp > activeTrade.entryPrice,
				fmt.Sprintf("LONG take profit=%.4f should be > entryPrice=%.4f",
					activeTrade.tp, activeTrade.entryPrice))
		} else {
			// Short: TP should be below entry
			assert(activeTrade.tp < activeTrade.entryPrice,
				fmt.Sprintf("SHORT take profit=%.4f should be < entryPrice=%.4f",
					activeTrade.tp, activeTrade.entryPrice))
		}

		// TP and SL should not overlap
		if activeTrade.dir == 1 {
			assert(activeTrade.tp > activeTrade.sl,
				fmt.Sprintf("LONG TP=%.4f should be > SL=%.4f", activeTrade.tp, activeTrade.sl))
		} else {
			assert(activeTrade.tp < activeTrade.sl,
				fmt.Sprintf("SHORT TP=%.4f should be < SL=%.4f", activeTrade.tp, activeTrade.sl))
		}
	}
}

// assertFeatureInvariants checks feature-related invariants
func assertFeatureInvariants(inv Invariants, f Features, s Series, t int) {
	if !inv.Enabled {
		return
	}

	// Check 1: All features have values at index t
	for i, feat := range f.F {
		assert(t < len(feat),
			fmt.Sprintf("feature[%d] (%s) has length %d, but accessing index t=%d",
				i, f.Names[i], len(feat), t))
	}

	// Check 2: Feature values are not NaN or Inf
	for i, feat := range f.F {
		if t < len(feat) {
			val := feat[t]
			assert(!isNaNInv(val),
				fmt.Sprintf("feature[%d] (%s) at t=%d is NaN", i, f.Names[i], t))
			assert(!isInfInv(val),
				fmt.Sprintf("feature[%d] (%s) at t=%d is Inf", i, f.Names[i], t))
		}
	}
}

// assertPositionStateInvariants checks position state consistency
func assertPositionStateInvariants(inv Invariants, position Position, activeTrade *ActiveTrade, pending *PendingEntry) {
	if !inv.Enabled {
		return
	}

	// Check 1: If active trade exists, position should not be Flat
	if activeTrade != nil {
		assert(position.State != Flat,
			fmt.Sprintf("activeTrade exists but position.State=%d (Flat)", position.State))

		// Check 2: Direction matches
		if position.State == Long {
			assert(activeTrade.dir == 1,
				fmt.Sprintf("position.State=LONG but activeTrade.dir=%d", activeTrade.dir))
		} else if position.State == Short {
			assert(activeTrade.dir == -1,
				fmt.Sprintf("position.State=SHORT but activeTrade.dir=%d", activeTrade.dir))
		}
	}

	// Check 2: If pending entry exists, position should be Flat
	if pending != nil {
		assert(position.State == Flat,
			fmt.Sprintf("pending entry exists but position.State=%d (not Flat)", position.State))
		assert(activeTrade == nil,
			"pending entry exists with activeTrade (should be mutually exclusive)")
	}
}

// assert is a helper function that logs and exits if condition is false
func assert(condition bool, message string) {
	if !condition {
		log.Fatalf("INVARIANT VIOLATION: %s\n", message)
	}
}

// isNaN checks if a float32 is NaN
func isNaNInv(f float32) bool {
	return f != f
}

// isInf checks if a float32 is Inf
func isInfInv(f float32) bool {
	return f > 1e30 || f < -1e30
}

// absInv returns absolute value of float32
func absInv(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// InvariantChecker provides a convenient way to check invariants throughout backtest
type InvariantChecker struct {
	config Invariants
	series Series
	features Features
	strategy Strategy
}

// NewInvariantChecker creates a new invariant checker
func NewInvariantChecker(inv Invariants, s Series, f Features, st Strategy) InvariantChecker {
	return InvariantChecker{
		config: inv,
		series: s,
		features: f,
		strategy: st,
	}
}

// CheckEntry checks invariants at entry
func (ic *InvariantChecker) CheckEntry(activeTrade ActiveTrade, t int) {
	assertEntryInvariants(ic.config, activeTrade, ic.series, ic.strategy, t)
	assertStopLossInvariants(ic.config, activeTrade, ic.series, ic.strategy, t)
}

// CheckExit checks invariants at exit
func (ic *InvariantChecker) CheckExit(tr Trade, t int) {
	assertExitInvariants(ic.config, tr, ic.series, ic.features, ic.strategy, t)
	assertExecutionModel(ic.config, tr)
}

// CheckFeature checks feature invariants
func (ic *InvariantChecker) CheckFeature(t int) {
	assertFeatureInvariants(ic.config, ic.features, ic.series, t)
}

// CheckPositionState checks position state consistency
func (ic *InvariantChecker) CheckPositionState(position Position, activeTrade *ActiveTrade, pending *PendingEntry) {
	assertPositionStateInvariants(ic.config, position, activeTrade, pending)
}
