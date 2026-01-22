package logx

import (
	"fmt"
	"strings"
	"time"
)

const eventSep = "═══════════════════════════════════════════════════════════════════"

// LogSignalBlock - signal detected event block at bar close (t)
// bar: true bar index (absolute position in full dataset)
// ts: timestamp of the signal bar
// closePrice: close price at signal detection
// regimeResult: result of regime filter evaluation
// entryResult: result of entry rule evaluation
// subConditions: optional string showing T/F for each sub-condition in OR/AND tree
func LogSignalBlock(bar int, ts time.Time, closePrice float64,
	regimeResult, entryResult bool, subConditions string,
) {
	subCondLine := ""
	if subConditions != "" {
		subCondLine = fmt.Sprintf("Sub-conditions: %s\n", subConditions)
	}
	fmt.Printf("%s\n%s  [SIG ]  SIGNAL DETECT (t close)\nTimestamp:    %s\nBar Index:    %d\nClose Price:  %.6f\nRegime Filter: %v\nEntry Rule:    %v\n%s%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, ts.UTC().Format("2006-01-02 15:04:05.000Z")),
		bar,
		closePrice,
		boolStr(regimeResult),
		boolStr(entryResult),
		subCondLine,
		eventSep,
	)
}

// LeafProof provides mathematical proof of leaf evaluation to demonstrate no lookahead bias
type LeafProof struct {
	Kind          string     // "CrossUp", "Rising", "SlopeGT", etc.
	Operator      string     // The actual operator used
	FeatureA      string     // Feature name
	FeatureB      string     // For Cross operators
	BarIndex      int        // Current bar t
	Values        []float64  // All values used in computation
	Comparisons   []string   // Step-by-step comparison results
	GuardChecks   []string   // t>=1, t>=lookback, NaN checks, eps checks
	ComputedSlope float64    // For SlopeGT/SlopeLT
	Threshold     float64    // X value compared against
	Result        bool       // Final result (matches returned bool)
}

// formatProofBlock formats a single leaf proof for display
func formatProofBlock(proof LeafProof) string {
	var b strings.Builder

	// Header line with operation and features
	header := proof.Kind
	if proof.FeatureB != "" {
		header = fmt.Sprintf("%s %s %s", proof.Kind, proof.FeatureA, proof.FeatureB)
		if proof.Threshold > 0 {
			header = fmt.Sprintf("%s %s %.4f[%d]", proof.Kind, proof.FeatureA, proof.Threshold, int(proof.Threshold))
		} else if proof.Operator != "" {
			header = fmt.Sprintf("%s %s %s %.2f", proof.Kind, proof.FeatureA, proof.Operator, proof.Values[1])
		}
	} else if proof.Threshold > 0 {
		header = fmt.Sprintf("%s %s %.4f", proof.Kind, proof.FeatureA, proof.Threshold)
	} else {
		header = fmt.Sprintf("%s %s[%d]", proof.Kind, proof.FeatureA, int(proof.Threshold))
	}

	b.WriteString(fmt.Sprintf("  %s at t=%d:\n", header, proof.BarIndex))

	// Guard checks
	if len(proof.GuardChecks) > 0 {
		b.WriteString("    Guards:\n")
		for _, check := range proof.GuardChecks {
			b.WriteString(fmt.Sprintf("      %s\n", check))
		}
	}

	// Values section
	if len(proof.Values) > 0 {
		b.WriteString("    Values:\n")
		if proof.Kind == "CrossUp" || proof.Kind == "CrossDown" {
			b.WriteString(fmt.Sprintf("      A[t-1]=%.2f, B[t-1]=%.2f\n", proof.Values[0], proof.Values[1]))
			b.WriteString(fmt.Sprintf("      A[t]=%.2f,  B[t]=%.2f\n", proof.Values[2], proof.Values[3]))
		} else if proof.Kind == "Rising" || proof.Kind == "Falling" {
			b.WriteString(fmt.Sprintf("      X[t-lookback]=%.2f, X[t]=%.2f\n", proof.Values[0], proof.Values[1]))
		} else if proof.Kind == "SlopeGT" || proof.Kind == "SlopeLT" {
			b.WriteString(fmt.Sprintf("      start=%.2f, end=%.2f, n=%.0f\n", proof.Values[0], proof.Values[1], proof.Values[1]-proof.Values[0]))
		} else {
			for i, v := range proof.Values {
				b.WriteString(fmt.Sprintf("      value[%d]=%.4f\n", i, v))
			}
		}
	}

	// Comparisons section
	if len(proof.Comparisons) > 0 {
		b.WriteString("    Comparisons:\n")
		if proof.Kind == "CrossUp" || proof.Kind == "CrossDown" {
			b.WriteString("      Movement check (eps=1e-6):\n")
			b.WriteString(fmt.Sprintf("        %s\n", proof.Comparisons[0]))
			b.WriteString(fmt.Sprintf("        %s\n", proof.Comparisons[1]))
			b.WriteString("      Comparison:\n")
			for i := 2; i < len(proof.Comparisons); i++ {
				b.WriteString(fmt.Sprintf("        %s\n", proof.Comparisons[i]))
			}
		} else {
			for _, comp := range proof.Comparisons {
				b.WriteString(fmt.Sprintf("      %s\n", comp))
			}
		}
	}

	// Final result
	resultStr := "false"
	if proof.Result {
		resultStr = "true"
	}
	b.WriteString(fmt.Sprintf("    result=%s\n", resultStr))

	return b.String()
}

// LogSignalBlockWithProof - signal detected with mathematical proof
// Used for first 3 executed trades to prove no lookahead bias
func LogSignalBlockWithProof(bar int, ts time.Time, closePrice float64,
	regimeResult, entryResult bool, subConditions string, proofs []LeafProof,
) {
	subCondLine := ""
	if subConditions != "" {
		subCondLine = fmt.Sprintf("Sub-conditions: %s\n", subConditions)
	}

	// Build proof section
	proofSection := ""
	if len(proofs) > 0 {
		proofSection = "\n  === Mathematical Proof (No Lookahead Bias) ===\n"
		for _, proof := range proofs {
			proofSection += formatProofBlock(proof)
		}
	}

	fmt.Printf("%s\n%s  [SIG ]  SIGNAL DETECT (t close)\nTimestamp:    %s\nBar Index:    %d\nClose Price:  %.6f\nRegime Filter: %v\nEntry Rule:    %v\n%s%s%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, ts.UTC().Format("2006-01-02 15:04:05.000Z")),
		bar,
		closePrice,
		boolStr(regimeResult),
		boolStr(entryResult),
		subCondLine,
		proofSection,
		eventSep,
	)
}

// boolStr converts bool to T/F string
func boolStr(b bool) string {
	if b {
		return C(green, "T")
	}
	return C(red, "F")
}

// LogEntryBlock - entry execution event block at next bar open (t+1)
// bar: bar index where entry executes
// ts: timestamp of entry execution bar
// entryPrice: entry price including slippage/fee
// dir: LONG or SHORT
func LogEntryBlock(bar int, ts time.Time, entryPrice float64, dir string) {
	fmt.Printf("%s\n%s  [ENT ]  ENTRY EXEC (t+1 open)\nTimestamp:    %s\nBar Index:    %d\nSide:         %s\nEntry Price:  %.6f (incl. slippage/fee)\n%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, ts.UTC().Format("2006-01-02 15:04:05.000Z")),
		bar,
		dir,
		entryPrice,
		eventSep,
	)
}

// LogExitBlock - exit event block at bar close (t) with execution at next bar open (t+1)
// evalBar: bar index where exit signal was evaluated (t close)
// execBar: bar index where exit was executed (t+1 open)
// evalTs: timestamp when exit signal became true
// execTs: timestamp when exit was executed
// exitPrice: exit price including slippage/fee
// reason: exit_rule / SL / TP / trail_hit
// pnl: profit/loss percentage
func LogExitBlock(evalBar, execBar int, evalTs, execTs time.Time, exitPrice float64, reason, pnl string) {
	reasonStr := reason
	switch reason {
	case "sl_hit", "sl_gap_open":
		reasonStr = C(red, reason)
	case "tp_hit", "tp_gap_open":
		reasonStr = C(green, reason)
	case "trail_hit":
		reasonStr = C(yellow, reason)
	case "exit_rule":
		reasonStr = C(cyan, reason)
	case "max_hold":
		reasonStr = C(magenta, reason)
	}
	fmt.Printf("%s\n%s  [EXT ]  EXIT (t close + t+1 open)\nEval Timestamp: %s (bar %d)\nExec Timestamp: %s (bar %d)\nReason:       %s\nExit Price:    %.6f (incl. slippage/fee)\nPnL:          %s\n%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, evalTs.UTC().Format("2006-01-02 15:04:05.000Z")), evalBar,
		C(gray, execTs.UTC().Format("2006-01-02 15:04:05.000Z")), execBar,
		reasonStr,
		exitPrice,
		pnl,
		eventSep,
	)
}

// ExitTriggerStatus captures the state of all exit triggers at exit time
type ExitTriggerStatus struct {
	GapOpenSL      bool    // Gap-open SL hit (open[t+1] <= sl)
	GapOpenTP      bool    // Gap-open TP hit (open[t+1] >= tp)
	IntrabarSL     bool    // Intrabar SL hit (low[t] <= sl)
	IntrabarTP     bool    // Intrabar TP hit (high[t] >= tp)
	ExitRule       bool    // Exit rule triggered
	MaxHold        bool    // Max hold exceeded
	ActualReason   string  // The actual reason chosen by precedence

	// Price details for context
	EntryPrice     float64 // Entry price
	ExitPrice      float64 // Exit price
	CloseT         float64 // Close price at bar t
	OpenT1         float64 // Open price at bar t+1
	HighT          float64 // High price at bar t
	LowT           float64 // Low price at bar t
	SLEntry        float64 // SL at entry
	TPEntry        float64 // TP at entry
	SLExit         float64 // SL at exit time
	TPExit         float64 // TP at exit time
	HoldBars       int     // Number of bars held
	MaxHoldBars    int     // Max hold bars setting
}

// LogExitBlockWithTriggers - exit event with full trigger status and precedence
// Shows all exit trigger states and which one was chosen according to precedence rules
func LogExitBlockWithTriggers(evalBar, execBar int, evalTs, execTs time.Time, exitPrice float64, reason, pnl string, triggers ExitTriggerStatus) {
	// Color the actual reason
	reasonStr := reason
	switch reason {
	case "sl_hit", "sl_gap_open":
		reasonStr = C(red, reason)
	case "tp_hit", "tp_gap_open":
		reasonStr = C(green, reason)
	case "trail_hit":
		reasonStr = C(yellow, reason)
	case "exit_rule":
		reasonStr = C(cyan, reason)
	case "max_hold":
		reasonStr = C(magenta, reason)
	}

	// Build trigger section
	triggerSection := "\n  === Exit Trigger Status ===\n"

	// Gap-open check
	triggerSection += fmt.Sprintf("  Gap-open check (at open[t+1]):\n")
	triggerSection += fmt.Sprintf("    open[t+1]=%.2f\n", triggers.OpenT1)
	triggerSection += fmt.Sprintf("    gap_open_sl: %v (%.2f %s sl=%.2f)\n",
		boolStr(triggers.GapOpenSL),
		triggers.OpenT1,
		map[bool]string{true: "<=", false: ">"}[true],
		triggers.SLEntry)
	triggerSection += fmt.Sprintf("    gap_open_tp: %v (%.2f %s tp=%.2f)\n",
		boolStr(triggers.GapOpenTP),
		triggers.OpenT1,
		map[bool]string{true: ">=", false: "<"}[true],
		triggers.TPEntry)

	// Intrabar check
	triggerSection += fmt.Sprintf("\n  Intrabar check (using high[t], low[t]):\n")
	triggerSection += fmt.Sprintf("    high[t]=%.2f, low[t]=%.2f\n", triggers.HighT, triggers.LowT)
	triggerSection += fmt.Sprintf("    sl_hit_intrabar: %v (low[t]=%.2f %s sl=%.2f)\n",
		boolStr(triggers.IntrabarSL),
		triggers.LowT,
		map[bool]string{true: "<=", false: ">"}[true],
		triggers.SLEntry)
	triggerSection += fmt.Sprintf("    tp_hit_intrabar: %v (high[t]=%.2f %s tp=%.2f)\n",
		boolStr(triggers.IntrabarTP),
		triggers.HighT,
		map[bool]string{true: ">=", false: "<"}[true],
		triggers.TPEntry)

	// Exit rule check
	triggerSection += fmt.Sprintf("\n  Exit rule check (evaluated at bar t):\n")
	triggerSection += fmt.Sprintf("    exit_rule: %v\n", boolStr(triggers.ExitRule))
	triggerSection += fmt.Sprintf("    max_hold: %v (held %d bars %s max=%d)\n",
		boolStr(triggers.MaxHold),
		triggers.HoldBars,
		map[bool]string{true: ">=", false: "<"}[true],
		triggers.MaxHoldBars)

	// Precedence decision
	triggerSection += "\n  === Precedence Decision ===\n"
	triggerSection += fmt.Sprintf("  Final reason: %s\n", reasonStr)

	// Price details
	triggerSection += "\n  === Price Details ===\n"
	triggerSection += fmt.Sprintf("  Entry Price:  %.2f\n", triggers.EntryPrice)
	triggerSection += fmt.Sprintf("  Exit Price:   %.2f\n", triggers.ExitPrice)
	triggerSection += fmt.Sprintf("  Close[t]:     %.2f\n", triggers.CloseT)
	triggerSection += fmt.Sprintf("  Open[t+1]:    %.2f\n", triggers.OpenT1)
	triggerSection += fmt.Sprintf("  High[t]:      %.2f\n", triggers.HighT)
	triggerSection += fmt.Sprintf("  Low[t]:       %.2f\n", triggers.LowT)
	triggerSection += "\n  SL/TP Levels:\n"
	triggerSection += fmt.Sprintf("  SL at entry:  %.2f (%.2f%% below entry)\n",
		triggers.SLEntry, (triggers.EntryPrice-triggers.SLEntry)/triggers.EntryPrice*100)
	triggerSection += fmt.Sprintf("  TP at entry:  %.2f (%.2f%% above entry)\n",
		triggers.TPEntry, (triggers.TPEntry-triggers.EntryPrice)/triggers.EntryPrice*100)
	triggerSection += fmt.Sprintf("  SL at exit:   %.2f\n", triggers.SLEntry)
	triggerSection += fmt.Sprintf("  TP at exit:   %.2f\n", triggers.TPEntry)

	fmt.Printf("%s\n%s  [EXT ]  EXIT (t close + t+1 open)\nEval Timestamp: %s (bar %d)\nExec Timestamp: %s (bar %d)\nReason:       %s\nExit Price:    %.6f (incl. slippage/fee)\nPnL:          %s%s\n%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, evalTs.UTC().Format("2006-01-02 15:04:05.000Z")), evalBar,
		C(gray, execTs.UTC().Format("2006-01-02 15:04:05.000Z")), execBar,
		reasonStr,
		exitPrice,
		pnl,
		triggerSection,
		eventSep,
	)
}

// LogHoldingBlock - holding heartbeat event (optional, prints every N bars)
// bar: current bar index
// ts: timestamp of current bar
// currentPnL: current unrealized PnL percentage
// stopPrice: current stop loss price
// tpPrice: current take profit price
// trailPrice: current trailing stop price (0 if not active)
// holdBars: number of bars held so far
func LogHoldingBlock(bar int, ts time.Time, currentPnL float64, stopPrice, tpPrice, trailPrice float64, holdBars int) {
	trailLine := ""
	if trailPrice > 0 {
		trailLine = fmt.Sprintf("Trail Stop:  %.6f\n", trailPrice)
	}
	pnlStr := fmt.Sprintf("%.2f%%", currentPnL)
	if currentPnL > 0 {
		pnlStr = C(green, pnlStr)
	} else if currentPnL < 0 {
		pnlStr = C(red, pnlStr)
	}
	fmt.Printf("%s\n%s  [HLD ]  HOLDING (heartbeat)\nTimestamp:    %s\nBar Index:    %d (held %d bars)\nCurrent PnL:  %s\nStop Loss:    %.6f\nTake Profit:  %.6f\n%s%s\n",
		eventSep,
		C(cyan, time.Now().UTC().Format("15:04:05.000Z")),
		C(gray, ts.UTC().Format("2006-01-02 15:04:05.000Z")),
		bar, holdBars,
		pnlStr,
		stopPrice,
		tpPrice,
		trailLine,
		eventSep,
	)
}

// LogValidationReject - validation rejection event block
func LogValidationReject(score, ret, pf, exp, dd float32, trades int, reason string) {
	fmt.Printf("%s\n%s  %s  %s\n",
		eventSep,
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("VAL "),
		C(bold, "VALIDATION REJECT"),
	)
	fmt.Printf("  score=%.4f  ret=%.2f%%  pf=%.2f  exp=%.5f  dd=%.3f  trades=%d\n",
		score, ret*100, pf, exp, dd, trades,
	)
	fmt.Printf("  reason: %s\n%s\n", reason, eventSep)
}

// LogRejectionStatsHeader - rejection statistics header
func LogRejectionStatsHeader() {
	fmt.Printf("\n%s  %s  REJECTION STATISTICS\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("VAL "),
	)
}

// LogRejectionStatsSummary - rejection statistics summary
func LogRejectionStatsSummary(totalEval, totalRejected, strategiesPassed int64) {
	fmt.Printf("Total Evaluated: %d\n", totalEval)
	fmt.Printf("Strategies Passed: %d\n", strategiesPassed)
	fmt.Printf("Total Rejected: %d\n", totalRejected)
}

// LogRejectionStatsScreen - screen stage failures
func LogRejectionStatsScreen(entryRateLow, entryRateHigh, trades, tooManyTrades, dd int64, totalRejected int64) {
	if totalRejected > 0 {
		fmt.Printf("\nScreen Stage Failures:\n")
		fmt.Printf("  Entry rate too low (<3 edges): %d (%.1f%%)\n", entryRateLow, float64(entryRateLow)*100/float64(totalRejected))
		fmt.Printf("  Entry rate too high (>120 edges): %d (%.1f%%)\n", entryRateHigh, float64(entryRateHigh)*100/float64(totalRejected))
		fmt.Printf("  Not enough trades: %d (%.1f%%)\n", trades, float64(trades)*100/float64(totalRejected))
		fmt.Printf("  Too many trades (>2000): %d (%.1f%%)\n", tooManyTrades, float64(tooManyTrades)*100/float64(totalRejected))
		fmt.Printf("  Drawdown too high: %d (%.1f%%)\n", dd, float64(dd)*100/float64(totalRejected))
	}
}

// LogRejectionStatsTrain - train stage failures
func LogRejectionStatsTrain(trades, tooManyTrades, dd int64, totalRejected int64) {
	if totalRejected > 0 {
		fmt.Printf("\nTrain Stage Failures:\n")
		fmt.Printf("  Not enough trades: %d (%.1f%%)\n", trades, float64(trades)*100/float64(totalRejected))
		fmt.Printf("  Too many trades (>5000): %d (%.1f%%)\n", tooManyTrades, float64(tooManyTrades)*100/float64(totalRejected))
		fmt.Printf("  Drawdown too high: %d (%.1f%%)\n", dd, float64(dd)*100/float64(totalRejected))
	}
}

// LogRejectionStatsFooter - rejection statistics footer
func LogRejectionStatsFooter(zeroTrades, crossSanityMutation, crossSanitySurrogate, crossSanityLoad int64) {
	fmt.Printf("\nZero-Trade Strategies: %d\n", zeroTrades)
	totalCrossSanity := crossSanityMutation + crossSanitySurrogate + crossSanityLoad
	if totalCrossSanity > 0 {
		fmt.Printf("\nCross Sanity Check Rejections:\n")
		fmt.Printf("  rejected_sanity_cross_mutation: %d\n", crossSanityMutation)
		fmt.Printf("  rejected_sanity_cross_surrogate: %d\n", crossSanitySurrogate)
		fmt.Printf("  rejected_sanity_cross_load: %d\n", crossSanityLoad)
		fmt.Printf("  TOTAL: %d\n", totalCrossSanity)
	}
	fmt.Printf("===========================\n\n")
}

// LogAutoAdjust - automatic adjustment message
func LogAutoAdjust(paramName string, oldVal, newVal float32) {
	fmt.Printf("%s  %s  AUTO-ADJUST: %s %.4f->%.4f\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		paramName, oldVal, newVal,
	)
}

// LogAutoAdjustInt - automatic adjustment message for int values
func LogAutoAdjustInt(paramName string, oldVal, newVal int) {
	fmt.Printf("%s  %s  AUTO-ADJUST: %s %d->%d\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		paramName, oldVal, newVal,
	)
}

// LogHOFRestore - HOF restore message
func LogHOFRestore(validCount, totalCount, droppedCount int) {
	if droppedCount > 0 {
		fmt.Printf("%s  %s  HOF restored: %d/%d valid (%d dropped due to feature map mismatch)\n",
			C(gray, time.Now().UTC().Format("15:04:05Z")),
			Channel("VAL "),
			validCount, totalCount, droppedCount,
		)
	}
}

// LogArchiveRestore - archive restore message
func LogArchiveRestore(validCount, totalCount, droppedCount int) {
	if droppedCount > 0 {
		fmt.Printf("%s  %s  Archive restored: %d/%d valid (%d dropped due to feature map mismatch)\n",
			C(gray, time.Now().UTC().Format("15:04:05Z")),
			Channel("VAL "),
			validCount, totalCount, droppedCount,
		)
	}
}
