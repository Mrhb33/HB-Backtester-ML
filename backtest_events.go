package main

import (
	"fmt"
	"strings"
	"time"
)

// leafOpToString converts LeafKind to string for throttling
func leafOpToString(kind LeafKind) string {
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
		return "Unknown"
	}
}

// dirString converts direction int to string
func dirString(d int) string {
	if d == 1 {
		return "LONG"
	}
	return "SHORT"
}

// buildRegimeLine creates the Regime line for signal event block
// Shows raw rule string + resolved boolean result (only when debug enabled)
func buildRegimeLine(st Strategy, f Features, t int) string {
	if st.RegimeFilter.Root == nil {
		return "none=T"
	}
	result := evaluateCompiled(st.RegimeCompiled.Code, f.F, t)

	// Show rule string (compact) with result
	ruleStr := ruleTreeToStringWithNames(st.RegimeFilter.Root, f)
	if len(ruleStr) > 60 {
		ruleStr = ruleStr[:57] + "..."
	}
	return fmt.Sprintf("%s=%v", ruleStr, result)
}

// buildEntryLines creates the two Entry lines for signal event block
// Computes both regimeOk and entryOk separately - ACTION=ARMED only if both true
// Returns: (line1, line2) where line2 contains the full Result
func buildEntryLines(st Strategy, f Features, t int, busted bool, consecLosses int, maxConsecLosses int) (string, string) {
	// Evaluate regime filter
	regimeOk := st.RegimeFilter.Root == nil
	if st.RegimeCompiled.Code != nil {
		regimeOk = evaluateCompiled(st.RegimeCompiled.Code, f.F, t)
	}

	// Evaluate entry rule
	entryOk := false
	if st.EntryCompiled.Code != nil {
		entryOk = evaluateCompiled(st.EntryCompiled.Code, f.F, t)
	}

	// Build action string
	action := "ARMED (enter next bar open)"
	if busted {
		action = fmt.Sprintf("BLOCKED (busted: %d consecutive losses, MaxConsecLosses=%d)", consecLosses, maxConsecLosses)
	} else if !regimeOk {
		action = "BLOCKED (regime filter failed)"
	} else if !entryOk {
		action = "BLOCKED (no entry signal)"
	}

	// Line 1: individual component results
	// Line 2: final result/action (this will be the "Result:" line in the event block)
	line1 := fmt.Sprintf("entry_rule=%v  regime_ok=%v", entryOk, regimeOk)
	line2 := fmt.Sprintf("Result: %s", action)
	return line1, line2
}

// formatTime formats a timestamp consistently for event blocks
func formatTime(t time.Time) string {
	return t.UTC().Format("15:04:05Z")
}

// buildEntryRuleResultString builds a detailed string showing all entry rule component results
// This is for verbose debugging of entry rules
func buildEntryRuleResultString(st Strategy, f Features, t int) string {
	if st.EntryCompiled.Code == nil {
		return "no entry rule"
	}

	result := evaluateCompiled(st.EntryCompiled.Code, f.F, t)
	return fmt.Sprintf("entry_rule=%v", result)
}

// buildRegimeRuleResultString builds a detailed string showing regime filter result
func buildRegimeRuleResultString(st Strategy, f Features, t int) string {
	if st.RegimeFilter.Root == nil {
		return "no regime filter"
	}

	result := evaluateCompiled(st.RegimeCompiled.Code, f.F, t)
	return fmt.Sprintf("regime_filter=%v", result)
}

// formatExitReason formats an exit reason with additional context
func formatExitReason(reason string, trailActive bool, holdBars int, maxHoldBars int) string {
	switch reason {
	case "tp_hit":
		return "tp_hit"
	case "tp_gap_open":
		return "tp_gap_open"
	case "sl_hit":
		return "sl_hit"
	case "sl_gap_open":
		return "sl_gap_open"
	case "trail_hit":
		return "trail_hit"
	case "exit_rule":
		if trailActive {
			return "exit_rule (trailing_active)"
		}
		return "exit_rule"
	case "max_hold":
		return fmt.Sprintf("max_hold (%d bars)", holdBars)
	default:
		return reason
	}
}

// formatPnLString formats a PnL value as a string with color indicators
func formatPnLString(pnl float32) string {
	if pnl > 0 {
		return fmt.Sprintf("+%.2f%%", pnl)
	}
	return fmt.Sprintf("%.2f%%", pnl)
}

// buildTradeSummary builds a summary string for a trade
func buildTradeSummary(trade Trade) string {
	return fmt.Sprintf("%s: %s @ %.2f -> %.2f (%.2f%%), held %d bars, reason=%s",
		dirString(trade.Direction),
		formatTime(trade.EntryTime),
		trade.EntryPrice,
		trade.ExitPrice,
		trade.PnL,
		trade.HoldBars,
		trade.Reason,
	)
}

// buildSubConditionString evaluates and returns T/F results for each sub-condition
// in an OR/AND tree. Returns empty string if the rule is not an OR/AND tree.
func buildSubConditionString(st Strategy, f Features, t int) string {
	if st.EntryRule.Root == nil {
		return ""
	}

	// Only build sub-conditions for OR/AND trees (non-leaf nodes)
	if st.EntryRule.Root.Op == OpLeaf {
		return ""
	}

	var results []string
	evaluateSubConditions(st.EntryRule.Root, st, f, t, &results)

	if len(results) == 0 {
		return ""
	}
	return fmt.Sprintf("[%s]", strings.Join(results, ", "))
}

// evaluateSubConditions recursively evaluates sub-conditions and appends T/F results
// IMPORTANT: Uses bytecode evaluation to match actual execution path
func evaluateSubConditions(node *RuleNode, st Strategy, f Features, t int, results *[]string) {
	if node == nil {
		return
	}

	if node.Op == OpLeaf {
		// Evaluate this leaf using bytecode to match actual execution
		compiled := compileRuleTree(node)
		result := evaluateCompiled(compiled.Code, f.F, t)
		conditionStr := leafToStringWithNames(&node.Leaf, f)
		*results = append(*results, fmt.Sprintf("%s=%v", conditionStr, result))
		return
	}

	// Recursively evaluate left and right children
	evaluateSubConditions(node.L, st, f, t, results)
	evaluateSubConditions(node.R, st, f, t, results)
}

// buildSubConditionProofs collects detailed proofs for each sub-condition leaf
// Uses bytecode evaluation to match actual execution path
func buildSubConditionProofs(node *RuleNode, f Features, t int) []LeafProof {
	if node == nil {
		return nil
	}

	if node.Op == OpLeaf {
		// Get detailed proof with values (using AST for detail capture)
		_, proof := evaluateLeafWithProof(&node.Leaf, f.F, &f, t)

		// Override result with bytecode evaluation to match actual execution
		compiled := compileRuleTree(node)
		proof.Result = evaluateCompiled(compiled.Code, f.F, t)

		return []LeafProof{proof}
	}

	// Recursively collect proofs from both children
	leftProofs := buildSubConditionProofs(node.L, f, t)
	rightProofs := buildSubConditionProofs(node.R, f, t)

	return append(leftProofs, rightProofs...)
}

// buildSubConditionProofsThrottled collects detailed proofs with throttling per leaf kind
// Only builds proofs for the first maxCount occurrences of each leaf kind (e.g., "CrossUp", "Rising")
// This prevents excessive proof logging that slows down backtesting (76.8/s -> 51.2/s)
func buildSubConditionProofsThrottled(node *RuleNode, f Features, t int, proofCounts map[string]int, maxCount int) []LeafProof {
	if node == nil {
		return nil
	}

	if node.Op == OpLeaf {
		// Determine leaf kind for throttling (e.g., "CrossUp", "Rising", "SlopeGT")
		leafKind := leafOpToString(node.Leaf.Kind)

		// Check throttle: skip if we've already logged enough of this kind
		if proofCounts[leafKind] >= maxCount {
			return nil // Throttled: skip building proof for this leaf
		}

		// Get detailed proof with values (using AST for detail capture)
		_, proof := evaluateLeafWithProof(&node.Leaf, f.F, &f, t)

		// Override result with bytecode evaluation to match actual execution
		compiled := compileRuleTree(node)
		proof.Result = evaluateCompiled(compiled.Code, f.F, t)

		// Increment counter for this leaf kind
		proofCounts[leafKind]++

		return []LeafProof{proof}
	}

	// Recursively collect proofs from both children
	leftProofs := buildSubConditionProofsThrottled(node.L, f, t, proofCounts, maxCount)
	rightProofs := buildSubConditionProofsThrottled(node.R, f, t, proofCounts, maxCount)

	return append(leftProofs, rightProofs...)
}
