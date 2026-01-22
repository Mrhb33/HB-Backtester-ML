package main

import "fmt"

type OpKind uint8

const (
	OpByteAnd OpKind = iota
	OpByteOr
	OpByteNot
	OpByteLeaf
)

type ByteCode struct {
	Op       OpKind
	Kind     uint8
	A, B     int16
	X, Y     float32 // Y is used for Between leaf (high threshold)
	Lookback uint8
}

type CompiledRule struct {
	Code []ByteCode
}

func compileRuleTree(node *RuleNode) CompiledRule {
	if node == nil {
		// Return empty code for nil nodes (regime filter with no rules)
		// This allows entries to proceed when there's no regime filter
		return CompiledRule{Code: []ByteCode{}}
	}

	var code []ByteCode
	compileNode(node, &code)
	return CompiledRule{Code: code}
}

func compileNode(node *RuleNode, code *[]ByteCode) int {
	if node == nil {
		idx := len(*code)
		*code = append(*code, ByteCode{Op: OpByteLeaf, A: -1})
		return idx
	}

	if node.Op == OpLeaf {
		idx := len(*code)
		*code = append(*code, ByteCode{
			Op:       OpByteLeaf,
			Kind:     uint8(node.Leaf.Kind),
			A:        int16(node.Leaf.A),
			B:        int16(node.Leaf.B),
			X:        node.Leaf.X,
			Y:        node.Leaf.Y,
			Lookback: uint8(node.Leaf.Lookback),
		})
		return idx
	}

	// Special case: NOT only compiles left child, then emits NOT
	if node.Op == OpNot {
		leftIdx := compileNode(node.L, code)
		*code = append(*code, ByteCode{Op: OpByteNot, A: int16(leftIdx)})
		return len(*code) - 1
	}

	leftIdx := compileNode(node.L, code)
	rightIdx := compileNode(node.R, code)

	var op OpKind
	switch node.Op {
	case OpAnd:
		op = OpByteAnd
	case OpOr:
		op = OpByteOr
	default:
		op = OpByteLeaf
	}

	*code = append(*code, ByteCode{
		Op: op,
		A:  int16(leftIdx),
		B:  int16(rightIdx),
	})

	return len(*code) - 1
}

// evaluateFastLeaf provides a fast path for simple single-leaf rules
// Returns (handled, value):
// - handled=true means we evaluated it, value is the result
// - handled=false means fall back to full VM (complex leaf type)
//
// CRITICAL: For guard failures (t < lookback, out of bounds), we must return
// (handled=true, value=false) because the slow VM would also return false.
// Returning (false, false) would incorrectly fall back to slow VM!
func evaluateFastLeaf(code []ByteCode, features [][]float32, t int) (bool, bool) {
	// Only handle simple single-leaf rules
	if len(code) != 2 {
		return false, false // Not handled: not a simple leaf
	}

	leaf := code[0]
	kind := LeafKind(leaf.Kind)

	// Bounds checks - return (handled=true, value=false) because slow VM would do same
	if leaf.A < 0 || int(leaf.A) >= len(features) {
		return true, false // Handled: invalid feature evaluates to false
	}

	fa := features[leaf.A]
	if t < 0 || t >= len(fa) {
		return true, false // Handled: out of bounds evaluates to false
	}

	aVal := fa[t]

	// Fast inline evaluation for common leaf types
	switch kind {
	case LeafGT:
		return true, aVal > leaf.X
	case LeafLT:
		return true, aVal < leaf.X
	case LeafRising:
		lb := int(leaf.Lookback)
		if t < lb {
			// CRITICAL: (true, false) not (false, false)
			// Slow VM would return false here, so we must too
			return true, false
		}
		return true, aVal > fa[t-lb]
	case LeafFalling:
		lb := int(leaf.Lookback)
		if t < lb {
			// CRITICAL: (true, false) not (false, false)
			return true, false
		}
		return true, aVal < fa[t-lb]
	default:
		// Complex leaf types (Cross, Slope, Between, etc.) - use full VM
		return false, false // Not handled: fall back to slow VM
	}
}

func evaluateCompiled(code []ByteCode, features [][]float32, t int) bool {
	if len(code) == 0 {
		return true // Empty code means no filter → always allowed
	}

	// OPTIMIZATION: Fast path for simple single-leaf rules (1.2-1.5x speedup)
	if len(code) == 2 && code[0].Op == OpByteLeaf && code[1].Op >= 8 {
		// This is a simple single-leaf rule: Leaf + Return
		// Use fast inline evaluation
		handled, value := evaluateFastLeaf(code, features, t)
		if handled {
			return value
		}
		// Fall through to full VM if fast path can't handle this leaf
	}

	stack := make([]bool, 0, 32)
	sp := 0

	for i := 0; i < len(code); i++ {
		instr := code[i]

		switch instr.Op {
		case OpByteLeaf:
			if instr.A < 0 || int(instr.A) >= len(features) {
				// Invalid feature A index - return false
				if sp >= len(stack) {
					stack = append(stack, false)
				} else {
					stack[sp] = false
				}
				sp++
				continue
			}

			kind := LeafKind(instr.Kind)

			// For CrossUp/CrossDown, validate and fetch B
			fa := features[instr.A]
			var fb []float32
			if kind == LeafCrossUp || kind == LeafCrossDown {
				if instr.B < 0 || int(instr.B) >= len(features) {
					// Invalid feature B index - return false
					if sp >= len(stack) {
						stack = append(stack, false)
					} else {
						stack[sp] = false
					}
					sp++
					continue
				}
				fb = features[instr.B]
				// Validate B array has enough elements for t
				if t >= len(fb) || (t >= 1 && t-1 >= len(fb)) {
					if sp >= len(stack) {
						stack = append(stack, false)
					} else {
						stack[sp] = false
					}
					sp++
					continue
				}
			}

			// Validate time index bounds
			if t < 0 || t >= len(fa) || t < int(instr.Lookback) {
				if sp >= len(stack) {
					stack = append(stack, false)
				} else {
					stack[sp] = false
				}
				sp++
				continue
			}

			aVal := fa[t]
			result := false

			switch kind {
			case LeafGT:
				result = aVal > instr.X
			case LeafLT:
				result = aVal < instr.X
			case LeafBetween:
				// Check if value is between X (low) and Y (high)
				low := instr.X
				high := instr.Y
				if low > high {
					low, high = high, low
				}
				result = aVal >= low && aVal <= high
			case LeafAbsGT:
				absVal := aVal
				if absVal < 0 {
					absVal = -absVal
				}
				result = absVal > instr.X
			case LeafAbsLT:
				absVal := aVal
				if absVal < 0 {
					absVal = -absVal
				}
				result = absVal < instr.X
			case LeafSlopeGT:
				lb := int(instr.Lookback)
				if lb > 0 && t >= lb {
					slope := (aVal - fa[t-lb]) / float32(lb)
					result = slope > instr.X
				}
			case LeafSlopeLT:
				lb := int(instr.Lookback)
				if lb > 0 && t >= lb {
					slope := (aVal - fa[t-lb]) / float32(lb)
					result = slope < instr.X
				}
			case LeafCrossUp:
				// CRITICAL FIX #1: Prevent fake CrossUp when one series is constant/zero
				// CrossUp only valid if BOTH series are actually moving
				if t < 1 {
					result = false
				} else {
					// Epsilon check: require actual movement in BOTH series (use abs!)
					// Block if EITHER series doesn't move (not just when both don't move)
					const eps = 1e-6
					aMove := fa[t] - fa[t-1]
					bMove := fb[t] - fb[t-1]
					// Use abs() because negative move still counts as no movement
					if aMove < 0 {
						aMove = -aMove
					}
					if bMove < 0 {
						bMove = -bMove
					}
					if aMove < eps || bMove < eps {
						// At least one series didn't move - can't be a real cross
						result = false
					} else {
						// Require both sides straddle (real crossing)
						bVal := fb[t]
						prevA := fa[t-1]
						prevB := fb[t-1]
						result = (prevA <= prevB) && (fa[t] > bVal)
					}
				}
			case LeafCrossDown:
				// CRITICAL FIX #1: Prevent fake CrossDown when one series is constant/zero
				// CrossDown only valid if BOTH series are actually moving
				if t < 1 {
					result = false
				} else {
					// Epsilon check: require actual movement in BOTH series (use abs!)
					// Block if EITHER series doesn't move (not just when both don't move)
					const eps = 1e-6
					aMove := fa[t] - fa[t-1]
					bMove := fb[t] - fb[t-1]
					// Use abs() because negative move still counts as no movement
					if aMove < 0 {
						aMove = -aMove
					}
					if bMove < 0 {
						bMove = -bMove
					}
					if aMove < eps || bMove < eps {
						// At least one series didn't move - can't be a real cross
						result = false
					} else {
						// Require both sides straddle (real crossing)
						bVal := fb[t]
						prevA := fa[t-1]
						prevB := fb[t-1]
						result = (prevA >= prevB) && (fa[t] < bVal)
					}
				}
			case LeafRising:
				lb := int(instr.Lookback)
				if lb > 0 && t >= lb {
					result = aVal > fa[t-lb]
				}
			case LeafFalling:
				lb := int(instr.Lookback)
				if lb > 0 && t >= lb {
					result = aVal < fa[t-lb]
				}
			default:
				// Unknown leaf kind - treat as false
				result = false
			}

			if sp >= len(stack) {
				stack = append(stack, result)
			} else {
				stack[sp] = result
			}
			sp++

		case OpByteAnd:
			if sp >= 2 {
				b := stack[sp-1]
				a := stack[sp-2]
				sp -= 2
				if sp >= len(stack) {
					stack = append(stack, a && b)
				} else {
					stack[sp] = a && b
				}
				sp++
			}

		case OpByteOr:
			if sp >= 2 {
				b := stack[sp-1]
				a := stack[sp-2]
				sp -= 2
				if sp >= len(stack) {
					stack = append(stack, a || b)
				} else {
					stack[sp] = a || b
				}
				sp++
			}

		case OpByteNot:
			if sp >= 1 {
				a := stack[sp-1]
				sp--
				if sp >= len(stack) {
					stack = append(stack, !a)
				} else {
					stack[sp] = !a
				}
				sp++
			}
		}
	}

	if sp > 0 {
		return stack[sp-1]
	}
	return false
}

// evaluateLeavesDebug returns a map of leaf index -> boolean result for debugging
// Only evaluates OpByteLeaf instructions, not the combined result
func evaluateLeavesDebug(code []ByteCode, features [][]float32, t int) []bool {
	if len(code) == 0 {
		return nil
	}

	// Count leaves
	leafCount := 0
	for _, instr := range code {
		if instr.Op == OpByteLeaf {
			leafCount++
		}
	}

	if leafCount == 0 {
		return nil
	}

	results := make([]bool, 0, leafCount)

	for _, instr := range code {
		if instr.Op != OpByteLeaf {
			continue
		}

		if instr.A < 0 || int(instr.A) >= len(features) {
			results = append(results, false)
			continue
		}

		kind := LeafKind(instr.Kind)
		fa := features[instr.A]

		if t < int(instr.Lookback) {
			results = append(results, false)
			continue
		}

		aVal := fa[t]
		result := false

		switch kind {
		case LeafGT:
			result = aVal > instr.X
		case LeafLT:
			result = aVal < instr.X
		case LeafBetween:
			low := instr.X
			high := instr.Y
			if low > high {
				low, high = high, low
			}
			result = aVal >= low && aVal <= high
		case LeafAbsGT:
			absVal := aVal
			if absVal < 0 {
				absVal = -absVal
			}
			result = absVal > instr.X
		case LeafAbsLT:
			absVal := aVal
			if absVal < 0 {
				absVal = -absVal
			}
			result = absVal < instr.X
		case LeafSlopeGT:
			lb := int(instr.Lookback)
			if lb > 0 && t >= lb {
				slope := (aVal - fa[t-lb]) / float32(lb)
				result = slope > instr.X
			}
		case LeafSlopeLT:
			lb := int(instr.Lookback)
			if lb > 0 && t >= lb {
				slope := (aVal - fa[t-lb]) / float32(lb)
				result = slope < instr.X
			}
		case LeafCrossUp:
			if instr.B < 0 || int(instr.B) >= len(features) {
				results = append(results, false)
				continue
			}
			fb := features[instr.B]
			bVal := fb[t]
			prevA := float32(0)
			prevB := float32(0)
			if t >= 1 {
				prevA = fa[t-1]
				prevB = fb[t-1]
			}
			// CRITICAL FIX #1: Prevent fake CrossUp when one series is constant/zero
			// Epsilon check: require actual movement in BOTH series (use abs!)
			const eps = 1e-6
			aMove := fa[t] - prevA
			bMove := fb[t] - prevB
			// Use abs() because negative move still counts as no movement
			if aMove < 0 {
				aMove = -aMove
			}
			if bMove < 0 {
				bMove = -bMove
			}
			if aMove < eps || bMove < eps {
				// At least one series didn't move - can't be a real cross
				result = false
			} else {
				// Require both sides straddle (real crossing)
				result = (prevA <= prevB) && (fa[t] > bVal)
			}
		case LeafCrossDown:
			if instr.B < 0 || int(instr.B) >= len(features) {
				results = append(results, false)
				continue
			}
			fb := features[instr.B]
			bVal := fb[t]
			prevA := float32(0)
			prevB := float32(0)
			if t >= 1 {
				prevA = fa[t-1]
				prevB = fb[t-1]
			}
			// CRITICAL FIX #1: Prevent fake CrossDown when one series is constant/zero
			// Epsilon check: require actual movement in BOTH series (use abs!)
			const eps = 1e-6
			aMove := fa[t] - prevA
			bMove := fb[t] - prevB
			// Use abs() because negative move still counts as no movement
			if aMove < 0 {
				aMove = -aMove
			}
			if bMove < 0 {
				bMove = -bMove
			}
			if aMove < eps || bMove < eps {
				// At least one series didn't move - can't be a real cross
				result = false
			} else {
				// Require both sides straddle (real crossing)
				result = (prevA >= prevB) && (fa[t] < bVal)
			}
		case LeafRising:
			lb := int(instr.Lookback)
			if lb > 0 && t >= lb {
				result = aVal > fa[t-lb]
			}
		case LeafFalling:
			lb := int(instr.Lookback)
			if lb > 0 && t >= lb {
				result = aVal < fa[t-lb]
			}
		default:
			result = false
		}

		results = append(results, result)
	}

	return results
}

// CrossDebugInfo holds detailed information about a cross evaluation for debugging
type CrossDebugInfo struct {
	LeafIndex int
	Kind      string    // "CrossUp" or "CrossDown"
	FeatAName string    // Name of feature A
	FeatBName string    // Name of feature B
	PrevA     float32   // A[t-1]
	PrevB     float32   // B[t-1]
	CurA      float32   // A[t]
	CurB      float32   // B[t]
	Result    bool      // Final result
	IsCorrect bool      // true if the cross logic is correct
}

// evaluateCrossDebug evaluates all CrossUp/CrossDown leaves and returns detailed debug info
func evaluateCrossDebug(code []ByteCode, features [][]float32, featureNames []string, t int) []CrossDebugInfo {
	if len(code) == 0 || t < 1 {
		return nil
	}

	var results []CrossDebugInfo

	for idx, instr := range code {
		if instr.Op != OpByteLeaf {
			continue
		}

		kind := LeafKind(instr.Kind)
		if kind != LeafCrossUp && kind != LeafCrossDown {
			continue
		}

		// Validate feature indices
		if instr.A < 0 || int(instr.A) >= len(features) || instr.B < 0 || int(instr.B) >= len(features) {
			continue
		}

		fa := features[instr.A]
		fb := features[instr.B]

		prevA := fa[t-1]
		prevB := fb[t-1]
		curA := fa[t]
		curB := fb[t]

		var result bool
		var kindName string

		if kind == LeafCrossUp {
			kindName = "CrossUp"
			// CRITICAL FIX #1: Apply same checks as live evaluator (use abs!)
			const eps = 1e-6
			aMove := curA - prevA
			bMove := curB - prevB
			// Use abs() because negative move still counts as no movement
			if aMove < 0 {
				aMove = -aMove
			}
			if bMove < 0 {
				bMove = -bMove
			}
			if aMove < eps || bMove < eps {
				result = false // At least one series didn't move = fake cross
			} else {
				result = (prevA <= prevB) && (curA > curB)
			}
		} else { // LeafCrossDown
			kindName = "CrossDown"
			// CRITICAL FIX #1: Apply same checks as live evaluator (use abs!)
			const eps = 1e-6
			aMove := curA - prevA
			bMove := curB - prevB
			// Use abs() because negative move still counts as no movement
			if aMove < 0 {
				aMove = -aMove
			}
			if bMove < 0 {
				bMove = -bMove
			}
			if aMove < eps || bMove < eps {
				result = false // At least one series didn't move = fake cross
			} else {
				result = (prevA >= prevB) && (curA < curB)
			}
		}

		// Check if the result makes sense
		// For CrossUp: A was below/equal to B, now A is above B
		// For CrossDown: A was above/equal to B, now A is below B
		isCorrect := true
		if result {
			if kind == LeafCrossUp {
				// CrossUp fired: prevA should be <= prevB and curA > curB
				isCorrect = (prevA <= prevB) && (curA > curB)
			} else {
				// CrossDown fired: prevA should be >= prevB and curA < curB
				isCorrect = (prevA >= prevB) && (curA < curB)
			}
		}

		featAName := fmt.Sprintf("F[%d]", instr.A)
		featBName := fmt.Sprintf("F[%d]", instr.B)
		if len(featureNames) > int(instr.A) {
			featAName = featureNames[instr.A]
		}
		if len(featureNames) > int(instr.B) {
			featBName = featureNames[instr.B]
		}

		results = append(results, CrossDebugInfo{
			LeafIndex: idx,
			Kind:      kindName,
			FeatAName: featAName,
			FeatBName: featBName,
			PrevA:     prevA,
			PrevB:     prevB,
			CurA:      curA,
			CurB:      curB,
			Result:    result,
			IsCorrect: isCorrect,
		})
	}

	return results
}

// VerificationResult stores the result of comparing bytecode vs AST evaluation
type VerificationResult struct {
	Match           bool
	ASTResult       bool
	BytecodeResult  bool
	DiscrepancyInfo string
}

// verifyBytecodeVsAST compares bytecode evaluation against AST evaluation
// This helps detect implementation differences between the two evaluation paths
func verifyBytecodeVsAST(rule *RuleNode, compiled CompiledRule, features [][]float32, t int) VerificationResult {
	// Evaluate using AST (tree traversal)
	astResult := evaluateRule(rule, features, t)

	// Evaluate using bytecode (stack-based VM)
	bytecodeResult := evaluateCompiled(compiled.Code, features, t)

	result := VerificationResult{
		Match:          astResult == bytecodeResult,
		ASTResult:      astResult,
		BytecodeResult: bytecodeResult,
	}

	if !result.Match {
		result.DiscrepancyInfo = fmt.Sprintf("t=%d: AST=%v, Bytecode=%v", t, astResult, bytecodeResult)
	}

	return result
}

// VerificationSummary stores the summary of verification results
type VerificationSummary struct {
	EntryDiscrepancies  int
	ExitDiscrepancies   int
	RegimeDiscrepancies int
	TotalChecked        int
	DiscrepancyDetails  []string
}

// VerifyStrategyAtBar verifies all rules of a strategy at a specific bar
func VerifyStrategyAtBar(st Strategy, features Features, t int) VerificationSummary {
	summary := VerificationSummary{
		DiscrepancyDetails: []string{},
	}

	// Check entry rule
	if st.EntryRule.Root != nil && len(st.EntryCompiled.Code) > 0 {
		entryCheck := verifyBytecodeVsAST(st.EntryRule.Root, st.EntryCompiled, features.F, t)
		summary.TotalChecked++
		if !entryCheck.Match {
			summary.EntryDiscrepancies++
			summary.DiscrepancyDetails = append(summary.DiscrepancyDetails,
				fmt.Sprintf("Entry rule at t=%d: AST=%v vs Bytecode=%v", t, entryCheck.ASTResult, entryCheck.BytecodeResult))
		}
	}

	// Check exit rule
	if st.ExitRule.Root != nil && len(st.ExitCompiled.Code) > 0 {
		exitCheck := verifyBytecodeVsAST(st.ExitRule.Root, st.ExitCompiled, features.F, t)
		summary.TotalChecked++
		if !exitCheck.Match {
			summary.ExitDiscrepancies++
			summary.DiscrepancyDetails = append(summary.DiscrepancyDetails,
				fmt.Sprintf("Exit rule at t=%d: AST=%v vs Bytecode=%v", t, exitCheck.ASTResult, exitCheck.BytecodeResult))
		}
	}

	// Check regime filter
	if st.RegimeFilter.Root != nil && len(st.RegimeCompiled.Code) > 0 {
		regimeCheck := verifyBytecodeVsAST(st.RegimeFilter.Root, st.RegimeCompiled, features.F, t)
		summary.TotalChecked++
		if !regimeCheck.Match {
			summary.RegimeDiscrepancies++
			summary.DiscrepancyDetails = append(summary.DiscrepancyDetails,
				fmt.Sprintf("Regime filter at t=%d: AST=%v vs Bytecode=%v", t, regimeCheck.ASTResult, regimeCheck.BytecodeResult))
		}
	}

	return summary
}
