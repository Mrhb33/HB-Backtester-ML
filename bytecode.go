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

func evaluateCompiled(code []ByteCode, features [][]float32, t int) bool {
	if len(code) == 0 {
		return true // Empty code means no filter → always allowed
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
			}

			if t < int(instr.Lookback) {
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
				bVal := fb[t]
				prevA := float32(0)
				prevB := float32(0)
				if t >= 1 {
					prevA = fa[t-1]
					prevB = fb[t-1]
				}
				result = prevA <= prevB && aVal > bVal
			case LeafCrossDown:
				bVal := fb[t]
				prevA := float32(0)
				prevB := float32(0)
				if t >= 1 {
					prevA = fa[t-1]
					prevB = fb[t-1]
				}
				result = prevA >= prevB && aVal < bVal
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
			result = prevA <= prevB && aVal > bVal
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
			result = prevA >= prevB && aVal < bVal
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
			result = prevA <= prevB && curA > curB
		} else { // LeafCrossDown
			kindName = "CrossDown"
			result = prevA >= prevB && curA < curB
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
