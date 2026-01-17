package main

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
