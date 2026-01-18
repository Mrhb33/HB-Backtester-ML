package main

import (
	"math/rand"
	"sort"
	"sync"
)

type Elite struct {
	Strat    Strategy
	Train    Result
	Val      Result
	ValScore float32
}

type HallOfFame struct {
	mu     sync.RWMutex
	K      int
	Elites []Elite
}

func NewHallOfFame(k int) *HallOfFame {
	return &HallOfFame{K: k}
}

func (h *HallOfFame) Add(e Elite) {
	h.mu.Lock()
	defer h.mu.Unlock()

	skeleton := e.Strat.SkeletonFingerprint()

	// Count elites with this skeleton and track worst one
	skeletonCount := 0
	worstIndex := -1
	worstScore := e.ValScore

	for i, elite := range h.Elites {
		if elite.Strat.SkeletonFingerprint() == skeleton {
			skeletonCount++
			if elite.ValScore < worstScore {
				worstScore = elite.ValScore
				worstIndex = i
			}
		}
	}

	// If skeleton is already at cap (2 elites), replace worst if new is better
	// Reduced from 4 to 2 to force more diversity and prevent inbreeding
	if skeletonCount >= 2 {
		if e.ValScore <= worstScore || worstIndex == -1 {
			// Not better than worst or no skeleton found (shouldn't happen), reject
			return
		}
		// Replace the worst elite of this skeleton
		h.Elites[worstIndex] = e
	} else {
		// Under cap, add normally
		h.Elites = append(h.Elites, e)
	}

	// Keep sorted by score
	sort.Slice(h.Elites, func(i, j int) bool { return h.Elites[i].ValScore > h.Elites[j].ValScore })

	// Enforce overall capacity
	if len(h.Elites) > h.K {
		h.Elites = h.Elites[:h.K]
	}
}

func (h *HallOfFame) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.Elites)
}

// tournament selection (fast + good)
func (h *HallOfFame) Sample(rng *rand.Rand) (Elite, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	n := len(h.Elites)
	if n == 0 {
		return Elite{}, false
	}
	k := 4
	best := h.Elites[rng.Intn(n)]
	for i := 1; i < k; i++ {
		c := h.Elites[rng.Intn(n)]
		if c.ValScore > best.ValScore {
			best = c
		}
	}
	return best, true
}

func cloneRule(node *RuleNode) *RuleNode {
	if node == nil {
		return nil
	}
	cp := &RuleNode{Op: node.Op, Leaf: node.Leaf}
	cp.L = cloneRule(node.L)
	cp.R = cloneRule(node.R)
	return cp
}

func mutateLeaf(rng *rand.Rand, leaf *Leaf, feats Features) {
	switch rng.Intn(4) {
	case 0: // change threshold (scale-aware mutation)
		if leaf.A < len(feats.Stats) {
			stats := feats.Stats[leaf.A]
			if stats.Std > 0 {
				// Mutate by a fraction of std dev (0.5-2.0 sigma)
				sigmaChange := float32(rng.NormFloat64()) * stats.Std * (0.5 + rng.Float32()*1.5)
				leaf.X += sigmaChange

				// Clamp to reasonable range (mean ± 5 std)
				lower := stats.Mean - 5*stats.Std
				upper := stats.Mean + 5*stats.Std
				if leaf.X < lower {
					leaf.X = lower
				}
				if leaf.X > upper {
					leaf.X = upper
				}
			} else {
				// Fallback for zero-variance features
				leaf.X += float32(rng.NormFloat64() * 0.1)
			}
		}
	case 1: // change feature A
		leaf.A = rng.Intn(len(feats.F))
		// Ensure B is valid for cross leaves
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
			if leaf.B == leaf.A || leaf.B >= len(feats.F) {
				leaf.B = (leaf.A + 1) % len(feats.F)
				if leaf.B == leaf.A {
					leaf.B = (leaf.B + 1) % len(feats.F)
				}
			}
		}
	case 2: // change kind
		kinds := []LeafKind{LeafGT, LeafLT, LeafCrossUp, LeafCrossDown, LeafRising, LeafFalling}
		leaf.Kind = kinds[rng.Intn(len(kinds))]
		// Ensure B is valid for cross leaves
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
			if leaf.B == leaf.A || leaf.B >= len(feats.F) {
				leaf.B = (leaf.A + 1) % len(feats.F)
				if leaf.B == leaf.A {
					leaf.B = (leaf.B + 1) % len(feats.F)
				}
			}
		}
	case 3: // tweak lookback (skip for CrossUp/CrossDown)
		if leaf.Kind != LeafCrossUp && leaf.Kind != LeafCrossDown {
			leaf.Lookback += rng.Intn(5) - 2
			// Use minimum of 2 for Rising/Falling to avoid noise (lookback=1 is too short)
			if leaf.Lookback < 2 {
				leaf.Lookback = 2
			}
		}
	}
}

func mutateRuleTree(rng *rand.Rand, root *RuleNode, feats Features, maxDepth int) *RuleNode {
	if root == nil {
		return root
	}

	// Choose mutation type:
	// 0-3: leaf mutations (existing behavior, 40%)
	// 4: subtree replace (20%)
	// 5: op flip (10%)
	// 6: NOT insert/remove (10%)
	// 7: swap branches (10%)
	// 8: prune (10%)
	mutType := rng.Intn(9)

	switch mutType {
	case 0, 1, 2, 3: // leaf mutations (pick a random leaf)
		// walk down tree and mutate a leaf
		cur := root
		for cur.Op != OpLeaf && rng.Float32() < 0.7 {
			if rng.Float32() < 0.5 && cur.L != nil {
				cur = cur.L
			} else if cur.R != nil {
				cur = cur.R
			} else {
				break
			}
		}
		if cur.Op == OpLeaf {
			mutateLeaf(rng, &cur.Leaf, feats)
		}
	case 4: // subtree replace
		// pick a random node and replace it with new random subtree
		newSubtree := randomRuleNode(rng, feats, 0, maxDepth)
		if rng.Float32() < 0.5 {
			// replace root
			return newSubtree
		} else {
			// replace a random internal node
			return replaceRandomSubtree(rng, root, newSubtree)
		}
	case 5: // op flip (AND <-> OR)
		// pick a random internal node and flip its op
		if !applyOpFlip(rng, root) {
			// fallback: flip root
			if root.Op == OpAnd {
				root.Op = OpOr
			} else if root.Op == OpOr {
				root.Op = OpAnd
			}
		}
	case 6: // NOT insert/remove
		if rng.Float32() < 0.5 {
			// insert NOT at random internal node
			if !applyNOTInsert(rng, root) {
				// fallback: wrap root in NOT
				if root.Op != OpLeaf && root.Op != OpNot {
					oldRoot := cloneRule(root)
					return &RuleNode{Op: OpNot, L: oldRoot, R: nil}
				}
			}
		} else {
			// remove NOT from random location
			if !applyNOTRemove(rng, root) {
				// fallback: unwrap root NOT
				if root.Op == OpNot && root.L != nil {
					return root.L
				}
			}
		}
	case 7: // swap branches
		// swap at random internal node
		if !applySwapBranches(rng, root) {
			// fallback: swap root branches
			if root.L != nil && root.R != nil {
				root.L, root.R = root.R, root.L
			}
		}
	case 8: // prune - replace internal node with one of its children
		if !applyPrune(rng, root) {
			// fallback: prune root
			if root.Op != OpLeaf {
				if root.L != nil && root.R == nil {
					return root.L
				} else if root.R != nil && root.L == nil {
					return root.R
				} else if root.L != nil && root.R != nil {
					// 50/50 chance to keep left or right child
					if rng.Float32() < 0.5 {
						return root.L
					} else {
						return root.R
					}
				}
			}
		}
	}

	return root
}

// Helper: apply op flip to a random internal node
func applyOpFlip(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		if node.Op == OpAnd {
			node.Op = OpOr
			return true
		} else if node.Op == OpOr {
			node.Op = OpAnd
			return true
		}
	}

	// try children
	if applyOpFlip(rng, node.L) {
		return true
	}
	return applyOpFlip(rng, node.R)
}

// Helper: insert NOT at a random internal node
func applyNOTInsert(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf || node.Op == OpNot {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		// wrap this node's children with NOT
		if rng.Float32() < 0.5 && node.L != nil {
			oldL := node.L
			node.L = &RuleNode{Op: OpNot, L: oldL, R: nil}
			return true
		} else if node.R != nil {
			oldR := node.R
			node.R = &RuleNode{Op: OpNot, L: oldR, R: nil}
			return true
		}
	}

	// try children
	if applyNOTInsert(rng, node.L) {
		return true
	}
	return applyNOTInsert(rng, node.R)
}

// Helper: remove NOT from a random location
func applyNOTRemove(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf {
		return false
	}

	// 30% chance to apply at this node
	if node.Op == OpNot {
		// unwrap this NOT
		if node.L != nil {
			*node = *node.L
			return true
		}
	}

	// try children
	if applyNOTRemove(rng, node.L) {
		return true
	}
	return applyNOTRemove(rng, node.R)
}

// Helper: swap branches at random internal node
func applySwapBranches(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		if node.L != nil && node.R != nil {
			node.L, node.R = node.R, node.L
			return true
		}
	}

	// try children
	if applySwapBranches(rng, node.L) {
		return true
	}
	return applySwapBranches(rng, node.R)
}

// Helper: prune at random internal node
func applyPrune(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		if node.L != nil && node.R == nil {
			*node = *node.L
			return true
		} else if node.R != nil && node.L == nil {
			*node = *node.R
			return true
		} else if node.L != nil && node.R != nil {
			// 50/50 chance to keep left or right child
			if rng.Float32() < 0.5 {
				*node = *node.L
			} else {
				*node = *node.R
			}
			return true
		}
	}

	// try children
	if applyPrune(rng, node.L) {
		return true
	}
	return applyPrune(rng, node.R)
}

func replaceRandomSubtree(rng *rand.Rand, root *RuleNode, replacement *RuleNode) *RuleNode {
	if root == nil {
		return root
	}

	// 50% chance to replace this node if it's internal
	if root.Op != OpLeaf && rng.Float32() < 0.3 {
		return cloneRule(replacement) // clone to avoid shared pointers
	}

	root.L = replaceRandomSubtree(rng, root.L, replacement)
	root.R = replaceRandomSubtree(rng, root.R, replacement)
	return root
}

func mutateStrategy(rng *rand.Rand, parent Strategy, feats Features) Strategy {
	child := parent
	child.Seed = rng.Int63()

	// clone trees so we don't mutate the parent in HOF
	child.EntryRule.Root = cloneRule(parent.EntryRule.Root)
	child.ExitRule.Root = cloneRule(parent.ExitRule.Root)
	child.RegimeFilter.Root = cloneRule(parent.RegimeFilter.Root)

	// mutate 1–3 parts
	for i := 0; i < 1+rng.Intn(3); i++ {
		switch rng.Intn(6) {
		case 0:
			child.EntryRule.Root = mutateRuleTree(rng, child.EntryRule.Root, feats, 4)
		case 1:
			child.ExitRule.Root = mutateRuleTree(rng, child.ExitRule.Root, feats, 3)
		case 2:
			child.RegimeFilter.Root = mutateRuleTree(rng, child.RegimeFilter.Root, feats, 2)
		case 3:
			// SL tweak - prefer looser ATR stops
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 0.3) // More conservative
				if child.StopLoss.ATRMult < 1.0 {
					child.StopLoss.ATRMult = 1.0 // Looser floor (was 0.5)
				}
				if child.StopLoss.ATRMult > 8.0 {
					child.StopLoss.ATRMult = 8.0
				}
			}
		case 4:
			// TP tweak - prefer larger ATR TPs to let winners run
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 0.4) // More variation
				if child.TakeProfit.ATRMult < 1.0 {
					child.TakeProfit.ATRMult = 1.0 // Looser floor (was 0.5)
				}
				if child.TakeProfit.ATRMult > 15.0 {
					child.TakeProfit.ATRMult = 15.0 // Higher ceiling (was 10.0)
				}
			}
		case 5:
			// trail toggle - ensure valid kind/params when enabling
			child.Trail.Active = !child.Trail.Active
			if child.Trail.Active {
				// Ensure valid kind and params when activating trail
				if child.Trail.Kind != "atr" && child.Trail.Kind != "swing" {
					// Initialize with valid ATR trail if no valid kind
					child.Trail.Kind = "atr"
					child.Trail.ATRMult = 2.5 // Use looser default
				}
				if child.Trail.Kind == "atr" && child.Trail.ATRMult <= 0 {
					// Ensure valid ATR multiplier with looser default
					child.Trail.ATRMult = 2.5
				}
			}
		}
	}

	// Recompile rules after mutation
	child.EntryCompiled = compileRuleTree(child.EntryRule.Root)
	child.ExitCompiled = compileRuleTree(child.ExitRule.Root)
	child.RegimeCompiled = compileRuleTree(child.RegimeFilter.Root)

	return child
}

func crossover(rng *rand.Rand, a, b Strategy) Strategy {
	child := a
	child.Seed = rng.Int63()

	// mix parts
	if rng.Float32() < 0.5 {
		child.EntryRule.Root = cloneRule(a.EntryRule.Root)
	} else {
		child.EntryRule.Root = cloneRule(b.EntryRule.Root)
	}
	if rng.Float32() < 0.5 {
		child.ExitRule.Root = cloneRule(a.ExitRule.Root)
	} else {
		child.ExitRule.Root = cloneRule(b.ExitRule.Root)
	}
	if rng.Float32() < 0.5 {
		child.RegimeFilter.Root = cloneRule(a.RegimeFilter.Root)
	} else {
		child.RegimeFilter.Root = cloneRule(b.RegimeFilter.Root)
	}

	// mix risk models lightly
	if rng.Float32() < 0.5 {
		child.StopLoss = a.StopLoss
	} else {
		child.StopLoss = b.StopLoss
	}
	if rng.Float32() < 0.5 {
		child.TakeProfit = a.TakeProfit
	} else {
		child.TakeProfit = b.TakeProfit
	}
	if rng.Float32() < 0.5 {
		child.Trail = a.Trail
	} else {
		child.Trail = b.Trail
	}

	// Recompile rules after crossover
	child.EntryCompiled = compileRuleTree(child.EntryRule.Root)
	child.ExitCompiled = compileRuleTree(child.ExitRule.Root)
	child.RegimeCompiled = compileRuleTree(child.RegimeFilter.Root)

	return child
}

// bigMutation creates a radical mutation to escape local maxima
// Mutates 4-6 parts simultaneously with larger step sizes
func bigMutation(rng *rand.Rand, parent Strategy, feats Features) Strategy {
	child := parent
	child.Seed = rng.Int63()

	// clone trees so we don't mutate parent in HOF
	child.EntryRule.Root = cloneRule(parent.EntryRule.Root)
	child.ExitRule.Root = cloneRule(parent.ExitRule.Root)
	child.RegimeFilter.Root = cloneRule(parent.RegimeFilter.Root)

	// mutate 4-7 parts (more than normal 1-3, increased upper bound)
	numMutations := 4 + rng.Intn(4)
	for i := 0; i < numMutations; i++ {
		switch rng.Intn(7) { // Added new case 6
		case 0:
			child.EntryRule.Root = mutateRuleTree(rng, child.EntryRule.Root, feats, 4)
		case 1:
			child.ExitRule.Root = mutateRuleTree(rng, child.ExitRule.Root, feats, 3)
		case 2:
			child.RegimeFilter.Root = mutateRuleTree(rng, child.RegimeFilter.Root, feats, 2)
		case 3:
			// Big SL tweak - larger jumps
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 2.5) // 8x bigger step (was 1.5)
				if child.StopLoss.ATRMult < 0.5 {
					child.StopLoss.ATRMult = 0.5
				}
				if child.StopLoss.ATRMult > 12.0 { // Increased ceiling
					child.StopLoss.ATRMult = 12.0
				}
			} else if child.StopLoss.Kind == "fixed" {
				child.StopLoss.Value += float32(rng.NormFloat64() * 3.0) // Larger jumps
				if child.StopLoss.Value < 0.3 {
					child.StopLoss.Value = 0.3
				}
				if child.StopLoss.Value > 20.0 { // Increased ceiling
					child.StopLoss.Value = 20.0
				}
			}
		case 4:
			// Big TP tweak - larger jumps
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 3.0) // 7.5x bigger step (was 2.0)
				if child.TakeProfit.ATRMult < 0.5 {
					child.TakeProfit.ATRMult = 0.5
				}
				if child.TakeProfit.ATRMult > 25.0 { // Increased ceiling
					child.TakeProfit.ATRMult = 25.0
				}
			} else if child.TakeProfit.Kind == "fixed" {
				child.TakeProfit.Value += float32(rng.NormFloat64() * 7.0) // Larger jumps
				if child.TakeProfit.Value < 0.5 {
					child.TakeProfit.Value = 0.5
				}
				if child.TakeProfit.Value > 50.0 { // Increased ceiling
					child.TakeProfit.Value = 50.0
				}
			}
		case 5:
			// Randomize direction (flip long <-> short)
			if rng.Float32() < 0.3 {
				child.Direction = -child.Direction
			}
		case 6:
			// NEW: Randomize SL/TP kind (swap ATR<->fixed)
			if rng.Float32() < 0.5 {
				if child.StopLoss.Kind == "atr" {
					// Convert to fixed
					child.StopLoss.Kind = "fixed"
					child.StopLoss.Value = child.StopLoss.ATRMult * 3.0 // Rough conversion
					child.StopLoss.ATRMult = 0
				} else if child.StopLoss.Kind == "fixed" {
					// Convert to ATR
					child.StopLoss.Kind = "atr"
					child.StopLoss.ATRMult = child.StopLoss.Value / 3.0 // Rough conversion
					child.StopLoss.Value = 0
				}
			} else {
				if child.TakeProfit.Kind == "atr" {
					// Convert to fixed
					child.TakeProfit.Kind = "fixed"
					child.TakeProfit.Value = child.TakeProfit.ATRMult * 3.0
					child.TakeProfit.ATRMult = 0
				} else if child.TakeProfit.Kind == "fixed" {
					// Convert to ATR
					child.TakeProfit.Kind = "atr"
					child.TakeProfit.ATRMult = child.TakeProfit.Value / 3.0
					child.TakeProfit.Value = 0
				}
			}
		}
	}

	// Recompile rules after big mutation
	child.EntryCompiled = compileRuleTree(child.EntryRule.Root)
	child.ExitCompiled = compileRuleTree(child.ExitRule.Root)
	child.RegimeCompiled = compileRuleTree(child.RegimeFilter.Root)

	return child
}
