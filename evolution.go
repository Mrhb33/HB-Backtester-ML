package main

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
)

type Elite struct {
	Strat     Strategy
	Train     Result
	Val       Result
	ValScore  float32
	IsPreElite bool // True if added via recovery mode (not fully validated)
}

type HallOfFame struct {
	mu       sync.RWMutex
	K        int
	Elites   []Elite
	snapshot atomic.Value // Stores []Elite for lock-free reads
}

func NewHallOfFame(k int) *HallOfFame {
	return &HallOfFame{K: k}
}

func (h *HallOfFame) Add(e Elite) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Use CoarseFingerprint for diversity tracking (threshold buckets)
	// This allows more threshold variation than SkeletonFingerprint while preventing duplicates
	fingerprint := e.Strat.CoarseFingerprint()

	// Count elites with this fingerprint and track worst one
	fpCount := 0
	worstIndex := -1
	worstScore := e.ValScore

	for i, elite := range h.Elites {
		if elite.Strat.CoarseFingerprint() == fingerprint {
			fpCount++
			if elite.ValScore < worstScore {
				worstScore = elite.ValScore
				worstIndex = i
			}
		}
	}

	// If fingerprint is already at cap (5 elites), replace worst if new is better
	// EDIT #3: Changed from 2 to 5 - allows more diversity per fingerprint family
	if fpCount >= 5 {
		if e.ValScore <= worstScore || worstIndex == -1 {
			// DEBUG: Log why elite was rejected
			fmt.Printf("[HOF-REJECT] fpCount=%d newScore=%.4f <= worstScore=%.4f\n",
				fpCount, e.ValScore, worstScore)
			// Not better than worst or no fingerprint found (shouldn't happen), reject
			return
		}
		// Replace the worst elite of this fingerprint family
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

	// Update atomic snapshot AFTER modification
	snapshot := make([]Elite, len(h.Elites))
	copy(snapshot, h.Elites)
	h.snapshot.Store(snapshot)
}

func (h *HallOfFame) Len() int {
	if elites := h.snapshot.Load(); elites != nil {
		return len(elites.([]Elite))
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.Elites)
}

// tournament selection (fast + good) - now lock-free
func (h *HallOfFame) Sample(rng *rand.Rand) (Elite, bool) {
	// Lock-free read from atomic snapshot
	elitesInterface := h.snapshot.Load()
	var elites []Elite

	if elitesInterface != nil {
		elites = elitesInterface.([]Elite)
	} else {
		// Fallback for initialization
		h.mu.RLock()
		defer h.mu.RUnlock()
		elites = h.Elites
	}

	n := len(elites)
	if n == 0 {
		return Elite{}, false
	}
	k := 4
	best := elites[rng.Intn(n)]
	for i := 1; i < k; i++ {
		c := elites[rng.Intn(n)]
		if c.ValScore > best.ValScore {
			best = c
		}
	}
	return best, true
}

// InitSnapshot initializes the atomic snapshot after loading elites from checkpoint
// This MUST be called after loading elites into HOF to enable lock-free reads
func (h *HallOfFame) InitSnapshot() {
	h.mu.Lock()
	defer h.mu.Unlock()
	snapshot := make([]Elite, len(h.Elites))
	copy(snapshot, h.Elites)
	h.snapshot.Store(snapshot)
}

// minInt returns the smaller of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SeedFromCandidates adds the best N candidates as elites when HOF is empty
// This is an emergency recovery mechanism when population collapses
// CHANGES SEARCH DYNAMICS: Results won't be comparable to runs without seeding
func (h *HallOfFame) SeedFromCandidates(candidates []Elite, maxSeeds int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.Elites) > 0 {
		return // Already has elites, don't seed
	}

	// Sort candidates by validation score
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].ValScore > candidates[j].ValScore
	})

	// CRITICAL FIX: Use h.Add() instead of direct append to respect fingerprint diversity caps
	// This prevents seeding 5 identical fingerprints which would block progress
	seedCount := minInt(maxSeeds, len(candidates))
	seeded := 0
	for i := 0; i < seedCount; i++ {
		// Try to add via normal Add() path which enforces fingerprint diversity
		beforeLen := len(h.Elites)
		h.Add(candidates[i])
		if len(h.Elites) > beforeLen {
			seeded++
		}
	}

	// Update atomic snapshot
	snapshot := make([]Elite, len(h.Elites))
	copy(snapshot, h.Elites)
	h.snapshot.Store(snapshot)

	fmt.Printf("[EMERGENCY-SEED] Seeded %d/%d elites from candidates (HOF was empty)\n", seeded, seedCount)
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
				// FIX #2: Larger mutations for more diversity (1.0-4.0 sigma, was 0.5-2.0)
				sigmaChange := float32(rng.NormFloat64()) * stats.Std * (1.0 + rng.Float32()*3.0)
				leaf.X += sigmaChange

				// Clamp to reasonable range (mean ± 6 std, was 5)
				lower := stats.Mean - 6*stats.Std
				upper := stats.Mean + 6*stats.Std
				if leaf.X < lower {
					leaf.X = lower
				}
				if leaf.X > upper {
					leaf.X = upper
				}
			} else {
				// Fallback for zero-variance features
				leaf.X += float32(rng.NormFloat64() * 0.2)  // FIX #2: Increased from 0.1
			}
		}
	case 1: // change feature A
		leaf.A = rng.Intn(len(feats.F))
		// SANITY CHECK: For cross leaves, ensure B is compatible with new A
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
			// Get type of new feature A
			var typeA FeatureType
			if leaf.A < len(feats.Types) {
				typeA = feats.Types[leaf.A]
			}
			// Find a compatible B
			maxAttempts := 50
			found := false
			for attempt := 0; attempt < maxAttempts; attempt++ {
				leaf.B = rng.Intn(len(feats.F))
				if leaf.B == leaf.A {
					continue
				}
				if leaf.B < len(feats.Types) {
					typeB := feats.Types[leaf.B]
					if canCrossFeatures(typeA, typeB) {
						found = true
						break // Found compatible B
					}
				}
			}
			// FIX: If no compatible B found, fall back to Rising
			if !found {
				leaf.Kind = LeafRising
				leaf.Lookback = 2 + rng.Intn(9) // Reduced from 5+16 to 2-10 for more trades
			}
		}
	case 2: // change kind
		kinds := []LeafKind{LeafGT, LeafLT, LeafCrossUp, LeafCrossDown, LeafRising, LeafFalling}
		leaf.Kind = kinds[rng.Intn(len(kinds))]
		// SANITY CHECK: For new cross leaves, ensure A and B are compatible
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
			if leaf.A >= len(feats.F) {
				leaf.A = rng.Intn(len(feats.F))
			}
			var typeA FeatureType
			if leaf.A < len(feats.Types) {
				typeA = feats.Types[leaf.A]
			}
			// Find a compatible B
			maxAttempts := 50
			found := false
			for attempt := 0; attempt < maxAttempts; attempt++ {
				leaf.B = rng.Intn(len(feats.F))
				if leaf.B == leaf.A {
					continue
				}
				if leaf.B < len(feats.Types) {
					typeB := feats.Types[leaf.B]
					if canCrossFeatures(typeA, typeB) {
						found = true
						break // Found compatible B
					}
				}
			}
			// FIX: If no compatible B found, fall back to Rising
			if !found {
				leaf.Kind = LeafRising
				leaf.Lookback = 2 + rng.Intn(9) // Reduced from 5+16 to 2-10 for more trades
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

// sanitizeCrossOperations walks the tree and fixes invalid CrossUp/CrossDown operations
// Returns the number of invalid cross operations found and fixed
func sanitizeCrossOperations(rng *rand.Rand, root *RuleNode, feats Features) int {
	if root == nil {
		return 0
	}

	fixedCount := 0

	var walk func(node *RuleNode)
	walk = func(node *RuleNode) {
		if node == nil {
			return
		}

		if node.Op == OpLeaf {
			leaf := &node.Leaf
			// Check CrossUp and CrossDown leaves
			if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
				// Validate feature indices
				if leaf.A < 0 || leaf.A >= len(feats.Types) || leaf.B < 0 || leaf.B >= len(feats.Types) {
					// Invalid indices - change to a safe leaf kind (Rising/Falling)
					leaf.Kind = LeafRising
					leaf.Lookback = 2 + rng.Intn(6) // Reduced from 5+10 to 2-7 for more trades
					fixedCount++
					return
				}

				typeA := feats.Types[leaf.A]
				typeB := feats.Types[leaf.B]

				// Check if types are compatible
				if !canCrossFeatures(typeA, typeB) {
					// Try to find a compatible B for feature A
					maxAttempts := 30
					found := false
					for attempt := 0; attempt < maxAttempts; attempt++ {
						newB := rng.Intn(len(feats.Types))
						if newB == leaf.A {
							continue
						}
						if canCrossFeatures(typeA, feats.Types[newB]) {
							leaf.B = newB
							found = true
							break
						}
					}

					if !found {
						// Couldn't find compatible B - change to a safe leaf kind
						leaf.Kind = LeafRising
						leaf.Lookback = 2 + rng.Intn(6) // Reduced from 5+10 to 2-7 for more trades
					}
					fixedCount++
				}
			}
			return
		}

		// Recurse into children
		walk(node.L)
		walk(node.R)
	}

	walk(root)
	return fixedCount
}

func mutateStrategy(rng *rand.Rand, parent Strategy, feats Features) Strategy {
	child := parent
	child.Seed = rng.Int63()

	// clone trees so we don't mutate the parent in HOF
	child.EntryRule.Root = cloneRule(parent.EntryRule.Root)
	child.ExitRule.Root = cloneRule(parent.ExitRule.Root)
	child.RegimeFilter.Root = cloneRule(parent.RegimeFilter.Root)

	// FIX #2: mutate 2–4 parts (was 1–3) for more diversity
	for i := 0; i < 2+rng.Intn(3); i++ {
		switch rng.Intn(6) {
		case 0:
			child.EntryRule.Root = mutateRuleTree(rng, child.EntryRule.Root, feats, 4)
		case 1:
			child.ExitRule.Root = mutateRuleTree(rng, child.ExitRule.Root, feats, 3)
		case 2:
			child.RegimeFilter.Root = mutateRuleTree(rng, child.RegimeFilter.Root, feats, 2)
		case 3:
			// SL tweak - FIX #2: Larger steps for more diversity
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 1.2)  // FIX #2: Increased from 0.3
				if child.StopLoss.ATRMult < 4.0 {
					child.StopLoss.ATRMult = 4.0  // FIX #2: Raised floor to match minimum
				}
				if child.StopLoss.ATRMult > 14.0 {  // FIX #2: Raised ceiling
					child.StopLoss.ATRMult = 14.0
				}
			}
		case 4:
			// TP tweak - FIX #2: Larger steps for more diversity
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 2.0)  // FIX #2: Increased from 0.4
				if child.TakeProfit.ATRMult < 8.0 {  // FIX #2: Raised floor
					child.TakeProfit.ATRMult = 8.0
				}
				if child.TakeProfit.ATRMult > 30.0 {  // FIX #2: Raised ceiling
					child.TakeProfit.ATRMult = 30.0
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
					child.Trail.ATRMult = 3.5 // Use looser default (was 2.5)
				}
				if child.Trail.Kind == "atr" && child.Trail.ATRMult <= 0 {
					// Ensure valid ATR multiplier with looser default
					child.Trail.ATRMult = 3.5 // Use looser default (was 2.5)
				}
			}
		}
	}

	// SANITY CHECK: Fix any invalid CrossUp/CrossDown operations after mutation
	// This prevents nonsense operations from surviving mutations
	fixedEntry := sanitizeCrossOperations(rng, child.EntryRule.Root, feats)
	fixedExit := sanitizeCrossOperations(rng, child.ExitRule.Root, feats)
	fixedRegime := sanitizeCrossOperations(rng, child.RegimeFilter.Root, feats)

	// Track total fixes for debugging
	totalFixed := fixedEntry + fixedExit + fixedRegime
	if totalFixed > 0 {
		// Increment counter for debugging
		// (Note: we don't reject here, we fix - but tracking helps identify issues)
	}

	// Recompile rules after mutation and sanitization
	child.EntryCompiled = compileRuleTree(child.EntryRule.Root)
	child.ExitCompiled = compileRuleTree(child.ExitRule.Root)
	child.RegimeCompiled = compileRuleTree(child.RegimeFilter.Root)

	return child
}

func crossover(rng *rand.Rand, a, b Strategy, feats Features) Strategy {
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

	// SANITY CHECK: Fix any invalid CrossUp/CrossDown operations after crossover
	// This prevents nonsense operations from surviving crossover
	sanitizeCrossOperations(rng, child.EntryRule.Root, feats)
	sanitizeCrossOperations(rng, child.ExitRule.Root, feats)
	sanitizeCrossOperations(rng, child.RegimeFilter.Root, feats)

	// Recompile rules after crossover and sanitization
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
			// Big SL tweak - FIX #2: Larger jumps with proper minimums
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 3.5)  // FIX #2: Increased from 2.5
				if child.StopLoss.ATRMult < 4.0 {  // FIX #2: Raised floor to 4.0
					child.StopLoss.ATRMult = 4.0
				}
				if child.StopLoss.ATRMult > 14.0 {  // FIX #2: Raised ceiling
					child.StopLoss.ATRMult = 14.0
				}
			} else if child.StopLoss.Kind == "fixed" {
				child.StopLoss.Value += float32(rng.NormFloat64() * 4.0)  // FIX #2: Increased from 3.0
				if child.StopLoss.Value < 4.0 {  // FIX #2: Raised floor to 4.0%
					child.StopLoss.Value = 4.0
				}
				if child.StopLoss.Value > 20.0 {
					child.StopLoss.Value = 20.0
				}
			}
		case 4:
			// Big TP tweak - FIX #2: Larger jumps with proper minimums
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 4.0)  // FIX #2: Increased from 3.0
				if child.TakeProfit.ATRMult < 8.0 {  // FIX #2: Raised floor to 8.0
					child.TakeProfit.ATRMult = 8.0
				}
				if child.TakeProfit.ATRMult > 30.0 {  // FIX #2: Raised ceiling
					child.TakeProfit.ATRMult = 30.0
				}
			} else if child.TakeProfit.Kind == "fixed" {
				child.TakeProfit.Value += float32(rng.NormFloat64() * 8.0)  // FIX #2: Increased from 7.0
				if child.TakeProfit.Value < 10.0 {  // FIX #2: Raised floor to 10%
					child.TakeProfit.Value = 10.0
				}
				if child.TakeProfit.Value > 50.0 {
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

	// SANITY CHECK: Fix any invalid CrossUp/CrossDown operations after big mutation
	// This prevents nonsense operations from surviving big mutations
	sanitizeCrossOperations(rng, child.EntryRule.Root, feats)
	sanitizeCrossOperations(rng, child.ExitRule.Root, feats)
	sanitizeCrossOperations(rng, child.RegimeFilter.Root, feats)

	// Recompile rules after big mutation and sanitization
	child.EntryCompiled = compileRuleTree(child.EntryRule.Root)
	child.ExitCompiled = compileRuleTree(child.ExitRule.Root)
	child.RegimeCompiled = compileRuleTree(child.RegimeFilter.Root)

	return child
}
