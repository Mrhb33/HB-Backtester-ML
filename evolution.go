package main

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"

	"hb_bactest_checker/logx"
)

// MaxElitesPerFingerprint is the maximum number of elites allowed per fingerprint family
const MaxElitesPerFingerprint = 5

type Elite struct {
	Strat      Strategy
	Train      Result
	Val        Result
	ValScore   float32
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

// Add adds an elite to the Hall of Fame with proper locking.
func (h *HallOfFame) Add(e Elite) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.addLocked(e)
}

// addLocked adds an elite assuming the lock is already held.
// This is an internal helper for Add() and SeedFromCandidates().
func (h *HallOfFame) addLocked(e Elite) {
	// CRITICAL FIX: Reject elites with invalid scores (sentinel corruption)
	// Sentinel -1e30 means uninitialized/overflow score - these should not be in HallOfFame
	if IsSentinelScore(e.ValScore) {
		if Verbose {
			fmt.Printf("[HOF-REJECT] Invalid score=%s - rejecting corrupted elite\n", CleanScoreForDisplay(e.ValScore))
		}
		return
	}

	// CRITICAL FIX: Reject candidates that don't improve the pool when full
	// Once elite pool is full, only admit strategies better than the worst elite
	// This prevents rejecting improving candidates with negative scores
	poolFull := len(h.Elites) >= h.K
	if poolFull {
		worstOverallScore := float32(0.0)
		if len(h.Elites) > 0 {
			worstOverallScore = h.Elites[len(h.Elites)-1].ValScore
		}
		// Only reject if not better than worst (not if <= 0)
		if e.ValScore <= worstOverallScore {
			if Verbose {
				fmt.Printf("[HOF-REJECT] Pool full (%d/%d), newScore=%.4f <= worstScore=%.4f\n",
					len(h.Elites), h.K, e.ValScore, worstOverallScore)
			}
			return
		}
	}

	// Use CoarseFingerprint for diversity tracking (threshold buckets)
	// This allows more threshold variation than SkeletonFingerprint while preventing duplicates
	fingerprint := e.Strat.CoarseFingerprint()

	// Count elites with this fingerprint and track worst one
	fpCount := 0
	worstIndex := -1
	worstScore := e.ValScore

	// Track previous best score for event logging
	prevBestScore := float32(0)
	if len(h.Elites) > 0 {
		prevBestScore = h.Elites[0].ValScore
	}

	for i, elite := range h.Elites {
		if elite.Strat.CoarseFingerprint() == fingerprint {
			fpCount++
			if elite.ValScore < worstScore {
				worstScore = elite.ValScore
				worstIndex = i
			}
		}
	}

	// If fingerprint is already at cap, replace worst if new is better
	// EDIT #3: Changed from 2 to MaxElitesPerFingerprint - allows more diversity per fingerprint family
	if fpCount >= MaxElitesPerFingerprint {
		if e.ValScore <= worstScore || worstIndex == -1 {
			// DEBUG: Log why elite was rejected
			if Verbose {
				fmt.Printf("[HOF-REJECT] fpCount=%d newScore=%.4f <= worstScore=%.4f\n",
					fpCount, e.ValScore, worstScore)
			}
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

	// Log elite added event to TUI
	logx.LogEliteAdded(e.ValScore, len(h.Elites))

	// Log if best score improved
	if len(h.Elites) > 0 && h.Elites[0].ValScore > prevBestScore {
		logx.LogNewBestScore(prevBestScore, h.Elites[0].ValScore)
	}
}

func (h *HallOfFame) Len() int {
	if elites := h.snapshot.Load(); elites != nil {
		elitesSlice, ok := elites.([]Elite)
		if ok {
			return len(elitesSlice)
		}
		// Type mismatch - fall through to locked read
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
		var ok bool
		elites, ok = elitesInterface.([]Elite)
		if !ok {
			// Type mismatch - fallback to locked read
			h.mu.RLock()
			defer h.mu.RUnlock()
			elites = h.Elites
		}
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
// CRITICAL FIX: Hold lock throughout and use addLocked() to prevent deadlock
// The outer lock protects all h.Elites access and prevents race conditions
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

	// CRITICAL FIX: Use h.addLocked() instead of h.Add() to avoid deadlock
	// addLocked() assumes lock is already held, avoiding nested lock acquisition
	seedCount := minInt(maxSeeds, len(candidates))
	seeded := 0
	for i := 0; i < seedCount; i++ {
		// Try to add via addLocked() path which enforces fingerprint diversity
		beforeLen := len(h.Elites)
		h.addLocked(candidates[i])
		if len(h.Elites) > beforeLen {
			seeded++
		}
	}

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
				leaf.X += float32(rng.NormFloat64() * 0.2) // FIX #2: Increased from 0.1
			}
		}
	case 1: // change feature A
		leaf.A = rng.Intn(len(feats.F))
		// SANITY CHECK: For cross leaves, ensure B is compatible with new A
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
			// Get type of new feature A
			var typeA FeatureType
			if leaf.A < len(feats.Types) {
				typeA = feats.Types[leaf.A]
			}
			// FIX: If A is EventFlag, can't cross - switch to GT/LT immediately
			if typeA == FeatTypeEventFlag {
				gtLtKinds := []LeafKind{LeafGT, LeafLT}
				leaf.Kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
			} else {
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
				// FIX: If no compatible B found, fall back to GT/LT (not Rising)
				if !found {
					gtLtKinds := []LeafKind{LeafGT, LeafLT}
					leaf.Kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
				}
			}
		}
	case 2: // change kind
		kinds := []LeafKind{LeafGT, LeafLT, LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown, LeafRising, LeafFalling}
		leaf.Kind = kinds[rng.Intn(len(kinds))]
		// SANITY CHECK: For new cross leaves, ensure A and B are compatible
		if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
			if leaf.A >= len(feats.F) {
				leaf.A = rng.Intn(len(feats.F))
			}
			var typeA FeatureType
			if leaf.A < len(feats.Types) {
				typeA = feats.Types[leaf.A]
			}
			// FIX: If A is EventFlag, can't cross - switch to GT/LT immediately
			if typeA == FeatTypeEventFlag {
				gtLtKinds := []LeafKind{LeafGT, LeafLT}
				leaf.Kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
			} else {
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
				// FIX: If no compatible B found, fall back to GT/LT (not Rising)
				if !found {
					gtLtKinds := []LeafKind{LeafGT, LeafLT}
					leaf.Kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
				}
			}
		}
	case 3: // tweak lookback (skip for CrossUp/CrossDown/BreakUp/BreakDown)
		if leaf.Kind != LeafCrossUp && leaf.Kind != LeafCrossDown && leaf.Kind != LeafBreakUp && leaf.Kind != LeafBreakDown {
			leaf.Lookback += rng.Intn(5) - 2
			// Use minimum of 2 for Rising/Falling to avoid noise (lookback=1 is too short)
			if leaf.Lookback < 2 {
				leaf.Lookback = 2
			}
		}
	}
}

func mutateRuleTree(rng *rand.Rand, root *RuleNode, feats Features, maxDepth int, ruleType string) *RuleNode {
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
	// PROBLEM A FIX: 9: simplify/remove constraint (NEW - for rate-aware mutation)
	mutType := rng.Intn(10)

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
		var newSubtree *RuleNode
		if ruleType == "exit" {
			newSubtree = randomExitRuleNode(rng, feats, 0, maxDepth)
		} else {
			newSubtree = randomRuleNode(rng, feats, 0, maxDepth)
		}
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
				// CRITICAL FIX #2: Skip if root contains rare event leaves
				if root.Op != OpLeaf && root.Op != OpNot && !containsRareEventLeaf(root) {
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
	case 9: // PROBLEM A FIX: simplify/remove constraint to increase entry rate
		// For AND nodes, replace with one child (removes constraint)
		// For NOT(AND), remove NOT to simplify
		if !applySimplifyMutation(rng, root) {
			// fallback: prune root to one child
			if root.Op == OpAnd && root.L != nil && root.R != nil {
				// Keep the simpler child (fewer nodes)
				if countNodes(root.L) <= countNodes(root.R) {
					return root.L
				} else {
					return root.R
				}
			} else if root.Op == OpNot && root.L != nil {
				return root.L
			}
		}
	}

	// Simplify the tree by removing redundant patterns after mutation
	// This cleans up things like (A AND A) -> A, (A AND True) -> A, NOT(NOT(A)) -> A
	root = pruneRedundantNodes(root)
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

// Helper: check if a rule tree contains rare event leaves (Cross/Break)
// These create "always true" when wrapped in NOT, producing 0 edges
func containsRareEventLeaf(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.Op == OpLeaf {
		// Check if this is a rare event leaf
		switch node.Leaf.Kind {
		case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
			return true
		}
		return false
	}
	// Recursively check children
	return containsRareEventLeaf(node.L) || containsRareEventLeaf(node.R)
}

// Helper: insert NOT at a random internal node
// CRITICAL FIX #2: Skip NOT on rare event leaves to prevent "always true" rules
func applyNOTInsert(rng *rand.Rand, node *RuleNode) bool {
	if node == nil || node.Op == OpLeaf || node.Op == OpNot {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		// wrap this node's children with NOT
		if rng.Float32() < 0.5 && node.L != nil {
			// Skip if left child contains rare event leaves
			if !containsRareEventLeaf(node.L) {
				oldL := node.L
				node.L = &RuleNode{Op: OpNot, L: oldL, R: nil}
				return true
			}
		} else if node.R != nil {
			// Skip if right child contains rare event leaves
			if !containsRareEventLeaf(node.R) {
				oldR := node.R
				node.R = &RuleNode{Op: OpNot, L: oldR, R: nil}
				return true
			}
		}
	}

	// try children
	if applyNOTInsert(rng, node.L) {
		return true
	}
	return applyNOTInsert(rng, node.R)
}

// Helper: add a new constraint by AND-ing a new leaf into the tree
func applyAddConstraint(rng *rand.Rand, node *RuleNode, newLeaf Leaf) bool {
	if node == nil {
		return false
	}

	// 30% chance to apply at this node
	if rng.Float32() < 0.3 {
		newNode := &RuleNode{Op: OpLeaf, Leaf: newLeaf}
		if node.Op == OpLeaf {
			// Wrap leaf with AND (old leaf AND new leaf)
			old := cloneRule(node)
			*node = RuleNode{Op: OpAnd, L: old, R: newNode}
			return true
		}
		if node.L == nil {
			node.L = newNode
			return true
		}
		if node.R == nil {
			node.R = newNode
			return true
		}
		// AND the new leaf into a random side to add a constraint
		if rng.Float32() < 0.5 {
			old := cloneRule(node.L)
			node.L = &RuleNode{Op: OpAnd, L: old, R: newNode}
		} else {
			old := cloneRule(node.R)
			node.R = &RuleNode{Op: OpAnd, L: old, R: newNode}
		}
		return true
	}

	// try children
	if applyAddConstraint(rng, node.L, newLeaf) {
		return true
	}
	return applyAddConstraint(rng, node.R, newLeaf)
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
			// Check CrossUp, CrossDown, BreakUp, and BreakDown leaves
			if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
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

	// Balanced mutations: 2-4 parts with more useful complexity exploration
	for i := 0; i < 2+rng.Intn(3); i++ {
		// Balanced distribution:
		// 0-19: 20% - Replace leaf/parameter (entry)
		// 20-29: 10% - Replace leaf/parameter (exit)
		// 30-39: 10% - Replace leaf/parameter (regime)
		// 40-49: 10% - SL tweak
		// 50-59: 10% - TP tweak
		// 60-69: 10% - Replace subtree (same complexity)
		// 70-79: 10% - Op flip (same complexity)
		// 80-89: 10% - ADD constraint (increase complexity)
		// 90-99: 10% - DELETE subtree (reduce complexity)
		mutType := rng.Intn(100)

		switch mutType {
		case 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
			10, 11, 12, 13, 14, 15, 16, 17, 18, 19: // 20% - Replace leaf/parameter (entry)
			child.EntryRule.Root = mutateRuleTree(rng, child.EntryRule.Root, feats, 4, "entry")
		case 20, 21, 22, 23, 24, 25, 26, 27, 28, 29: // 10% - Replace leaf/parameter (exit)
			child.ExitRule.Root = mutateRuleTree(rng, child.ExitRule.Root, feats, 3, "exit")
		case 30, 31, 32, 33, 34, 35, 36, 37, 38, 39: // 10% - Replace leaf/parameter (regime)
			child.RegimeFilter.Root = mutateRuleTree(rng, child.RegimeFilter.Root, feats, 2, "regime")
		case 40, 41, 42, 43, 44, 45, 46, 47, 48, 49: // 10% - SL tweak
			// SL tweak - Larger steps for more diversity
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 1.2)
				if child.StopLoss.ATRMult < 4.0 {
					child.StopLoss.ATRMult = 4.0
				}
				if child.StopLoss.ATRMult > 10.0 { // CRITICAL FIX: Lowered ceiling to prevent high DD
					child.StopLoss.ATRMult = 10.0
				}
			}
		case 50, 51, 52, 53, 54, 55, 56, 57, 58, 59: // 10% - TP tweak
			// TP tweak - Larger steps for more diversity
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 2.0)
				if child.TakeProfit.ATRMult < 8.0 {
					child.TakeProfit.ATRMult = 8.0
				}
				if child.TakeProfit.ATRMult > 20.0 { // CRITICAL FIX: Lowered ceiling to prevent extreme R:R
					child.TakeProfit.ATRMult = 20.0 // FIX: Was 30.0, matches comment
				}
			}
		case 60, 61, 62, 63, 64, 65, 66, 67, 68, 69: // 10% - Replace subtree (same complexity)
			// Replace random subtree with new random tree
			// CRITICAL FIX: Use exit-specific subtree for exit rules to enforce meaningful exit signals
			roll := rng.Float32()
			if roll < 0.3 {
				newSubtree := randomRuleNode(rng, feats, 0, 3)
				child.EntryRule.Root = replaceRandomSubtree(rng, child.EntryRule.Root, newSubtree)
			} else if roll < 0.6 {
				newSubtree := randomExitRuleNode(rng, feats, 0, 3)
				child.ExitRule.Root = replaceRandomSubtree(rng, child.ExitRule.Root, newSubtree)
			} else {
				newSubtree := randomRuleNode(rng, feats, 0, 3)
				child.RegimeFilter.Root = replaceRandomSubtree(rng, child.RegimeFilter.Root, newSubtree)
			}
		case 70, 71, 72, 73, 74, 75, 76, 77, 78, 79: // 10% - Op flip (same complexity)
			// Flip AND <-> OR at random internal node
			if !applyOpFlip(rng, child.EntryRule.Root) {
				if child.EntryRule.Root != nil && child.EntryRule.Root.Op == OpAnd {
					child.EntryRule.Root.Op = OpOr
				} else if child.EntryRule.Root != nil && child.EntryRule.Root.Op == OpOr {
					child.EntryRule.Root.Op = OpAnd
				}
			}
		case 80, 81, 82, 83, 84, 85, 86, 87, 88, 89: // 10% - ADD constraint (increase complexity)
			newLeaf := randomEntryLeaf(rng, feats, true)
			if child.EntryRule.Root == nil {
				child.EntryRule.Root = &RuleNode{Op: OpLeaf, Leaf: newLeaf}
			} else {
				applyAddConstraint(rng, child.EntryRule.Root, newLeaf)
			}
		case 90, 91, 92, 93, 94, 95, 96, 97, 98, 99: // 10% - DELETE subtree (reduce complexity)
			// Prune entry rule subtree
			child.EntryRule.Root = pruneRandomSubtree(rng, child.EntryRule.Root)
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

	// CRITICAL: Validate and repair entry rule after mutation
	// Mutation can produce empty entry rules (e.g., prune deletes all nodes)
	// This ensures every strategy has a valid entry rule that can open positions
	if !validateAndRepairEntryRule(rng, &child, feats) {
		// Repair failed - this strategy is broken, return parent instead
		// (Don't waste population slots on dead strategies)
		return parent
	}

	// Ensure regime filter remains non-empty and compilable
	ensureRegimeFilterValid(rng, &child, feats)

	// PROBLEM 10 FIX: Fold-aware hold/cooldown sampling during mutation
	// Instead of clamping after inheritance, sample within fold constraints
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	barsPerDay := int(1440.0 / float64(tfMinutes))
	typicalFoldBars := barsPerDay * 90 // 90-day fold

	// Sample hold_bars based on fold size: max = testBars/10
	maxHoldBars := typicalFoldBars / 10
	if maxHoldBars < 20 {
		maxHoldBars = 20
	}
	if maxHoldBars > 120 {
		maxHoldBars = 120
	}
	// Sample within 25% to 100% of max (more variation than initial generation)
	child.MaxHoldBars = (maxHoldBars / 4) + rng.Intn(maxHoldBars*3/4+1)

	// Sample cooldown_bars based on fold size: max = testBars/30
	maxCooldownBars := typicalFoldBars / 30
	if maxCooldownBars < 10 {
		maxCooldownBars = 10
	}
	if maxCooldownBars > 48 {
		maxCooldownBars = 48
	}
	child.CooldownBars = (maxCooldownBars / 4) + rng.Intn(maxCooldownBars*3/4+1)

	return child
}

// pruneRandomSubtree removes a random internal node and replaces it with one of its children
// This reduces tree complexity by removing a subtree
func pruneRandomSubtree(rng *rand.Rand, root *RuleNode) *RuleNode {
	if root == nil {
		return nil
	}

	// Collect all internal nodes (non-leaf)
	var internalNodes []*RuleNode
	var collectInternal func(node *RuleNode)
	collectInternal = func(node *RuleNode) {
		if node == nil {
			return
		}
		if node.Op != OpLeaf {
			internalNodes = append(internalNodes, node)
			collectInternal(node.L)
			collectInternal(node.R)
		}
	}
	collectInternal(root)

	if len(internalNodes) == 0 {
		// No internal nodes to prune, return as is
		return root
	}

	// Pick random internal node to prune
	target := internalNodes[rng.Intn(len(internalNodes))]

	// Replace with one of its children (prefer left if both exist)
	if target.L != nil && target.R == nil {
		*target = *target.L
	} else if target.R != nil && target.L == nil {
		*target = *target.R
	} else if target.L != nil && target.R != nil {
		if rng.Float32() < 0.5 {
			*target = *target.L
		} else {
			*target = *target.R
		}
	}

	return root
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

	// CRITICAL: Validate and repair entry rule after crossover
	// Crossover can produce empty entry rules if parent had nil entry
	if !validateAndRepairEntryRule(rng, &child, feats) {
		// Repair failed - return parent A as fallback
		return a
	}

	// Ensure regime filter remains non-empty and compilable
	ensureRegimeFilterValid(rng, &child, feats)

	// PROBLEM 10 FIX: Fold-aware hold/cooldown sampling during crossover
	// Same logic as mutation to prevent clamped duplicates
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	barsPerDay := int(1440.0 / float64(tfMinutes))
	typicalFoldBars := barsPerDay * 90 // 90-day fold

	// Sample hold_bars based on fold size: max = testBars/10
	maxHoldBars := typicalFoldBars / 10
	if maxHoldBars < 20 {
		maxHoldBars = 20
	}
	if maxHoldBars > 120 {
		maxHoldBars = 120
	}
	child.MaxHoldBars = (maxHoldBars / 4) + rng.Intn(maxHoldBars*3/4+1)

	// Sample cooldown_bars based on fold size: max = testBars/30
	maxCooldownBars := typicalFoldBars / 30
	if maxCooldownBars < 10 {
		maxCooldownBars = 10
	}
	if maxCooldownBars > 48 {
		maxCooldownBars = 48
	}
	child.CooldownBars = (maxCooldownBars / 4) + rng.Intn(maxCooldownBars*3/4+1)

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
			child.EntryRule.Root = mutateRuleTree(rng, child.EntryRule.Root, feats, 4, "entry")
		case 1:
			child.ExitRule.Root = mutateRuleTree(rng, child.ExitRule.Root, feats, 3, "exit")
		case 2:
			child.RegimeFilter.Root = mutateRuleTree(rng, child.RegimeFilter.Root, feats, 2, "regime")
		case 3:
			// Big SL tweak - FIX #2: Larger jumps with proper minimums
			// CRITICAL FIX: Capped SL ATRMult to [4.0, 10.0] to prevent 60%+ DD monsters
			if child.StopLoss.Kind == "atr" {
				child.StopLoss.ATRMult += float32(rng.NormFloat64() * 3.5) // FIX #2: Increased from 2.5
				if child.StopLoss.ATRMult < 4.0 {                          // FIX #2: Raised floor to 4.0
					child.StopLoss.ATRMult = 4.0
				}
				if child.StopLoss.ATRMult > 10.0 { // CRITICAL FIX: Lowered ceiling to prevent high DD
					child.StopLoss.ATRMult = 10.0
				}
			} else if child.StopLoss.Kind == "fixed" {
				child.StopLoss.Value += float32(rng.NormFloat64() * 4.0) // FIX #2: Increased from 3.0
				if child.StopLoss.Value < 4.0 {                          // FIX #2: Raised floor to 4.0%
					child.StopLoss.Value = 4.0
				}
				if child.StopLoss.Value > 10.0 { // CRITICAL FIX: Lowered max from 20% to 10% to reduce DD
					child.StopLoss.Value = 10.0
				}
			}
		case 4:
			// Big TP tweak - FIX #2: Larger jumps with proper minimums
			// CRITICAL FIX: Capped TP ATRMult to [8.0, 20.0] for better R:R ratios
			if child.TakeProfit.Kind == "atr" {
				child.TakeProfit.ATRMult += float32(rng.NormFloat64() * 4.0) // FIX #2: Increased from 3.0
				if child.TakeProfit.ATRMult < 8.0 {                          // FIX #2: Raised floor to 8.0
					child.TakeProfit.ATRMult = 8.0
				}
				if child.TakeProfit.ATRMult > 20.0 { // CRITICAL FIX: Lowered ceiling to prevent extreme R:R
					child.TakeProfit.ATRMult = 20.0
				}
			} else if child.TakeProfit.Kind == "fixed" {
				child.TakeProfit.Value += float32(rng.NormFloat64() * 8.0) // FIX #2: Increased from 7.0
				if child.TakeProfit.Value < 10.0 {                         // FIX #2: Raised floor to 10%
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
					// CRITICAL FIX: Clamp converted value to prevent Fixed SL > 10%
					if child.StopLoss.Value > 10.0 {
						child.StopLoss.Value = 10.0
					}
					if child.StopLoss.Value < 4.0 {
						child.StopLoss.Value = 4.0
					}
					child.StopLoss.ATRMult = 0
				} else if child.StopLoss.Kind == "fixed" {
					// Convert to ATR
					child.StopLoss.Kind = "atr"
					child.StopLoss.ATRMult = child.StopLoss.Value / 3.0 // Rough conversion
					// Clamp converted ATR to [4.0, 10.0]
					if child.StopLoss.ATRMult > 10.0 {
						child.StopLoss.ATRMult = 10.0
					}
					if child.StopLoss.ATRMult < 4.0 {
						child.StopLoss.ATRMult = 4.0
					}
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

	// CRITICAL: Validate and repair entry rule after big mutation
	// Big mutation can drastically change rules, potentially emptying entry
	if !validateAndRepairEntryRule(rng, &child, feats) {
		// Repair failed - return parent as fallback
		return parent
	}

	// Ensure regime filter remains non-empty and compilable
	ensureRegimeFilterValid(rng, &child, feats)

	// PROBLEM E FIX: Clamp MaxHoldBars and CooldownBars in bigMutation
	const maxHoldClampBig = 120
	const maxCooldownClampBig = 48
	if child.MaxHoldBars > maxHoldClampBig {
		child.MaxHoldBars = maxHoldClampBig
	}
	if child.CooldownBars > maxCooldownClampBig {
		child.CooldownBars = maxCooldownClampBig
	}

	return child
}

// PROBLEM A FIX: applySimplifyMutation removes constraints to increase entry rate
// For AND(A, B), returns either A or B (removes one constraint)
// For NOT(X), returns X (removes negation)
func applySimplifyMutation(rng *rand.Rand, root *RuleNode) bool {
	if root == nil {
		return false
	}

	// Collect all AND and NOT nodes in the tree
	type NodePath struct {
		node   **RuleNode
		parent **RuleNode
		isLeft bool
	}
	var andNodes []NodePath
	var notNodes []NodePath

	var walk func(node **RuleNode, parent **RuleNode, isLeft bool)
	walk = func(node **RuleNode, parent **RuleNode, isLeft bool) {
		if *node == nil {
			return
		}
		if (*node).Op == OpAnd {
			andNodes = append(andNodes, NodePath{node: node, parent: parent, isLeft: isLeft})
		} else if (*node).Op == OpNot {
			notNodes = append(notNodes, NodePath{node: node, parent: parent, isLeft: isLeft})
		}
		walk(&(*node).L, node, true)
		walk(&(*node).R, node, false)
	}
	walk(&root, nil, false)

	// Prefer simplifying AND nodes (removes constraints)
	if len(andNodes) > 0 && rng.Float32() < 0.7 {
		// Pick a random AND node and replace with one of its children
		target := andNodes[rng.Intn(len(andNodes))]
		andNode := *target.node
		if andNode.L != nil && andNode.R != nil {
			// Keep the simpler child (fewer nodes = higher entry rate)
			if countNodes(andNode.L) <= countNodes(andNode.R) {
				*target.node = andNode.L
			} else {
				*target.node = andNode.R
			}
			return true
		}
	}

	// Fallback: remove NOT negations
	if len(notNodes) > 0 {
		target := notNodes[rng.Intn(len(notNodes))]
		notNode := *target.node
		if notNode.L != nil {
			*target.node = notNode.L
			return true
		}
	}

	return false
}
