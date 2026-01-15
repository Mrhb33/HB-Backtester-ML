package main

import (
	"fmt"
	"math/rand"
)

type Op uint8

const (
	OpAnd Op = iota
	OpOr
	OpNot
	OpLeaf
)

type LeafKind uint8

const (
	LeafGT LeafKind = iota
	LeafLT
	LeafCrossUp
	LeafCrossDown
	LeafRising
	LeafFalling
	LeafBetween
	LeafAbsGT
	LeafAbsLT
	LeafZScoreGT
	LeafZScoreLT
	LeafSlopeGT
	LeafSlopeLT
)

type Leaf struct {
	Kind     LeafKind
	A        int
	B        int
	X        float32
	Y        float32 // For Between leaf (high threshold)
	Lookback int
}

type RuleNode struct {
	Op   Op
	Leaf Leaf
	L, R *RuleNode
}

type StopModel struct {
	Kind     string
	Value    float32
	ATRMult  float32
	SwingIdx int
}

type TPModel struct {
	Kind    string
	Value   float32
	ATRMult float32
}

type TrailModel struct {
	Kind    string
	ATRMult float32
	Active  bool
}

type Strategy struct {
	Seed           uint64
	FeeBps         float32
	SlippageBps    float32
	RiskPct        float32
	Direction      int
	EntryRule      RuleTree
	EntryCompiled  CompiledRule
	ExitRule       RuleTree
	ExitCompiled   CompiledRule
	StopLoss       StopModel
	TakeProfit     TPModel
	Trail          TrailModel
	RegimeFilter   RuleTree
	RegimeCompiled CompiledRule
}

// Fingerprint creates a unique string representation of the strategy (excluding seed)
// This is used for deduplication to avoid having duplicate strategies in the leaderboard
func (s Strategy) Fingerprint() string {
	return ruleTreeToString(s.EntryRule.Root) + "|" +
		ruleTreeToString(s.ExitRule.Root) + "|" +
		ruleTreeToString(s.RegimeFilter.Root) + "|" +
		stopModelToString(s.StopLoss) + "|" +
		tpModelToString(s.TakeProfit) + "|" +
		trailModelToString(s.Trail)
}

// SkeletonFingerprint creates a structure-only fingerprint (ignores numeric thresholds)
// This tracks strategy "families" - same operators + features + SL/TP type
func (s Strategy) SkeletonFingerprint() string {
	return ruleTreeToSkeleton(s.EntryRule.Root) + "|" +
		ruleTreeToSkeleton(s.ExitRule.Root) + "|" +
		ruleTreeToSkeleton(s.RegimeFilter.Root) + "|" +
		stopModelToSkeleton(s.StopLoss) + "|" +
		tpModelToSkeleton(s.TakeProfit) + "|" +
		trailModelToSkeleton(s.Trail)
}

type RuleTree struct {
	Root *RuleNode
}

func randomStrategy(rng *rand.Rand, feats Features) Strategy {
	entryRoot := randomRuleNode(rng, feats, 0, 4)
	exitRoot := randomRuleNode(rng, feats, 0, 3)

	// Regime filter: make it more likely to be non-empty to reduce noise entries
	// 85% chance of having a regime filter (increased from 70% for rarer, higher-quality entries)
	regimeRoot := randomRuleNode(rng, feats, 0, 2)
	if rng.Float32() < 0.15 {
		// 15% chance to disable regime filter (nil = always true)
		regimeRoot = nil
	}

	// Randomly set direction: 1 for long-only, -1 for short-only
	// Removed Direction=0 (both) to make scores deterministic for search/test consistency
	direction := 1 // default: long-only
	randDir := rng.Float32()
	if randDir < 0.5 {
		direction = 1 // long-only
	} else {
		direction = -1 // short-only
	}

	s := Strategy{
		Seed:           uint64(rng.Int63()),
		FeeBps:         40, // 0.4% fee per trade
		SlippageBps:    40, // 0.4% slippage per trade (total 0.8% per trade)
		RiskPct:        0.01,
		Direction:      direction,
		EntryRule:      RuleTree{Root: entryRoot},
		EntryCompiled:  compileRuleTree(entryRoot),
		ExitRule:       RuleTree{Root: exitRoot},
		ExitCompiled:   compileRuleTree(exitRoot),
		StopLoss:       randomStopModel(rng),
		TakeProfit:     randomTPModel(rng),
		Trail:          randomTrailModel(rng),
		RegimeFilter:   RuleTree{Root: regimeRoot},
		RegimeCompiled: compileRuleTree(regimeRoot),
	}
	return s
}

func randomRuleNode(rng *rand.Rand, feats Features, depth, maxDepth int) *RuleNode {
	if depth >= maxDepth || rng.Float32() < 0.3 {
		return &RuleNode{
			Op:   OpLeaf,
			Leaf: randomLeaf(rng, feats),
		}
	}

	op := OpAnd
	randOp := rng.Float32()

	// NOT has 10% probability when not at leaf level
	if randOp < 0.10 {
		op = OpNot
	} else if randOp < 0.55 {
		op = OpOr
	}

	if op == OpNot {
		// NOT has only one child (Left), Right stays empty
		return &RuleNode{
			Op: op,
			L:  randomRuleNode(rng, feats, depth+1, maxDepth),
			R:  nil,
		}
	}

	return &RuleNode{
		Op: op,
		L:  randomRuleNode(rng, feats, depth+1, maxDepth),
		R:  randomRuleNode(rng, feats, depth+1, maxDepth),
	}
}

func randomLeaf(rng *rand.Rand, feats Features) Leaf {
	// Prefer triggers (Cross, Rising, Falling) over simple thresholds to capture real moves
	// 60% trigger leaves, 30% threshold leaves, 10% slope leaves
	triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafRising, LeafFalling}
	thresholdKinds := []LeafKind{LeafGT, LeafLT, LeafBetween, LeafAbsGT, LeafAbsLT}
	slopeKinds := []LeafKind{LeafSlopeGT, LeafSlopeLT}

	var kind LeafKind
	if len(feats.F) < 2 {
		// No cross leaves possible - use threshold or rising/falling
		kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
	} else if rng.Float32() < 0.60 {
		kind = triggerKinds[rng.Intn(len(triggerKinds))]
	} else if rng.Float32() < 0.90 {
		kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
	} else {
		kind = slopeKinds[rng.Intn(len(slopeKinds))]
	}

	a := rng.Intn(len(feats.F))

	var b int
	var lookback int
	var threshold float32
	var leafY float32 // For Between leaf (high threshold)

	switch kind {
	case LeafCrossUp, LeafCrossDown:
		// Keep picking B until it's different from A
		for {
			b = rng.Intn(len(feats.F))
			if b != a {
				break
			}
		}
		lookback = 0  // Cross leaves don't use lookback
		threshold = 0 // Cross leaves don't use threshold
	case LeafRising, LeafFalling:
		// Increase lookback to capture real moves, not micro-noise
		// Range: 5-20 bars (was 3-12 before)
		lookback = rng.Intn(16) + 5
		threshold = 0 // Rising/Falling don't use threshold
	case LeafBetween:
		// Between needs two thresholds (X and Y)
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			// Generate two thresholds around mean
			k1 := rng.NormFloat64() * 1.0
			k2 := rng.NormFloat64() * 1.0
			threshold = stats.Mean + float32(k1)*stats.Std
			leafY = stats.Mean + float32(k2)*stats.Std

			// Ensure X < Y for "Between" (low < high)
			if threshold > leafY {
				threshold, leafY = leafY, threshold
			}
		} else {
			threshold = float32(rng.NormFloat64()*20 + 50)
			leafY = threshold + float32(rng.NormFloat64()*10)
			if threshold < 0 {
				threshold = 0
			}
		}
	case LeafAbsGT, LeafAbsLT:
		// Absolute value comparison
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			k := rng.NormFloat64() * 1.0
			threshold = stats.Mean + float32(k)*stats.Std
			if threshold < 0 {
				threshold = -threshold // Absolute thresholds should be positive
			}
		} else {
			threshold = float32(rng.NormFloat64()*20 + 50)
			if threshold < 0 {
				threshold = 0
			}
		}
	case LeafZScoreGT, LeafZScoreLT:
		// Z-score comparison (threshold is in standard deviations)
		b = a
		threshold = float32(rng.NormFloat64() * 1.5) // -1.5 to 1.5 std devs by default
		// Clamp to reasonable range
		if threshold < -3.0 {
			threshold = -3.0
		}
		if threshold > 3.0 {
			threshold = 3.0
		}
	case LeafSlopeGT, LeafSlopeLT:
		// Slope comparison over lookback period
		b = a
		lookback = rng.Intn(16) + 5 // 5-20 bars
		threshold = float32(rng.NormFloat64() * 0.1) // Small slope values
	default:
		// For GT/LT, use feature-specific statistics for realistic threshold
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			// Pick X around the feature's mean with random multiplier of std
			// Using mean ± k*std where k is random between -2 and 2
			k := rng.NormFloat64() * 1.0 // Normal distribution centered at 0, std=1
			threshold = stats.Mean + float32(k)*stats.Std

			// Add small bounds check to prevent extreme values (outlier protection)
			if stats.Std > 0 {
				minThreshold := stats.Mean - 3*stats.Std
				maxThreshold := stats.Mean + 3*stats.Std
				if threshold < minThreshold {
					threshold = minThreshold
				}
				if threshold > maxThreshold {
					threshold = maxThreshold
				}
			}
		} else {
			// Fallback if stats not available
			threshold = float32(rng.NormFloat64()*20 + 50)
			if threshold < 0 {
				threshold = 0
			}
			if threshold > 100 {
				threshold = 100
			}
		}
	}

	return Leaf{
		Kind:     kind,
		A:        a,
		B:        b,
		X:        threshold,
		Y:        leafY,
		Lookback: lookback,
	}
}

func randomStopModel(rng *rand.Rand) StopModel {
	// Prefer ATR-based stops (70% ATR, 20% fixed, 10% swing)
	// ATR adapts to volatility, which is better for fat trades
	kind := rng.Intn(10)
	if kind < 7 { // 70% ATR
		atr := rng.NormFloat64()*1.0 + 2.0
		if atr < 0.5 {
			atr = 0.5
		}
		if atr > 6.0 {
			atr = 6.0
		}
		return StopModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Value:   0,
		}
	} else if kind < 9 { // 20% fixed
		val := rng.NormFloat64()*1.5 + 2.0
		if val < 0.3 {
			val = 0.3
		}
		if val > 10.0 {
			val = 10.0
		}
		return StopModel{
			Kind:    "fixed",
			Value:   float32(val),
			ATRMult: 0,
		}
	} else { // 10% swing
		return StopModel{
			Kind:     "swing",
			SwingIdx: rng.Intn(5) + 3,
			Value:    0,
			ATRMult:  0,
		}
	}
}

func randomTPModel(rng *rand.Rand) TPModel {
	// Prefer ATR-based TPs (80% ATR, 20% fixed)
	// ATR adapts to volatility, letting winners run
	kind := rng.Intn(10)
	if kind < 8 { // 80% ATR
		atr := rng.NormFloat64()*2.0 + 3.0
		if atr < 0.5 {
			atr = 0.5
		}
		if atr > 10.0 {
			atr = 10.0
		}
		return TPModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Value:   0,
		}
	} else { // 20% fixed
		val := rng.NormFloat64()*3.0 + 4.0
		if val < 0.5 {
			val = 0.5
		}
		if val > 30.0 {
			val = 30.0
		}
		return TPModel{
			Kind:    "fixed",
			Value:   float32(val),
			ATRMult: 0,
		}
	}
}

func randomTrailModel(rng *rand.Rand) TrailModel {
	// Prefer ATR-based trails or no trail (70% none, 30% loose ATR)
	// Removed swing trails since they're not implemented in backtest logic
	// Avoid tight trails that stop winners too early
	kind := rng.Intn(10)
	if kind < 7 { // 70% no trail
		return TrailModel{
			Kind:    "none",
			ATRMult: 0,
			Active:  false,
		}
	} else { // 30% loose ATR trail
		atr := rng.NormFloat64()*1.0 + 2.5 // Looser: mean 2.5 (was 1.5)
		if atr < 1.5 {
			atr = 1.5
		}
		if atr > 5.0 {
			atr = 5.0
		}
		return TrailModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Active:  true,
		}
	}
}

// Skeleton helper functions - ignore numeric thresholds, keep only structure
func ruleTreeToSkeleton(node *RuleNode) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		return leafToSkeleton(&node.Leaf)
	}

	leftStr := ruleTreeToSkeleton(node.L)
	rightStr := ruleTreeToSkeleton(node.R)

	switch node.Op {
	case OpAnd:
		return "(AND " + leftStr + " " + rightStr + ")"
	case OpOr:
		return "(OR " + leftStr + " " + rightStr + ")"
	case OpNot:
		return "(NOT " + leftStr + ")"
	default:
		return ""
	}
}

func leafToSkeleton(leaf *Leaf) string {
	kindNames := map[LeafKind]string{
		LeafGT:        "GT",
		LeafLT:        "LT",
		LeafCrossUp:   "CrossUp",
		LeafCrossDown: "CrossDown",
		LeafRising:    "Rising",
		LeafFalling:   "Falling",
		LeafBetween:   "Between",
		LeafAbsGT:     "AbsGT",
		LeafAbsLT:     "AbsLT",
		LeafSlopeGT:   "SlopeGT",
		LeafSlopeLT:   "SlopeLT",
	}
	kindName := kindNames[leaf.Kind]

	// Include feature IDs but ignore numeric thresholds
	switch leaf.Kind {
	case LeafGT, LeafLT, LeafAbsGT, LeafAbsLT, LeafZScoreGT, LeafZScoreLT:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " THRESH)"
	case LeafBetween:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " THRESH THRESH)"
	case LeafSlopeGT, LeafSlopeLT:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " THRESH LB)"
	case LeafCrossUp, LeafCrossDown:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "] F[" + fmt.Sprint(leaf.B) + "])"
	case LeafRising, LeafFalling:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "] LB)"
	default:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "])"
	}
}

func stopModelToSkeleton(sm StopModel) string {
	// Only keep kind, ignore numeric values
	switch sm.Kind {
	case "fixed":
		return "Fixed"
	case "atr":
		return "ATR"
	case "swing":
		return "Swing"
	default:
		return sm.Kind
	}
}

func tpModelToSkeleton(tp TPModel) string {
	// Only keep kind, ignore numeric values
	switch tp.Kind {
	case "fixed":
		return "Fixed"
	case "atr":
		return "ATR"
	default:
		return tp.Kind
	}
}

func trailModelToSkeleton(tm TrailModel) string {
	if !tm.Active {
		return "none"
	}
	// Only keep kind, ignore numeric values
	switch tm.Kind {
	case "atr":
		return "ATR"
	case "swing":
		return "swing"
	default:
		return tm.Kind
	}
}

func evaluateRule(rule *RuleNode, features [][]float32, t int) bool {
	if rule == nil {
		return true // nil rule = no filter = allow (consistent with empty compiled code)
	}

	if rule.Op == OpLeaf {
		return evaluateLeaf(&rule.Leaf, features, t)
	}

	left := evaluateRule(rule.L, features, t)
	right := evaluateRule(rule.R, features, t)

	switch rule.Op {
	case OpAnd:
		return left && right
	case OpOr:
		return left || right
	case OpNot:
		return !left
	default:
		return false
	}
}

func evaluateLeaf(leaf *Leaf, features [][]float32, t int) bool {
	if t < leaf.Lookback {
		return false
	}

	// Guard against invalid feature indices (safety for mutated strategies)
	if leaf.A < 0 || leaf.A >= len(features) {
		return false // Invalid index - treat as false
	}
	if leaf.B < 0 || leaf.B >= len(features) {
		return false // Invalid index - treat as false
	}

	fa := features[leaf.A]
	fb := features[leaf.B]

	aVal := fa[t]
	bVal := fb[t]
	prevA := fa[t-1]
	prevB := fb[t-1]

	switch leaf.Kind {
	case LeafGT:
		return aVal > leaf.X
	case LeafLT:
		return aVal < leaf.X
	case LeafCrossUp:
		return prevA <= prevB && aVal > bVal
	case LeafCrossDown:
		return prevA >= prevB && aVal < bVal
	case LeafRising:
		return aVal > fa[t-leaf.Lookback]
	case LeafFalling:
		return aVal < fa[t-leaf.Lookback]
	case LeafBetween:
		// Check if value is between X (low) and Y (high)
		low := leaf.X
		high := leaf.Y
		if low > high {
			low, high = high, low
		}
		return aVal >= low && aVal <= high
	case LeafAbsGT:
		absVal := aVal
		if absVal < 0 {
			absVal = -absVal
		}
		return absVal > leaf.X
	case LeafAbsLT:
		absVal := aVal
		if absVal < 0 {
			absVal = -absVal
		}
		return absVal < leaf.X
	case LeafSlopeGT:
		// Compute slope over lookback period
		if t >= leaf.Lookback {
			slope := (aVal - fa[t-leaf.Lookback]) / float32(leaf.Lookback)
			return slope > leaf.X
		}
		return false
	case LeafSlopeLT:
		// Compute slope over lookback period
		if t >= leaf.Lookback {
			slope := (aVal - fa[t-leaf.Lookback]) / float32(leaf.Lookback)
			return slope < leaf.X
		}
		return false
	default:
		return false
	}
}
