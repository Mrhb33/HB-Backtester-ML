package main

import (
	"fmt"
	"math/rand"
	"strings"
	"sync/atomic"
)

// isHighFrequencyFeature returns true if the feature name indicates a high-frequency indicator
// These features produce values every bar (vs sparse/event features like Swing*, BOS, FVG)
func isHighFrequencyFeature(name string) bool {
	highFreqPrefixes := []string{
		"EMA", "RSI", "ATR", "BB_", "MACD", "ROC", "Vol", "ADX",
		"PlusDI", "MinusDI", "MFI", "BuyRatio", "Imbalance", "VolPerTrade",
		"Active", "HighLowDiff", "Body", "RangeWidth",
	}
	for _, prefix := range highFreqPrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// randomFeatureIndex selects a random feature, preferring high-frequency ones
func randomFeatureIndex(rng *rand.Rand, feats Features) int {
	// Build list of high-frequency feature indices
	highFreqIndices := make([]int, 0, len(feats.Names)/2)
	for i, name := range feats.Names {
		if isHighFrequencyFeature(name) {
			highFreqIndices = append(highFreqIndices, i)
		}
	}

	// 70% chance to pick from high-frequency features, 30% from all features
	if len(highFreqIndices) > 0 && rng.Float32() < 0.70 {
		return highFreqIndices[rng.Intn(len(highFreqIndices))]
	}
	return rng.Intn(len(feats.F))
}

// clampThresholdByFeature applies feature-specific threshold bounds to prevent impossible conditions
// CRITICAL FIX #4: Prevent nonsense thresholds like "VolZ20 > 317" or "RSI7 > 150"
func clampThresholdByFeature(threshold float32, featName string) float32 {
	// Oscillator 0-100 features: RSI7, PlusDI, MFI
	switch featName {
	case "RSI7":
		if threshold < 5 {
			return 5
		}
		if threshold > 95 {
			return 95
		}
		return threshold
	case "PlusDI":
		if threshold < 5 {
			return 5
		}
		if threshold > 70 {
			return 70
		}
		return threshold
	case "MFI":
		if threshold < 5 {
			return 5
		}
		if threshold > 95 {
			return 95
		}
		return threshold
	}

	// Z-score features: VolZ20, VolZ50 (typically -3 to +3, hard cap at ±3.5)
	switch featName {
	case "VolZ20", "VolZ50":
		if threshold < -3.5 {
			return -3.5
		}
		if threshold > 3.5 {
			return 3.5
		}
		return threshold
	}

	// Normalized bounded features: Imbalance [-1, +1]
	if featName == "Imbalance" {
		if threshold < -1.0 {
			return -1.0
		}
		if threshold > 1.0 {
			return 1.0
		}
		return threshold
	}

	// Price-based features for BTC 5m: Body, RangeWidth [0, 200]
	// BB_Lower50, SwingHigh should be within typical price range
	switch featName {
	case "Body", "RangeWidth":
		// BTC 5m typical body is 0-200 price units
		if threshold < 0 {
			return 0
		}
		if threshold > 200 {
			return 200
		}
		return threshold
	case "BB_Lower50", "SwingHigh", "SwingLow":
		// Price level features - clamp to reasonable BTC price range
		// Min BTC price (2020-2024): ~3000, Max: ~74000
		if threshold < 3000 {
			return 3000
		}
		if threshold > 74000 {
			return 74000
		}
		return threshold
	}

	// Default: no clamping for other features
	return threshold
}

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
	Seed            int64
	FeeBps          float32
	SlippageBps     float32
	RiskPct         float32
	Direction       int
	EntryRule       RuleTree
	EntryCompiled   CompiledRule
	ExitRule        RuleTree
	ExitCompiled    CompiledRule
	StopLoss        StopModel
	TakeProfit      TPModel
	Trail           TrailModel
	RegimeFilter    RuleTree
	RegimeCompiled  CompiledRule
	MaxHoldBars     int // time-based exit: max bars to hold a position
	MaxConsecLosses int // busted trades protection: stop after N consecutive losses
	CooldownBars    int // optional: pause after busted streak (0 = no cooldown, stop completely)
}

// Fingerprint creates a unique string representation of the strategy (excluding seed)
// This is used for deduplication to avoid having duplicate strategies in the leaderboard
func (s Strategy) Fingerprint() string {
	return ruleTreeToString(s.EntryRule.Root) + "|" +
		ruleTreeToString(s.ExitRule.Root) + "|" +
		ruleTreeToString(s.RegimeFilter.Root) + "|" +
		stopModelToString(s.StopLoss) + "|" +
		tpModelToString(s.TakeProfit) + "|" +
		trailModelToString(s.Trail) + "|" +
		fmt.Sprintf("hold=%d|loss=%d|cd=%d", s.MaxHoldBars, s.MaxConsecLosses, s.CooldownBars)
}

// SkeletonFingerprint creates a structure-only fingerprint (ignores numeric thresholds)
// This tracks strategy "families" - same operators + features + SL/TP type
func (s Strategy) SkeletonFingerprint() string {
	return ruleTreeToSkeleton(s.EntryRule.Root) + "|" +
		ruleTreeToSkeleton(s.ExitRule.Root) + "|" +
		ruleTreeToSkeleton(s.RegimeFilter.Root) + "|" +
		stopModelToSkeleton(s.StopLoss) + "|" +
		tpModelToSkeleton(s.TakeProfit) + "|" +
		trailModelToSkeleton(s.Trail) + "|" +
		fmt.Sprintf("hold|loss|cd")
}

type RuleTree struct {
	Root *RuleNode
}

// Global bootstrap mode flag (accessed atomically for thread safety)
// When true: use lower cooldown (0-50) and lower MaxHoldBars (50-150) to speed up learning
// Once elites exist, set to false for normal operation
var globalBootstrapMode int32 = 1 // Start in bootstrap mode

func isBootstrapMode() bool {
	return atomic.LoadInt32(&globalBootstrapMode) > 0
}

func setBootstrapMode(enabled bool) {
	if enabled {
		atomic.StoreInt32(&globalBootstrapMode, 1)
	} else {
		atomic.StoreInt32(&globalBootstrapMode, 0)
	}
}

func randomStrategy(rng *rand.Rand, feats Features) Strategy {
	return randomStrategyWithCosts(rng, feats, 30, 8) // Default production costs
}

// hasHighFrequencyPrimitive checks if a rule tree contains at least one high-frequency primitive
func hasHighFrequencyPrimitive(node *RuleNode, feats Features) bool {
	if node == nil {
		return false
	}

	if node.Op == OpLeaf {
		// Check if the leaf uses a high-frequency feature
		if node.Leaf.A < len(feats.Names) && isHighFrequencyFeature(feats.Names[node.Leaf.A]) {
			return true
		}
		// For Cross leaves, also check feature B
		if (node.Leaf.Kind == LeafCrossUp || node.Leaf.Kind == LeafCrossDown) &&
			node.Leaf.B < len(feats.Names) && isHighFrequencyFeature(feats.Names[node.Leaf.B]) {
			return true
		}
		return false
	}

	// Recursively check children
	if hasHighFrequencyPrimitive(node.L, feats) {
		return true
	}
	if hasHighFrequencyPrimitive(node.R, feats) {
		return true
	}
	return false
}

// replaceOneLeafWithHighFrequency replaces one leaf in the tree with a high-frequency primitive
func replaceOneLeafWithHighFrequency(node *RuleNode, rng *rand.Rand, feats Features) bool {
	if node == nil {
		return false
	}

	if node.Op == OpLeaf {
		// Replace this leaf with one that uses a high-frequency feature
		newLeaf := randomLeaf(rng, feats)
		// Force feature A to be high-frequency
		highFreqIndices := make([]int, 0, len(feats.Names)/2)
		for i, name := range feats.Names {
			if isHighFrequencyFeature(name) {
				highFreqIndices = append(highFreqIndices, i)
			}
		}
		if len(highFreqIndices) > 0 {
			newLeaf.A = highFreqIndices[rng.Intn(len(highFreqIndices))]
			node.Leaf = newLeaf
			return true
		}
		return false
	}

	// Try left child first, then right
	if replaceOneLeafWithHighFrequency(node.L, rng, feats) {
		return true
	}
	if replaceOneLeafWithHighFrequency(node.R, rng, feats) {
		return true
	}
	return false
}

// randomEntryRuleNode generates an entry rule with constrained tree shape
// to prevent "always true" strategies. Rules:
// 1. Root must be AND (not OR)
// 2. OR operators are limited to 1 per tree
// 3. This forces selectivity and prevents easy OR(...) where one side is always true
func randomEntryRuleNode(rng *rand.Rand, feats Features, depth, maxDepth int, orCount *int) *RuleNode {
	if depth >= maxDepth || rng.Float32() < 0.3 {
		return &RuleNode{
			Op:   OpLeaf,
			Leaf: randomLeaf(rng, feats),
		}
	}

	op := OpAnd
	randOp := rng.Float32()

	// For entry rules: use AND more aggressively, limit OR count
	// NOT has 10% probability when not at leaf level
	if randOp < 0.10 {
		op = OpNot
	} else if randOp < 0.40 && *orCount < 1 {
		// Only allow OR if we haven't used it yet (30% chance)
		op = OpOr
		*orCount++
	}

	if op == OpNot {
		// NOT has only one child (Left), Right stays empty
		return &RuleNode{
			Op: op,
			L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, orCount),
			R:  nil,
		}
	}

	return &RuleNode{
		Op: op,
		L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, orCount),
		R:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, orCount),
	}
}

// randomStrategyWithCosts creates a strategy with specified fee and slippage costs
// This allows consistent costs across train/val/test for realistic evaluation
func randomStrategyWithCosts(rng *rand.Rand, feats Features, feeBps, slipBps float32) Strategy {
	// Entry rule: use constrained generation to prevent "always true" strategies
	// Root must be AND, and OR count is limited to 1
	orCount := 0
	entryRoot := randomEntryRuleNode(rng, feats, 0, 4, &orCount)

	// CRITICAL: Ensure entry rule contains at least one high-frequency primitive
	// to avoid "rare by construction" strategies that never trigger
	if !hasHighFrequencyPrimitive(entryRoot, feats) {
		replaceOneLeafWithHighFrequency(entryRoot, rng, feats)
	}
	exitRoot := randomRuleNode(rng, feats, 0, 3)

	// Regime filter: 65% chance of having a regime filter (reduced from 85%)
	// Fewer regime filters = more trades = better chance to pass gates
	regimeRoot := randomRuleNode(rng, feats, 0, 2)
	if rng.Float32() < 0.35 {
		// 35% chance to disable regime filter (nil = always true)
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
		Seed:            rng.Int63(),
		FeeBps:          feeBps,   // Use specified production costs
		SlippageBps:     slipBps,  // Use specified production costs
		RiskPct:         0.01,
		Direction:       direction,
		EntryRule:       RuleTree{Root: entryRoot},
		EntryCompiled:   compileRuleTree(entryRoot),
		ExitRule:        RuleTree{Root: exitRoot},
		ExitCompiled:    compileRuleTree(exitRoot),
		StopLoss:        randomStopModel(rng),
		TakeProfit:      TPModel{}, // Placeholder, will be set below with RR constraint
		Trail:           randomTrailModel(rng),
		RegimeFilter:    RuleTree{Root: regimeRoot},
		RegimeCompiled:  compileRuleTree(regimeRoot),
		MaxHoldBars:     150 + rng.Intn(180), // 150..329 bars (searchable/evolved, reduced early exits)
		MaxConsecLosses: 20,                 // stop after 20 consecutive losses
		CooldownBars:    200,                // pause for 200 bars after busted (realistic)
	}

	// CRITICAL FIX #2: Bootstrap mode - use lower cooldown and MaxHoldBars to speed up learning
	// When no elites exist yet, use 0-50 cooldown and 50-150 MaxHoldBars for faster feedback
	if isBootstrapMode() {
		s.CooldownBars = rng.Intn(51)      // 0-50 bars during bootstrap
		s.MaxHoldBars = 50 + rng.Intn(101) // 50-150 bars during bootstrap
	}

	// CRITICAL FIX #1: Force SL/TP to be same kind and enforce RR >= 1.3
	// This prevents negative expectancy when SL is ATR-based and TP is fixed percent
	// which creates TP < SL when ATR is large vs price
	if s.StopLoss.Kind == "atr" {
		s.TakeProfit.Kind = "atr"
		// Enforce RR >= 1.3: TP must be at least 1.3x SL
		minTP := s.StopLoss.ATRMult * 1.3
		if minTP < 1.5 {
			minTP = 1.5 // Still enforce absolute minimum from randomTPModel
		}
		if s.TakeProfit.ATRMult < minTP {
			s.TakeProfit.ATRMult = minTP + rng.Float32()*2.0 // Add some variation above minimum
		}
		if s.TakeProfit.ATRMult > 10 {
			s.TakeProfit.ATRMult = 10
		}
		// TP.Value stays 0 for ATR-based
		s.TakeProfit.Value = 0
	} else if s.StopLoss.Kind == "fixed" {
		s.TakeProfit.Kind = "fixed"
		// Enforce RR >= 1.3: TP must be at least 1.3x SL
		minTP := s.StopLoss.Value * 1.3
		if minTP < 1.0 {
			minTP = 1.0 // Still enforce absolute minimum from randomTPModel
		}
		if s.TakeProfit.Value < minTP {
			s.TakeProfit.Value = minTP + rng.Float32()*2.0 // Add some variation above minimum
		}
		if s.TakeProfit.Value > 30 {
			s.TakeProfit.Value = 30
		}
		// TP.ATRMult stays 0 for fixed-based
		s.TakeProfit.ATRMult = 0
	} else {
		// For swing stops, use ATR-based TP with RR >= 1.3
		// Estimate swing stop in ATR terms (rough approximation)
		estSLATR := float32(2.0) // rough estimate for swing
		s.TakeProfit.Kind = "atr"
		minTP := estSLATR * 1.3
		if minTP < 1.5 {
			minTP = 1.5
		}
		s.TakeProfit.ATRMult = minTP + rng.Float32()*2.0
		if s.TakeProfit.ATRMult > 10 {
			s.TakeProfit.ATRMult = 10
		}
		s.TakeProfit.Value = 0
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

	// Use smart feature selection that prefers high-frequency features
	a := randomFeatureIndex(rng, feats)

	// Get feature name for threshold clamping
	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	var b int
	var lookback int
	var threshold float32
	var leafY float32 // For Between leaf (high threshold)

	switch kind {
	case LeafCrossUp, LeafCrossDown:
		// CRITICAL FIX #5: Keep picking B until it's different from A AND compatible types
		// Only allow crossing within same feature type to prevent nonsense operations
		var typeA FeatureType
		if a < len(feats.Types) {
			typeA = feats.Types[a]
		}
		maxAttempts := 100
		for attempt := 0; attempt < maxAttempts; attempt++ {
			b = rng.Intn(len(feats.F))
			if b == a {
				continue
			}
			// Check feature type compatibility
			if b < len(feats.Types) {
				typeB := feats.Types[b]
				if canCrossFeatures(typeA, typeB) {
					break // Found compatible B
				}
			} else if typeA == FeatTypeUnknown {
				// If typeA is unknown, allow crossing with unknown typeB
				break
			}
			// If we've tried many times, give up and use the last one
			if attempt == maxAttempts-1 {
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
			// Use quantiles: (20th-40th) and (60th-80th) percentile
			// Low threshold: 20th-40th percentile (below mean)
			k1 := -0.5 - rng.Float32()*0.3 // range: -0.8 to -0.5
			// High threshold: 60th-80th percentile (above mean)
			k2 := 0.25 + rng.Float32()*0.55 // range: 0.25 to 0.8
			threshold = stats.Mean + float32(k1)*stats.Std
			leafY = stats.Mean + float32(k2)*stats.Std

			// CRITICAL FIX #4: Clamp to feature-specific bounds
			threshold = clampThresholdByFeature(threshold, featName)
			leafY = clampThresholdByFeature(leafY, featName)

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
			// CRITICAL FIX #4: Clamp to feature-specific bounds
			threshold = clampThresholdByFeature(threshold, featName)
			leafY = clampThresholdByFeature(leafY, featName)
		}
	case LeafAbsGT, LeafAbsLT:
		// Absolute value comparison - use quantiles for selectivity
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			// For Abs leaves, use positive quantiles
			// AbsLT: pick from 10th-40th percentile (lower values)
			// AbsGT: pick from 60th-90th percentile (higher values)
			var k float32
			if kind == LeafAbsLT {
				k = -0.5 - rng.Float32()*0.7 // range: -1.2 to -0.5
			} else { // LeafAbsGT
				k = 0.25 + rng.Float32()*0.95 // range: 0.25 to 1.2
			}
			threshold = stats.Mean + k*stats.Std
			if threshold < 0 {
				threshold = -threshold // Absolute thresholds should be positive
			}
		} else {
			threshold = float32(rng.NormFloat64()*20 + 50)
			if threshold < 0 {
				threshold = 0
			}
		}
		// CRITICAL FIX #4: Clamp to feature-specific bounds
		threshold = clampThresholdByFeature(threshold, featName)
	case LeafSlopeGT, LeafSlopeLT:
		// Slope comparison over lookback period
		b = a
		lookback = rng.Intn(16) + 5                  // 5-20 bars
		threshold = float32(rng.NormFloat64() * 0.1) // Small slope values
		// CRITICAL FIX #4: Clamp to feature-specific bounds
		threshold = clampThresholdByFeature(threshold, featName)
	default:
		// For GT/LT, use quantile-based thresholds to avoid "always true" leaves
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			// Use quantiles instead of raw random values to make conditions naturally selective
			// LT (10th-40th percentile): Mean - (1.2 to 0.25) * Std
			// GT (60th-90th percentile): Mean + (0.25 to 1.2) * Std
			var k float32
			if kind == LeafLT {
				// LT: pick from 10th-40th percentile (below mean)
				k = -0.25 - rng.Float32()*0.95 // range: -1.2 to -0.25
			} else { // LeafGT
				// GT: pick from 60th-90th percentile (above mean)
				k = 0.25 + rng.Float32()*0.95 // range: 0.25 to 1.2
			}
			threshold = stats.Mean + k*stats.Std

			// Bounds check to prevent extreme values
			if stats.Std > 0 {
				minThreshold := stats.Mean - 2*stats.Std
				maxThreshold := stats.Mean + 2*stats.Std
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
		// CRITICAL FIX #4: Clamp to feature-specific bounds (prevents VolZ20 > 317)
		threshold = clampThresholdByFeature(threshold, featName)
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
		if atr < 1.0 { // Raised from 0.5 to 1.0 for reasonable SL
			atr = 1.0
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
		if val < 0.5 { // Raised from 0.3 to 0.5
			val = 0.5
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
	// RAISED MINIMUMS to prevent cost-losers (TP must be larger than SL)
	kind := rng.Intn(10)
	if kind < 8 { // 80% ATR
		atr := rng.NormFloat64()*2.0 + 3.5 // Raised mean from 3.0 to 3.5
		if atr < 1.5 { // Raised from 0.5 to 1.5 - TP must be >= 1.5x ATR
			atr = 1.5
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
		if val < 1.0 { // Raised from 0.5 to 1.0 - TP must be >= 1%
			val = 1.0
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
	// Prefer ATR-based trails or no trail (60% none, 30% loose ATR, 10% swing)
	// Avoid tight trails that stop winners too early
	kind := rng.Intn(10)
	if kind < 6 { // 60% no trail
		return TrailModel{
			Kind:    "none",
			ATRMult: 0,
			Active:  false,
		}
	} else if kind < 9 { // 30% loose ATR trail
		atr := rng.NormFloat64()*1.5 + 3.5 // Looser: mean 3.5 (was 2.5)
		if atr < 2.5 {
			atr = 2.5
		}
		if atr > 6.0 {
			atr = 6.0
		}
		return TrailModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Active:  true,
		}
	} else { // 10% swing trail
		return TrailModel{
			Kind:    "swing",
			ATRMult: 0,
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
	case LeafGT, LeafLT, LeafAbsGT, LeafAbsLT:
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
	// FIX: Cross operators need t >= 1 to access t-1 values
	// Check this BEFORE accessing previous values (prevents t=0 crash)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		if t < 1 {
			return false // Can't detect cross without previous bar
		}
	}

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

	// CRITICAL FIX #2: Only compute prevA/prevB for Cross operations (prevents t=0 crash)
	// Also make Cross logic identical to bytecode evaluator (same eps rule)
	switch leaf.Kind {
	case LeafGT:
		return aVal > leaf.X
	case LeafLT:
		return aVal < leaf.X
	case LeafCrossUp:
		// CRITICAL FIX #1: Prevent fake CrossUp - match bytecode logic
		const eps = 1e-6
		prevA := fa[t-1]
		prevB := fb[t-1]
		aMove := aVal - prevA
		bMove := bVal - prevB
		// Use abs() because negative move still counts as no movement
		if aMove < 0 {
			aMove = -aMove
		}
		if bMove < 0 {
			bMove = -bMove
		}
		// Block if EITHER series doesn't move
		if aMove < eps || bMove < eps {
			return false // At least one series didn't move - can't be a real cross
		}
		return prevA <= prevB && aVal > bVal
	case LeafCrossDown:
		// CRITICAL FIX #1: Prevent fake CrossDown - match bytecode logic
		const eps2 = 1e-6
		prevA := fa[t-1]
		prevB := fb[t-1]
		aMove := aVal - prevA
		bMove := bVal - prevB
		// Use abs() because negative move still counts as no movement
		if aMove < 0 {
			aMove = -aMove
		}
		if bMove < 0 {
			bMove = -bMove
		}
		// Block if EITHER series doesn't move
		if aMove < eps2 || bMove < eps2 {
			return false // At least one series didn't move - can't be a real cross
		}
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
