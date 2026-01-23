package main

import (
	"fmt"
	"math"
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
// During bootstrap/recovery mode, heavily weights toward high-frequency features to reduce dead strategies
func randomFeatureIndex(rng *rand.Rand, feats Features) int {
	// Build list of high-frequency feature indices
	highFreqIndices := make([]int, 0, len(feats.Names)/2)
	for i, name := range feats.Names {
		if isHighFrequencyFeature(name) {
			highFreqIndices = append(highFreqIndices, i)
		}
	}

	// During bootstrap/recovery: 85% chance to pick from high-frequency features
	// Normal mode: 70% chance to pick from high-frequency features
	highFreqProb := float32(0.70)
	if RecoveryMode.Load() || isBootstrapMode() {
		highFreqProb = 0.85 // Stronger preference during bootstrap/recovery
	}

	if len(highFreqIndices) > 0 && rng.Float32() < highFreqProb {
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
	LeafNode      Leaf       // Original leaf node for bytecode re-evaluation
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
	FeatureMapHash  string // Fingerprint of feature ordering when strategy was created
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
		fmt.Sprintf("hold=%d|loss=%d|cd=%d", s.MaxHoldBars, s.MaxConsecLosses, s.CooldownBars) +
		includeRiskParams(s) + "|" +
		s.FeatureMapHash
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
		fmt.Sprintf("hold|loss|cd") +
		"|dir=" + fmt.Sprint(s.Direction)
}

// CoarseFingerprint creates a fingerprint with coarse threshold buckets
// This reduces duplicates by bucketing similar thresholds together
// More granular than SkeletonFingerprint (which ignores thresholds entirely)
// Less granular than Fingerprint() (which uses exact values)
func (s Strategy) CoarseFingerprint() string {
	return ruleTreeToCoarse(s.EntryRule.Root) + "|" +
		ruleTreeToCoarse(s.ExitRule.Root) + "|" +
		ruleTreeToCoarse(s.RegimeFilter.Root) + "|" +
		stopModelToCoarse(s.StopLoss) + "|" +
		tpModelToCoarse(s.TakeProfit) + "|" +
		trailModelToCoarse(s.Trail) + "|" +
		fmt.Sprintf("hold=%d|loss=%d|cd=%d",
			coarseInt(s.MaxHoldBars, 50),   // Bucket by 50
			coarseInt(s.MaxConsecLosses, 2), // Bucket by 2
			coarseInt(s.CooldownBars, 10)) + // Bucket by 10
		"|dir=" + fmt.Sprint(s.Direction) +
		"|fees=" + fmt.Sprintf("%.0f", math.Floor(float64(s.FeeBps))) +
		"|slip=" + fmt.Sprintf("%.0f", math.Floor(float64(s.SlippageBps))) +
		"|" + s.FeatureMapHash
}

// coarseInt buckets an integer into coarse ranges
func coarseInt(val, bucket int) int {
	if bucket <= 0 {
		return val
	}
	return (val / bucket) * bucket
}

// includeRiskParams returns a string fragment with risk parameters for fingerprints
// Quantizes floats to 4 decimal places to avoid "same but formatted different"
func includeRiskParams(st Strategy) string {
	feeQ := math.Floor(float64(st.FeeBps)*10000) / 10000
	slipQ := math.Floor(float64(st.SlippageBps)*10000) / 10000
	riskQ := math.Floor(float64(st.RiskPct)*10000) / 10000
	return fmt.Sprintf("|fees=%.4f|slip=%.4f|risk=%.4f|dir=%d", feeQ, slipQ, riskQ, st.Direction)
}

type RuleTree struct {
	Root *RuleNode
}

// Global bootstrap mode flag (accessed atomically for thread safety)
// When true: use lower cooldown (0-50) and lower MaxHoldBars (50-150) to speed up learning
// Once elites exist, set to false for normal operation
var globalBootstrapMode int32 = 0 // DISABLED - want longer holds for bigger winners

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

// FeatureCategory groups feature types for complexity validation
type FeatureCategory uint8

const (
	CatPriceMarket FeatureCategory = iota // PriceLevel, Oscillator, Momentum
	CatVolumeVolatility                   // Volume, ATR
	CatOther                              // ZScore, Normalized, EventFlag, etc.
)

// getFeatureCategory returns the category group for a feature type
func getFeatureCategory(featType FeatureType) FeatureCategory {
	switch featType {
	case FeatTypePriceLevel, FeatTypeOscillator, FeatTypeMomentum:
		return CatPriceMarket
	case FeatTypeVolume, FeatTypeATR:
		return CatVolumeVolatility
	default:
		return CatOther
	}
}

// getRuleFeatureCategories returns the set of feature categories used in a rule tree
func getRuleFeatureCategories(node *RuleNode, feats Features) map[FeatureCategory]bool {
	categories := make(map[FeatureCategory]bool)
	if node == nil {
		return categories
	}

	if node.Op == OpLeaf {
		// Add category for feature A
		if node.Leaf.A >= 0 && node.Leaf.A < len(feats.Types) {
			cat := getFeatureCategory(feats.Types[node.Leaf.A])
			categories[cat] = true
		}
		// For Cross leaves, also check feature B
		if (node.Leaf.Kind == LeafCrossUp || node.Leaf.Kind == LeafCrossDown) &&
			node.Leaf.B >= 0 && node.Leaf.B < len(feats.Types) {
			cat := getFeatureCategory(feats.Types[node.Leaf.B])
			categories[cat] = true
		}
		return categories
	}

	// Recursively collect categories from children
	leftCats := getRuleFeatureCategories(node.L, feats)
	rightCats := getRuleFeatureCategories(node.R, feats)

	for cat := range leftCats {
		categories[cat] = true
	}
	for cat := range rightCats {
		categories[cat] = true
	}

	return categories
}

// hasMinimumComplexity checks if an entry rule has minimum required complexity
// Requires at least one Price/Market condition AND one Volume/Volatility condition
// Rejects volume-only entries like "AbsGT VolSMA50 X"
func hasMinimumComplexity(node *RuleNode, feats Features) bool {
	categories := getRuleFeatureCategories(node, feats)

	hasPriceMarket := categories[CatPriceMarket]
	hasVolumeVolatility := categories[CatVolumeVolatility]

	// Must have BOTH price/market AND volume/volatility features
	return hasPriceMarket && hasVolumeVolatility
}

// addMissingFeatureCategory adds a leaf with the required feature category to the tree
// Returns true if successful, false otherwise
func addMissingFeatureCategory(node *RuleNode, rng *rand.Rand, feats Features, requiredCat FeatureCategory) bool {
	if node == nil {
		return false
	}

	if node.Op == OpLeaf {
		// Replace this leaf with one that uses the required category
		newLeaf := randomLeaf(rng, feats)
		// Build list of features with required category
		candidateIndices := make([]int, 0)
		for i, featType := range feats.Types {
			if getFeatureCategory(featType) == requiredCat {
				candidateIndices = append(candidateIndices, i)
			}
		}
		if len(candidateIndices) > 0 {
			newLeaf.A = candidateIndices[rng.Intn(len(candidateIndices))]
			// For Cross leaves, also ensure B is from the same category
			if newLeaf.Kind == LeafCrossUp || newLeaf.Kind == LeafCrossDown {
				newLeaf.B = candidateIndices[rng.Intn(len(candidateIndices))]
			}
			node.Leaf = newLeaf
			return true
		}
		return false
	}

	// Try left child first, then right
	if addMissingFeatureCategory(node.L, rng, feats, requiredCat) {
		return true
	}
	if addMissingFeatureCategory(node.R, rng, feats, requiredCat) {
		return true
	}
	return false
}

// maxLookback computes the maximum lookback required by a rule tree
// This is the maximum lookback of any leaf in the tree
func maxLookback(node *RuleNode) int {
	if node == nil {
		return 0
	}

	if node.Op == OpLeaf {
		return node.Leaf.Lookback
	}

	leftLB := maxLookback(node.L)
	rightLB := maxLookback(node.R)

	if leftLB > rightLB {
		return leftLB
	}
	return rightLB
}

// computeWarmup computes the warmup period needed for a strategy
// This is the maximum lookback of entry, exit, and regime rules
func computeWarmup(st Strategy) int {
	entryLB := maxLookback(st.EntryRule.Root)
	exitLB := maxLookback(st.ExitRule.Root)
	regimeLB := maxLookback(st.RegimeFilter.Root)

	warmup := entryLB
	if exitLB > warmup {
		warmup = exitLB
	}
	if regimeLB > warmup {
		warmup = regimeLB
	}

	// Minimum warmup of 10 bars to ensure indicators have some history
	if warmup < 10 {
		warmup = 10
	}

	return warmup
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
// 2. OR operators are limited to 3 per tree
// 3. This forces selectivity and prevents easy OR(...) where one side is always true
func randomEntryRuleNode(rng *rand.Rand, feats Features, depth, maxDepth int, entryOrCount *int) *RuleNode {
	if depth >= maxDepth || rng.Float32() < 0.3 {
		return &RuleNode{
			Op:   OpLeaf,
			Leaf: randomEntryLeaf(rng, feats),
		}
	}

	op := OpAnd
	randOp := rng.Float32()

	// For entry rules: use AND more aggressively, limit OR count
	// NOT has 10% probability when not at leaf level
	if randOp < 0.10 {
		op = OpNot
	} else if randOp < 0.40 && *entryOrCount < 3 {
		// Allow up to 3 ORs per entry rule
		op = OpOr
		*entryOrCount++
	}

	if op == OpNot {
		// NOT has only one child (Left), Right stays empty
		return &RuleNode{
			Op: op,
			L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount),
			R:  nil,
		}
	}

	return &RuleNode{
		Op: op,
		L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount),
		R:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount),
	}
}

// randomStrategyWithCosts creates a strategy with specified fee and slippage costs
// This allows consistent costs across train/val/test for realistic evaluation
func randomStrategyWithCosts(rng *rand.Rand, feats Features, feeBps, slipBps float32) Strategy {
	// Entry rule: use constrained generation to prevent "always true" strategies
	// Root must be AND, and OR count is limited to 1

	// RECOVERY MODE: Define helper functions for constraint validation
	// Defined ONCE outside the attempts loop for efficiency
	var countSparseFeatures func(n *RuleNode) int
	var checkNOTDepth func(n *RuleNode, depth int) bool

	if RecoveryMode.Load() {
		// Helper: count sparse features (SwingHigh, SwingLow, BOS, FVG are sparse)
		// Sparse features produce few trading signals, leading to dead strategies
		countSparseFeatures = func(n *RuleNode) int {
			if n == nil {
				return 0
			}
			if n.Op == OpLeaf {
				featName := feats.Names[n.Leaf.A]
				sparsePrefixes := []string{"Swing", "BOS", "FVG", "Breakout"}
				for _, prefix := range sparsePrefixes {
					if strings.HasPrefix(featName, prefix) {
						return 1
					}
				}
				return 0
			}
			return countSparseFeatures(n.L) + countSparseFeatures(n.R)
		}

		// Helper: check NOT nesting (allow NOT at root, but NOT(NOT(...)) is rejected)
		// Deeply nested NOTs create overly restrictive rules that rarely trigger
		checkNOTDepth = func(n *RuleNode, depth int) bool {
			if n == nil {
				return true
			}
			if n.Op == OpNot {
				if depth >= 1 { // Reject NOT(NOT(...)) patterns
					return false
				}
				return checkNOTDepth(n.L, depth+1)
			}
			return checkNOTDepth(n.L, depth) && checkNOTDepth(n.R, depth)
		}
	}

	var entryRoot *RuleNode

	if RecoveryMode.Load() {
		// Recovery mode: bounded loop to find a rule that passes constraints
		for attempts := 0; attempts < 50; attempts++ {
			entryOrCount := 0
			entryRoot = randomEntryRuleNode(rng, feats, 0, 4, &entryOrCount)

			// Validate sparse feature constraint
			sparseCount := countSparseFeatures(entryRoot)
			if sparseCount > 1 {
				continue // Too many sparse features, try again
			}

			// Validate NOT nesting constraint
			if !checkNOTDepth(entryRoot, 0) {
				continue // NOT nesting too deep, try again
			}

			// All constraints passed - break and use this rule
			break
		}
		// If we exhausted attempts, we use the last generated rule (fallback)
	} else {
		// Normal mode: original generation
		entryOrCount := 0
		entryRoot = randomEntryRuleNode(rng, feats, 0, 4, &entryOrCount)
	}

	// CRITICAL: Ensure entry rule contains at least one high-frequency primitive
	// to avoid "rare by construction" strategies that never trigger
	if !hasHighFrequencyPrimitive(entryRoot, feats) {
		replaceOneLeafWithHighFrequency(entryRoot, rng, feats)
	}

	// CRITICAL FIX #6: Enforce minimum complexity for entry rules
	// Require at least one Price/Market feature AND one Volume/Volatility feature
	// This prevents volume-only junk strategies like "AbsGT VolSMA50 X"
	if !hasMinimumComplexity(entryRoot, feats) {
		categories := getRuleFeatureCategories(entryRoot, feats)
		if !categories[CatPriceMarket] {
			// Add a price/market feature
			addMissingFeatureCategory(entryRoot, rng, feats, CatPriceMarket)
		}
		if !categories[CatVolumeVolatility] {
			// Add a volume/volatility feature
			addMissingFeatureCategory(entryRoot, rng, feats, CatVolumeVolatility)
		}
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

	// TREND ALIGNMENT: Ensure entry has trend guard for direction
	// This adds EMA50/EMA200 cross to align entries with the trend, reducing bad entries
	ensureTrendGuard(entryRoot, direction, feats, rng)

	// FORCE TRIGGER: Ensure at least one trigger leaf to prevent entry_rate_dead
	ensureTriggerLeaf(entryRoot, rng, feats)

	s := Strategy{
		Seed:            rng.Int63(),
		FeeBps:          feeBps,   // Use specified production costs
		SlippageBps:     slipBps,  // Use specified production costs
		RiskPct:         1.0,  // Set to 100% to report actual returns (not scaled)
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
		MaxHoldBars:     500 + rng.Intn(500), // 500..999 bars (~2-4 days for 5min) - allow big winners to run
		MaxConsecLosses: 20,                 // stop after 20 consecutive losses
		CooldownBars:    200,                // pause for 200 bars after busted (realistic)
	}

	// CRITICAL FIX #2: Bootstrap mode - use lower cooldown and MaxHoldBars to speed up learning
	// When no elites exist yet, use 0-50 cooldown and 50-150 MaxHoldBars for faster feedback
	if isBootstrapMode() {
		s.CooldownBars = rng.Intn(51)      // 0-50 bars during bootstrap
		s.MaxHoldBars = 200 + rng.Intn(301) // 200-500 bars during bootstrap (~17-42 hours)
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

// isAbsoluteThresholdKind returns true if the leaf kind uses absolute thresholds
// CRITICAL FIX #7: PriceLevel features (EMA, BB, Swing) should NOT use absolute thresholds
// because BTC price scale changes massively from 2017→2026
func isAbsoluteThresholdKind(kind LeafKind) bool {
	switch kind {
	case LeafGT, LeafLT, LeafBetween, LeafAbsGT, LeafAbsLT:
		return true
	default:
		return false
	}
}

// isTriggerKind returns true if the leaf kind is a trigger
// Trigger leaf kinds: CrossUp, CrossDown, Between, GT, LT, Rising, Falling
// These are the leaves that actually cause entries to trigger (not just static conditions)
func isTriggerKind(kind LeafKind) bool {
	switch kind {
	case LeafCrossUp, LeafCrossDown, LeafBetween, LeafGT, LeafLT, LeafRising, LeafFalling:
		return true
	default:
		return false
	}
}

// findFeatureIndexByContains finds feature index by substring search
func findFeatureIndexByContains(feats Features, substr string) int {
	for i, name := range feats.Names {
		if strings.Contains(name, substr) {
			return i
		}
	}
	return -1
}

// treeHasTrendGuard checks if tree contains EMA cross for direction
func treeHasTrendGuard(node *RuleNode, dir int, ema50, ema200 int) bool {
	if node == nil {
		return false
	}
	if node.Op == OpLeaf {
		if dir == 1 {
			return node.Leaf.Kind == LeafCrossUp && node.Leaf.A == ema50 && node.Leaf.B == ema200
		}
		return node.Leaf.Kind == LeafCrossDown && node.Leaf.A == ema50 && node.Leaf.B == ema200
	}
	return treeHasTrendGuard(node.L, dir, ema50, ema200) ||
		treeHasTrendGuard(node.R, dir, ema50, ema200)
}

// replaceRandomLeaf replaces a random leaf in the tree
func replaceRandomLeaf(root *RuleNode, rng *rand.Rand, replacement Leaf) {
	if root == nil {
		return
	}

	var walk func(node *RuleNode) bool
	walk = func(node *RuleNode) bool {
		if node == nil {
			return false
		}
		if node.Op == OpLeaf {
			// Replace this leaf
			node.Leaf = replacement
			return true
		}
		// Randomly try left or right first
		if rng.Float32() < 0.5 {
			if walk(node.L) {
				return true
			}
			return walk(node.R)
		} else {
			if walk(node.R) {
				return true
			}
			return walk(node.L)
		}
	}
	walk(root)
}

// ensureTrendGuard adds EMA cross trend guard if missing
func ensureTrendGuard(root *RuleNode, dir int, feats Features, rng *rand.Rand) {
	ema200 := findFeatureIndexByContains(feats, "EMA200")
	ema50 := findFeatureIndexByContains(feats, "EMA50")
	if ema200 < 0 || ema50 < 0 {
		return // Features not available
	}

	if treeHasTrendGuard(root, dir, ema50, ema200) {
		return // Already has trend guard
	}

	var guard Leaf
	if dir == 1 {
		guard = Leaf{Kind: LeafCrossUp, A: ema50, B: ema200}
	} else {
		guard = Leaf{Kind: LeafCrossDown, A: ema50, B: ema200}
	}
	replaceRandomLeaf(root, rng, guard)
}

// hasTriggerLeaf checks if tree contains any trigger leaf
// Trigger leaf kinds: CrossUp, CrossDown, Between, GT, LT, Rising, Falling
// These are the leaves that actually cause entries to trigger (not just static conditions)
func hasTriggerLeaf(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.Op == OpLeaf {
		switch node.Leaf.Kind {
		case LeafCrossUp, LeafCrossDown, LeafBetween, LeafGT, LeafLT, LeafRising, LeafFalling:
			return true
		}
		return false
	}
	return hasTriggerLeaf(node.L) || hasTriggerLeaf(node.R)
}

// ensureTriggerLeaf forces at least one trigger leaf in entry rule
// This prevents "entry_rate_dead" rejections by ensuring strategy can trigger
// Trigger leaf kinds: CrossUp, CrossDown, Between, GT, LT, Rising, Falling
func ensureTriggerLeaf(root *RuleNode, rng *rand.Rand, feats Features) {
	if hasTriggerLeaf(root) {
		return // Already has trigger leaf
	}
	// Replace a random leaf with a trigger leaf
	// Include all trigger kinds: CrossUp, CrossDown, Between, GT, LT, Rising, Falling
	triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBetween, LeafGT, LeafLT, LeafRising, LeafFalling}
	triggerKind := triggerKinds[rng.Intn(len(triggerKinds))]
	newLeaf := randomEntryLeaf(rng, feats)
	newLeaf.Kind = triggerKind
	replaceRandomLeaf(root, rng, newLeaf)
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
	var kind LeafKind

	// RECOVERY MODE + BOOTSTRAP MODE: Weight toward high-frequency operators to reduce dead strategies
	// When population is empty or in recovery, we need strategies that actually trigger
	if RecoveryMode.Load() || isBootstrapMode() {
		// Bootstrap/Recovery mode: 80% high-freq (biased toward Cross/Between), 15% mid-freq, 5% rare
		// Increased Cross/CrossDown/Between probability to improve entry rate hit rate
		crossKinds := []LeafKind{
			LeafCrossUp, LeafCrossDown, // Highest entry rate - trigger on price movement
			LeafBetween,                // High entry rate - range-based trigger
		}
		highFreqKinds := []LeafKind{
			LeafGT, LeafLT, // Simple comparisons
		}
		midFreqKinds := []LeafKind{
			LeafRising, LeafFalling, // Trend-based (less frequent)
		}
		rareKinds := []LeafKind{
			LeafAbsGT, LeafAbsLT,    // Rare by construction
			LeafSlopeGT, LeafSlopeLT, // Trend-based, sparse
		}

		roll := rng.Float32()
		if roll < 0.5 {
			// 50% Cross/Between - highest entry rate operators
			kind = crossKinds[rng.Intn(len(crossKinds))]
		} else if roll < 0.8 {
			// 30% GT/LT - simple comparisons
			kind = highFreqKinds[rng.Intn(len(highFreqKinds))]
		} else if roll < 0.95 {
			// 15% Rising/Falling - trend-based
			kind = midFreqKinds[rng.Intn(len(midFreqKinds))]
		} else {
			// 5% rare - reduced from 10%
			kind = rareKinds[rng.Intn(len(rareKinds))]
		}
	} else {
		// Normal mode: biased toward Cross/Between for better entry rate
		crossKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBetween}
		triggerKinds := []LeafKind{LeafRising, LeafFalling}
		thresholdKinds := []LeafKind{LeafGT, LeafLT, LeafAbsGT, LeafAbsLT}
		slopeKinds := []LeafKind{LeafSlopeGT, LeafSlopeLT}

		if len(feats.F) < 2 {
			// No cross leaves possible - use threshold or rising/falling
			kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
		} else if rng.Float32() < 0.45 {
			// 45% Cross/Between - increased from 30% for better entry rate
			kind = crossKinds[rng.Intn(len(crossKinds))]
		} else if rng.Float32() < 0.65 {
			// 20% Rising/Falling
			kind = triggerKinds[rng.Intn(len(triggerKinds))]
		} else if rng.Float32() < 0.90 {
			// 25% threshold
			kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
		} else {
			// 10% slope
			kind = slopeKinds[rng.Intn(len(slopeKinds))]
		}
	}

	// Use smart feature selection that prefers high-frequency features
	a := randomFeatureIndex(rng, feats)

	// Get feature name for threshold clamping
	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	// CRITICAL FIX #7: Ban absolute price thresholds on PriceLevel features
	// PriceLevel features (EMA*, BB_Upper/Lower*, SwingHigh/Low) should NOT use
	// absolute thresholds because BTC price scale changes massively from 2017→2026
	// Instead, force Cross/Rising/Falling comparisons (relative)
	if a < len(feats.Types) && feats.Types[a] == FeatTypePriceLevel && isAbsoluteThresholdKind(kind) {
		// Force a trigger kind instead (CrossUp, CrossDown, Rising, or Falling)
		triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafRising, LeafFalling}
		if len(feats.F) < 2 {
			// No cross leaves possible - use Rising or Falling
			kind = triggerKinds[2+rng.Intn(2)] // Rising or Falling
		} else {
			kind = triggerKinds[rng.Intn(len(triggerKinds))]
		}
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
		found := false
		for attempt := 0; attempt < maxAttempts; attempt++ {
			b = rng.Intn(len(feats.F))
			if b == a {
				continue
			}
			// Check feature type compatibility
			if b < len(feats.Types) {
				typeB := feats.Types[b]
				if canCrossFeatures(typeA, typeB) {
					found = true
					break
				}
			}
		}
		// FIX: If no compatible B found, fall back to Rising instead of using incompatible B
		if !found {
			kind = LeafRising
			lookback = 5 + rng.Intn(16)
			b = a // Not used for Rising
			threshold = 0
			break // Skip the default setup below
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

// randomEntryLeaf generates leaves biased toward high entry rate operators
// 70% triggers (Cross/Rising/Falling), 25% thresholds (near mean), 5% slopes
func randomEntryLeaf(rng *rand.Rand, feats Features) Leaf {
	// Trigger kinds produce more entry edges
	triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafRising, LeafFalling}
	thresholdKinds := []LeafKind{LeafGT, LeafLT, LeafBetween}
	slopeKinds := []LeafKind{LeafSlopeGT, LeafSlopeLT}

	var kind LeafKind
	p := rng.Float32()
	switch {
	case p < 0.70:
		kind = triggerKinds[rng.Intn(len(triggerKinds))]
	case p < 0.95:
		kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
	default:
		kind = slopeKinds[rng.Intn(len(slopeKinds))]
	}

	a := randomFeatureIndex(rng, feats)
	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	leaf := Leaf{Kind: kind, A: a, B: a}

	// Shorter lookback for Rising/Falling -> more edges
	switch kind {
	case LeafRising, LeafFalling:
		leaf.Lookback = 3 + rng.Intn(10) // 3-12 (shorter than normal 5-20)
		return leaf
	}

	// For thresholds: use "near mean" instead of extreme quantiles
	if a < len(feats.Stats) {
		st := feats.Stats[a]
		if st.Std > 0 {
			k := (rng.Float32()*1.2 - 0.6) // -0.6 to +0.6 sigma (centered)
			leaf.X = st.Mean + k*st.Std
			if kind == LeafBetween {
				leaf.Y = leaf.X + rng.Float32()*0.5*st.Std // Small band
			}
			leaf.X = clampThresholdByFeature(leaf.X, featName)
			leaf.Y = clampThresholdByFeature(leaf.Y, featName)
		}
	}

	// For Cross operations: delegate to randomLeaf for compatibility
	if kind == LeafCrossUp || kind == LeafCrossDown {
		tmp := randomLeaf(rng, feats)
		tmp.Kind = kind
		tmp.A = a
		leaf = tmp
	}

	return leaf
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

// Coarse helper functions - bucket numeric thresholds into ranges
// This reduces duplicates while providing more discrimination than skeleton-only

func ruleTreeToCoarse(node *RuleNode) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		return leafToCoarse(&node.Leaf)
	}

	leftStr := ruleTreeToCoarse(node.L)
	rightStr := ruleTreeToCoarse(node.R)

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

func leafToCoarse(leaf *Leaf) string {
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

	// Include feature IDs + bucketed thresholds
	// Threshold bucket: use adaptive bucketing (0.1 for small values, 5.0 for large)
	// Lookback bucket: round to nearest 5 bars
	switch leaf.Kind {
	case LeafGT, LeafLT, LeafAbsGT, LeafAbsLT:
		threshBucket := adaptiveCoarseFloat32(leaf.X)
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " " + fmt.Sprint(threshBucket) + ")"
	case LeafBetween:
		t1Bucket := adaptiveCoarseFloat32(leaf.X)
		t2Bucket := adaptiveCoarseFloat32(leaf.Y)
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " " + fmt.Sprint(t1Bucket) + " " + fmt.Sprint(t2Bucket) + ")"
	case LeafSlopeGT, LeafSlopeLT:
		threshBucket := coarseFloat32(leaf.X, 0.01) // Slope is small, bucket by 0.01
		lbBucket := coarseInt(leaf.Lookback, 5)     // Lookback bucket by 5
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + " " + fmt.Sprint(threshBucket) + " LB=" + fmt.Sprint(lbBucket) + ")"
	case LeafCrossUp, LeafCrossDown:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "] F[" + fmt.Sprint(leaf.B) + "])"
	case LeafRising, LeafFalling:
		lbBucket := coarseInt(leaf.Lookback, 5) // Lookback bucket by 5
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "] LB=" + fmt.Sprint(lbBucket) + ")"
	default:
		return "(" + kindName + " F[" + fmt.Sprint(leaf.A) + "])"
	}
}

// coarseFloat32 buckets a float32 into coarse ranges
func coarseFloat32(val float32, bucket float32) float32 {
	if bucket <= 0 {
		return val
	}
	return float32(int(val/bucket)) * bucket
}

// adaptiveCoarseFloat32 buckets a float32 with adaptive bucket size
// For small values (|x| < 10): use 0.1 bucket for better precision
// For large values (|x| >= 10): use 5.0 bucket to group similar values
// This reduces false "seen" collisions while still grouping similar strategies
func adaptiveCoarseFloat32(val float32) float32 {
	if val < 0 {
		val = -val
	}
	if val < 10 {
		// Small values: use 0.1 bucket for fine-grained discrimination
		return float32(int(val*10)) / 10
	}
	// Large values: use 5.0 bucket to group similar thresholds
	return float32(int(val/5)) * 5
}

func stopModelToCoarse(sm StopModel) string {
	// Keep kind + bucketed numeric values
	switch sm.Kind {
	case "fixed":
		pctBucket := coarseFloat32(sm.Value, 0.01) // Bucket by 1%
		return "Fixed_" + fmt.Sprint(pctBucket)
	case "atr":
		multBucket := coarseFloat32(sm.ATRMult, 0.5) // Bucket by 0.5x ATR
		return "ATR_" + fmt.Sprint(multBucket)
	case "swing":
		return "Swing"
	default:
		return sm.Kind
	}
}

func tpModelToCoarse(tp TPModel) string {
	// Keep kind + bucketed numeric values
	switch tp.Kind {
	case "fixed":
		pctBucket := coarseFloat32(tp.Value, 0.01) // Bucket by 1%
		return "Fixed_" + fmt.Sprint(pctBucket)
	case "atr":
		multBucket := coarseFloat32(tp.ATRMult, 0.5) // Bucket by 0.5x ATR
		return "ATR_" + fmt.Sprint(multBucket)
	default:
		return tp.Kind
	}
}

func trailModelToCoarse(tm TrailModel) string {
	if !tm.Active {
		return "none"
	}
	// Keep kind + bucketed numeric values
	switch tm.Kind {
	case "atr":
		multBucket := coarseFloat32(tm.ATRMult, 0.5) // Bucket by 0.5x ATR
		return "ATR_" + fmt.Sprint(multBucket)
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
	result, _ := evaluateLeafWithProof(leaf, features, nil, t)
	return result
}

// evaluateLeafWithProof returns both the boolean result and mathematical proof
// The feats parameter is optional (used for feature names in proof)
func evaluateLeafWithProof(leaf *Leaf, features [][]float32, feats *Features, t int) (bool, LeafProof) {
	proof := LeafProof{
		BarIndex: t,
		GuardChecks: []string{},
		Values: []float64{},
		Comparisons: []string{},
		LeafNode: *leaf, // Store original leaf for bytecode re-evaluation
	}

	// Map LeafKind to string
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
	proof.Kind = kindNames[leaf.Kind]

	// Get feature names if available
	getFeatName := func(idx int) string {
		if feats != nil && idx >= 0 && idx < len(feats.Names) {
			return feats.Names[idx]
		}
		return fmt.Sprintf("F[%d]", idx)
	}
	proof.FeatureA = getFeatName(leaf.A)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		proof.FeatureB = getFeatName(leaf.B)
	}

	// FIX: Cross operators need t >= 1 to access t-1 values
	// Check this BEFORE accessing previous values (prevents t=0 crash)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		if t < 1 {
			proof.GuardChecks = append(proof.GuardChecks, "t >= 1: false")
			proof.Result = false
			return false, proof
		}
		proof.GuardChecks = append(proof.GuardChecks, "t >= 1: true")
		proof.GuardChecks = append(proof.GuardChecks, "t-1 >= 0: true")
	} else if leaf.Kind == LeafRising || leaf.Kind == LeafFalling || leaf.Kind == LeafSlopeGT || leaf.Kind == LeafSlopeLT {
		if t < leaf.Lookback {
			proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("t >= lookback(%d): false", leaf.Lookback))
			proof.Result = false
			return false, proof
		}
		proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("t >= lookback(%d): true", leaf.Lookback))
	}

	if t < leaf.Lookback && leaf.Kind != LeafCrossUp && leaf.Kind != LeafCrossDown {
		proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("t >= lookback(%d): false", leaf.Lookback))
		proof.Result = false
		return false, proof
	}

	// Guard against invalid feature indices (safety for mutated strategies)
	if leaf.A < 0 || leaf.A >= len(features) {
		proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("feature A index %d valid: false", leaf.A))
		proof.Result = false
		return false, proof
	}
	// BUG FIX: Only check leaf.B for operators that actually use it (CrossUp, CrossDown)
	// SlopeGT, GT, LT, Rising, Falling operators don't use leaf.B (it's -1)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		if leaf.B < 0 || leaf.B >= len(features) {
			proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("feature B index %d valid: false", leaf.B))
			proof.Result = false
			return false, proof
		}
	}

	fa := features[leaf.A]
	var fb []float32
	// Only access leaf.B for operators that actually use it (CrossUp, CrossDown)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		fb = features[leaf.B]
	}

	aVal := fa[t]
	var bVal float32
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown {
		bVal = fb[t]
	}

	// NaN guard
	if math.IsNaN(float64(aVal)) {
		proof.GuardChecks = append(proof.GuardChecks, "A[t] is NaN: true")
		proof.Result = false
		return false, proof
	}
	proof.GuardChecks = append(proof.GuardChecks, "No NaN in features: true")

	// CRITICAL FIX #2: Only compute prevA/prevB for Cross operations (prevents t=0 crash)
	// Also make Cross logic identical to bytecode evaluator (same eps rule)
	var result bool
	switch leaf.Kind {
	case LeafGT:
		proof.Operator = ">"
		proof.Threshold = float64(leaf.X)
		proof.Values = []float64{float64(aVal), float64(leaf.X)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("%s > %.4f: %v", getFeatName(leaf.A), leaf.X, aVal > leaf.X))
		result = aVal > leaf.X
	case LeafLT:
		proof.Operator = "<"
		proof.Threshold = float64(leaf.X)
		proof.Values = []float64{float64(aVal), float64(leaf.X)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("%s < %.4f: %v", getFeatName(leaf.A), leaf.X, aVal < leaf.X))
		result = aVal < leaf.X
	case LeafCrossUp:
		// CRITICAL FIX #1: Prevent fake CrossUp - match bytecode logic
		const eps = 1e-6
		prevA := fa[t-1]
		prevB := fb[t-1]
		aMove := aVal - prevA
		bMove := bVal - prevB

		proof.Operator = "CrossUp"
		proof.Values = []float64{float64(prevA), float64(prevB), float64(aVal), float64(bVal)}

		// Use abs() because negative move still counts as no movement
		if aMove < 0 {
			aMove = -aMove
		}
		if bMove < 0 {
			bMove = -bMove
		}

		// Movement check with eps
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|A[t]-A[t-1]| = %.2f >= eps: %v", math.Abs(float64(aVal-prevA)), math.Abs(float64(aVal-prevA)) >= eps))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|B[t]-B[t-1]| = %.2f >= eps: %v", math.Abs(float64(bVal-prevB)), math.Abs(float64(bVal-prevB)) >= eps))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t-1] <= B[t-1]: %.2f <= %.2f: %v", prevA, prevB, prevA <= prevB))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t] > B[t]: %.2f > %.2f: %v", aVal, bVal, aVal > bVal))

		// Block if EITHER series doesn't move
		if aMove < eps || bMove < eps {
			result = false
		} else {
			result = prevA <= prevB && aVal > bVal
		}
	case LeafCrossDown:
		// CRITICAL FIX #1: Prevent fake CrossDown - match bytecode logic
		const eps2 = 1e-6
		prevA := fa[t-1]
		prevB := fb[t-1]
		aMove := aVal - prevA
		bMove := bVal - prevB

		proof.Operator = "CrossDown"
		proof.Values = []float64{float64(prevA), float64(prevB), float64(aVal), float64(bVal)}

		// Use abs() because negative move still counts as no movement
		if aMove < 0 {
			aMove = -aMove
		}
		if bMove < 0 {
			bMove = -bMove
		}

		// Movement check with eps
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|A[t]-A[t-1]| = %.2f >= eps: %v", math.Abs(float64(aVal-prevA)), math.Abs(float64(aVal-prevA)) >= eps2))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|B[t]-B[t-1]| = %.2f >= eps: %v", math.Abs(float64(bVal-prevB)), math.Abs(float64(bVal-prevB)) >= eps2))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t-1] >= B[t-1]: %.2f >= %.2f: %v", prevA, prevB, prevA >= prevB))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t] < B[t]: %.2f < %.2f: %v", aVal, bVal, aVal < bVal))

		// Block if EITHER series doesn't move
		if aMove < eps2 || bMove < eps2 {
			result = false
		} else {
			result = prevA >= prevB && aVal < bVal
		}
	case LeafRising:
		proof.Operator = "Rising"
		prevVal := fa[t-leaf.Lookback]
		proof.Threshold = float64(leaf.Lookback)
		proof.Values = []float64{float64(prevVal), float64(aVal)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t-lookback] = %.2f", prevVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t] = %.2f", aVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t] > X[t-lookback]: %.2f > %.2f: %v", aVal, prevVal, aVal > prevVal))
		result = aVal > prevVal
	case LeafFalling:
		proof.Operator = "Falling"
		prevVal := fa[t-leaf.Lookback]
		proof.Threshold = float64(leaf.Lookback)
		proof.Values = []float64{float64(prevVal), float64(aVal)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t-lookback] = %.2f", prevVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t] = %.2f", aVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("X[t] < X[t-lookback]: %.2f < %.2f: %v", aVal, prevVal, aVal < prevVal))
		result = aVal < prevVal
	case LeafBetween:
		proof.Operator = "Between"
		low := leaf.X
		high := leaf.Y
		if low > high {
			low, high = high, low
		}
		proof.Values = []float64{float64(low), float64(high), float64(aVal)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("low = %.4f", low))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("high = %.4f", high))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("%s >= low: %v", getFeatName(leaf.A), aVal >= low))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("%s <= high: %v", getFeatName(leaf.A), aVal <= high))
		result = aVal >= low && aVal <= high
	case LeafAbsGT:
		proof.Operator = "AbsGT"
		proof.Threshold = float64(leaf.X)
		absVal := aVal
		if absVal < 0 {
			absVal = -absVal
		}
		proof.Values = []float64{float64(absVal), float64(leaf.X)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|%s| = %.4f", getFeatName(leaf.A), absVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|%s| > %.4f: %v", getFeatName(leaf.A), leaf.X, absVal > leaf.X))
		result = absVal > leaf.X
	case LeafAbsLT:
		proof.Operator = "AbsLT"
		proof.Threshold = float64(leaf.X)
		absVal := aVal
		if absVal < 0 {
			absVal = -absVal
		}
		proof.Values = []float64{float64(absVal), float64(leaf.X)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|%s| = %.4f", getFeatName(leaf.A), absVal))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("|%s| < %.4f: %v", getFeatName(leaf.A), leaf.X, absVal < leaf.X))
		result = absVal < leaf.X
	case LeafSlopeGT:
		proof.Operator = "SlopeGT"
		proof.Threshold = float64(leaf.X)
		// Compute slope over lookback period
		if t >= leaf.Lookback {
			slope := (aVal - fa[t-leaf.Lookback]) / float32(leaf.Lookback)
			proof.ComputedSlope = float64(slope)
			proof.Values = []float64{float64(fa[t-leaf.Lookback]), float64(aVal), float64(slope)}
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("start = X[t-lookback] = %.2f", fa[t-leaf.Lookback]))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("end = X[t] = %.2f", aVal))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("n = %d", leaf.Lookback))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("slope = (end - start) / n = %.3f", slope))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("slope > threshold: %.3f > %.4f: %v", slope, leaf.X, slope > leaf.X))
			result = slope > leaf.X
		} else {
			result = false
		}
	case LeafSlopeLT:
		proof.Operator = "SlopeLT"
		proof.Threshold = float64(leaf.X)
		// Compute slope over lookback period
		if t >= leaf.Lookback {
			slope := (aVal - fa[t-leaf.Lookback]) / float32(leaf.Lookback)
			proof.ComputedSlope = float64(slope)
			proof.Values = []float64{float64(fa[t-leaf.Lookback]), float64(aVal), float64(slope)}
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("start = X[t-lookback] = %.2f", fa[t-leaf.Lookback]))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("end = X[t] = %.2f", aVal))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("n = %d", leaf.Lookback))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("slope = (end - start) / n = %.3f", slope))
			proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("slope < threshold: %.3f < %.4f: %v", slope, leaf.X, slope < leaf.X))
			result = slope < leaf.X
		} else {
			result = false
		}
	default:
		result = false
	}

	proof.Result = result
	return result, proof
}

// ruleTreeToStringWithNames converts a rule tree to a string with feature names instead of indices
// This helps debug "the strategy you think" == "the strategy being executed"
func ruleTreeToStringWithNames(node *RuleNode, feats Features) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		return leafToStringWithNames(&node.Leaf, feats)
	}

	leftStr := ruleTreeToStringWithNames(node.L, feats)
	rightStr := ruleTreeToStringWithNames(node.R, feats)

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

// leafToStringWithNames converts a leaf to string with feature names
func leafToStringWithNames(leaf *Leaf, feats Features) string {
	// Get feature names for indices
	getFeatName := func(idx int) string {
		if idx >= 0 && idx < len(feats.Names) {
			return feats.Names[idx]
		}
		return fmt.Sprintf("F[%d]", idx)
	}

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

	featA := getFeatName(leaf.A)

	switch leaf.Kind {
	case LeafGT, LeafLT, LeafAbsGT, LeafAbsLT:
		return "(" + kindName + " " + featA + " " + fmt.Sprintf("%.4f", leaf.X) + ")"
	case LeafBetween:
		return "(" + kindName + " " + featA + " " + fmt.Sprintf("%.4f", leaf.X) + " " + fmt.Sprintf("%.4f", leaf.Y) + ")"
	case LeafSlopeGT, LeafSlopeLT:
		return "(" + kindName + " " + featA + " " + fmt.Sprintf("%.4f", leaf.X) + " " + fmt.Sprintf("%d", leaf.Lookback) + ")"
	case LeafCrossUp, LeafCrossDown:
		featB := getFeatName(leaf.B)
		return "(" + kindName + " " + featA + " " + featB + ")"
	case LeafRising, LeafFalling:
		return "(" + kindName + " " + featA + " " + fmt.Sprintf("%d", leaf.Lookback) + ")"
	default:
		return "(" + kindName + " " + featA + ")"
	}
}
