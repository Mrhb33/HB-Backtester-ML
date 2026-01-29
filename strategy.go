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
		"BodyPct", "WickUpPct", "WickDownPct", "ClosePos", // Candle anatomy
		"HH_", "LL_", // Market structure levels
		"SweepUp_", "SweepDown_", // Sweep flags
		"HMA", "Kijun", // Original indicators
		// PHASE 1: HIGH-IMPACT NEW INDICATORS
		"KC_", "Stoch", "Donchian", "SuperTrend", "WilliamsR", "ForceIndex", "Momentum",
		// PHASE 2: EVENT FLAGS (these are sparse but should be considered for frequency checks)
		"Squeeze", "StochBullCross", "StochBearCross",
	}
	for _, prefix := range highFreqPrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// isCandleFeature returns true for pure candle-based features (no indicators)
func isCandleFeature(name string) bool {
	candlePrefixes := []string{
		"BodyPct", "WickUpPct", "WickDownPct", "ClosePos",
		"HH_", "LL_",
		"SweepUp_", "SweepDown_",
	}
	for _, prefix := range candlePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func isMomentumFeature(name string) bool {
	return strings.HasPrefix(name, "ROC") ||
		strings.Contains(name, "Momentum") ||
		strings.Contains(name, "MACD") ||
		strings.Contains(name, "Hist") ||
		strings.Contains(name, "SuperTrendDir") ||
		name == "ADX" || name == "PlusDI" || name == "MinusDI" ||
		strings.Contains(name, "RSI") ||
		strings.Contains(name, "Stoch") ||
		strings.Contains(name, "MFI")
}

func isVolatilityStateFeature(name string) bool {
	return strings.Contains(name, "ATR") ||
		strings.Contains(name, "VolZ") ||
		strings.Contains(name, "VOLRET") ||
		strings.Contains(name, "BB_Width") ||
		strings.Contains(name, "Compression") ||
		strings.Contains(name, "Range")
}

func isStructureFeature(name string) bool {
	return strings.HasPrefix(name, "HH_") ||
		strings.HasPrefix(name, "LL_") ||
		strings.HasPrefix(name, "Sweep") ||
		strings.HasPrefix(name, "BOS") ||
		strings.HasPrefix(name, "FVG") ||
		strings.HasPrefix(name, "Breakout")
}

// randomFeatureIndex selects a random feature, preferring high-frequency ones
// During bootstrap/recovery mode, heavily weights toward high-frequency features to reduce dead strategies
func randomFeatureIndex(rng *rand.Rand, feats Features) int {
	// Build lists of feature indices by type
	candleIndices := make([]int, 0)
	volumeIndices := make([]int, 0)
	highFreqIndices := make([]int, 0, len(feats.Names)/2)
	momentumIndices := make([]int, 0)
	volStateIndices := make([]int, 0)
	structureIndices := make([]int, 0)

	for i, name := range feats.Names {
		if isCandleFeature(name) {
			candleIndices = append(candleIndices, i)
		} else if isHighFrequencyFeature(name) {
			highFreqIndices = append(highFreqIndices, i)
		}
		if isMomentumFeature(name) {
			momentumIndices = append(momentumIndices, i)
		}
		if isVolatilityStateFeature(name) {
			volStateIndices = append(volStateIndices, i)
		}
		if isStructureFeature(name) {
			structureIndices = append(structureIndices, i)
		}
	}

	// Collect volume-derived features (FeatTypeVolume + FeatTypeVolumeDerived)
	for i := range feats.Names {
		if i < len(feats.Types) {
			if feats.Types[i] == FeatTypeVolume || feats.Types[i] == FeatTypeVolumeDerived {
				volumeIndices = append(volumeIndices, i)
			}
		}
	}

	// 20% chance to pick from momentum features
	if len(momentumIndices) > 0 && rng.Float32() < 0.20 {
		return momentumIndices[rng.Intn(len(momentumIndices))]
	}

	// 15% chance to pick from volatility/state features
	if len(volStateIndices) > 0 && rng.Float32() < 0.15 {
		return volStateIndices[rng.Intn(len(volStateIndices))]
	}

	// 10% chance to pick from structure features
	if len(structureIndices) > 0 && rng.Float32() < 0.10 {
		return structureIndices[rng.Intn(len(structureIndices))]
	}

	// 30% chance to pick from candle features (bias)
	if len(candleIndices) > 0 && rng.Float32() < 0.30 {
		return candleIndices[rng.Intn(len(candleIndices))]
	}

	// 25% chance to pick from volume features
	if len(volumeIndices) > 0 && rng.Float32() < 0.25 {
		return volumeIndices[rng.Intn(len(volumeIndices))]
	}

	// During bootstrap/recovery: 75% high-frequency
	// Normal mode: 60% high-frequency
	highFreqProb := float32(0.60)
	if RecoveryMode.Load() || isBootstrapMode() {
		highFreqProb = 0.75
	}

	if len(highFreqIndices) > 0 && rng.Float32() < highFreqProb {
		return highFreqIndices[rng.Intn(len(highFreqIndices))]
	}
	return rng.Intn(len(feats.F))
}

// randomFeatureIndexByGroup picks a feature index from a specific group (momentum/vol/structure)
// Returns -1 if no features match the group.
func randomFeatureIndexByGroup(rng *rand.Rand, feats Features, group string) int {
	candidates := make([]int, 0)
	for i, name := range feats.Names {
		switch group {
		case "momentum":
			if isMomentumFeature(name) {
				candidates = append(candidates, i)
			}
		case "vol":
			if isVolatilityStateFeature(name) {
				candidates = append(candidates, i)
			}
		case "structure":
			if isStructureFeature(name) {
				candidates = append(candidates, i)
			}
		}
	}
	if len(candidates) == 0 {
		return -1
	}
	return candidates[rng.Intn(len(candidates))]
}

// getCompatibleFeatureIndex returns a feature index compatible with the given feature type
// For Cross/Break operations, we need features of the same category to make valid comparisons
func getCompatibleFeatureIndex(rng *rand.Rand, feats Features, aIdx int, aName string) int {
	// Categorize features by type
	isOscillator := func(name string) bool {
		return strings.Contains(name, "RSI") || strings.Contains(name, "ADX") ||
			strings.Contains(name, "MFI") || strings.Contains(name, "Stoch")
	}
	isPrice := func(name string) bool {
		return strings.Contains(name, "EMA") || strings.Contains(name, "SMA") ||
			strings.Contains(name, "BB_") || strings.Contains(name, "VWAP") ||
			name == "Close" || name == "Open" || name == "High" || name == "Low"
	}
	isNormalized := func(name string) bool {
		return strings.Contains(name, "Ratio") || strings.Contains(name, "Pct") ||
			strings.Contains(name, "Imbalance") || strings.Contains(name, "Body")
	}
	isZScore := func(name string) bool {
		return strings.Contains(name, "VolZ") || strings.Contains(name, "ZScore")
	}

	// Find compatible features
	var compatible []int
	for i, name := range feats.Names {
		if i == aIdx {
			continue // Skip self
		}
		// Match by category
		if (isOscillator(aName) && isOscillator(name)) ||
			(isPrice(aName) && isPrice(name)) ||
			(isNormalized(aName) && isNormalized(name)) ||
			(isZScore(aName) && isZScore(name)) {
			compatible = append(compatible, i)
		}
	}

	if len(compatible) > 0 {
		return compatible[rng.Intn(len(compatible))]
	}
	return aIdx // Fallback to same feature (degenerate but won't crash)
}

// clampThresholdByFeature applies feature-specific threshold bounds to prevent impossible conditions
// CRITICAL FIX #4: Prevent nonsense thresholds like "VolZ20 > 317" or "RSI7 > 150"
func clampThresholdByFeature(threshold float32, featName string) float32 {
	// Oscillator 0-100 features: RSI7, PlusDI, MFI
	switch featName {
	case "RSI7":
		if threshold < 1 {
			return 1
		}
		if threshold > 99 {
			return 99
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

	// BuyRatio [0.0, 1.0] - volume buy ratio
	if featName == "BuyRatio" {
		if threshold < 0.0 {
			return 0.0
		}
		if threshold > 1.0 {
			return 1.0
		}
		return threshold
	}

	// Candle anatomy features (normalized 0-1)
	switch featName {
	case "BodyPct", "WickUpPct", "WickDownPct", "ClosePos":
		if threshold < 0.0 {
			return 0.0
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

	// ========== PHASE 1: NEW INDICATOR THRESHOLDS ==========
	// Stochastic Oscillator (0-100 bounded, use 1-99 like RSI)
	switch featName {
	case "StochK_14", "StochD_14", "StochK_5", "StochD_5":
		if threshold < 1 {
			return 1
		}
		if threshold > 99 {
			return 99
		}
		return threshold
	}

	// Williams %R (-100 to 0, use clamping to prevent extreme values)
	switch featName {
	case "WilliamsR_14", "WilliamsR_7":
		if threshold < -99 {
			return -99
		}
		if threshold > -1 {
			return -1
		}
		return threshold
	}

	// Keltner Channel, Donchian, SuperTrend (price level features)
	switch featName {
	case "KC_Upper20", "KC_Lower20", "KC_Middle20":
		if threshold < 3000 {
			return 3000
		}
		if threshold > 74000 {
			return 74000
		}
		return threshold
	case "Donchian_Upper20", "Donchian_Lower20", "Donchian_Upper55", "Donchian_Lower55":
		if threshold < 3000 {
			return 3000
		}
		if threshold > 74000 {
			return 74000
		}
		return threshold
	case "SuperTrend10":
		if threshold < 3000 {
			return 3000
		}
		if threshold > 74000 {
			return 74000
		}
		return threshold
	}

	// Force Index (can be positive or negative, very large values)
	// Use reasonable bounds based on typical BTC volume
	switch featName {
	case "ForceIndex2", "ForceIndex13":
		// Force index can be very large (volume × price change)
		// Clamp to reasonable range: -1e9 to +1e9
		if threshold < -1e9 {
			return -1e9
		}
		if threshold > 1e9 {
			return 1e9
		}
		return threshold
	}

	// Momentum (rate of return, typically -0.5 to +0.5)
	switch featName {
	case "Momentum60", "Momentum240":
		// Allow for large swings but clamp extreme values
		if threshold < -0.8 {
			return -0.8
		}
		if threshold > 1.0 {
			return 1.0
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
	LeafBreakUp   // NEW: A[t-1] <= B[t-1] && A[t] > B[t] (no movement requirement)
	LeafBreakDown // NEW: A[t-1] >= B[t-1] && A[t] < B[t] (no movement requirement)
)

type Leaf struct {
	Kind     LeafKind
	A        int
	B        int
	X        float32
	Y        float32 // For Between leaf (high threshold)
	Lookback int
}

// IsSelfComparison returns true if this leaf compares a feature to itself
// FIX B1: Self-comparisons like CrossUp(A,A), BreakDown(A,A), GT(A,A) are always
// true or false (degenerate) and produce 0 edges, 0 trades, garbage results
func (l Leaf) IsSelfComparison() bool {
	switch l.Kind {
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
		// These operators compare A vs B; if A == B, it's always false
		return l.A == l.B
	case LeafGT, LeafLT:
		// GT(A, A) means F[A] > F[A] which is always false
		// LT(A, A) means F[A] < F[A] which is always false
		// However, GT/LT typically compare feature to threshold X, not to B
		// Only flag as self-comparison if B is used and equals A
		// Note: In this codebase, GT/LT use X threshold, not B, so this is OK
		return false
	default:
		return false
	}
}

// IsDegenerate returns true if this leaf is invalid/degenerate and will produce garbage
// FIX B1 + B2 + C: Catches self-comparisons, needle Between, and "always true" patterns
func (l Leaf) IsDegenerate() bool {
	// Check self-comparison (always false/true)
	if l.IsSelfComparison() {
		return true
	}

	// FIX B2 + C: Check degenerate Between patterns
	if l.Kind == LeafBetween {
		// Case 1: Exact equality (needle Between)
		if l.X == l.Y {
			return true
		}
		// FIX C: Case 2: Very narrow range (effectively a needle)
		// Between(x, 0.001, 0.002) is degenerate - almost never triggers
		absDiff := l.X - l.Y
		if absDiff < 0 {
			absDiff = -absDiff
		}
		// For absolute values < 1: require at least 0.01 width
		// For larger values: require at least 1% relative width
		maxVal := l.X
		if l.Y > maxVal {
			maxVal = l.Y
		}
		var minWidth float32
		if maxVal < 1.0 {
			minWidth = 0.01 // 1% absolute minimum for small values
		} else {
			minWidth = maxVal * 0.01 // 1% relative minimum
		}
		if absDiff < minWidth {
			return true
		}
		// FIX C: Case 3: Between(x, 0, 0) or Between(x, tiny, tiny) for binary features
		// This creates "almost always false" rules
		if l.X <= 0.001 && l.Y <= 0.01 {
			return true // Between(x, 0, ~0) is degenerate
		}
	}

	return false
}

// LeafProof provides mathematical proof of leaf evaluation to demonstrate no lookahead bias
type LeafProof struct {
	Kind          string    // "CrossUp", "Rising", "SlopeGT", etc.
	Operator      string    // The actual operator used
	FeatureA      string    // Feature name
	FeatureB      string    // For Cross operators
	BarIndex      int       // Current bar t
	Values        []float64 // All values used in computation
	Comparisons   []string  // Step-by-step comparison results
	GuardChecks   []string  // t>=1, t>=lookback, NaN checks, eps checks
	ComputedSlope float64   // For SlopeGT/SlopeLT
	Threshold     float64   // X value compared against
	Result        bool      // Final result (matches returned bool)
	LeafNode      Leaf      // Original leaf node for bytecode re-evaluation
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

// VolFilterModel implements volatility regime filtering
// Only trades when ATR14 is above its historical average (high volatility periods)
type VolFilterModel struct {
	Enabled   bool    // true = filter is active
	ATRPeriod int     // ATR period to use (typically 14)
	SMAPeriod int     // SMA period for ATR average (typically 50)
	Threshold float32 // Multiplier: ATR must be > Threshold × SMA (1.0 = above average)
}

// IsActive returns true if volatility filter is enabled
func (v VolFilterModel) IsActive() bool {
	return v.Enabled && v.ATRPeriod > 0 && v.SMAPeriod > 0
}

type Strategy struct {
	Seed             int64
	FeeBps           float32
	SlippageBps      float32
	RiskPct          float32
	Direction        int
	EntryRule        RuleTree
	EntryCompiled    CompiledRule
	ExitRule         RuleTree
	ExitCompiled     CompiledRule
	StopLoss         StopModel
	TakeProfit       TPModel
	Trail            TrailModel
	RegimeFilter     RuleTree
	RegimeCompiled   CompiledRule
	VolatilityFilter VolFilterModel // Volatility regime filter
	MaxHoldBars      int            // time-based exit: max bars to hold a position
	MaxConsecLosses  int            // busted trades protection: stop after N consecutive losses
	CooldownBars     int            // optional: pause after busted streak (0 = no cooldown, stop completely)
	FeatureMapHash   string         // Fingerprint of feature ordering when strategy was created
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
		volFilterToString(s.VolatilityFilter) + "|" +
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
			coarseInt(s.MaxHoldBars, 50),    // Bucket by 50
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
	CatPriceMarket      FeatureCategory = iota // PriceLevel, Oscillator, Momentum
	CatVolumeVolatility                        // Volume, ATR
	CatOther                                   // ZScore, Normalized, EventFlag, etc.
)

// getFeatureCategory returns the category group for a feature type
func getFeatureCategory(featType FeatureType) FeatureCategory {
	switch featType {
	case FeatTypePriceLevel, FeatTypeOscillator, FeatTypeMomentum:
		return CatPriceMarket
	case FeatTypeVolume, FeatTypeATR, FeatTypeVolumeDerived:
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

// getRuleDepth returns the maximum depth of a rule tree
func getRuleDepth(node *RuleNode) int {
	if node == nil {
		return 0
	}
	if node.Op == OpLeaf {
		return 1
	}
	leftDepth := getRuleDepth(node.L)
	rightDepth := getRuleDepth(node.R)
	if leftDepth > rightDepth {
		return leftDepth + 1
	}
	return rightDepth + 1
}

// countNodes returns the total number of nodes in a rule tree
func countNodes(node *RuleNode) int {
	if node == nil {
		return 0
	}
	return 1 + countNodes(node.L) + countNodes(node.R)
}

// hasUglyDeepNesting checks for patterns like AND(OR(AND(NOT(...))))
// Targeted at 3+ levels of mixed nesting, not simple AND(OR(...))
func hasUglyDeepNesting(node *RuleNode, depth int) bool {
	if node == nil {
		return false
	}
	// Only problematic at depth 3+ (like AND(OR(AND(...))))
	if depth >= 3 {
		// Check for mixed nesting (AND child of OR, or vice versa)
		if node.Op == OpAnd || node.Op == OpOr {
			// If we have both AND and OR at this depth with children, it's ugly
			leftHasMixed := hasMixedOpAtDepth(node.L, depth+1)
			rightHasMixed := hasMixedOpAtDepth(node.R, depth+1)
			if leftHasMixed || rightHasMixed {
				return true
			}
		}
	}
	return hasUglyDeepNesting(node.L, depth+1) || hasUglyDeepNesting(node.R, depth+1)
}

// hasMixedOpAtDepth checks if children have different operators (AND vs OR)
func hasMixedOpAtDepth(node *RuleNode, depth int) bool {
	if node == nil || node.Op == OpLeaf || node.Op == OpNot {
		return false
	}
	// Check if children have different operators
	if node.L != nil && node.R != nil && node.L.Op != OpLeaf && node.R.Op != OpLeaf {
		if node.L.Op != node.R.Op {
			return true // AND(OR(...)) or OR(AND(...))
		}
	}
	return false
}

// isFragileFeatureFor1H checks if a feature is too fragile for 1H timeframes
// Fragile features produce sparse signals that don't work well on lower-resolution data
func isFragileFeatureFor1H(name string) bool {
	fragilePrefixes := []string{"FVG", "Swing", "BOS"} // Removed "VolZ"
	for _, prefix := range fragilePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// validateStrategyEventUsage validates that event flags aren't used in problematic ways
// Only blocks the specific bad pattern: Rising/Falling ON sparse event flags
func validateStrategyEventUsage(st Strategy, feats Features) bool {
	// Only check regime filter for the specific bad pattern
	if st.RegimeFilter.Root != nil {
		if hasRisingOrFallingOnEventFlags(st.RegimeFilter.Root, feats) {
			return false // Don't use Rising/Falling on event flags in regime filter
		}
	}

	return true
}

// hasRisingOrFallingOnEventFlags checks if Rising/Falling is used on sparse event flags
// This is the specific bad pattern that causes extreme sparsity
func hasRisingOrFallingOnEventFlags(node *RuleNode, feats Features) bool {
	if node == nil {
		return false
	}

	if node.Op == OpLeaf {
		leaf := node.Leaf

		// Check if this is a Rising/Falling operator
		if leaf.Kind == LeafRising || leaf.Kind == LeafFalling {
			// Check if feature A is a sparse event flag
			if leaf.A >= 0 && leaf.A < len(feats.Names) {
				featName := feats.Names[leaf.A]
				if isSparseEventFeature(featName) {
					return true // Bad: Rising/Falling on event flag
				}
			}
		}

		return false
	}

	// Recursively check children
	return hasRisingOrFallingOnEventFlags(node.L, feats) || hasRisingOrFallingOnEventFlags(node.R, feats)
}

// isSparseEventFeature checks if a feature is a sparse event flag
func isSparseEventFeature(name string) bool {
	sparsePrefixes := []string{
		"SweepUp_", "SweepDown_", // Liquidity sweeps
		"FVG",                              // Fair value gaps
		"BOS",                              // Break of structure
		"Squeeze",                          // BB-KC squeeze (rare event)
		"StochBullCross", "StochBearCross", // Stochastic crosses
	}
	for _, prefix := range sparsePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// hasStaticPriceThreshold checks if a rule tree contains static price/volume thresholds
// that will become invalid as BTC price/volume changes over time
// Examples: LT(LL_100_prev, 6490.39), GT(EMA20, 15466.71), LT(VolEMA50, 1420.66)
// These thresholds only worked during specific historical ranges
func hasStaticPriceThreshold(node *RuleNode, feats Features) bool {
	if node == nil {
		return false
	}

	if node.Op == OpLeaf {
		leaf := node.Leaf

		// Only check threshold-based operators (GT, LT, Between, AbsGT, AbsLT)
		if leaf.Kind != LeafGT && leaf.Kind != LeafLT &&
			leaf.Kind != LeafBetween && leaf.Kind != LeafAbsGT && leaf.Kind != LeafAbsLT {
			return false
		}

		// Get feature name and type
		if leaf.A < 0 || leaf.A >= len(feats.Names) {
			return false
		}
		featName := feats.Names[leaf.A]
		var featType FeatureType
		if leaf.A < len(feats.Types) {
			featType = feats.Types[leaf.A]
		}

		threshold := leaf.X

		// PRICE LEVEL FEATURES: EMA, HMA, BB, KC, Donchian, HH/LL, Swing, Kijun, SuperTrend
		// These should NOT use absolute BTC price thresholds (1000 to 500000 range)
		if featType == FeatTypePriceLevel {
			if threshold >= 1000 && threshold <= 500000 {
				return true // Static BTC price threshold - REJECT
			}
		}

		// VOLUME FEATURES: VolSMA, VolEMA, OBV, VolPerTrade
		// These should NOT use absolute volume thresholds (100 to 100000 range)
		// BTC volume has grown 10x+ from 2017 to 2024
		if featType == FeatTypeVolume || strings.HasPrefix(featName, "Vol") || featName == "OBV" || featName == "VolPerTrade" {
			if threshold >= 100 && threshold <= 100000 {
				return true // Static volume threshold - REJECT
			}
		}

		return false
	}

	// Recursively check children
	if hasStaticPriceThreshold(node.L, feats) {
		return true
	}
	if hasStaticPriceThreshold(node.R, feats) {
		return true
	}
	return false
}

// validateStaticPriceThresholds checks all rules in a strategy for static price thresholds
// Returns true if the strategy is valid (no static price thresholds)
func validateStaticPriceThresholds(st Strategy, feats Features) bool {
	// Check entry rule
	if hasStaticPriceThreshold(st.EntryRule.Root, feats) {
		return false
	}
	// Check regime filter
	if hasStaticPriceThreshold(st.RegimeFilter.Root, feats) {
		return false
	}
	return true
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
		if (node.Leaf.Kind == LeafCrossUp || node.Leaf.Kind == LeafCrossDown || node.Leaf.Kind == LeafBreakUp || node.Leaf.Kind == LeafBreakDown) &&
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

// FIX #3A: validateEntryRuleForComplexity checks if an entry rule has sufficient complexity
// to prevent "always true" strategies like (GT SweepUp_100 > 0)
// Returns true if the rule has at least 2 leaves AND at least one is range/thresholded
func validateEntryRuleForComplexity(n *RuleNode) (leafCount int, hasRangeThreshold bool) {
	if n == nil {
		return 0, false
	}

	if n.Op == OpLeaf {
		leafCount = 1
		// Check if this leaf is a range/threshold type (Between, AbsGT, AbsLT)
		hasRangeThreshold = (n.Leaf.Kind == LeafBetween || n.Leaf.Kind == LeafAbsGT || n.Leaf.Kind == LeafAbsLT)
		return leafCount, hasRangeThreshold
	}

	// Recurse into children
	leftCount, leftHasRange := validateEntryRuleForComplexity(n.L)
	rightCount, rightHasRange := validateEntryRuleForComplexity(n.R)

	leafCount = leftCount + rightCount
	hasRangeThreshold = leftHasRange || rightHasRange

	return leafCount, hasRangeThreshold
}

// PROBLEM C FIX: isDeadBetweenOnlyRule detects rules that are just "Between(feature, a, b)"
// without a second condition, which often leads to edges=0 strategies
// Returns true if the rule is a single Between leaf with no other constraints
func isDeadBetweenOnlyRule(n *RuleNode) bool {
	if n == nil {
		return false
	}

	// Check if this is just a single Between leaf
	if n.Op == OpLeaf && n.Leaf.Kind == LeafBetween {
		return true
	}

	// Check for simple AND tree with one Between leaf and one "always true" condition
	if n.Op == OpAnd {
		leftIsBetweenOnly := (n.L != nil && n.L.Op == OpLeaf && n.L.Leaf.Kind == LeafBetween)
		rightIsTrivial := (n.R == nil) || (n.R.Op == OpLeaf && isAlwaysTrueLeaf(n.R.Leaf))
		if leftIsBetweenOnly && rightIsTrivial {
			return true
		}
		rightIsBetweenOnly := (n.R != nil && n.R.Op == OpLeaf && n.R.Leaf.Kind == LeafBetween)
		leftIsTrivial := (n.L == nil) || (n.L.Op == OpLeaf && isAlwaysTrueLeaf(n.L.Leaf))
		if rightIsBetweenOnly && leftIsTrivial {
			return true
		}
	}

	// Recurse to check deeper structures
	return isDeadBetweenOnlyRule(n.L) || isDeadBetweenOnlyRule(n.R)
}

// isAlwaysTrueLeaf checks if a leaf is likely "always true" (GT/LT with extreme thresholds)
func isAlwaysTrueLeaf(l Leaf) bool {
	if l.Kind == LeafGT && l.X < -1e6 { // GT with very negative threshold
		return true
	}
	if l.Kind == LeafLT && l.X > 1e6 { // LT with very positive threshold
		return true
	}
	return false
}

// randomEntryRuleNode generates an entry rule with constrained tree shape
// to prevent "always true" strategies. Rules:
// 1. Root must be AND (not OR)
// 2. OR operators are limited to maxOrCount per tree
// 3. Between leaves are limited to maxBetweenCount per tree (default 1)
// 4. This forces selectivity and prevents easy OR(...) where one side is always true
func randomEntryRuleNode(rng *rand.Rand, feats Features, depth, maxDepth int, entryOrCount *int, maxOrCount int, entryBetweenCount *int, maxBetweenCount int) *RuleNode {
	// Keep root as AND for "setup + trigger" structure; allow deeper leaves below
	if depth >= maxDepth || (depth > 0 && rng.Float32() < 0.40) {
		// Check if we've reached the Between limit
		allowBetween := (*entryBetweenCount < maxBetweenCount)
		leaf := randomEntryLeaf(rng, feats, allowBetween)
		// Count Between leaves
		if leaf.Kind == LeafBetween {
			*entryBetweenCount++
		}
		return &RuleNode{
			Op:   OpLeaf,
			Leaf: leaf,
		}
	}

	op := OpAnd
	if depth == 0 {
		// Root is forced AND
		return &RuleNode{
			Op: op,
			L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount, maxOrCount, entryBetweenCount, maxBetweenCount),
			R:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount, maxOrCount, entryBetweenCount, maxBetweenCount),
		}
	}
	randOp := rng.Float32()

	// For entry rules: use AND more aggressively, limit OR count
	// NOT has 10% probability when not at leaf level
	if randOp < 0.10 {
		op = OpNot
	} else if randOp < 0.40 && *entryOrCount < maxOrCount {
		// Allow up to maxOrCount ORs per entry rule
		op = OpOr
		*entryOrCount++
	}

	if op == OpNot {
		// NOT has only one child (Left), Right stays empty
		return &RuleNode{
			Op: op,
			L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount, maxOrCount, entryBetweenCount, maxBetweenCount),
			R:  nil,
		}
	}

	return &RuleNode{
		Op: op,
		L:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount, maxOrCount, entryBetweenCount, maxBetweenCount),
		R:  randomEntryRuleNode(rng, feats, depth+1, maxDepth, entryOrCount, maxOrCount, entryBetweenCount, maxBetweenCount),
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
	var validateEntryComplexity func(n *RuleNode) bool

	// FIX #3A: Helper to validate entry rule complexity
	// Require at least 2 leaves AND at least one range/threshold type (Between/AbsGT/AbsLT)
	validateEntryComplexity = func(n *RuleNode) bool {
		leafCount, hasRangeThreshold := validateEntryRuleForComplexity(n)
		return leafCount >= 2 && hasRangeThreshold
	}

	if RecoveryMode.Load() {
		// Helper: count sparse features (SwingHigh, SwingLow, BOS, FVG are sparse)
		// Sparse features produce few trading signals, leading to dead strategies
		countSparseFeatures = func(n *RuleNode) int {
			if n == nil {
				return 0
			}
			if n.Op == OpLeaf {
				featName := feats.Names[n.Leaf.A]
				sparsePrefixes := []string{"Swing", "BOS", "FVG", "Breakout"} // EDIT: Removed BB_Lower, BB_Upper (they're dense, not sparse)
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

	// Detect 1H timeframe to apply simplified rule generation
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	is1H := tfMinutes >= 60

	// Set parameters for entry rule structure
	// Allow "setup + trigger" by increasing depth while keeping root AND and ORs limited
	maxEntryDepth := 3
	maxOrCount := 1
	maxBetweenCount := 1 // Limit Between leaves to max 1 per entry tree to reduce overtrading
	if is1H {
		maxEntryDepth = 3
		maxOrCount = 1
	}

	if RecoveryMode.Load() {
		// Recovery mode: bounded loop to find a rule that passes constraints
		for attempts := 0; attempts < 50; attempts++ {
			entryOrCount := 0
			entryBetweenCount := 0
			entryRoot = randomEntryRuleNode(rng, feats, 0, maxEntryDepth, &entryOrCount, maxOrCount, &entryBetweenCount, maxBetweenCount)

			// Validate sparse feature constraint
			sparseCount := countSparseFeatures(entryRoot)
			if sparseCount > 1 {
				continue // Too many sparse features, try again
			}

			// Validate NOT nesting constraint
			if !checkNOTDepth(entryRoot, 0) {
				continue // NOT nesting too deep, try again
			}

			// PROBLEM C FIX: Reject dead Between-only rules that lead to edges=0
			if isDeadBetweenOnlyRule(entryRoot) {
				continue // Between-only rule without second condition, try again
			}

			// FIX #3A: Validate entry complexity (at least 2 leaves + one range/threshold)
			if !validateEntryComplexity(entryRoot) {
				continue // Too simple (single leaf like "GT X > 0"), try again
			}

			// All constraints passed - break and use this rule
			break
		}
		// If we exhausted attempts, we use the last generated rule (fallback)
	} else {
		// Normal mode: also enforce entry complexity constraint
		for attempts := 0; attempts < 20; attempts++ {
			entryOrCount := 0
			entryBetweenCount := 0
			entryRoot = randomEntryRuleNode(rng, feats, 0, maxEntryDepth, &entryOrCount, maxOrCount, &entryBetweenCount, maxBetweenCount)

			// PROBLEM C FIX: Reject dead Between-only rules that lead to edges=0
			if isDeadBetweenOnlyRule(entryRoot) {
				continue // Between-only rule without second condition, try again
			}

			// FIX #3A: Validate entry complexity (at least 2 leaves + one range/threshold)
			if validateEntryComplexity(entryRoot) {
				break // Found a valid rule
			}
			// If not valid and we have attempts left, try again
		}
		// If exhausted attempts, we use the last generated rule (fallback)
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

	// FIX #2B: Exit rules must use meaningful exit signals (no "always active")
	// For 1H: exits are momentum fade (Falling ROC/RSI/MACD), vol contraction, or trend break
	// This prevents "enter signal + SL/TP only" strategies that create big left-tail months
	exitRoot := randomExitRuleNode(rng, feats, 0, 2)

	// PROBLEM C FIX: Allow optional regime filter
	// In recovery mode, allow nil regime filter (always active) to discover more strategies
	// In normal mode, require a regime filter for stability
	var regimeRoot *RuleNode
	if RecoveryMode.Load() {
		// Recovery mode: 50% chance of no regime filter (always active)
		if rng.Float32() < 0.5 {
			regimeRoot = nil
		} else {
			// Use slightly deeper regime (0-2) for more nuanced conditions
			regimeRoot = randomRuleNode(rng, feats, 0, 2)
		}
	} else {
		// Normal mode: require regime filter with shallow depth (0-1)
		regimeRoot = randomRuleNode(rng, feats, 0, 1)
	}
	// NOTE: nil regimeRoot means "always active" - no regime restriction

	// Randomly set direction: 1 for long-only, -1 for short-only
	// Removed Direction=0 (both) to make scores deterministic for search/test consistency
	direction := 1 // default: long-only
	randDir := rng.Float32()
	if randDir < 0.5 {
		direction = 1 // long-only
	} else {
		direction = -1 // short-only
	}

	// TREND ALIGNMENT: DISABLED - trend guard was causing 0-trade strategies
	// EMA50 > EMA200 can be false for entire periods, blocking all trades
	// Let the evolution find profitable entries without forcing trend alignment
	// ensureTrendGuard(entryRoot, direction, feats, rng)

	// FORCE TRIGGER: Ensure at least one trigger leaf to prevent entry_rate_dead
	ensureTriggerLeaf(entryRoot, rng, feats)
	// FORCE SETUP: Ensure at least one setup leaf for sustained-move alignment
	ensureSetupLeaf(entryRoot, rng, feats)

	// Fix C: Timeframe-aware risk sizing to reduce DD
	// 100% risk for all timeframes (full account position sizing)
	tfMinutes = atomic.LoadInt32(&globalTimeframeMinutes) // Reuse variable from line 781
	riskPct := float32(1.0) // 100% risk for all timeframes

	s := Strategy{
		Seed:             rng.Int63(),
		FeeBps:           feeBps,  // Use specified production costs
		SlippageBps:      slipBps, // Use specified production costs
		RiskPct:          riskPct, // Fix C: Timeframe-aware risk (1% for 1H, 10% for 5min/15min)
		Direction:        direction,
		EntryRule:        RuleTree{Root: entryRoot},
		EntryCompiled:    compileRuleTree(entryRoot),
		ExitRule:         RuleTree{Root: exitRoot},
		ExitCompiled:     compileRuleTree(exitRoot),
		StopLoss:         randomStopModel(rng),
		TakeProfit:       TPModel{}, // Placeholder, will be set below with RR constraint
		Trail:            randomTrailModel(rng),
		RegimeFilter:     RuleTree{Root: regimeRoot},
		RegimeCompiled:   compileRuleTree(regimeRoot),
		VolatilityFilter: randomVolFilterModel(rng), // Volatility regime filter
		MaxHoldBars:      500 + rng.Intn(500),       // 500..999 bars (~2-4 days for 5min) - allow big winners to run
		MaxConsecLosses:  20,                        // stop after 20 consecutive losses
		CooldownBars:     100,                       // Fix B: Raised from 50 to 100 for all timeframes
	}

	// FIX #C: Timeframe-aware cooldown - increase for 1H to reduce overtrading
	// Fix B: Raised cooldown from 150-250 to 200-300 bars to reduce overtrading
	tfMinutes = atomic.LoadInt32(&globalTimeframeMinutes) // Reuse variable from line 781
	if tfMinutes >= 60 {
		// 1H: reduce cooldown to allow more trade opportunities while still avoiding churn
		// 96-168 bars ~= 4-7 days on 1H
		s.CooldownBars = 96 + rng.Intn(73) // 96-168 bars for 1H
		// Shorten max hold to increase turnover without forcing early exits (SL/TP still primary)
		s.MaxHoldBars = 300 + rng.Intn(401) // 300-700 bars for 1H (~12-29 days)
	}

	// CRITICAL FIX #2: Bootstrap mode - use lower cooldown and MaxHoldBars to speed up learning
	// When no elites exist yet, use 0-50 cooldown and 50-150 MaxHoldBars for faster feedback
	if isBootstrapMode() {
		if tfMinutes >= 60 {
			// 1H bootstrap: shorten cooldown to accelerate discovery
			s.CooldownBars = 48 + rng.Intn(49) // 48-96 bars for 1H during bootstrap
		} else {
			// Fix B: 5min/15min bootstrap: raised from 0-50 to 50-100 bars
			s.CooldownBars = 50 + rng.Intn(51) // 50-100 bars during bootstrap for 5min/15min
		}
		if tfMinutes >= 60 {
			s.MaxHoldBars = 180 + rng.Intn(301) // 180-480 bars during bootstrap for 1H
		} else {
			s.MaxHoldBars = 200 + rng.Intn(301) // 200-500 bars during bootstrap (~17-42 hours)
		}
	}

	// CRITICAL FIX #1: Force SL/TP to be same kind and enforce RR >= 2.5
	// EDIT #2: Changed from 1.3 to 2.5 for more realistic RR ratio (trend following needs room to breathe)
	// This prevents negative expectancy when SL is ATR-based and TP is fixed percent
	// which creates TP < SL when ATR is large vs price
	if s.StopLoss.Kind == "atr" {
		s.TakeProfit.Kind = "atr"
		// Enforce RR >= 2.5: TP must be at least 2.5x SL (changed from 1.3)
		minTP := s.StopLoss.ATRMult * 2.5
		if minTP < 3.0 { // Raised from 1.5 to 3.0
			minTP = 3.0
		}
		if s.TakeProfit.ATRMult < minTP {
			s.TakeProfit.ATRMult = minTP + rng.Float32()*3.0 // More variation (changed from 2.0)
		}
		// FIX #2B: Cap TP max to ATR12 (was ATR15) to reduce churn and improve stability
		// 1H gets a wider cap to allow trend capture
		maxTP := float32(12.0)
		if tfMinutes >= 60 {
			maxTP = 18.0
		}
		if s.TakeProfit.ATRMult > maxTP {
			s.TakeProfit.ATRMult = maxTP
		}
		// TP.Value stays 0 for ATR-based
		s.TakeProfit.Value = 0
	} else if s.StopLoss.Kind == "fixed" {
		s.TakeProfit.Kind = "fixed"
		// Enforce RR >= 2.5: TP must be at least 2.5x SL (changed from 1.3)
		minTP := s.StopLoss.Value * 2.5
		if minTP < 1.0 {
			minTP = 1.0 // Still enforce absolute minimum from randomTPModel
		}
		if s.TakeProfit.Value < minTP {
			s.TakeProfit.Value = minTP + rng.Float32()*3.0 // More variation (changed from 2.0)
		}
		// FIX #2B: Cap fixed TP to 25% (was 30%) to reduce churn
		// 1H gets a wider cap to allow trend capture
		maxFixedTP := float32(25)
		if tfMinutes >= 60 {
			maxFixedTP = 35
		}
		if s.TakeProfit.Value > maxFixedTP {
			s.TakeProfit.Value = maxFixedTP
		}
		// TP.ATRMult stays 0 for fixed-based
		s.TakeProfit.ATRMult = 0
	} else {
		// For swing stops, use ATR-based TP with RR >= 2.5 (changed from 1.3)
		// Estimate swing stop in ATR terms (rough approximation)
		estSLATR := float32(2.0) // rough estimate for swing
		s.TakeProfit.Kind = "atr"
		minTP := estSLATR * 2.5 // Changed from 1.3
		if minTP < 3.0 {        // Raised from 1.5 to 3.0
			minTP = 3.0
		}
		s.TakeProfit.ATRMult = minTP + rng.Float32()*3.0 // More variation (changed from 2.0)
		// FIX #2B: Cap swing TP max to ATR12 (was ATR15) to reduce churn
		if s.TakeProfit.ATRMult > 12.0 {
			s.TakeProfit.ATRMult = 12.0
		}
		s.TakeProfit.Value = 0
	}

	// Ensure regime filter is non-empty and compilable
	ensureRegimeFilterValid(rng, &s, feats)

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
// Trigger leaf kinds: Cross/Break/Between/GT/LT
func isTriggerKind(kind LeafKind) bool {
	switch kind {
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown, LeafBetween, LeafGT, LeafLT:
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
		// Check for GT (long uptrend) or LT (short downtrend) trend guard
		if dir == 1 {
			return node.Leaf.Kind == LeafGT && node.Leaf.A == ema50 && node.Leaf.B == ema200
		}
		return node.Leaf.Kind == LeafLT && node.Leaf.A == ema50 && node.Leaf.B == ema200
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

	// FIX: Use GT/LT instead of CrossUp/CrossDown for more trade opportunities
	// CrossUp/CrossDown only trigger at crossover moment (rare)
	// GT/LT trigger whenever trend is in right direction (much more common)
	var guard Leaf
	if dir == 1 {
		// Long: EMA50 > EMA200 (uptrend) - triggers continuously in uptrend
		guard = Leaf{Kind: LeafGT, A: ema50, B: ema200}
	} else {
		// Short: EMA50 < EMA200 (downtrend) - triggers continuously in downtrend
		guard = Leaf{Kind: LeafLT, A: ema50, B: ema200}
	}
	replaceRandomLeaf(root, rng, guard)
}

// hasTriggerLeaf checks if tree contains any trigger leaf (fast or threshold)
func hasTriggerLeaf(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.Op == OpLeaf {
		switch node.Leaf.Kind {
		case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown, LeafBetween, LeafGT, LeafLT:
			return true
		}
		return false
	}
	return hasTriggerLeaf(node.L) || hasTriggerLeaf(node.R)
}

func hasSetupLeaf(node *RuleNode) bool {
	if node == nil {
		return false
	}
	if node.Op == OpLeaf {
		switch node.Leaf.Kind {
		case LeafRising, LeafFalling, LeafSlopeGT, LeafSlopeLT:
			return true
		}
		return false
	}
	return hasSetupLeaf(node.L) || hasSetupLeaf(node.R)
}

func buildSetupLeaf(rng *rand.Rand, feats Features) Leaf {
	setupKinds := []LeafKind{LeafRising, LeafFalling, LeafSlopeGT, LeafSlopeLT}
	kind := setupKinds[rng.Intn(len(setupKinds))]

	a := randomFeatureIndexByGroup(rng, feats, "momentum")
	if a < 0 {
		a = randomFeatureIndex(rng, feats)
	}
	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	leaf := Leaf{Kind: kind, A: a, B: a}
	switch kind {
	case LeafRising, LeafFalling:
		leaf.Lookback = 4 + rng.Intn(9) // 4-12 bars
	case LeafSlopeGT, LeafSlopeLT:
		leaf.Lookback = 5 + rng.Intn(11) // 5-15 bars
		leaf.X = float32(rng.NormFloat64() * 0.05)
		leaf.X = clampThresholdByFeature(leaf.X, featName)
	}
	return leaf
}

// ensureTriggerLeaf forces at least one trigger leaf in entry rule
// This prevents "entry_rate_dead" rejections by ensuring strategy can trigger
func ensureTriggerLeaf(root *RuleNode, rng *rand.Rand, feats Features) {
	if hasTriggerLeaf(root) {
		return // Already has trigger leaf
	}

	// Generate a valid trigger leaf based on kind
	// CRITICAL: Build leaf from scratch to guarantee A != B for Cross/Break
	// OLD BUG: newLeaf := randomEntryLeaf(...); newLeaf.Kind = triggerKind
	// This created CrossUp(A,A) which is always false!
	var triggerKind LeafKind
	roll := rng.Float32()
	switch {
	case roll < 0.25:
		crossKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown}
		triggerKind = crossKinds[rng.Intn(len(crossKinds))]
	case roll < 0.45:
		triggerKind = LeafBetween
	default:
		gtLtKinds := []LeafKind{LeafGT, LeafLT}
		triggerKind = gtLtKinds[rng.Intn(len(gtLtKinds))]
	}

	var newLeaf Leaf
	if triggerKind == LeafCrossUp || triggerKind == LeafCrossDown ||
		triggerKind == LeafBreakUp || triggerKind == LeafBreakDown {
		// For Cross/Break: pick A and compatible B where B != A
		a := randomFeatureIndex(rng, feats)
		var typeA FeatureType
		if a < len(feats.Types) {
			typeA = feats.Types[a]
		}

		// Try to find compatible B
		maxAttempts := 50
		found := false
		var b int
		for attempt := 0; attempt < maxAttempts; attempt++ {
			b = rng.Intn(len(feats.F))
			if b == a {
				continue
			}
			if b < len(feats.Types) {
				typeB := feats.Types[b]
				if canCrossFeatures(typeA, typeB) {
					found = true
					break
				}
			}
		}

		if !found {
			// Fallback to GT/LT (fast operators)
			if rng.Float32() < 0.5 {
				triggerKind = LeafGT
			} else {
				triggerKind = LeafLT
			}
			b = a
		}

		newLeaf = Leaf{
			Kind:     triggerKind,
			A:        a,
			B:        b,
			X:        0, // Set below
			Y:        0,
			Lookback: 0,
		}

		// Set threshold for GT/LT fallback
		if triggerKind == LeafGT || triggerKind == LeafLT {
			if a < len(feats.Stats) && a < len(feats.Names) {
				stats := feats.Stats[a]
				var k float32
				if triggerKind == LeafGT {
					k = 0.25 + rng.Float32()*0.59 // 60th-80th percentile
				} else {
					k = -0.84 + rng.Float32()*0.59 // 20th-40th percentile
				}
				newLeaf.X = stats.Mean + k*stats.Std
				newLeaf.X = clampThresholdByFeature(newLeaf.X, feats.Names[a])
			}
		}

	} else if triggerKind == LeafBetween {
		// For Between: B=A, set X<Y
		a := randomFeatureIndex(rng, feats)
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			k1 := -0.52 + rng.Float32()*0.27 // 30th-40th percentile
			k2 := 0.25 + rng.Float32()*0.27  // 60th-70th percentile
			newLeaf.X = stats.Mean + float32(k1)*stats.Std
			newLeaf.Y = stats.Mean + float32(k2)*stats.Std
			newLeaf.X = clampThresholdByFeature(newLeaf.X, feats.Names[a])
			newLeaf.Y = clampThresholdByFeature(newLeaf.Y, feats.Names[a])
		}
		newLeaf = Leaf{Kind: triggerKind, A: a, B: a, X: newLeaf.X, Y: newLeaf.Y, Lookback: 0}

	} else {
		// For GT/LT: B=A, set threshold from stats
		a := randomFeatureIndex(rng, feats)
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			var k float32
			if triggerKind == LeafGT {
				k = 0.25 + rng.Float32()*0.59
			} else {
				k = -0.84 + rng.Float32()*0.59
			}
			threshold := stats.Mean + k*stats.Std
			threshold = clampThresholdByFeature(threshold, feats.Names[a])
			newLeaf = Leaf{Kind: triggerKind, A: a, B: a, X: threshold, Y: 0, Lookback: 0}
		} else {
			newLeaf = Leaf{Kind: triggerKind, A: a, B: a, X: 50, Y: 0, Lookback: 0}
		}
	}

	replaceRandomLeaf(root, rng, newLeaf)
}

// ensureSetupLeaf forces at least one setup leaf (trend/momentum) in entry rule
func ensureSetupLeaf(root *RuleNode, rng *rand.Rand, feats Features) {
	if hasSetupLeaf(root) {
		return
	}
	setupLeaf := buildSetupLeaf(rng, feats)
	replaceRandomLeaf(root, rng, setupLeaf)
}

// isEntryRuleValid checks if entry rule exists and would compile to non-empty code
// Returns false if EntryRule.Root is nil (empty entry can never open positions)
func isEntryRuleValid(st Strategy) bool {
	return st.EntryRule.Root != nil
}

// repairEntryRule rebuilds a valid entry rule when the current one is empty/nil
// This repairs strategies that would otherwise be dead (can never open positions)
// Uses a simple guaranteed-valid leaf like CrossUp or BreakUp for discovery
func repairEntryRule(rng *rand.Rand, st *Strategy, feats Features) error {
	if st.EntryRule.Root == nil {
		// Build a simple guaranteed-valid entry leaf
		// Prefer simple GT/LT, but allow a trigger or Between for variety
		var kind LeafKind
		roll := rng.Float32()
		switch {
		case roll < 0.40:
			gtLtKinds := []LeafKind{LeafGT, LeafLT}
			kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
		case roll < 0.60:
			kind = LeafBetween
		default:
			triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown}
			kind = triggerKinds[rng.Intn(len(triggerKinds))]
		}

		// Pick two different feature indices for Cross/Break operations
		a := randomFeatureIndex(rng, feats)
		b := a
		if kind == LeafCrossUp || kind == LeafCrossDown || kind == LeafBreakUp || kind == LeafBreakDown {
			for attempts := 0; attempts < 10 && b == a; attempts++ {
				b = randomFeatureIndex(rng, feats)
			}
		}

		// Create a simple valid leaf
		if kind == LeafGT || kind == LeafLT {
			threshold := float32(0)
			if a < len(feats.Stats) && a < len(feats.Names) {
				stats := feats.Stats[a]
				if kind == LeafGT {
					threshold = stats.Mean + 0.25*stats.Std
				} else {
					threshold = stats.Mean - 0.25*stats.Std
				}
				threshold = clampThresholdByFeature(threshold, feats.Names[a])
			}
			st.EntryRule.Root = &RuleNode{
				Op:   OpLeaf,
				Leaf: Leaf{Kind: kind, A: a, B: a, X: threshold, Y: 0, Lookback: 0},
			}
			return nil
		}
		if kind == LeafBetween {
			threshold := float32(0)
			leafY := float32(0)
			if a < len(feats.Stats) && a < len(feats.Names) {
				stats := feats.Stats[a]
				threshold = stats.Mean - 0.25*stats.Std
				leafY = stats.Mean + 0.25*stats.Std
				threshold = clampThresholdByFeature(threshold, feats.Names[a])
				leafY = clampThresholdByFeature(leafY, feats.Names[a])
			}
			st.EntryRule.Root = &RuleNode{
				Op:   OpLeaf,
				Leaf: Leaf{Kind: kind, A: a, B: a, X: threshold, Y: leafY, Lookback: 0},
			}
			return nil
		}
		st.EntryRule.Root = &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind:     kind,
				A:        a,
				B:        b,
				Lookback: 1,
			},
		}

		// Compile the new entry rule
		st.EntryCompiled = compileRuleTree(st.EntryRule.Root)

		// Verify compilation produced non-empty code
		if st.EntryCompiled.Code == nil || len(st.EntryCompiled.Code) == 0 {
			return fmt.Errorf("repaired entry rule failed to compile")
		}

		return nil
	}

	// Entry rule exists but verify it compiles
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	if st.EntryCompiled.Code == nil || len(st.EntryCompiled.Code) == 0 {
		// Compilation failed, rebuild from scratch
		st.EntryRule.Root = nil
		return repairEntryRule(rng, st, feats)
	}

	return nil
}

// validateAndRepairEntryRule checks entry rule validity and repairs if needed
// Returns (isValid true) if entry rule is valid or was successfully repaired
// Returns (isValid false) if entry rule is broken and cannot be repaired
func validateAndRepairEntryRule(rng *rand.Rand, st *Strategy, feats Features) bool {
	if isEntryRuleValid(*st) {
		// Entry rule exists, verify compilation
		st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
		if st.EntryCompiled.Code != nil && len(st.EntryCompiled.Code) > 0 {
			// Enforce setup + trigger mix for sustained entries
			ensureTriggerLeaf(st.EntryRule.Root, rng, feats)
			ensureSetupLeaf(st.EntryRule.Root, rng, feats)
			st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
			return true // Valid
		}
	}

	// Entry rule is nil or compilation failed - attempt repair
	err := repairEntryRule(rng, st, feats)
	if err != nil {
		return false
	}
	ensureTriggerLeaf(st.EntryRule.Root, rng, feats)
	ensureSetupLeaf(st.EntryRule.Root, rng, feats)
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	return true
}

func ensureRegimeFilterValid(rng *rand.Rand, st *Strategy, feats Features) {
	if st.RegimeFilter.Root == nil || (st.RegimeFilter.Root.Op == OpNot && st.RegimeFilter.Root.L == nil) {
		st.RegimeFilter.Root = randomRuleNode(rng, feats, 0, 1)
	}
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)
	if st.RegimeCompiled.Code == nil || len(st.RegimeCompiled.Code) == 0 {
		st.RegimeFilter.Root = randomRuleNode(rng, feats, 0, 1)
		st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)
	}
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

// randomExitRuleNode generates exit rule trees using exit-specific leaves
// Uses randomExitLeaf which enforces meaningful exit signals (no "always active")
func randomExitRuleNode(rng *rand.Rand, feats Features, depth, maxDepth int) *RuleNode {
	if depth >= maxDepth || rng.Float32() < 0.4 {
		// Exit rules are simpler: 60% chance of leaf at any node
		return &RuleNode{
			Op:   OpLeaf,
			Leaf: randomExitLeaf(rng, feats),
		}
	}

	// For exit rules, prefer AND (all conditions must be met for exit)
	op := OpAnd
	if rng.Float32() < 0.15 {
		op = OpOr
	} else if rng.Float32() < 0.20 {
		op = OpNot
	}

	if op == OpNot {
		return &RuleNode{
			Op: op,
			L:  randomExitRuleNode(rng, feats, depth+1, maxDepth),
			R:  nil,
		}
	}

	return &RuleNode{
		Op: op,
		L:  randomExitRuleNode(rng, feats, depth+1, maxDepth),
		R:  randomExitRuleNode(rng, feats, depth+1, maxDepth),
	}
}

func randomLeaf(rng *rand.Rand, feats Features) Leaf {
	// Detect 1H timeframe from global variable
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	is1H := tfMinutes >= 60
	// Prefer triggers (Cross, Rising, Falling) over simple thresholds to capture real moves
	// 60% trigger leaves, 30% threshold leaves, 10% slope leaves
	var kind LeafKind

	// RECOVERY MODE + BOOTSTRAP MODE: Weight toward high-frequency operators to reduce dead strategies
	// When population is empty or in recovery, we need strategies that actually trigger
	if RecoveryMode.Load() || isBootstrapMode() {
		// Bootstrap/Recovery mode: 80% high-freq (biased toward Cross/Break/Between), 15% mid-freq, 5% rare
		// Increased Cross/CrossDown/BreakUp/BreakDown/Between probability to improve entry rate hit rate
		crossKinds := []LeafKind{
			LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown, // Highest entry rate - trigger on price movement
			LeafBetween, // High entry rate - range-based trigger
		}
		highFreqKinds := []LeafKind{
			LeafGT, LeafLT, // Simple comparisons
		}
		midFreqKinds := []LeafKind{
			LeafRising, LeafFalling, // Trend-based (less frequent)
		}
		rareKinds := []LeafKind{
			LeafAbsGT, LeafAbsLT, // Rare by construction
			LeafSlopeGT, LeafSlopeLT, // Trend-based, sparse
		}

		roll := rng.Float32()
		switch {
		case roll < 0.40:
			// 40% Cross/Between
			kind = crossKinds[rng.Intn(len(crossKinds))]
		case roll < 0.70:
			// 30% GT/LT
			kind = highFreqKinds[rng.Intn(len(highFreqKinds))]
		case roll < 0.90:
			// 20% Rising/Falling
			kind = midFreqKinds[rng.Intn(len(midFreqKinds))]
		default:
			// 10% rare
			kind = rareKinds[rng.Intn(len(rareKinds))]
		}
	} else {
		// Normal mode: biased toward Cross/Break/Between for better entry rate
		crossKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown, LeafBetween}
		triggerKinds := []LeafKind{LeafRising, LeafFalling}
		thresholdKinds := []LeafKind{LeafGT, LeafLT, LeafAbsGT, LeafAbsLT}
		slopeKinds := []LeafKind{LeafSlopeGT, LeafSlopeLT}

		if len(feats.F) < 2 {
			// No cross leaves possible - use threshold or rising/falling
			kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
		} else {
			roll := rng.Float32()
			switch {
			case roll < 0.25:
				// 25% Cross/Between
				kind = crossKinds[rng.Intn(len(crossKinds))]
			case roll < 0.55:
				// 30% GT/LT/Abs
				kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
			case roll < 0.80:
				// 25% Rising/Falling
				kind = triggerKinds[rng.Intn(len(triggerKinds))]
			default:
				// 20% slope
				kind = slopeKinds[rng.Intn(len(slopeKinds))]
			}
		}
	}

	// Use smart feature selection that prefers high-frequency features
	a := randomFeatureIndex(rng, feats)

	// Get feature name for threshold clamping
	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	// FIXED: Use loop instead of recursion to avoid stack blowup
	// For 1H: skip fragile feature types
	if is1H {
		for attempts := 0; attempts < 50; attempts++ {
			if !isFragileFeatureFor1H(featName) {
				break // Found non-fragile feature
			}
			// Try again with different feature
			a = randomFeatureIndex(rng, feats)
			if a < len(feats.Names) {
				featName = feats.Names[a]
			} else {
				break // Safety: index out of range
			}
		}
		// After 50 attempts, use whatever we got (fallback)
	}

	// CRITICAL FIX #7: Ban absolute price thresholds on PriceLevel features
	// PriceLevel features (EMA*, BB_Upper/Lower*, SwingHigh/Low) should NOT use
	// absolute thresholds because BTC price scale changes massively from 2017→2026
	if a < len(feats.Types) && feats.Types[a] == FeatTypePriceLevel && isAbsoluteThresholdKind(kind) {
		// Force a Cross operator (relative comparison, NOT Rising/Falling!)
		// Rising/Falling are too slow and cause low entry rates
		triggerKinds := []LeafKind{LeafCrossUp, LeafCrossDown}
		if len(feats.F) < 2 {
			// No cross leaves possible - use GT/LT instead (better than Rising/Falling)
			gtLtKinds := []LeafKind{LeafGT, LeafLT}
			kind = gtLtKinds[rng.Intn(len(gtLtKinds))]
		} else {
			kind = triggerKinds[rng.Intn(len(triggerKinds))]
		}
	}

	var b int
	var lookback int
	var threshold float32
	var leafY float32 // For Between leaf (high threshold)

	switch kind {
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
		// FIX B1: CRITICAL - All cross/break operators require A != B
		// BreakUp/BreakDown with A == B is always false (A[t-1] <= A[t-1] && A[t] > A[t])
		// CrossUp/CrossDown with A == B is also always false
		// Keep picking B until it's different from A AND compatible types
		var typeA FeatureType
		if a < len(feats.Types) {
			typeA = feats.Types[a]
		}
		maxAttempts := 100
		found := false
		for attempt := 0; attempt < maxAttempts; attempt++ {
			b = rng.Intn(len(feats.F))
			if b == a {
				continue // FIX B1: Must have A != B
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
		// FIX: If no compatible B found, fall back to GT/LT (fast threshold operators)
		// OLD BUG: kind = LeafRising - too slow, causes edges=0 in WF folds
		// NEW: Use GT/LT with immediate threshold computation
		if !found {
			// Fallback to fast threshold operator
			var thresholdValue float32
			if rng.Float32() < 0.5 {
				kind = LeafGT
				// 60th-80th percentile
				k := 0.25 + rng.Float32()*0.59
				thresholdValue = feats.Stats[a].Mean + k*feats.Stats[a].Std
			} else {
				kind = LeafLT
				// 20th-40th percentile
				k := -0.84 + rng.Float32()*0.59
				thresholdValue = feats.Stats[a].Mean + k*feats.Stats[a].Std
			}
			if a < len(feats.Names) {
				thresholdValue = clampThresholdByFeature(thresholdValue, feats.Names[a])
			}
			b = a
			lookback = 0
			threshold = thresholdValue

			// Return immediately - don't rely on default case
			// Fall through to Leaf below to handle GT/LT setup
		}
		lookback = 0  // Cross/Break leaves don't use lookback
		threshold = 0 // Cross/Break leaves don't use threshold
	case LeafRising, LeafFalling:
		// FIX 2: Very short lookback for faster triggers: 2-5 bars (was 2-10)
		lookback = rng.Intn(4) + 2 // 2, 3, 4, or 5 bars only
		threshold = 0              // Rising/Falling don't use threshold
	case LeafBetween:
		// Fix C: Prefer normalized volume features over absolute volume for Between leaves
		// Between(VolEMA20, 2892, 3247) is brittle due to BTC volume scaling over time
		// Use VolZ20, BuyRatio, Imbalance instead for regime-agnostic volume filtering
		if a < len(feats.Names) {
			featName = feats.Names[a]
		}
		// Check if feature is absolute volume (brittle for Between)
		isAbsVolume := strings.HasPrefix(featName, "VolSMA") ||
			strings.HasPrefix(featName, "VolEMA") ||
			featName == "OBV" || featName == "VolPerTrade"
		// If absolute volume, 70% chance to reroll to normalized volume feature
		if isAbsVolume && rng.Float32() < 0.70 {
			// Find normalized volume features: VolZ, BuyRatio, Imbalance
			normalizedVolFeatures := []string{"VolZ20", "VolZ50", "BuyRatio", "Imbalance"}
			candidates := []int{}
			for _, normName := range normalizedVolFeatures {
				if idx, ok := feats.Index[normName]; ok {
					candidates = append(candidates, idx)
				}
			}
			if len(candidates) > 0 {
				a = candidates[rng.Intn(len(candidates))]
				if a < len(feats.Names) {
					featName = feats.Names[a]
				}
			}
		}

		// Between needs two thresholds (X and Y)
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]

			// FIX C: Handle low-variance features (binary features like SweepUp, FVG, etc.)
			// When Std ≈ 0, quantile-based thresholds collapse to same value → degenerate rule
			const lowVarThreshold = 0.001
			if stats.Std < lowVarThreshold {
				// For binary/low-variance features, use explicit meaningful ranges
				// Event flags (0 or 1): use [0, 0.1], [0.1, 0.5], or [0.5, 1]
				if strings.HasPrefix(featName, "Sweep") || strings.HasPrefix(featName, "FVG") ||
					featName == "BOS" || featName == "Active" {
					// Randomly pick one of three meaningful ranges for binary features
					rangeChoice := rng.Intn(3)
					switch rangeChoice {
					case 0:
						threshold, leafY = 0, 0.1 // "mostly false" range
					case 1:
						threshold, leafY = 0.1, 0.5 // "sometimes" range
					case 2:
						threshold, leafY = 0.5, 1.0 // "mostly true" range
					}
				} else {
					// For other low-variance features, use a small fixed width around the mean
					halfWidth := float32(0.05)
					threshold = stats.Mean - halfWidth
					leafY = stats.Mean + halfWidth
					// Clamp to feature-specific bounds
					threshold = clampThresholdByFeature(threshold, featName)
					leafY = clampThresholdByFeature(leafY, featName)
				}
			} else {
				// PROBLEM C FIX: Use quantile-based thresholds (Q30-Q70) instead of random
				// This ensures Between thresholds are always in meaningful ranges
				// Low threshold: 30th-40th percentile (below mean)
				k1 := -0.52 + rng.Float32()*0.27 // range: -0.52 to -0.25 (approx 30th-40th percentile)
				// High threshold: 60th-70th percentile (above mean)
				k2 := 0.25 + rng.Float32()*0.27 // range: 0.25 to 0.52 (approx 60th-70th percentile)
				threshold = stats.Mean + float32(k1)*stats.Std
				leafY = stats.Mean + float32(k2)*stats.Std

				// CRITICAL FIX #4: Clamp to feature-specific bounds
				threshold = clampThresholdByFeature(threshold, featName)
				leafY = clampThresholdByFeature(leafY, featName)

				// Ensure X < Y for "Between" (low < high)
				if threshold > leafY {
					threshold, leafY = leafY, threshold
				}

				// FIX #4A: Ban Between(x, x) where min==max (degenerate "always true/false" rule)
				// Force minimum width epsilon based on feature type
				minWidth := float32(0.01) // Default: 1% minimum width
				if strings.Contains(featName, "Pct") || strings.Contains(featName, "Ratio") ||
					strings.Contains(featName, "RSI") || strings.Contains(featName, "MFI") {
					minWidth = 0.01 // 1% for percentage/ratio features
				} else {
					minWidth = 0.05 * stats.Std // For normalized features, use 5% of Std as minimum
				}
				if leafY-threshold < minWidth {
					// Band too narrow - widen it by moving Y away from X
					leafY = threshold + minWidth
					// Re-clamp Y to feature bounds
					leafY = clampThresholdByFeature(leafY, featName)
					// If still too narrow after clamping, swap direction
					if leafY-threshold < minWidth {
						leafY = threshold
						threshold = leafY - minWidth
						threshold = clampThresholdByFeature(threshold, featName)
					}
				}
			}
		} else {
			// FIX C: Fallback case - ensure threshold starts > 0 to avoid Between(x, 0, 0)
			// Start from a positive base, not centered at 0
			threshold = float32(rng.NormFloat64()*10 + 25) // Mean=25, allows some negatives but rare
			if threshold < 1.0 {
				threshold = 1.0 // Minimum threshold of 1.0
			}
			leafY = threshold + float32(rng.NormFloat64()*10+5) // Add at least ~5 width
			if leafY <= threshold {
				leafY = threshold + 5.0
			}
			// CRITICAL FIX #4: Clamp to feature-specific bounds
			threshold = clampThresholdByFeature(threshold, featName)
			leafY = clampThresholdByFeature(leafY, featName)

			// FIX #4A: Ensure minimum width for fallback case too
			minWidth := float32(1.0) // 1.0 for absolute scales
			if leafY-threshold < minWidth {
				leafY = threshold + minWidth
				leafY = clampThresholdByFeature(leafY, featName)
			}
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
		lookback = rng.Intn(9) + 2                   // 2-10 bars (reduced from 5-20)
		threshold = float32(rng.NormFloat64() * 0.1) // Small slope values
		// CRITICAL FIX #4: Clamp to feature-specific bounds
		threshold = clampThresholdByFeature(threshold, featName)
	default:
		// For GT/LT, use quantile-based thresholds to avoid "always true" leaves
		// Use Q20-Q80 range to reduce "entry rate too low" failures
		b = a
		if a < len(feats.Stats) {
			stats := feats.Stats[a]
			// Use quantiles instead of raw random values to make conditions naturally selective
			// LT (20th-40th percentile): Mean - (0.84 to 0.25) * Std
			// GT (60th-80th percentile): Mean + (0.25 to 0.84) * Std
			var k float32
			if kind == LeafLT {
				// LT: pick from 20th-40th percentile (below mean)
				k = -0.84 + rng.Float32()*0.59 // range: -0.84 to -0.25 (approx 20th-40th percentile)
			} else { // LeafGT
				// GT: pick from 60th-80th percentile (above mean)
				k = 0.25 + rng.Float32()*0.59 // range: 0.25 to 0.84 (approx 60th-80th percentile)
			}
			threshold = stats.Mean + k*stats.Std

			// Bounds check to prevent extreme values
			if stats.Std > 0 {
				minThreshold := stats.Mean - stats.Std
				maxThreshold := stats.Mean + stats.Std
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

// countBetweenLeaves counts LeafBetween nodes in a rule tree
func countBetweenLeaves(node *RuleNode) int {
	if node == nil {
		return 0
	}
	if node.Op == OpLeaf && node.Leaf.Kind == LeafBetween {
		return 1
	}
	return countBetweenLeaves(node.L) + countBetweenLeaves(node.R)
}

// randomEntryLeaf generates entry leaves with a bias toward sustained-move setups
// and simple thresholds, while still allowing fast triggers.
// allowBetween controls whether Between leaves are allowed (used to limit to max 1 per entry tree)
func randomEntryLeaf(rng *rand.Rand, feats Features, allowBetween bool) Leaf {
	fastCrossKinds := []LeafKind{LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown}
	triggerRangeKinds := []LeafKind{LeafBetween}
	thresholdKinds := []LeafKind{LeafGT, LeafLT}
	setupKinds := []LeafKind{LeafRising, LeafFalling, LeafSlopeGT, LeafSlopeLT}

	if !allowBetween {
		triggerRangeKinds = nil
	}

	var kind LeafKind
	p := rng.Float32()
	switch {
	case p < 0.20: // 20% Cross/Break (reduced to cut noise)
		kind = fastCrossKinds[rng.Intn(len(fastCrossKinds))]
	case p < 0.30 && len(triggerRangeKinds) > 0: // 10% Between
		kind = triggerRangeKinds[rng.Intn(len(triggerRangeKinds))]
	case p < 0.65: // 35% GT/LT
		kind = thresholdKinds[rng.Intn(len(thresholdKinds))]
	case p < 0.85: // 20% Rising/Falling
		kind = setupKinds[rng.Intn(2)]
	default: // 15% Slope
		kind = setupKinds[2+rng.Intn(2)]
	}

	// Bias feature selection by leaf type to combine trend/momentum + vol state + structure
	a := -1
	switch kind {
	case LeafRising, LeafFalling, LeafSlopeGT, LeafSlopeLT:
		if rng.Float32() < 0.70 {
			a = randomFeatureIndexByGroup(rng, feats, "momentum")
		}
	case LeafBetween:
		if rng.Float32() < 0.70 {
			a = randomFeatureIndexByGroup(rng, feats, "vol")
		}
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
		if rng.Float32() < 0.70 {
			a = randomFeatureIndexByGroup(rng, feats, "structure")
		}
	case LeafGT, LeafLT:
		roll := rng.Float32()
		if roll < 0.45 {
			a = randomFeatureIndexByGroup(rng, feats, "momentum")
		} else if roll < 0.75 {
			a = randomFeatureIndexByGroup(rng, feats, "vol")
		}
	}
	if a < 0 {
		a = randomFeatureIndex(rng, feats)
	}

	featName := ""
	if a < len(feats.Names) {
		featName = feats.Names[a]
	}

	leaf := Leaf{Kind: kind, A: a, B: a}

	// 1H: avoid fragile features for entry rules
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	if tfMinutes >= 60 {
		for attempts := 0; attempts < 30; attempts++ {
			if !isFragileFeatureFor1H(featName) {
				break
			}
			a = randomFeatureIndex(rng, feats)
			leaf.A = a
			leaf.B = a
			if a < len(feats.Names) {
				featName = feats.Names[a]
			} else {
				break
			}
		}
	}

	// Setup leaves: longer lookbacks to align with sustained moves
	switch kind {
	case LeafRising, LeafFalling:
		leaf.Lookback = 4 + rng.Intn(9) // 4-12 bars
		return leaf
	case LeafSlopeGT, LeafSlopeLT:
		leaf.Lookback = 5 + rng.Intn(11) // 5-15 bars
		leaf.X = float32(rng.NormFloat64() * 0.05)
		leaf.X = clampThresholdByFeature(leaf.X, featName)
		return leaf
	}

	// Cross/Break leaves: ensure A != B and compatible
	if kind == LeafCrossUp || kind == LeafCrossDown || kind == LeafBreakUp || kind == LeafBreakDown {
		b := getCompatibleFeatureIndex(rng, feats, a, featName)
		if b == a {
			for attempts := 0; attempts < 10 && b == a; attempts++ {
				b = randomFeatureIndex(rng, feats)
			}
		}
		leaf.B = b
		leaf.Lookback = 0
		leaf.X = 0
		leaf.Y = 0
		return leaf
	}

	// For thresholds: use "near mean" instead of extreme quantiles
	if a < len(feats.Stats) {
		st := feats.Stats[a]
		if st.Std > 0 {
			k := (rng.Float32()*1.2 - 0.6) // -0.6 to +0.6 sigma (centered)
			leaf.X = st.Mean + k*st.Std
			// FIX #4A: Ban thresholds at exact 0 for normalized features
			// Add small epsilon if threshold is exactly 0 (unless feature is naturally 0-centered)
			if leaf.X == 0 && !strings.Contains(featName, "Diff") && !strings.Contains(featName, "Osc") {
				leaf.X = 0.01 * st.Std // Small offset from 0
			}
			if kind == LeafBetween {
				leaf.Y = leaf.X + rng.Float32()*0.5*st.Std // Small band
				// FIX #4A: Ensure minimum width for Between leaves
				minWidth := float32(0.01) // 1% minimum
				if strings.Contains(featName, "Pct") || strings.Contains(featName, "Ratio") {
					minWidth = 0.01 // 1% for percentages
				} else {
					minWidth = 0.05 * st.Std // For other features, use 5% of Std
				}
				if leaf.Y-leaf.X < minWidth {
					leaf.Y = leaf.X + minWidth
				}
				// PROBLEM A FIX: Ensure Y > X (high > low) - reject invalid bounds
				// This prevents "Between X 23420.88 0.00" type junk strategies
				if leaf.Y <= leaf.X {
					// Invalid bounds - swap to make valid
					leaf.X, leaf.Y = leaf.Y, leaf.X
					// Ensure minimum width after swap
					if leaf.Y-leaf.X < minWidth {
						leaf.Y = leaf.X + minWidth
					}
				}
				// PROBLEM A FIX: Reject zero bounds unless feature is naturally 0-centered
				// Features like Diff/Osc are naturally near 0, but others should not be
				isZeroCentered := strings.Contains(featName, "Diff") ||
					strings.Contains(featName, "Osc") ||
					strings.Contains(featName, "Change") ||
					strings.Contains(featName, "Delta")
				if !isZeroCentered && (leaf.X == 0 || leaf.Y == 0) {
					// Add minimum offset to avoid zero bounds
					minOffset := 0.01 * st.Std
					if minOffset == 0 {
						minOffset = 0.01 // Fallback if Std is 0
					}
					if leaf.X == 0 {
						leaf.X = minOffset
					}
					if leaf.Y == 0 {
						leaf.Y = leaf.X + minOffset
					}
				}
			}
			leaf.X = clampThresholdByFeature(leaf.X, featName)
			leaf.Y = clampThresholdByFeature(leaf.Y, featName)
		}
	}

	// Fallback for when stats are not available or Std <= 0
	// This can happen for some features or during bootstrap
	if kind == LeafBetween && (leaf.X == 0 || leaf.Y == 0 || leaf.Y <= leaf.X) {
		// Generate valid bounds using feature-specific defaults
		isZeroCentered := strings.Contains(featName, "Diff") ||
			strings.Contains(featName, "Osc") ||
			strings.Contains(featName, "Change") ||
			strings.Contains(featName, "Delta")

		if isZeroCentered {
			// For zero-centered features, use symmetric bounds around 0
			leaf.X = -0.01
			leaf.Y = 0.01
		} else {
			// For non-zero-centered features, use small positive bounds
			leaf.X = 0.01
			leaf.Y = 0.02
		}
		leaf.X = clampThresholdByFeature(leaf.X, featName)
		leaf.Y = clampThresholdByFeature(leaf.Y, featName)
	}

	// FIX B1: For Cross/Break operations, ensure A and B are type-compatible
	// BreakUp/BreakDown with A == B is always false (degenerate rule)
	if kind == LeafCrossUp || kind == LeafCrossDown || kind == LeafBreakUp || kind == LeafBreakDown {
		// Get feature B that's compatible with A (same type category)
		b := getCompatibleFeatureIndex(rng, feats, a, featName)
		if b == a {
			// Couldn't find compatible feature, pick different one
			for attempts := 0; attempts < 5 && b == a; attempts++ {
				b = randomFeatureIndex(rng, feats)
			}
		}
		leaf.B = b
	}

	return leaf
}

// randomExitLeaf generates exit leaves that enforce meaningful exit signals (no "always active" exits)
// For 1H timeframe: exits must be momentum fade, volatility contraction, or trend break
// This prevents "enter signal + SL/TP only" strategies that create big left-tail months
func randomExitLeaf(rng *rand.Rand, feats Features) Leaf {
	// Detect 1H timeframe from global variable
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	is1H := tfMinutes >= 60

	// Exit leaf types - organized by category
	// Momentum fade: exit when momentum drops (Falling ROC, RSI drops, MACD cross down)
	momentumFadeKinds := []LeafKind{
		LeafFalling,   // Trend reverses (e.g., Falling ROC10)
		LeafCrossDown, // Cross below threshold (e.g., RSI crosses below 70)
		LeafLT,        // Below threshold (e.g., MACD_Hist < 0)
	}
	// Volatility contraction: exit when vol drops or trend breaks
	volContractionKinds := []LeafKind{
		LeafCrossDown, // ATR drops below threshold
		LeafLT,        // BB Width below threshold (squeeze)
		LeafFalling,   // ATR falling
	}
	// Trend break: exit when price cross below MA
	trendBreakKinds := []LeafKind{
		LeafCrossDown, // Price crosses below EMA
		LeafLT,        // Price below EMA
	}

	var kind LeafKind
	var featureGroup string // "momentum", "volatility", or "structure"

	// For 1H: enforce strict exit categories
	if is1H {
		p := rng.Float32()
		switch {
		case p < 0.65: // 65% momentum fade (primary exit type)
			kind = momentumFadeKinds[rng.Intn(len(momentumFadeKinds))]
			featureGroup = "momentum" // ROC, RSI, MACD, etc.
		case p < 0.90: // 25% volatility contraction
			kind = volContractionKinds[rng.Intn(len(volContractionKinds))]
			featureGroup = "volatility" // ATR, BB_Width
		default: // 10% trend break
			kind = trendBreakKinds[rng.Intn(len(trendBreakKinds))]
			featureGroup = "structure" // Price, EMA
		}
	} else {
		// For lower timeframes: more lenient but still avoid "always active"
		p := rng.Float32()
		switch {
		case p < 0.50: // 50% momentum fade
			kind = momentumFadeKinds[rng.Intn(len(momentumFadeKinds))]
			featureGroup = "momentum"
		case p < 0.75: // 25% volatility contraction
			kind = volContractionKinds[rng.Intn(len(volContractionKinds))]
			featureGroup = "volatility"
		default: // 25% trend break
			kind = trendBreakKinds[rng.Intn(len(trendBreakKinds))]
			featureGroup = "structure"
		}
	}

	// Select feature based on group
	a := randomFeatureIndexByGroup(rng, feats, featureGroup)
	// CRITICAL FIX: Handle case where feature group returns -1 (no features in that group)
	if a < 0 {
		// Fallback to random feature from any group
		a = randomFeatureIndex(rng, feats)
	}

	featName := ""
	if a >= 0 && a < len(feats.Names) {
		featName = feats.Names[a]
	}

	leaf := Leaf{Kind: kind, A: a, B: a}

	// 1H: avoid fragile features for exit rules too
	if is1H {
		for attempts := 0; attempts < 30; attempts++ {
			if !isFragileFeatureFor1H(featName) {
				break
			}
			a = randomFeatureIndexByGroup(rng, feats, featureGroup)
			// CRITICAL FIX: Handle -1 return from group lookup
			if a < 0 {
				a = randomFeatureIndex(rng, feats)
			}
			leaf.A = a
			leaf.B = a
			if a >= 0 && a < len(feats.Names) {
				featName = feats.Names[a]
			} else {
				break
			}
		}
	}

	// Setup leaves based on kind
	switch kind {
	case LeafFalling:
		// Short lookback for faster exit response (2-6 bars)
		leaf.Lookback = 2 + rng.Intn(5) // 2-6 bars
		leaf.X = 0
		leaf.Y = 0
		return leaf

	case LeafCrossDown, LeafCrossUp:
		// Cross leaves require A != B
		// For exit: use A crossing below/above B
		var b int
		if kind == LeafCrossDown || kind == LeafCrossUp {
			// Get compatible B feature
			b = getCompatibleFeatureIndex(rng, feats, a, featName)
			if b == a || b < 0 {
				// Fallback: pick different feature
				for attempts := 0; attempts < 10 && (b == a || b < 0); attempts++ {
					b = randomFeatureIndexByGroup(rng, feats, featureGroup)
					if b < 0 {
						b = randomFeatureIndex(rng, feats)
					}
				}
			}
		}
		leaf.B = b
		leaf.Lookback = 0
		leaf.X = 0
		leaf.Y = 0
		return leaf

	case LeafGT, LeafLT:
		// Threshold leaves - use selective thresholds to avoid "always true"
		if a >= 0 && a < len(feats.Stats) {
			st := feats.Stats[a]
			if st.Std > 0 {
				// For exits (LT): lower percentile threshold (20th-40th)
				// For exits (GT): higher percentile (60th-80th)
				var k float32
				if kind == LeafLT {
					// Below mean: 20th-40th percentile
					k = -0.84 + rng.Float32()*0.59 // -0.84 to -0.25
				} else {
					// Above mean: 60th-80th percentile
					k = 0.25 + rng.Float32()*0.59 // 0.25 to 0.84
				}
				leaf.X = st.Mean + k*st.Std

				// Clamp to feature bounds
				leaf.X = clampThresholdByFeature(leaf.X, featName)
			} else {
				// Fallback for zero Std
				leaf.X = st.Mean * 0.95 // Slightly below/above mean
			}
		} else {
			// Fallback for missing stats
			leaf.X = 0
		}
		leaf.Lookback = 0
		leaf.Y = 0
		return leaf

	default:
		// Fallback: simple threshold
		if a < len(feats.Stats) {
			leaf.X = feats.Stats[a].Mean * 0.95
		}
		return leaf
	}
}

func randomStopModel(rng *rand.Rand) StopModel {
	// Fix C: Timeframe-aware stop loss to reduce DD for 1H
	// 1H: tighter stops (1.5-4.5x ATR, 3-6% fixed)
	// 5min/15min: wider stops (2-6x ATR, 4-8% fixed)
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	is1H := tfMinutes >= 60

	// FIX #1: Enforce ATR-based stops (98% ATR, 2% fixed with strict minimum)
	kind := rng.Intn(100)
	if kind < 98 { // 98% ATR (increased from 70%)
		var atrMin, atrMax, atrMean float64
		if is1H {
			atrMin, atrMax, atrMean = 1.5, 4.5, 3.0 // Fix C: Tighter for 1H
		} else {
			atrMin, atrMax, atrMean = 2.0, 6.0, 4.5 // Normal for 5min/15min
		}
		atr := rng.NormFloat64()*1.5 + atrMean
		if atr < atrMin {
			atr = atrMin
		}
		if atr > atrMax {
			atr = atrMax
		}
		return StopModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Value:   0,
		}
	} else { // 2% fixed - must be large to match ATR minimum
		var valMin, valMax, valMean float64
		if is1H {
			valMin, valMax, valMean = 3.0, 6.0, 4.5 // Fix C: 3-6% for 1H (was 4-8%)
		} else {
			valMin, valMax, valMean = 4.0, 8.0, 6.0 // Normal: 4-8% for 5min/15min
		}
		val := rng.NormFloat64()*1.0 + valMean
		if val < valMin {
			val = valMin
		}
		if val > valMax {
			val = valMax
		}
		return StopModel{
			Kind:    "fixed",
			Value:   float32(val),
			ATRMult: 0,
		}
	}
}

func randomTPModel(rng *rand.Rand) TPModel {
	// Fix C: Timeframe-aware take profit to reduce DD for 1H
	// 1H: wider TP to allow trend capture (6-16x ATR, 8-20% fixed)
	// 5min/15min: wider TP (6-12x ATR, 5-15% fixed)
	tfMinutes := atomic.LoadInt32(&globalTimeframeMinutes)
	is1H := tfMinutes >= 60

	// FIX #1: Enforce ATR-based TPs (99% ATR, 1% fixed with strict minimum)
	// Fixed TPs must scale with SL to maintain RR >= 2.5
	kind := rng.Intn(100)
	if kind < 99 { // 99% ATR (increased from 95%)
		var atrMin, atrMax, atrMean float64
		if is1H {
			atrMin, atrMax, atrMean = 6.0, 16.0, 10.0 // 1H: broader to capture larger moves
		} else {
			atrMin, atrMax, atrMean = 6.0, 12.0, 9.0 // Normal for 5min/15min
		}
		atr := rng.NormFloat64()*2.0 + atrMean
		if atr < atrMin {
			atr = atrMin
		}
		if atr > atrMax {
			atr = atrMax
		}
		return TPModel{
			Kind:    "atr",
			ATRMult: float32(atr),
			Value:   0,
		}
	} else { // 1% fixed - must be large to maintain RR
		var valMin, valMax, valMean float64
		if is1H {
			valMin, valMax, valMean = 8.0, 20.0, 12.0 // 1H: broader to capture larger moves
		} else {
			valMin, valMax, valMean = 5.0, 15.0, 10.0 // Normal: 5-15% for 5min/15min
		}
		val := rng.NormFloat64()*2.5 + valMean
		if val < valMin {
			val = valMin
		}
		if val > valMax {
			val = valMax
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

// randomVolFilterModel generates a volatility regime filter
// Only trades when ATR14 is above its historical average (high volatility periods)
func randomVolFilterModel(rng *rand.Rand) VolFilterModel {
	// REDUCED to 40% enable (was 70%) to allow more trades
	// 60% chance to disable (trade all the time)
	enabled := rng.Float32() < 0.4

	if !enabled {
		return VolFilterModel{
			Enabled:   false,
			ATRPeriod: 14,
			SMAPeriod: 50,
			Threshold: 1.0,
		}
	}

	// Generate random parameters for enabled filter
	// ATR period: mostly 14 (standard)
	atrPeriod := 14

	// SMA period: 30-70 bars (mostly around 50)
	smaPeriod := 30 + rng.Intn(41) // 30-70

	// Threshold: 0.5-1.0 (was 0.8-1.3) - LOWERED to allow more trades
	// 0.7 = ATR must be 70% of average (more permissive)
	// 1.0 = ATR must be above average
	threshold := 0.5 + rng.Float32()*0.5

	return VolFilterModel{
		Enabled:   true,
		ATRPeriod: atrPeriod,
		SMAPeriod: smaPeriod,
		Threshold: threshold,
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
		LeafBreakUp:   "BreakUp",
		LeafBreakDown: "BreakDown",
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
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
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
		LeafBreakUp:   "BreakUp",
		LeafBreakDown: "BreakDown",
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
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
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
		BarIndex:    t,
		GuardChecks: []string{},
		Values:      []float64{},
		Comparisons: []string{},
		LeafNode:    *leaf, // Store original leaf for bytecode re-evaluation
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
		LeafBreakUp:   "BreakUp",   // NEW
		LeafBreakDown: "BreakDown", // NEW
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
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
		proof.FeatureB = getFeatName(leaf.B)
	}

	// FIX: Cross operators need t >= 1 to access t-1 values
	// Check this BEFORE accessing previous values (prevents t=0 crash)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
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

	if t < leaf.Lookback && leaf.Kind != LeafCrossUp && leaf.Kind != LeafCrossDown && leaf.Kind != LeafBreakUp && leaf.Kind != LeafBreakDown {
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
	// BUG FIX: Only check leaf.B for operators that actually use it (CrossUp, CrossDown, BreakUp, BreakDown)
	// SlopeGT, GT, LT, Rising, Falling operators don't use leaf.B (it's -1)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
		if leaf.B < 0 || leaf.B >= len(features) {
			proof.GuardChecks = append(proof.GuardChecks, fmt.Sprintf("feature B index %d valid: false", leaf.B))
			proof.Result = false
			return false, proof
		}
	}

	fa := features[leaf.A]
	var fb []float32
	// Only access leaf.B for operators that actually use it (CrossUp, CrossDown, BreakUp, BreakDown)
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
		fb = features[leaf.B]
	}

	aVal := fa[t]
	var bVal float32
	if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
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
	case LeafBreakUp:
		// NO eps/movement check - pure breakout detection
		prevA := fa[t-1]
		prevB := fb[t-1]

		proof.Operator = "BreakUp"
		proof.Values = []float64{float64(prevA), float64(prevB), float64(aVal), float64(bVal)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t-1] <= B[t-1]: %.2f <= %.2f: %v", prevA, prevB, prevA <= prevB))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t] > B[t]: %.2f > %.2f: %v", aVal, bVal, aVal > bVal))

		result = prevA <= prevB && aVal > bVal
	case LeafBreakDown:
		// NO eps/movement check - pure breakdown detection
		prevA := fa[t-1]
		prevB := fb[t-1]

		proof.Operator = "BreakDown"
		proof.Values = []float64{float64(prevA), float64(prevB), float64(aVal), float64(bVal)}
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t-1] >= B[t-1]: %.2f >= %.2f: %v", prevA, prevB, prevA >= prevB))
		proof.Comparisons = append(proof.Comparisons, fmt.Sprintf("A[t] < B[t]: %.2f < %.2f: %v", aVal, bVal, aVal < bVal))

		result = prevA >= prevB && aVal < bVal
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
		LeafBreakUp:   "BreakUp",
		LeafBreakDown: "BreakDown",
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
	case LeafCrossUp, LeafCrossDown, LeafBreakUp, LeafBreakDown:
		featB := getFeatName(leaf.B)
		return "(" + kindName + " " + featA + " " + featB + ")"
	case LeafRising, LeafFalling:
		return "(" + kindName + " " + featA + " " + fmt.Sprintf("%d", leaf.Lookback) + ")"
	default:
		return "(" + kindName + " " + featA + ")"
	}
}
