package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"strings"
)

type FeatureStats struct {
	Min  float32
	Max  float32
	Mean float32
	Std  float32
}

// FeatureType defines the semantic type/scale of a feature for operator validation
// CRITICAL FIX #5: Typed features prevent nonsense operations like "CrossUp(Price, RSI)"
type FeatureType uint8

const (
	FeatTypeUnknown FeatureType = iota

	// Price level family (same scale)
	FeatTypePriceLevel // EMA*, HMA*, Kijun26, SwingHigh/Low, HH_, LL_, BB_Upper/Lower, KC_*, Donchian_*, SuperTrend*

	// Oscillator/Returns family (0-100 bounded or similar scale)
	FeatTypeOscillator // RSI*, MFI14, ADX, PlusDI, MinusDI, Stoch*, WilliamsR*

	// Momentum family (centered around 0, rate-of-change style)
	FeatTypeMomentum // ROC*, MACD*, MACD_Signal, MACD_Hist, ForceIndex*, Momentum*

	// Volume RAW family (raw volume units)
	FeatTypeVolumeRaw // OBV, VolPerTrade, VolSMA*, VolEMA*

	// Volume NORMALIZED family (z-scores, standardized)
	FeatTypeVolumeZScore // VolZ20, VolZ50

	// Range NORMALIZED family (0-1 ratios, percentages)
	FeatTypeRangeNorm // BB_Width*, RangeWidth, WickUpPct, WickDownPct, BodyPct, ClosePos, Displacement

	// Other volume-derived normalized metrics (0-1 range, but conceptually volume-based)
	FeatTypeVolumeDerivedNorm // Imbalance, BuyRatio

	// Price range (raw dollar units)
	FeatTypePriceRange // Body, HighLowDiff

	// ATR (volatility, dollar units)
	FeatTypeATR // ATR7, ATR14, ATR14_SMA50

	// Event flags (binary/discrete, not for crossing)
	FeatTypeEventFlag // BOS, FVG*, Sweep*, Active, Squeeze*, Stoch*Bear/BullCross, SuperTrendDir*
)

type Features struct {
	F     [][]float32
	Names []string
	Index map[string]int
	Stats []FeatureStats // Per-feature statistics for scale-aware mutations
	Types []FeatureType  // CRITICAL FIX #5: Feature type metadata for operator validation
}

// ComputeFeatureMapHash creates a fingerprint of the feature ordering
// Returns a sha256 hash of "0:Name1|1:Name2|..." format
// This allows detection of feature order changes between strategy generation and execution
func ComputeFeatureMapHash(f Features) string {
	parts := make([]string, len(f.Names))
	for i, name := range f.Names {
		parts[i] = fmt.Sprintf("%d:%s", i, name)
	}
	combined := strings.Join(parts, "|")
	hash := sha256.Sum256([]byte(combined))
	return hex.EncodeToString(hash[:])[:16] // First 16 chars of SHA256 (enough for fingerprint)
}

// GetFeatureMapVersion returns a human-readable version string based on key features
// This helps identify feature order changes without comparing full hashes
func GetFeatureMapVersion(f Features) string {
	// Sample a few key feature indices to detect major rearrangements
	checkFeatures := []string{"EMA10", "EMA20", "EMA50", "VolSMA20", "MinusDI", "BB_Width50", "MACD", "SwingHigh"}
	indices := make([]string, 0, len(checkFeatures))
	for _, name := range checkFeatures {
		if idx, ok := f.Index[name]; ok {
			indices = append(indices, fmt.Sprintf("%s@F[%d]", name, idx))
		}
	}
	return strings.Join(indices, ",")
}

// getFeatureType determines the semantic type of a feature from its name
// CRITICAL FIX #5: This enables operator validation to prevent nonsense operations
// Groups features into compatible scale groups for CrossUp/CrossDown/BreakUp/BreakDown
func getFeatureType(name string) FeatureType {
	switch {
	// ========== PRICE_LEVEL family (same scale: price dollars) ==========
	case name == "EMA10" || name == "EMA20" || name == "EMA50" || name == "EMA100" || name == "EMA200":
		return FeatTypePriceLevel
	case name == "HMA9" || name == "HMA20" || name == "HMA50" || name == "HMA100" || name == "HMA200":
		return FeatTypePriceLevel
	case name == "Kijun26":
		return FeatTypePriceLevel
	case name == "BB_Lower20" || name == "BB_Lower50" || name == "BB_Upper20" || name == "BB_Upper50":
		return FeatTypePriceLevel
	case name == "SwingHigh" || name == "SwingLow":
		return FeatTypePriceLevel
	case name == "HH_20_prev" || name == "HH_50_prev" || name == "HH_100_prev" || name == "HH_200_prev":
		return FeatTypePriceLevel
	case name == "LL_20_prev" || name == "LL_50_prev" || name == "LL_100_prev" || name == "LL_200_prev":
		return FeatTypePriceLevel
	case name == "KC_Middle20" || name == "KC_Upper20" || name == "KC_Lower20":
		return FeatTypePriceLevel
	case name == "Donchian_Upper20" || name == "Donchian_Lower20":
		return FeatTypePriceLevel
	case name == "Donchian_Upper55" || name == "Donchian_Lower55":
		return FeatTypePriceLevel
	case name == "SuperTrend10":
		return FeatTypePriceLevel

	// ========== OSCILLATOR family (0-100 bounded or similar) ==========
	case name == "RSI7" || name == "RSI14" || name == "RSI21":
		return FeatTypeOscillator
	case name == "MFI14":
		return FeatTypeOscillator
	case name == "ADX":
		return FeatTypeOscillator
	case name == "PlusDI" || name == "MinusDI":
		return FeatTypeOscillator
	case name == "StochK_14" || name == "StochD_14" || name == "StochK_5" || name == "StochD_5":
		return FeatTypeOscillator
	case name == "WilliamsR_14" || name == "WilliamsR_7":
		return FeatTypeOscillator

	// ========== MOMENTUM family (centered around 0, rate-of-change) ==========
	case name == "ROC5" || name == "ROC10" || name == "ROC20":
		return FeatTypeMomentum
	case name == "MACD" || name == "MACD_Signal" || name == "MACD_Hist":
		return FeatTypeMomentum
	case name == "ForceIndex2" || name == "ForceIndex13":
		return FeatTypeMomentum
	case name == "Momentum60" || name == "Momentum240":
		return FeatTypeMomentum
	case name == "SLOPE_20":
		return FeatTypeMomentum

	// ========== VOLUME_RAW family (raw volume units) ==========
	case name == "OBV":
		return FeatTypeVolumeRaw
	case name == "VolPerTrade":
		return FeatTypeVolumeRaw
	case name == "VolSMA20" || name == "VolSMA50" || name == "VolEMA20" || name == "VolEMA50":
		return FeatTypeVolumeRaw

	// ========== VOLUME_ZSCORE family (standardized volume) ==========
	case name == "VolZ20" || name == "VolZ50":
		return FeatTypeVolumeZScore

	// ========== RANGE_NORM family (0-1 ratios, percentages) ==========
	case name == "BB_Width20" || name == "BB_Width50":
		return FeatTypeRangeNorm
	case name == "RangeWidth":
		return FeatTypeRangeNorm
	case name == "BodyPct" || name == "WickUpPct" || name == "WickDownPct" || name == "ClosePos":
		return FeatTypeRangeNorm
	case name == "Displacement":
		return FeatTypeRangeNorm

	// ========== VOLUME_DERIVED_NORM family (0-1 volume-based metrics) ==========
	case name == "Imbalance":
		return FeatTypeVolumeDerivedNorm
	case name == "BuyRatio":
		return FeatTypeVolumeDerivedNorm

	// ========== PRICE_RANGE family (raw dollar units) ==========
	case name == "Body":
		return FeatTypePriceRange
	case name == "HighLowDiff":
		return FeatTypePriceRange

	// ========== ATR family (volatility in dollar units) ==========
	case name == "ATR7" || name == "ATR14" || name == "ATR14_SMA50":
		return FeatTypeATR

	// ========== EVENT_FLAG family (binary/discrete, NOT for crossing) ==========
	case name == "BOS" || name == "Sweep" || name == "FVGUp" || name == "FVGDown":
		return FeatTypeEventFlag
	case name == "SweepUp_20" || name == "SweepUp_50" || name == "SweepUp_100" || name == "SweepUp_200":
		return FeatTypeEventFlag
	case name == "SweepDown_20" || name == "SweepDown_50" || name == "SweepDown_100" || name == "SweepDown_200":
		return FeatTypeEventFlag
	case name == "Active":
		return FeatTypeEventFlag
	case name == "SuperTrendDir10":
		return FeatTypeEventFlag
	case name == "Squeeze20" || name == "SqueezeBreakUp" || name == "SqueezeBreakDown":
		return FeatTypeEventFlag
	case name == "StochBullCross" || name == "StochBearCross":
		return FeatTypeEventFlag

	// ========== Legacy/Other ==========
	case name == "VOLRET_20":
		return FeatTypeMomentum // Volatility-of-returns

	default:
		return FeatTypeUnknown
	}
}

// canCrossFeatures returns true if two feature types are compatible for cross operations
// CRITICAL FIX #5: Only allow crossing within same semantic scale group.
// This blocks nonsense like: CrossUp(VolEMA20, VolZ20) - mixing raw volume with z-score
//
// Compatible groups (can cross within group):
// - price_level: EMA*, HMA*, Kijun26, SwingHigh/Low, HH_, LL_, BB_Upper/Lower, KC_*, Donchian_*, SuperTrend*
// - returns_osc: RSI*, ROC*, MACD*, ADX, PlusDI, MinusDI, MFI14, Stoch*, WilliamsR*
// - volume_raw: VolSMA*, VolEMA*, VolPerTrade, OBV
// - volume_norm: VolZ20, VolZ50 (z-scores, standardized)
// - range_norm: BB_Width*, RangeWidth, WickUpPct/WickDownPct, BodyPct, ClosePos, Displacement
// - volume_derived_norm: Imbalance, BuyRatio (volume-based 0-1 metrics)
//
// Price-related family (can cross within):
// - price_level, price_range, ATR
//
// IMPORTANT: EventFlags are NEVER allowed in cross operations!
func canCrossFeatures(typeA, typeB FeatureType) bool {
	// EventFlags can NEVER be crossed (they're binary/discrete, not continuous)
	if typeA == FeatTypeEventFlag || typeB == FeatTypeEventFlag {
		return false
	}

	if typeA == FeatTypeUnknown || typeB == FeatTypeUnknown {
		return false
	}

	// Same type can always cross (except EventFlag handled above)
	if typeA == typeB {
		return true
	}

	// Price family: PriceLevel / PriceRange / ATR (all dollar units)
	priceFamily := []FeatureType{FeatTypePriceLevel, FeatTypePriceRange, FeatTypeATR}
	if inSlice(priceFamily, typeA) && inSlice(priceFamily, typeB) {
		return true
	}

	// Oscillator + Momentum family (similar scale: oscillating around center)
	oscMomFamily := []FeatureType{FeatTypeOscillator, FeatTypeMomentum}
	if inSlice(oscMomFamily, typeA) && inSlice(oscMomFamily, typeB) {
		return true
	}

	// Volume RAW + VolumeDerivedNorm family (raw volume units)
	// Note: VolZ* (z-scores) NOT included - they're in separate group
	volRawFamily := []FeatureType{FeatTypeVolumeRaw, FeatTypeVolumeDerivedNorm}
	if inSlice(volRawFamily, typeA) && inSlice(volRawFamily, typeB) {
		return true
	}

	// Volume ZScore family (standardized) - ONLY cross within itself
	// CANNOT mix with raw volume or other normalized features

	// RangeNorm family (0-1 ratios) - ONLY cross within itself
	// CANNOT mix with volume_zscore or other families

	// All other combinations are INCOMPATIBLE
	return false
}

// inSlice checks if a FeatureType is in a slice
func inSlice(slice []FeatureType, item FeatureType) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}

// validateCrossSanity checks all CrossUp/CrossDown/BreakUp/BreakDown nodes in a rule tree for:
// 1. Feature type compatibility (can these feature types be crossed?)
// 2. FIX B1: Self-comparison (A == B, which is always false)
// 3. FIX B2: Degenerate leaves (Between with min==max)
// Returns (isValid, invalidCount) - if invalidCount > 0, the tree has invalid cross operations
func validateCrossSanity(root *RuleNode, feats Features) (bool, int) {
	if root == nil {
		return true, 0
	}

	invalidCount := 0

	// Walk the tree and validate all cross nodes
	var walk func(node *RuleNode)
	walk = func(node *RuleNode) {
		if node == nil {
			return
		}

		if node.Op == OpLeaf {
			leaf := node.Leaf

			// FIX B1: Check for degenerate self-comparison leaves
			// CrossUp(A,A), BreakDown(A,A), etc. are always false
			if leaf.IsDegenerate() {
				invalidCount++
				return
			}

			// Check CrossUp, CrossDown, BreakUp, and BreakDown leaves for type compatibility
			if leaf.Kind == LeafCrossUp || leaf.Kind == LeafCrossDown || leaf.Kind == LeafBreakUp || leaf.Kind == LeafBreakDown {
				// Validate feature indices
				if leaf.A >= 0 && leaf.A < len(feats.Types) && leaf.B >= 0 && leaf.B < len(feats.Types) {
					typeA := feats.Types[leaf.A]
					typeB := feats.Types[leaf.B]
					if !canCrossFeatures(typeA, typeB) {
						invalidCount++
					}
				} else {
					// Invalid feature indices
					invalidCount++
				}
			}
			return
		}

		// Recurse into children
		walk(node.L)
		walk(node.R)
	}

	walk(root)
	return invalidCount == 0, invalidCount
}

// validateLoadedStrategy checks all CrossUp/CrossDown/BreakUp/BreakDown operations in a complete strategy
// Returns error if any invalid cross operations are found
// Use this when loading strategies from disk/checkpoint to reject invalid ones
func validateLoadedStrategy(s Strategy, feats *Features) error {
	var totalInvalid int

	// Validate all three rule trees
	if s.EntryRule.Root != nil {
		_, entryInvalid := validateCrossSanity(s.EntryRule.Root, *feats)
		totalInvalid += entryInvalid
	}
	if s.ExitRule.Root != nil {
		_, exitInvalid := validateCrossSanity(s.ExitRule.Root, *feats)
		totalInvalid += exitInvalid
	}
	if s.RegimeFilter.Root != nil {
		_, regimeInvalid := validateCrossSanity(s.RegimeFilter.Root, *feats)
		totalInvalid += regimeInvalid
	}

	if totalInvalid > 0 {
		return fmt.Errorf("strategy has %d invalid CrossUp/CrossDown/BreakUp/BreakDown operations", totalInvalid)
	}
	return nil
}

func computeAllFeatures(s Series) Features {
	n := s.T
	f := Features{}

	f.Names = make([]string, 0, 60)
	f.F = make([][]float32, 0, 60)
	f.Index = make(map[string]int, 60)
	f.Types = make([]FeatureType, 0, 60) // CRITICAL FIX #5: Initialize types slice

	addFeature := func(name string, arr []float32) {
		f.Index[name] = len(f.F)
		f.F = append(f.F, arr)
		f.Names = append(f.Names, name)
		f.Types = append(f.Types, getFeatureType(name)) // CRITICAL FIX #5: Set type
	}

	emaPeriods := []int{10, 20, 50, 100, 200}
	for _, p := range emaPeriods {
		arr := make([]float32, n)
		computeEMA(s.Close, arr, p)
		addFeature(fmt.Sprintf("EMA%d", p), arr)
	}

	// HMA - Hull Moving Average
	hmaPeriods := []int{9, 20, 50, 100, 200}
	for _, p := range hmaPeriods {
		arr := make([]float32, n)
		computeHMA(s.Close, arr, p)
		addFeature(fmt.Sprintf("HMA%d", p), arr)
	}

	// Ichimoku Kijun-sen (Base Line)
	kijun := make([]float32, n)
	computeIchimokuKijun(s.High, s.Low, kijun, 26)
	addFeature("Kijun26", kijun)

	rsiPeriods := []int{7, 14, 21}
	for _, p := range rsiPeriods {
		arr := make([]float32, n)
		computeRSI(s.Close, arr, p)
		addFeature(fmt.Sprintf("RSI%d", p), arr)
	}

	for _, p := range []int{20, 50} {
		upper := make([]float32, n)
		lower := make([]float32, n)
		width := make([]float32, n)
		computeBollinger(s.Close, upper, lower, width, p, 2.0)
		addFeature(fmt.Sprintf("BB_Upper%d", p), upper)
		addFeature(fmt.Sprintf("BB_Lower%d", p), lower)
		addFeature(fmt.Sprintf("BB_Width%d", p), width)
	}

	macd := make([]float32, n)
	signal := make([]float32, n)
	hist := make([]float32, n)
	computeMACD(s.Close, macd, signal, hist, 12, 26, 9)
	addFeature("MACD", macd)
	addFeature("MACD_Signal", signal)
	addFeature("MACD_Hist", hist)

	obv := make([]float32, n)
	computeOBV(s.Close, s.Volume, obv)
	addFeature("OBV", obv)

	for _, p := range []int{10, 20} {
		roc := make([]float32, n)
		computeROC(s.Close, roc, p)
		addFeature(fmt.Sprintf("ROC%d", p), roc)
	}

	atrPeriods := []int{7, 14}
	for _, p := range atrPeriods {
		arr := make([]float32, n)
		computeATR(s.High, s.Low, s.Close, arr, p)
		addFeature(fmt.Sprintf("ATR%d", p), arr)
	}

	// Add ATR14 SMA for volatility regime filter
	atr14Idx := f.Index["ATR14"]
	atr14 := f.F[atr14Idx]
	atr14SMA50 := make([]float32, n)
	computeSMA(atr14, atr14SMA50, 50)
	addFeature("ATR14_SMA50", atr14SMA50)

	adxPeriods := []int{14}
	for _, p := range adxPeriods {
		adx, plusDI, minusDI := make([]float32, n), make([]float32, n), make([]float32, n)
		computeADX(s.High, s.Low, s.Close, adx, plusDI, minusDI, p)
		addFeature("ADX", adx)
		addFeature("PlusDI", plusDI)
		addFeature("MinusDI", minusDI)
	}

	mfi := make([]float32, n)
	computeMFI(s.High, s.Low, s.Close, s.Volume, mfi, 14)
	addFeature("MFI14", mfi)

	for _, p := range []int{20, 50} {
		volSMA := make([]float32, n)
		volEMA := make([]float32, n)
		volZ := make([]float32, n)
		computeSMA(s.Volume, volSMA, p)
		computeEMA(s.Volume, volEMA, p)
		computeZScore(s.Volume, volSMA, volZ, p)
		addFeature(fmt.Sprintf("VolSMA%d", p), volSMA)
		addFeature(fmt.Sprintf("VolEMA%d", p), volEMA)
		addFeature(fmt.Sprintf("VolZ%d", p), volZ)
	}

	// Compute volSMA20 for Active calculation (already added as feature above, retrieve index)
	volSMA20Idx := f.Index["VolSMA20"]
	volSMA20 := f.F[volSMA20Idx]

	buyRatio := make([]float32, n)
	imbalance := make([]float32, n)
	volPerTrade := make([]float32, n)
	active := make([]float32, n)

	for i := 0; i < n; i++ {
		if s.Volume[i] > 0 {
			buyRatio[i] = s.TakerBuyBase[i] / s.Volume[i]
			sellBase := s.Volume[i] - s.TakerBuyBase[i]
			imbalance[i] = (s.TakerBuyBase[i] - sellBase) / s.Volume[i]
		}
		if s.Trades[i] > 0 {
			volPerTrade[i] = s.Volume[i] / float32(s.Trades[i])
		}
		// FIX: Active = 1 when volume >= 10% of average (more permissive threshold)
		// Guard: i >= 19 for warmup, volSMA20[i] > 0 to avoid early bars
		if i >= 19 && volSMA20[i] > 0 && s.Volume[i] >= volSMA20[i]*0.1 {
			active[i] = 1
		}
	}

	addFeature("BuyRatio", buyRatio)
	addFeature("Imbalance", imbalance)
	addFeature("VolPerTrade", volPerTrade)
	addFeature("Active", active)

	highLowDiff := make([]float32, n)
	body := make([]float32, n)
	rangeWidth := make([]float32, n)
	for i := 0; i < n; i++ {
		highLowDiff[i] = s.High[i] - s.Low[i]
		body[i] = float32(math.Abs(float64(s.Close[i] - s.Open[i])))
		if highLowDiff[i] > 0 {
			rangeWidth[i] = body[i] / highLowDiff[i]
		}
	}
	addFeature("HighLowDiff", highLowDiff)
	addFeature("Body", body)
	addFeature("RangeWidth", rangeWidth)

	// ========== CANDLE ANATOMY FEATURES (normalized 0-1) ==========
	// Pure price action without indicators - body/wick/position ratios
	bodyPct := make([]float32, n)
	wickUpPct := make([]float32, n)
	wickDownPct := make([]float32, n)
	closePos := make([]float32, n)

	const epsCandle = 1e-9

	for i := 0; i < n; i++ {
		range_val := s.High[i] - s.Low[i]
		maxRange := range_val
		if maxRange < epsCandle {
			maxRange = epsCandle
		}

		openVal := s.Open[i]
		closeVal := s.Close[i]
		highVal := s.High[i]
		lowVal := s.Low[i]

		// BodyPct = |Close - Open| / Range (0..1)
		bodyPct[i] = float32(math.Abs(float64(closeVal - openVal))) / maxRange

		// WickUpPct = (High - max(Open,Close)) / Range (clamp >= 0)
		upperWick := highVal - openVal
		if closeVal > openVal {
			upperWick = highVal - closeVal
		}
		if upperWick < 0 {
			upperWick = 0
		}
		wickUpPct[i] = upperWick / maxRange

		// WickDownPct = (min(Open,Close) - Low) / Range (clamp >= 0)
		lowerWick := openVal - lowVal
		if closeVal < openVal {
			lowerWick = closeVal - lowVal
		}
		if lowerWick < 0 {
			lowerWick = 0
		}
		wickDownPct[i] = lowerWick / maxRange

		// ClosePos = (Close - Low) / Range (clamp 0..1)
		closePos[i] = (closeVal - lowVal) / maxRange
		if closePos[i] < 0 {
			closePos[i] = 0
		}
		if closePos[i] > 1 {
			closePos[i] = 1
		}
	}

	addFeature("BodyPct", bodyPct)
	addFeature("WickUpPct", wickUpPct)
	addFeature("WickDownPct", wickDownPct)
	addFeature("ClosePos", closePos)

	swingHigh := make([]float32, n)
	swingLow := make([]float32, n)
	sweep := make([]float32, n)
	bos := make([]float32, n)
	displacement := make([]float32, n)
	fvgUp := make([]float32, n)
	fvgDown := make([]float32, n)

	// Rolling max/min swing detection (no lookahead)
	// Each bar gets the max high / min low from the lookback window
	// This is NOT a pivot-style swing - it's a rolling window for stable levels
	lookback := 20
	lastHigh, lastLow := float32(0), float32(0)

	for i := 0; i < n; i++ {
		swingHigh[i] = lastHigh
		swingLow[i] = lastLow
		if i >= lookback-1 {
			maxH := s.High[i-lookback+1]
			minL := s.Low[i-lookback+1]
			for j := i - lookback + 2; j <= i; j++ {
				if s.High[j] > maxH {
					maxH = s.High[j]
				}
				if s.Low[j] < minL {
					minL = s.Low[j]
				}
			}
			lastHigh, lastLow = maxH, minL
			swingHigh[i], swingLow[i] = lastHigh, lastLow
		}
	}

	for i := 4; i < n; i++ {
		if swingHigh[i-4] > 0 {
			swingHighPrice := swingHigh[i-4]
			if s.Low[i] < swingHighPrice {
				if s.Close[i] > swingHighPrice {
					sweep[i] = 1
				}
			}
		}
	}

	// BOS: Break of Structure - EVENT TRIGGER (only on crossing bar)
	// Use swingHigh[i-1] as reference to avoid same-bar update/break artifacts
	lastSwingHigh := float32(-1)
	for i := 1; i < n; i++ {
		if swingHigh[i-1] > 0 {
			lastSwingHigh = swingHigh[i-1]
		}
		if lastSwingHigh > 0 && s.Close[i-1] <= lastSwingHigh && s.Close[i] > lastSwingHigh {
			bos[i] = 1
		}
	}

	for i := 5; i < n; i++ {
		recentRange := s.High[i-5 : i+1]
		minR := recentRange[0]
		maxR := recentRange[0]
		for _, v := range recentRange {
			if v < minR {
				minR = v
			}
			if v > maxR {
				maxR = v
			}
		}
		if maxR-minR > 0 {
			displacement[i] = (s.Close[i] - minR) / (maxR - minR)
		}
	}

	for i := 2; i < n; i++ {
		if s.Low[i-1] > s.High[i] {
			fvgUp[i] = s.Low[i-1] - s.High[i]
		}
		if s.High[i-1] < s.Low[i] {
			fvgDown[i] = s.Low[i] - s.High[i-1]
		}
	}

	addFeature("SwingHigh", swingHigh)
	addFeature("SwingLow", swingLow)
	addFeature("Sweep", sweep)
	addFeature("BOS", bos)
	addFeature("Displacement", displacement)
	addFeature("FVGUp", fvgUp)
	addFeature("FVGDown", fvgDown)

	// ========== MARKET STRUCTURE LEVELS (HH/LL) ==========
	// Rolling max/min price levels (NO lookahead - exclude current candle)
	hhLookbacks := []int{20, 50, 100, 200}
	for _, lookback := range hhLookbacks {
		hh := make([]float32, n)
		ll := make([]float32, n)

		for i := 0; i < n; i++ {
			if i >= lookback {
				// Window is [i-lookback ... i-1] - EXCLUDE current candle i
				maxH := s.High[i-lookback]
				minL := s.Low[i-lookback]
				for j := i - lookback + 1; j < i; j++ {
					if s.High[j] > maxH {
						maxH = s.High[j]
					}
					if s.Low[j] < minL {
						minL = s.Low[j]
					}
				}
				hh[i] = maxH
				ll[i] = minL
			}
			// For i < lookback, hh[i] and ll[i] remain 0 (warmup period)
		}

		addFeature(fmt.Sprintf("HH_%d_prev", lookback), hh)
		addFeature(fmt.Sprintf("LL_%d_prev", lookback), ll)
	}

	// ========== LIQUIDITY SWEEP FLAGS ==========
	// Liquidity sweep detection (false breakouts) - EventFlag features (0/1)
	sweepLookbacks := []int{20, 50, 100, 200}
	for _, lookback := range sweepLookbacks {
		sweepUp := make([]float32, n)
		sweepDown := make([]float32, n)

		// Fetch HH/LL indices and slices ONCE per lookback (not per iteration)
		hhName := fmt.Sprintf("HH_%d_prev", lookback)
		llName := fmt.Sprintf("LL_%d_prev", lookback)
		hhIdx := f.Index[hhName]
		llIdx := f.Index[llName]
		hh := f.F[hhIdx]
		ll := f.F[llIdx]

		// Loop with i++ (CRITICAL: was missing in original plan)
		for i := 1; i < n; i++ {
			// SweepUp: High > HH_prev && Close < HH_prev (bullish trap)
			if hh[i] > 0 && s.High[i] > hh[i] && s.Close[i] < hh[i] {
				sweepUp[i] = 1
			}

			// SweepDown: Low < LL_prev && Close > LL_prev (bearish trap)
			if ll[i] > 0 && s.Low[i] < ll[i] && s.Close[i] > ll[i] {
				sweepDown[i] = 1
			}
		}

		addFeature(fmt.Sprintf("SweepUp_%d", lookback), sweepUp)
		addFeature(fmt.Sprintf("SweepDown_%d", lookback), sweepDown)
	}

	// Compute statistics for scale-aware mutations (will be recomputed on train window in main)
	f.Stats = make([]FeatureStats, len(f.F))

	return f
}

// computeStatsOnWindow computes feature statistics only on a specific window (train)
func computeStatsOnWindow(f *Features, startIdx, endIdx int) {
	for i := range f.F {
		arr := f.F[i]
		if len(arr) == 0 || endIdx <= startIdx {
			continue
		}

		// Clamp indices to array bounds
		start := startIdx
		end := endIdx
		if start < 0 {
			start = 0
		}
		if end > len(arr) {
			end = len(arr)
		}
		if start >= end {
			continue
		}

		// Compute min, max, mean on window only
		min := arr[start]
		max := arr[start]
		sum := float32(0)
		count := int32(0)
		for j := start; j < end; j++ {
			v := arr[j]
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
			count++
		}
		mean := sum / float32(count)

		// Compute standard deviation on window only
		variance := float32(0)
		for j := start; j < end; j++ {
			diff := arr[j] - mean
			variance += diff * diff
		}
		variance /= float32(count)
		std := float32(math.Sqrt(float64(variance)))

		f.Stats[i] = FeatureStats{
			Min:  min,
			Max:  max,
			Mean: mean,
			Std:  std,
		}
	}
}

func computeSMA(src, dst []float32, period int) {
	if len(src) < period {
		return
	}
	sum := float32(0.0)
	for i := 0; i < len(src); i++ {
		sum += src[i]
		if i >= period {
			sum -= src[i-period]
		}
		if i >= period-1 {
			dst[i] = sum / float32(period)
		}
	}
}

func computeEMA(src, dst []float32, period int) {
	if len(src) < period {
		return
	}

	// TradingView method: SMA for first period values
	sum := float32(0)
	for i := 0; i < period; i++ {
		sum += src[i]
	}
	sma := sum / float32(period)

	// Fill first period-1 with SMA
	for i := 0; i < period-1; i++ {
		dst[i] = sma
	}
	dst[period-1] = sma

	// EMA formula from period onwards
	multiplier := float32(2.0 / float32(period+1))
	for i := period; i < len(src); i++ {
		dst[i] = (src[i]-dst[i-1])*multiplier + dst[i-1]
	}
}

func computeWMA(src, dst []float32, period int) {
	if len(src) < period {
		return
	}
	for i := period - 1; i < len(src); i++ {
		sum := float32(0)
		weight := float32(1)
		for j := i; j >= i-period+1; j-- {
			sum += src[j] * weight
			weight++
		}
		dst[i] = sum / (float32(period) * float32(period+1) / 2)
	}
}

func computeHMA(src, dst []float32, period int) {
	n := len(src)
	if n < period {
		return
	}

	halfPeriod := period / 2
	sqrtPeriod := int(math.Sqrt(float64(period)))

	wmaHalf := make([]float32, n)
	wmaFull := make([]float32, n)
	temp := make([]float32, n)

	computeWMA(src, wmaHalf, halfPeriod)
	computeWMA(src, wmaFull, period)

	for i := 0; i < n; i++ {
		temp[i] = 2*wmaHalf[i] - wmaFull[i]
	}

	computeWMA(temp, dst, sqrtPeriod)
}

func computeIchimokuKijun(high, low, dst []float32, period int) {
	if len(high) < period {
		return
	}
	for i := period - 1; i < len(high); i++ {
		highestHigh := high[i]
		lowestLow := low[i]
		for j := i - period + 1; j <= i; j++ {
			if high[j] > highestHigh {
				highestHigh = high[j]
			}
			if low[j] < lowestLow {
				lowestLow = low[j]
			}
		}
		dst[i] = (highestHigh + lowestLow) / 2
	}
}

// computeWilderEMA uses Wilder's smoothing method (α = 1/period)
// This is the standard smoothing used for RSI, ATR, and ADX in TradingView
// For period N: alpha = 1/N (NOT 2/(N+1) like EMA)
func computeWilderEMA(src, dst []float32, period int) {
	if len(src) < 1 {
		return
	}
	alpha := float32(1.0 / float32(period)) // Wilder: α = 1/period
	dst[0] = src[0]
	for i := 1; i < len(src); i++ {
		dst[i] = (src[i]-dst[i-1])*alpha + dst[i-1]
	}
}

func computeRSI(src, dst []float32, period int) {
	if len(src) < period+1 {
		return
	}
	gain := make([]float32, len(src))
	loss := make([]float32, len(src))

	for i := 1; i < len(src); i++ {
		change := src[i] - src[i-1]
		if change > 0 {
			gain[i] = change
			loss[i] = 0
		} else {
			gain[i] = 0
			loss[i] = -change
		}
	}

	avgGain := float32(0)
	avgLoss := float32(0)

	for i := 1; i <= period; i++ {
		avgGain += gain[i]
		avgLoss += loss[i]
	}
	avgGain /= float32(period)
	avgLoss /= float32(period)

	if avgLoss == 0 {
		dst[period] = 100
	} else {
		dst[period] = 100 - (100 / (1 + avgGain/avgLoss))
	}

	for i := period + 1; i < len(src); i++ {
		avgGain = (avgGain*float32(period-1) + gain[i]) / float32(period)
		avgLoss = (avgLoss*float32(period-1) + loss[i]) / float32(period)
		if avgLoss == 0 {
			dst[i] = 100
		} else {
			dst[i] = 100 - (100 / (1 + avgGain/avgLoss))
		}
	}
}

func computeATR(high, low, close, dst []float32, period int) {
	if len(high) < period+1 {
		return
	}
	tr := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		hl := high[i] - low[i]
		hc := float32(math.Abs(float64(high[i] - close[i-1])))
		lc := float32(math.Abs(float64(low[i] - close[i-1])))
		tr[i] = maxFloat32(hl, maxFloat32(hc, lc))
	}

	sum := float32(0)
	for i := 1; i <= period; i++ {
		sum += tr[i]
	}
	dst[period] = sum / float32(period)

	for i := period + 1; i < len(high); i++ {
		dst[i] = (dst[i-1]*float32(period-1) + tr[i]) / float32(period)
	}
}

func computeBollinger(src, upper, lower, width []float32, period int, stdDev float64) {
	sma := make([]float32, len(src))
	computeSMA(src, sma, period)

	for i := period - 1; i < len(src); i++ {
		sumSq := float32(0)
		for j := i - period + 1; j <= i; j++ {
			diff := src[j] - sma[i]
			sumSq += diff * diff
		}
		std := float32(math.Sqrt(float64(sumSq / float32(period))))
		upper[i] = sma[i] + float32(stdDev)*std
		lower[i] = sma[i] - float32(stdDev)*std
		if sma[i] > 0 {
			width[i] = (upper[i] - lower[i]) / sma[i]
		}
		// CRITICAL FIX #3: BB_Width must never be 0 after warmup
		// Add numerical floor to prevent division issues
		if width[i] < 1e-9 {
			width[i] = 1e-9
		}
	}
}

func computeMACD(src, macd, signal, hist []float32, fastPeriod, slowPeriod, signalPeriod int) {
	fastEMA := make([]float32, len(src))
	slowEMA := make([]float32, len(src))
	computeEMA(src, fastEMA, fastPeriod)
	computeEMA(src, slowEMA, slowPeriod)

	for i := 0; i < len(src); i++ {
		macd[i] = fastEMA[i] - slowEMA[i]
	}
	computeEMA(macd, signal, signalPeriod)

	for i := 0; i < len(src); i++ {
		hist[i] = macd[i] - signal[i]
	}
}

func computeOBV(close, volume, dst []float32) {
	if len(close) < 1 {
		return
	}
	dst[0] = volume[0]
	for i := 1; i < len(close); i++ {
		if close[i] > close[i-1] {
			dst[i] = dst[i-1] + volume[i]
		} else if close[i] < close[i-1] {
			dst[i] = dst[i-1] - volume[i]
		} else {
			dst[i] = dst[i-1]
		}
	}
}

func computeADX(high, low, close, adx, plusDI, minusDI []float32, period int) {
	if len(high) < period*2+1 {
		return
	}

	tr := make([]float32, len(high))
	plusDM := make([]float32, len(high))
	minusDM := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		hl := high[i] - low[i]
		hc := float32(math.Abs(float64(high[i] - close[i-1])))
		lc := float32(math.Abs(float64(low[i] - close[i-1])))
		tr[i] = maxFloat32(hl, maxFloat32(hc, lc))

		upMove := high[i] - high[i-1]
		downMove := low[i-1] - low[i]
		if upMove > downMove && upMove > 0 {
			plusDM[i] = upMove
		} else {
			plusDM[i] = 0
		}
		if downMove > upMove && downMove > 0 {
			minusDM[i] = downMove
		} else {
			minusDM[i] = 0
		}
	}

	smoothedTR := make([]float32, len(high))
	smoothedPlusDM := make([]float32, len(high))
	smoothedMinusDM := make([]float32, len(high))

	for i := 1; i <= period; i++ {
		smoothedTR[period] += tr[i]
		smoothedPlusDM[period] += plusDM[i]
		smoothedMinusDM[period] += minusDM[i]
	}

	for i := period + 1; i < len(high); i++ {
		smoothedTR[i] = smoothedTR[i-1] - smoothedTR[i-1]/float32(period) + tr[i]
		smoothedPlusDM[i] = smoothedPlusDM[i-1] - smoothedPlusDM[i-1]/float32(period) + plusDM[i]
		smoothedMinusDM[i] = smoothedMinusDM[i-1] - smoothedMinusDM[i-1]/float32(period) + minusDM[i]
	}

	for i := period; i < len(high); i++ {
		if smoothedTR[i] > 0 {
			plusDI[i] = 100 * smoothedPlusDM[i] / smoothedTR[i]
			minusDI[i] = 100 * smoothedMinusDM[i] / smoothedTR[i]
		}
	}

	dx := make([]float32, len(high))
	for i := period; i < len(high); i++ {
		sum := plusDI[i] + minusDI[i]
		if sum > 0 {
			dx[i] = 100 * float32(math.Abs(float64(plusDI[i]-minusDI[i]))) / sum
		}
	}

	// FIX: Use Wilder smoothing (α = 1/period) instead of EMA (α = 2/(period+1))
	// TradingView uses Wilder smoothing for ADX, which matches the standard ADX calculation
	// Without this fix, ADX values are ~1.4x higher than TradingView
	computeWilderEMA(dx, adx, period)
}

func computeMFI(high, low, close, volume, dst []float32, period int) {
	if len(high) < period+1 {
		return
	}

	typicalPrice := make([]float32, len(high))
	rawMF := make([]float32, len(high))

	for i := 0; i < len(high); i++ {
		typicalPrice[i] = (high[i] + low[i] + close[i]) / 3
		rawMF[i] = typicalPrice[i] * volume[i]
	}

	posMF := make([]float32, len(high))
	negMF := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		if typicalPrice[i] > typicalPrice[i-1] {
			posMF[i] = rawMF[i]
		} else {
			negMF[i] = rawMF[i]
		}
	}

	sumPos := float32(0)
	sumNeg := float32(0)

	for i := 1; i <= period; i++ {
		sumPos += posMF[i]
		sumNeg += negMF[i]
	}

	if sumNeg == 0 {
		dst[period] = 100
	} else {
		dst[period] = 100 - (100 / (1 + sumPos/sumNeg))
	}

	for i := period + 1; i < len(high); i++ {
		sumPos = sumPos - posMF[i-period] + posMF[i]
		sumNeg = sumNeg - negMF[i-period] + negMF[i]

		if sumNeg == 0 {
			dst[i] = 100
		} else {
			dst[i] = 100 - (100 / (1 + sumPos/sumNeg))
		}
	}
}

func computeROC(src, dst []float32, period int) {
	if len(src) < period {
		return
	}
	for i := period; i < len(src); i++ {
		if src[i-period] != 0 {
			dst[i] = 100 * (src[i] - src[i-period]) / src[i-period]
		}
	}
}

func computeZScore(src, mean, dst []float32, period int) {
	if len(src) < period || len(mean) < period {
		return
	}
	for i := period - 1; i < len(src); i++ {
		if i >= len(mean) {
			break
		}
		meanVal := mean[i]
		if meanVal == 0 {
			continue // Avoid weird warmup spikes
		}

		// Calculate variance from mean
		sumSqDiff := float32(0)
		for j := i - period + 1; j <= i; j++ {
			diff := src[j] - meanVal
			sumSqDiff += diff * diff
		}
		variance := sumSqDiff / float32(period)

		if variance > 0 {
			std := float32(math.Sqrt(float64(variance)))
			if std > 0 {
				dst[i] = (src[i] - meanVal) / std
			}
		}
	}
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
