package main

import "math"

// ExtendFeatures appends derived features to the base Features struct.
// It uses raw OHLCV data from Series to compute momentum and volatility features.
func ExtendFeatures(f *Features, s Series) {
	n := s.T

	// Helper to append a new feature column (matches the pattern in features.go)
	addFeature := func(name string, arr []float32) {
		f.Index[name] = len(f.F)
		f.F = append(f.F, arr)
		f.Names = append(f.Names, name)
		f.Types = append(f.Types, getFeatureType(name)) // CRITICAL FIX: prevents Unknown type
		// Stats will be computed later in computeStatsOnWindow
		f.Stats = append(f.Stats, FeatureStats{})
	}

	// ---------- Feature #1: ROC / Momentum ----------
	// ROC_n = (close[t] - close[t-n]) / close[t-n]
	// Note: ROC10 and ROC20 already exist, adding ROC5 here
	roc5 := make([]float32, n)
	for t := 0; t < n; t++ {
		if t < 5 {
			roc5[t] = 0
			continue
		}
		prev := s.Close[t-5]
		if prev == 0 {
			roc5[t] = 0
			continue
		}
		roc5[t] = (s.Close[t] - prev) / prev
	}
	addFeature("ROC5", roc5)

	// ---------- Feature #2: Rolling Volatility (std of returns) ----------
	// VOLRET_n = std( (close[t]-close[t-1])/close[t-1] ) over window n
	rollStdReturns := func(window int) []float32 {
		col := make([]float32, n)
		for t := 0; t < n; t++ {
			if t < window+1 {
				col[t] = 0
				continue
			}
			// mean
			var sum float64
			count := 0
			for k := t - window + 1; k <= t; k++ {
				prev := s.Close[k-1]
				if prev == 0 {
					continue
				}
				r := float64((s.Close[k] - prev) / prev)
				sum += r
				count++
			}
			if count == 0 {
				col[t] = 0
				continue
			}
			mean := sum / float64(count)

			// variance
			var v float64
			for k := t - window + 1; k <= t; k++ {
				prev := s.Close[k-1]
				if prev == 0 {
					continue
				}
				r := float64((s.Close[k] - prev) / prev)
				d := r - mean
				v += d * d
			}
			v /= float64(count)
			col[t] = float32(math.Sqrt(v))
		}
		return col
	}

	addFeature("VOLRET_20", rollStdReturns(20))

	// ---------- Feature #3: Trend Slope ----------
	// SLOPE_n = (close[t] - close[t-n]) / n
	slope := func(period int) []float32 {
		col := make([]float32, n)
		for t := 0; t < n; t++ {
			if t < period {
				col[t] = 0
				continue
			}
			col[t] = (s.Close[t] - s.Close[t-period]) / float32(period)
		}
		return col
	}
	addFeature("SLOPE_20", slope(20))

	// ---------- Feature #4: Range Compression ----------
	// RANGE = high - low
	// COMPRESSION = RANGE / (VOLRET_20 + eps)
	rangeCol := make([]float32, n)
	comp := make([]float32, n)
	eps := float32(1e-6)

	// Get VOLRET_20 index
	volretIdx, hasVolret := f.Index["VOLRET_20"]

	for t := 0; t < n; t++ {
		r := s.High[t] - s.Low[t]
		rangeCol[t] = r
		if hasVolret && volretIdx >= 0 && volretIdx < len(f.F) {
			volret := f.F[volretIdx][t]
			comp[t] = r / (volret + eps)
		} else {
			comp[t] = 0
		}
	}
	addFeature("RANGE", rangeCol)
	addFeature("COMPRESSION", comp)

	// ---------- Feature #5: Volume Regime ----------
	// VOLRATIO_n = vol[t] / SMA(vol, n)
	volRatio := func(period int) []float32 {
		col := make([]float32, n)
		for t := 0; t < n; t++ {
			if t < period {
				col[t] = 1
				continue
			}
			var sum float32
			for k := t - period + 1; k <= t; k++ {
				sum += s.Volume[k]
			}
			mean := sum / float32(period)
			if mean == 0 {
				col[t] = 1
			} else {
				col[t] = s.Volume[t] / mean
			}
		}
		return col
	}
	addFeature("VOLRATIO_20", volRatio(20))

	// ========== PHASE 1: HIGH-IMPACT NEW INDICATORS ==========

	// ---------- Keltner Channels ----------
	// KC Middle = EMA(20), KC Upper/Lower = EMA(20) ± ATR(14) × 2
	// Used for BB-KC squeeze detection (volatility breakout strategy)
	ema20 := make([]float32, n)
	computeEMA(s.Close, ema20, 20)

	atr14 := make([]float32, n)
	computeATR(s.High, s.Low, s.Close, atr14, 14)

	kcUpper20 := make([]float32, n)
	kcLower20 := make([]float32, n)
	kcMiddle20 := make([]float32, n)

	for i := 0; i < n; i++ {
		kcMiddle20[i] = ema20[i]
		kcUpper20[i] = ema20[i] + atr14[i]*2.0
		kcLower20[i] = ema20[i] - atr14[i]*2.0
	}
	addFeature("KC_Middle20", kcMiddle20)
	addFeature("KC_Upper20", kcUpper20)
	addFeature("KC_Lower20", kcLower20)

	// ---------- Stochastic Oscillator ----------
	// Stochastic K and D for both 14-period (slow) and 5-period (fast)
	// K = (Close - LL) / (HH - LL) × 100, D = SMA(K, 3)
	// Complements RSI for oscillator-based signals
	stochK14 := computeStochasticK(s, 14)
	stochD14 := make([]float32, n)
	computeSMA(stochK14, stochD14, 3)

	stochK5 := computeStochasticK(s, 5)
	stochD5 := make([]float32, n)
	computeSMA(stochK5, stochD5, 3)

	addFeature("StochK_14", stochK14)
	addFeature("StochD_14", stochD14)
	addFeature("StochK_5", stochK5)
	addFeature("StochD_5", stochD5)

	// ---------- Donchian Channels ----------
	// Turtle Trading system breakout levels
	// Upper = Highest High over N bars, Lower = Lowest Low over N bars
	donchianUpper20 := make([]float32, n)
	donchianLower20 := make([]float32, n)
	donchianUpper55 := make([]float32, n)
	donchianLower55 := make([]float32, n)

	for i := 0; i < n; i++ {
		// Donchian 20
		if i >= 20 {
			maxH := s.High[i-19]
			minL := s.Low[i-19]
			for j := i - 18; j <= i; j++ {
				if s.High[j] > maxH {
					maxH = s.High[j]
				}
				if s.Low[j] < minL {
					minL = s.Low[j]
				}
			}
			donchianUpper20[i] = maxH
			donchianLower20[i] = minL
		}

		// Donchian 55 (Turtle System 2)
		if i >= 55 {
			maxH := s.High[i-54]
			minL := s.Low[i-54]
			for j := i - 53; j <= i; j++ {
				if s.High[j] > maxH {
					maxH = s.High[j]
				}
				if s.Low[j] < minL {
					minL = s.Low[j]
				}
			}
			donchianUpper55[i] = maxH
			donchianLower55[i] = minL
		}
	}
	addFeature("Donchian_Upper20", donchianUpper20)
	addFeature("Donchian_Lower20", donchianLower20)
	addFeature("Donchian_Upper55", donchianUpper55)
	addFeature("Donchian_Lower55", donchianLower55)

	// ---------- SuperTrend ----------
	// Trend-following indicator using ATR bands
	// HL2 ± (multiplier × ATR). Direction flips when price crosses band
	superTrend10, superTrendDir10 := computeSuperTrend(s, 10, 3.0)
	addFeature("SuperTrend10", superTrend10)
	addFeature("SuperTrendDir10", superTrendDir10)

	// ---------- Williams %R ----------
	// Momentum oscillator similar to Stochastic but inverted (0 to -100)
	// R = (HH - Close) / (HH - LL) × -100
	williamsR14 := computeWilliamsR(s, 14)
	williamsR7 := computeWilliamsR(s, 7)
	addFeature("WilliamsR_14", williamsR14)
	addFeature("WilliamsR_7", williamsR7)

	// ---------- Force Index ----------
	// Volume-weighted momentum from Elder's Triple Screen
	// EMA((Close - PrevClose) × Volume, period)
	forceIndex2 := computeForceIndex(s, 2)
	forceIndex13 := computeForceIndex(s, 13)
	addFeature("ForceIndex2", forceIndex2)
	addFeature("ForceIndex13", forceIndex13)

	// ---------- Momentum Features ----------
	// Dual momentum: (Close - Close[t-n]) / Close[t-n]
	// Used for regime filtering (absolute momentum > 0 = uptrend)
	momentum60 := make([]float32, n)
	momentum240 := make([]float32, n)

	for i := 0; i < n; i++ {
		// Momentum 60 (one day for 1H timeframe)
		if i >= 60 {
			baseClose := s.Close[i-60]
			if baseClose > 0 {
				momentum60[i] = (s.Close[i] - baseClose) / baseClose
			}
		}

		// Momentum 240 (four days for 1H timeframe)
		if i >= 240 {
			baseClose := s.Close[i-240]
			if baseClose > 0 {
				momentum240[i] = (s.Close[i] - baseClose) / baseClose
			}
		}
	}
	addFeature("Momentum60", momentum60)
	addFeature("Momentum240", momentum240)

	// ========== PHASE 2: HIGH-VALUE EVENT FLAGS ==========

	// ---------- BB-KC Squeeze Detection ----------
	// Squeeze = BB inside KC (low volatility period, expansion expected)
	// SqueezeBreakUp/Down = squeeze ends + price breaks BB band
	squeeze20 := make([]float32, n)
	squeezeBreakUp := make([]float32, n)
	squeezeBreakDown := make([]float32, n)

	// Get BB and KC indices (already computed above)
	bbUpper20Idx, hasBBUpper := f.Index["BB_Upper20"]
	bbLower20Idx, hasBBLower := f.Index["BB_Lower20"]
	kcUpper20Idx, hasKcUpper := f.Index["KC_Upper20"]
	kcLower20Idx, hasKcLower := f.Index["KC_Lower20"]

	var wasSqueezed bool
	for i := 20; i < n; i++ {
		isSqueeze := false
		if hasBBUpper && hasBBLower && hasKcUpper && hasKcLower {
			bbUpper := f.F[bbUpper20Idx][i]
			bbLower := f.F[bbLower20Idx][i]
			kcUpper := f.F[kcUpper20Idx][i]
			kcLower := f.F[kcLower20Idx][i]

			// Squeeze = BB completely inside KC
			if bbUpper <= kcUpper && bbLower >= kcLower {
				isSqueeze = true
			}
		}

		squeeze20[i] = boolToFloat32(isSqueeze)

		// Detect squeeze break (previous bar was in squeeze, now price breaks BB)
		if i > 0 && wasSqueezed && !isSqueeze {
			if hasBBUpper && hasBBLower {
				bbUpper := f.F[bbUpper20Idx][i]
				bbLower := f.F[bbLower20Idx][i]
				if s.Close[i] > bbUpper {
					squeezeBreakUp[i] = 1
				} else if s.Close[i] < bbLower {
					squeezeBreakDown[i] = 1
				}
			}
		}

		wasSqueezed = isSqueeze
	}
	addFeature("Squeeze20", squeeze20)
	addFeature("SqueezeBreakUp", squeezeBreakUp)
	addFeature("SqueezeBreakDown", squeezeBreakDown)

	// ---------- Stochastic Cross Signals ----------
	// K crosses above D while both < 20 (bullish oversold cross)
	// K crosses below D while both > 80 (bearish overbought cross)
	stochBullCross := make([]float32, n)
	stochBearCross := make([]float32, n)

	stochK14Idx, _ := f.Index["StochK_14"]
	stochD14Idx, _ := f.Index["StochD_14"]

	for i := 1; i < n; i++ {
		k := f.F[stochK14Idx]
		d := f.F[stochD14Idx]

		// Bullish cross: K crosses above D while both in oversold (< 20)
		if k[i] > d[i] && k[i-1] <= d[i-1] && k[i] < 30 && d[i] < 30 {
			stochBullCross[i] = 1
		}

		// Bearish cross: K crosses below D while both in overbought (> 80)
		if k[i] < d[i] && k[i-1] >= d[i-1] && k[i] > 70 && d[i] > 70 {
			stochBearCross[i] = 1
		}
	}
	addFeature("StochBullCross", stochBullCross)
	addFeature("StochBearCross", stochBearCross)
}

// computeStochasticK calculates the %K line for Stochastic oscillator
// %K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100
func computeStochasticK(s Series, period int) []float32 {
	n := s.T
	k := make([]float32, n)

	for i := period - 1; i < n; i++ {
		highestHigh := s.High[i]
		lowestLow := s.Low[i]

		for j := i - period + 1; j <= i; j++ {
			if s.High[j] > highestHigh {
				highestHigh = s.High[j]
			}
			if s.Low[j] < lowestLow {
				lowestLow = s.Low[j]
			}
		}

		range_val := highestHigh - lowestLow
		if range_val > 0 {
			k[i] = (s.Close[i] - lowestLow) / range_val * 100
		} else {
			k[i] = 50 // Neutral value when no range
		}
	}

	return k
}

// computeSuperTrend calculates SuperTrend indicator
// Returns (trend line values, direction: +1 bullish, -1 bearish)
func computeSuperTrend(s Series, atrPeriod int, multiplier float32) ([]float32, []float32) {
	n := s.T
	trend := make([]float32, n)
	direction := make([]float32, n)

	// Calculate ATR and HL2
	atr := make([]float32, n)
	computeATR(s.High, s.Low, s.Close, atr, atrPeriod)

	hl2 := make([]float32, n)
	for i := 0; i < n; i++ {
		hl2[i] = (s.High[i] + s.Low[i]) / 2
	}

	// SuperTrend bands
	upperBand := make([]float32, n)
	lowerBand := make([]float32, n)

	for i := 0; i < n; i++ {
		upperBand[i] = hl2[i] + multiplier*atr[i]
		lowerBand[i] = hl2[i] - multiplier*atr[i]
	}

	// Track current trend direction
	currentDir := float32(1) // Start bullish

	for i := 1; i < n; i++ {
		if currentDir == 1 {
			// Bullish trend - check if price closes below lower band
			if s.Close[i] < lowerBand[i] {
				currentDir = -1
				trend[i] = upperBand[i]
			} else {
				trend[i] = lowerBand[i]
				// Rising band effect
				if trend[i] < trend[i-1] && trend[i-1] > 0 {
					trend[i] = trend[i-1]
				}
			}
		} else {
			// Bearish trend - check if price closes above upper band
			if s.Close[i] > upperBand[i] {
				currentDir = 1
				trend[i] = lowerBand[i]
			} else {
				trend[i] = upperBand[i]
				// Falling band effect
				if trend[i] > trend[i-1] && trend[i-1] > 0 {
					trend[i] = trend[i-1]
				}
			}
		}
		direction[i] = currentDir
	}

	return trend, direction
}

// computeWilliamsR calculates Williams %R oscillator
// %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
func computeWilliamsR(s Series, period int) []float32 {
	n := s.T
	r := make([]float32, n)

	for i := period - 1; i < n; i++ {
		highestHigh := s.High[i]
		lowestLow := s.Low[i]

		for j := i - period + 1; j <= i; j++ {
			if s.High[j] > highestHigh {
				highestHigh = s.High[j]
			}
			if s.Low[j] < lowestLow {
				lowestLow = s.Low[j]
			}
		}

		range_val := highestHigh - lowestLow
		if range_val > 0 {
			r[i] = (highestHigh - s.Close[i]) / range_val * -100
		} else {
			r[i] = -50 // Neutral value
		}
	}

	return r
}

// computeForceIndex calculates the Force Index
// ForceIndex = EMA((Close - PrevClose) × Volume, period)
func computeForceIndex(s Series, period int) []float32 {
	n := s.T
	force := make([]float32, n)

	// Calculate raw force: price change × volume
	rawForce := make([]float32, n)
	for i := 1; i < n; i++ {
		rawForce[i] = (s.Close[i] - s.Close[i-1]) * s.Volume[i]
	}

	// Apply EMA smoothing
	computeEMA(rawForce, force, period)

	return force
}

// boolToFloat32 converts boolean to float32 (1.0 for true, 0.0 for false)
func boolToFloat32(b bool) float32 {
	if b {
		return 1
	}
	return 0
}
