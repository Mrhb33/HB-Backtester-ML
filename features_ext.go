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
}
