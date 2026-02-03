package main

import (
	"time"
)

// Window defines a time-based backtesting window with warmup history
type Window struct {
	Start  int // trade starts here (inclusive)
	End    int // trade ends here (exclusive)
	Warmup int // how many candles of history to include before Start
}

// idxAtOrAfter finds the first index where times[i] >= ms using binary search
func idxAtOrAfter(times []int64, ms int64) int {
	lo, hi := 0, len(times)
	for lo < hi {
		mid := (lo + hi) / 2
		if times[mid] < ms {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// SliceSeries creates a new Series slice from i0 (inclusive) to i1 (exclusive)
func SliceSeries(s Series, i0, i1 int) Series {
	if i0 < 0 {
		i0 = 0
	}
	if i1 > s.T {
		i1 = s.T
	}
	if i0 >= i1 {
		return Series{}
	}

	return Series{
		T: i1 - i0,

		OpenTimeMs:  s.OpenTimeMs[i0:i1],
		CloseTimeMs: s.CloseTimeMs[i0:i1],

		Open:  s.Open[i0:i1],
		High:  s.High[i0:i1],
		Low:   s.Low[i0:i1],
		Close: s.Close[i0:i1],

		Volume:        s.Volume[i0:i1],
		QuoteVolume:   s.QuoteVolume[i0:i1],
		TakerBuyBase:  s.TakerBuyBase[i0:i1],
		TakerBuyQuote: s.TakerBuyQuote[i0:i1],
		Trades:        s.Trades[i0:i1],
	}
}

// SliceFeatures creates a new Features slice from i0 (inclusive) to i1 (exclusive)
// OPTIMIZATION: Reuses Index map and Stats since feature order doesn't change (only bar range changes)
func SliceFeatures(f Features, i0, i1 int) Features {
	if i0 < 0 {
		i0 = 0
	}
	if i1 > len(f.F[0]) {
		i1 = len(f.F[0])
	}
	if i0 >= i1 {
		return Features{}
	}

	out := Features{
		F:     make([][]float32, len(f.F)),
		Names: f.Names,
		// OPTIMIZATION: Reuse Index map - slicing bars doesn't change feature indices
		// Index maps feature names to feature slot indices, which remain the same
		Index: f.Index,
		// OPTIMIZATION: Reuse Types and Stats - they're per-feature metadata, not per-bar
		Types: f.Types,
		Stats: f.Stats,
	}
	for k := range f.F {
		out.F[k] = f.F[k][i0:i1]
	}

	return out
}

// GetSplitIndices computes train/validation/test split indices based on dates
func GetSplitIndices(times []int64) (trainStart, trainEnd, valEnd int) {
	trainStart = idxAtOrAfter(times, time.Date(2017, 8, 17, 0, 0, 0, 0, time.UTC).UnixMilli())
	trainEnd = idxAtOrAfter(times, time.Date(2022, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli())
	valEnd = idxAtOrAfter(times, time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli())

	return trainStart, trainEnd, valEnd
}

// GetSplitWindows creates Window structs for train/validation/test splits
func GetSplitWindows(trainStart, trainEnd, valEnd, totalCandles int, warmup int) (trainW, valW, testW Window) {
	trainW = Window{Start: trainStart, End: trainEnd, Warmup: warmup}
	valW = Window{Start: trainEnd, End: valEnd, Warmup: warmup}
	testW = Window{Start: valEnd, End: totalCandles, Warmup: warmup}

	return trainW, valW, testW
}

// GetCustomWindow creates a custom window for time-based replay testing
// Critical: Compute features on FULL dataset first, then slice with warmup buffer
// to ensure indicators are "real" (not affected by slice boundaries)
func GetCustomWindow(times []int64, fromIdx, toIdx int, fromStr, toStr string, warmup int, totalCandles int) Window {
	// Resolve from/to indices from various inputs
	start := -1
	end := totalCandles

	// Priority 1: Use explicit index if provided
	if fromIdx >= 0 {
		start = fromIdx
	}
	if toIdx >= 0 {
		end = toIdx
	}

	// Priority 2: Parse time strings if provided
	if start < 0 && fromStr != "" {
		// Parse time string "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
		layout := "2006-01-02 15:04:05"
		if len(fromStr) <= 10 {
			layout = "2006-01-02"
		}
		fromTs, err := time.Parse(layout, fromStr)
		if err == nil {
			start = idxAtOrAfter(times, fromTs.UnixMilli())
		}
	}

	if end == totalCandles && toStr != "" {
		layout := "2006-01-02 15:04:05"
		if len(toStr) <= 10 {
			layout = "2006-01-02"
		}
		toTs, err := time.Parse(layout, toStr)
		if err == nil {
			end = idxAtOrAfter(times, toTs.UnixMilli())
		}
	}

	// Default to full range if still unresolved
	if start < 0 {
		start = 0
	}
	if end > totalCandles {
		end = totalCandles
	}

	// Validate range
	if start < 0 {
		start = 0
	}
	if end > totalCandles {
		end = totalCandles
	}
	if start >= end {
		start = 0
		end = totalCandles
	}

	return Window{
		Start:  start, // First bar where trades are ALLOWED
		End:    end,
		Warmup: warmup, // Warmup bars before tradeStart
	}
}
