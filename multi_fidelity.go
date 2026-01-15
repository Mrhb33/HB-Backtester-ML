package main

import (
	"math"
)

// FidelityLevel represents evaluation depth
type FidelityLevel int

const (
	FidelityScreen FidelityLevel = iota // Fast screen (3-6 months)
	FidelityFull                     // Full train window
	FidelityVal                      // Full validation
)

// evaluateMultiFidelity runs strategy through multi-fidelity pipeline
// Returns: (passedScreen, passedFull, trainResult, valResult)
func evaluateMultiFidelity(full Series, fullF Features, st Strategy, screenW, trainW, valW Window, testedCount int64) (bool, bool, Result, Result) {
	// Stage 1: Fast screen (quick filter)
	screenResult := evaluateStrategyWindow(full, fullF, st, screenW)

	// Compute DSR-lite score for screen
	screenScore := computeScore(
		screenResult.Return,
		screenResult.MaxDD,
		screenResult.Expectancy,
		screenResult.Trades,
		0, // No deflation penalty for screen
	)

	// Quick filter: screen must pass basic criteria
	// Less strict than full criteria (we just want to filter obvious junk)
	minScreenScore := float32(0.0) // Very permissive at screen stage
	minScreenTrades := 30 // Reduced from 50 to let more candidates through
	maxScreenDD := float32(0.70) // Increased from 0.60 for aggressive exploration

	if screenScore < minScreenScore || screenResult.Trades < minScreenTrades || screenResult.MaxDD >= maxScreenDD {
		return false, false, screenResult, Result{}
	}

	// Stage 2: Full train window evaluation
	trainResult := evaluateStrategyWindow(full, fullF, st, trainW)

	// Compute DSR-lite score for train (no penalty)
	trainScore := computeScore(
		trainResult.Return,
		trainResult.MaxDD,
		trainResult.Expectancy,
		trainResult.Trades,
		0, // No deflation penalty for train phase
	)

	// Basic train filter
	minTrainScore := float32(0.0)
	minTrainTrades := 80
	maxTrainDD := float32(0.50)

	if trainScore < minTrainScore || trainResult.Trades < minTrainTrades || trainResult.MaxDD >= maxTrainDD {
		return true, false, trainResult, Result{}
	}

	// Stage 3: Validation (only if passed train)
	valResult := evaluateStrategyWindow(full, fullF, st, valW)

	// Compute DSR-lite score for validation (WITH deflation penalty)
	valScore := computeScore(
		valResult.Return,
		valResult.MaxDD,
		valResult.Expectancy,
		valResult.Trades,
		testedCount, // Apply DSR-lite penalty here!
	)

	// Update result with DSR-lite score
	valResult.Score = valScore

	return true, true, trainResult, valResult
}

// computeDeflatedScore applies DSR-lite deflation penalty
// Formula: deflated = baseScore - k * log(1 + testedCount / 10000)
func computeDeflatedScore(baseScore float32, testedCount int64) float32 {
	if testedCount <= 0 {
		return baseScore
	}

	// k = 0.5 gives gentle penalty that scales with testing
	deflationPenalty := float32(0.5 * math.Log(float64(1.0+testedCount)/10000.0))
	return baseScore - deflationPenalty
}

// getScreenWindow creates a smaller screening window (3-6 months)
// Use last 6 months of train window for screening
func getScreenWindow(trainW Window) Window {
	// Assume 5min candles, 6 months ≈ 6 * 30 * 24 * 12 = 51,840 candles
	// Reduced to 3 months for faster screening (25,000 candles)
	screenCandles := 25000

	screenStart := trainW.End - screenCandles
	if screenStart < trainW.Start {
		screenStart = trainW.Start
	}

	return Window{
		Start:  screenStart,
		End:    trainW.End,
		Warmup: trainW.Warmup,
	}
}
