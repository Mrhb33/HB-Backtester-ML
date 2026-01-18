package main

import (
	"math"
	"sync/atomic"
)

// Global screen relax level (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
// Accessed atomically for thread-safe reads from workers
var globalScreenRelaxLevel int32 = 3 // Default to very relaxed (unblock mode)

// getScreenRelaxLevel returns the current screen relax level (thread-safe)
func getScreenRelaxLevel() int {
	return int(atomic.LoadInt32(&globalScreenRelaxLevel))
}

// setScreenRelaxLevel sets the screen relax level (thread-safe)
func setScreenRelaxLevel(level int) {
	atomic.StoreInt32(&globalScreenRelaxLevel, int32(level))
}

// FidelityLevel represents evaluation depth
type FidelityLevel int

const (
	FidelityScreen FidelityLevel = iota // Fast screen (3-6 months)
	FidelityFull                        // Full train window
	FidelityVal                         // Full validation
)

// evaluateMultiFidelity runs strategy through multi-fidelity pipeline
// Returns: (passedScreen, passedFull, trainResult, valResult)
func evaluateMultiFidelity(full Series, fullF Features, st Strategy, screenW, trainW, valW Window, testedCount int64) (bool, bool, Result, Result) {
	// Get relax level from meta (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
	// Default to very relaxed (3) to unblock candidate flow
	relaxLevel := getScreenRelaxLevel()

	// Set gates based on relax level
	var minScreenScore float32
	var minScreenTrades int
	var maxScreenDD float32
	var minTrainScore float32
	var minTrainTrades int
	var maxTrainDD float32

	switch relaxLevel {
	case 0: // Strict
		minScreenScore = 0.0
		minScreenTrades = 30
		maxScreenDD = 0.70
		minTrainScore = -0.20
		minTrainTrades = 80
		maxTrainDD = 0.50
	case 1: // Normal
		minScreenScore = 0.0
		minScreenTrades = 20
		maxScreenDD = 0.80
		minTrainScore = -0.20
		minTrainTrades = 60
		maxTrainDD = 0.55
	case 2: // Relaxed
		minScreenScore = 0.0
		minScreenTrades = 15
		maxScreenDD = 0.85
		minTrainScore = -0.20
		minTrainTrades = 40
		maxTrainDD = 0.60
	case 3: // Very Relaxed (unblock mode)
		minScreenScore = 0.0
		minScreenTrades = 10
		maxScreenDD = 0.90
		minTrainScore = -0.20
		minTrainTrades = 30
		maxTrainDD = 0.65
	default: // Default to very relaxed
		minScreenScore = 0.0
		minScreenTrades = 10
		maxScreenDD = 0.90
		minTrainScore = -0.20
		minTrainTrades = 30
		maxTrainDD = 0.65
	}

	// Stage 1: Fast screen (quick filter)
	screenResult := evaluateStrategyWindow(full, fullF, st, screenW)

	// Compute DSR-lite score for screen with smoothness
	screenScore := computeScoreWithSmoothness(
		screenResult.Return,
		screenResult.MaxDD,
		screenResult.Expectancy,
		screenResult.SmoothVol,
		screenResult.DownsideVol,
		screenResult.Trades,
		0, // No deflation penalty for screen
	)

	// Quick filter: screen must pass basic criteria
	// Less strict than full criteria (we just want to filter obvious junk)
	if screenScore < minScreenScore || screenResult.Trades < minScreenTrades || screenResult.MaxDD >= maxScreenDD {
		return false, false, screenResult, Result{}
	}

	// Stage 2: Full train window evaluation
	trainResult := evaluateStrategyWindow(full, fullF, st, trainW)

	// Compute DSR-lite score for train with smoothness (no penalty)
	trainScore := computeScoreWithSmoothness(
		trainResult.Return,
		trainResult.MaxDD,
		trainResult.Expectancy,
		trainResult.SmoothVol,
		trainResult.DownsideVol,
		trainResult.Trades,
		0, // No deflation penalty for train phase
	)

	// Basic train filter (relaxed to get candidates flowing to validation)
	if trainScore < minTrainScore || trainResult.Trades < minTrainTrades || trainResult.MaxDD >= maxTrainDD {
		return true, false, trainResult, Result{}
	}

	// Stage 3: Validation (only if passed train)
	valResult := evaluateStrategyWindow(full, fullF, st, valW)

	// Compute DSR-lite score for validation with smoothness (WITH deflation penalty)
	valScore := computeScoreWithSmoothness(
		valResult.Return,
		valResult.MaxDD,
		valResult.Expectancy,
		valResult.SmoothVol,
		valResult.DownsideVol,
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
