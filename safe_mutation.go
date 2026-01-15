package main

import (
	"math"
	"math/rand"
)

// SignalSimilarityResult holds similarity metrics
type SignalSimilarityResult struct {
	AgreementPct float32
	MaxDiffBars  int
}

// compareSignals compares parent and child strategy signals
// Returns percentage of bars where signals agree
// sampleSize: number of bars to sample (default 1000)
func compareSignals(parent, child Strategy, series Series, feats Features, sampleSize int) SignalSimilarityResult {
	// Sample indices from middle of data (avoid warmup)
	totalBars := len(series.Close)
	if totalBars < sampleSize+200 {
		// Not enough data, sample all
		sampleSize = totalBars - 200
	}
	if sampleSize < 100 {
		// Too little data, return high similarity (can't check)
		return SignalSimilarityResult{AgreementPct: 1.0, MaxDiffBars: 0}
	}

	start := 200
	end := start + sampleSize
	if end > totalBars {
		end = totalBars
		sampleSize = end - start
	}

	// Sample every Nth bar (reduce computation)
	sampleStep := 10
	if sampleSize/sampleStep < 100 {
		sampleStep = max(1, sampleSize/100)
	}

	agreements := 0
	totalSamples := 0
	maxDiffBars := 0
	currentDiffBars := 0

	for t := start; t < end; t += sampleStep {
		// Get parent entry signal
		parentEntry := evaluateCompiled(parent.EntryCompiled.Code, feats.F, t)
		parentExit := evaluateCompiled(parent.ExitCompiled.Code, feats.F, t)
		parentRegime := parent.RegimeFilter.Root == nil || evaluateCompiled(parent.RegimeCompiled.Code, feats.F, t)

		// Get child entry signal
		childEntry := evaluateCompiled(child.EntryCompiled.Code, feats.F, t)
		childExit := evaluateCompiled(child.ExitCompiled.Code, feats.F, t)
		childRegime := child.RegimeFilter.Root == nil || evaluateCompiled(child.RegimeCompiled.Code, feats.F, t)

		// Determine if strategies agree (both would take same action)
		parentAction := parentRegime && parentEntry && !parentExit
		childAction := childRegime && childEntry && !childExit

		if parentAction == childAction {
			agreements++
			currentDiffBars = 0
		} else {
			currentDiffBars++
			if currentDiffBars > maxDiffBars {
				maxDiffBars = currentDiffBars
			}
		}

		totalSamples++
	}

	if totalSamples == 0 {
		return SignalSimilarityResult{AgreementPct: 1.0, MaxDiffBars: 0}
	}

	agreementPct := float32(agreements) / float32(totalSamples)

	return SignalSimilarityResult{
		AgreementPct: agreementPct,
		MaxDiffBars:  maxDiffBars,
	}
}

// isMutationSafe checks if mutation is acceptable
// Returns: (isSafe, penaltyMultiplier)
// penaltyMultiplier: 1.0 = no penalty, 0.5 = heavy penalty, 0.0 = reject
func isMutationSafe(sim SignalSimilarityResult) (bool, float32) {
	const minAgreement = 0.60  // 60% minimum agreement
	const dangerousAgreement = 0.40  // Below 40% = catastrophic change
	const maxDiffBars = 50  // Max consecutive bars of disagreement

	// Check for catastrophic behavior jumps
	if sim.AgreementPct < dangerousAgreement {
		// Reject completely
		return false, 0.0
	}

	// Check for excessive consecutive differences
	if sim.MaxDiffBars > maxDiffBars {
		// This could cause massive position flips - penalize heavily
		return true, 0.3
	}

	// Check minimum agreement threshold
	if sim.AgreementPct < minAgreement {
		// Below threshold but not catastrophic - moderate penalty
		penalty := (sim.AgreementPct - dangerousAgreement) / (minAgreement - dangerousAgreement)
		return true, penalty
	}

	// Safe mutation - no penalty
	return true, 1.0
}

// applySafeMutation wraps mutateStrategy with safety check
// Returns: (child, isSafe, penaltyMultiplier)
func applySafeMutation(rng *rand.Rand, parent Strategy, feats Features, series Series, fullF Features) (Strategy, bool, float32) {
	// Perform normal mutation
	child := mutateStrategy(rng, parent, feats)

	// Check safety
	sim := compareSignals(parent, child, series, fullF, 1000)
	isSafe, penalty := isMutationSafe(sim)

	return child, isSafe, penalty
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// applySafeCrossover wraps crossover with safety check
func applySafeCrossover(rng *rand.Rand, a, b Strategy, feats Features, series Series, fullF Features) (Strategy, bool, float32) {
	// Perform normal crossover
	child := crossover(rng, a, b)

	// Check safety against both parents
	simA := compareSignals(a, child, series, fullF, 1000)
	simB := compareSignals(b, child, series, fullF, 1000)

	isSafeA, penaltyA := isMutationSafe(simA)
	isSafeB, penaltyB := isMutationSafe(simB)

	// Use minimum penalty (most conservative)
	isSafe := isSafeA && isSafeB
	penalty := math.Min(float64(penaltyA), float64(penaltyB))

	return child, isSafe, float32(penalty)
}
