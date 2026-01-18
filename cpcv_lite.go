package main

import (
	"sort"
)

// CPCVFold represents a validation fold
type CPCVFold struct {
	TrainStart int
	TrainEnd   int
	TestStart  int
	TestEnd    int
}

// CPCVResult holds results across all folds
type CPCVResult struct {
	Scores          []float32
	Returns         []float32
	MaxDDs          []float32
	WinRates        []float32
	Trades          []int
	MeanScore       float32
	MedianScore     float32 // NEW: median score for stability
	MedianReturn    float32 // NEW: median return
	WorstScore      float32
	MinFoldScore    float32 // Simple min score without DSR penalty
	PassingCount    int
	WeakFoldCount   int // NEW: count of weak folds (trades 20-29)
	ProfitableFolds int // NEW: count of profitable folds
	Stability       float32
}

// generateCPCVFolds creates purged walk-forward folds
// Splits train window into 6 blocks, creates 3-fold combinations
// Uses embargo/purge gaps between blocks
func generateCPCVFolds(trainStart, trainEnd int, purgePct float32) []CPCVFold {
	const numBlocks = 6
	const numFolds = 3 // Use top 3 combinations

	// Calculate block size
	totalCandles := trainEnd - trainStart
	blockSize := totalCandles / numBlocks

	// Purge gap: percentage of block size
	purgeGap := int(float32(blockSize) * purgePct)

	// Generate block boundaries
	blocks := make([]struct{ start, end int }, numBlocks)
	for i := 0; i < numBlocks; i++ {
		start := trainStart + i*blockSize
		end := start + blockSize
		if i == numBlocks-1 {
			end = trainEnd
		}
		blocks[i] = struct{ start, end int }{start, end}
	}

	// Create 3 test folds:
	// 1. Train on blocks 0-3, test on block 4 (with purge)
	// 2. Train on blocks 1-4, test on block 5 (with purge)
	// 3. Train on blocks 0,1,3,4 (skip 2), test on block 5
	folds := make([]CPCVFold, 0, numFolds)

	// Fold 1: Train blocks 0-3, test block 4
	test1Start := blocks[4].start
	test1End := blocks[4].end
	train1End := blocks[3].end - purgeGap
	if train1End > test1Start-purgeGap {
		train1End = test1Start - purgeGap
	}
	folds = append(folds, CPCVFold{
		TrainStart: blocks[0].start,
		TrainEnd:   train1End,
		TestStart:  test1Start,
		TestEnd:    test1End,
	})

	// Fold 2: Train blocks 1-4, test block 5
	test2Start := blocks[5].start
	test2End := blocks[5].end
	train2End := blocks[4].end - purgeGap
	if train2End > test2Start-purgeGap {
		train2End = test2Start - purgeGap
	}
	folds = append(folds, CPCVFold{
		TrainStart: blocks[1].start,
		TrainEnd:   train2End,
		TestStart:  test2Start,
		TestEnd:    test2End,
	})

	// Fold 3: Train on 0,1,3,4 (skip 2 for variety), test on 5
	test3Start := blocks[5].start
	test3End := blocks[5].end
	train3Start := blocks[0].start
	train3End := blocks[4].end - purgeGap
	if train3End > test3Start-purgeGap {
		train3End = test3Start - purgeGap
	}
	folds = append(folds, CPCVFold{
		TrainStart: train3Start,
		TrainEnd:   train3End,
		TestStart:  test3Start,
		TestEnd:    test3End,
	})

	return folds
}

// evaluateCPCV runs strategy through all CPCV folds
func evaluateCPCV(full Series, fullF Features, st Strategy, trainStart, trainEnd int, testedCount int64, minScoreThreshold float32) CPCVResult {
	// Generate folds with 10% purge gap
	folds := generateCPCVFolds(trainStart, trainEnd, 0.10)

	result := CPCVResult{
		Scores:   make([]float32, 0, len(folds)),
		Returns:  make([]float32, 0, len(folds)),
		MaxDDs:   make([]float32, 0, len(folds)),
		WinRates: make([]float32, 0, len(folds)),
		Trades:   make([]int, 0, len(folds)),
	}

	for _, fold := range folds {
		// Create test window for this fold
		testW := Window{
			Start:  fold.TestStart,
			End:    fold.TestEnd,
			Warmup: 200,
		}

		// Evaluate on test (this is what matters)
		testResult := evaluateStrategyWindow(full, fullF, st, testW)

		// Simple score without DSR penalty for MinFoldScore check
		simpleFoldScore := testResult.Return / (testResult.MaxDD + 1e-4)
		if len(result.Scores) == 0 || simpleFoldScore < result.MinFoldScore || result.MinFoldScore == 0 {
			result.MinFoldScore = simpleFoldScore
		}

		// Compute DSR-lite score for this fold with smoothness
		foldScore := computeScoreWithSmoothness(
			testResult.Return,
			testResult.MaxDD,
			testResult.Expectancy,
			testResult.SmoothVol,
			testResult.DownsideVol,
			testResult.Trades,
			testedCount, // Apply DSR-lite penalty
		)

		result.Scores = append(result.Scores, foldScore)
		result.Returns = append(result.Returns, testResult.Return)
		result.MaxDDs = append(result.MaxDDs, testResult.MaxDD)
		result.WinRates = append(result.WinRates, testResult.WinRate)
		result.Trades = append(result.Trades, testResult.Trades)

		// Track profitable folds
		if testResult.Return > 0 {
			result.ProfitableFolds++
		}

		// Check if this fold passes threshold
		// FINAL-MODE STRICT: Use production DD limits matching test mode
		// Level 3 (unblock): 0.45, Level 0-2: 0.35 (matching test mode)
		cpcvMaxDDThreshold := float32(0.35) // Strict: matches test mode MaxDD<=0.35
		if getScreenRelaxLevel() >= 3 {
			cpcvMaxDDThreshold = 0.45 // Slightly relaxed during unblock mode
		}

		// Fold passing logic with final-mode trade requirements
		// Require minTrades >= 50 for passing fold (matching test mode)
		minTradesPerFold := 50
		if getScreenRelaxLevel() >= 3 {
			minTradesPerFold = 30 // Relaxed during unblock mode
		}

		if foldScore >= minScoreThreshold && testResult.MaxDD < cpcvMaxDDThreshold {
			if testResult.Trades >= minTradesPerFold {
				result.PassingCount++ // Fully passing fold (meets final requirements)
			} else if testResult.Trades >= 30 {
				result.WeakFoldCount++ // Weak fold (trades 30-49)
			}
			// Trades < 30: don't count at all (insufficient signal)
		}
	}

	// Compute aggregate metrics (including median for stability)
	if len(result.Scores) > 0 {
		// Mean score
		sum := float32(0)
		worst := result.Scores[0]
		for _, score := range result.Scores {
			sum += score
			if score < worst {
				worst = score
			}
		}
		result.MeanScore = sum / float32(len(result.Scores))
		result.WorstScore = worst

		// Median score (sort and pick middle)
		sortedScores := make([]float32, len(result.Scores))
		copy(sortedScores, result.Scores)
		// Simple bubble sort for small slices (3 elements)
		for i := 0; i < len(sortedScores); i++ {
			for j := i + 1; j < len(sortedScores); j++ {
				if sortedScores[i] > sortedScores[j] {
					sortedScores[i], sortedScores[j] = sortedScores[j], sortedScores[i]
				}
			}
		}
		result.MedianScore = sortedScores[len(sortedScores)/2]

		// Median return (sort and pick middle)
		sortedReturns := make([]float32, len(result.Returns))
		copy(sortedReturns, result.Returns)
		for i := 0; i < len(sortedReturns); i++ {
			for j := i + 1; j < len(sortedReturns); j++ {
				if sortedReturns[i] > sortedReturns[j] {
					sortedReturns[i], sortedReturns[j] = sortedReturns[j], sortedReturns[i]
				}
			}
		}
		result.MedianReturn = sortedReturns[len(sortedReturns)/2]

		// Stability: % of profitable folds with weak fold penalty
		// Weak folds (trades 20-29) count as 0.5 instead of 1.0
		effectiveProfitableFolds := float32(result.ProfitableFolds) - float32(result.WeakFoldCount)*0.5
		result.Stability = effectiveProfitableFolds / float32(len(folds))
	}

	return result
}

// CPCVPassCriteria checks if CPCV results are acceptable
// Uses final production requirements to match test-mode gates
func CPCVPassCriteria(cpcv CPCVResult, minMeanScore float32, minStability float32) bool {
	if len(cpcv.Scores) == 0 {
		return false
	}

	// Must have passing mean score
	if cpcv.MeanScore < minMeanScore {
		return false
	}

	// Must have passing median score (more robust than mean)
	if cpcv.MedianScore < minMeanScore*0.8 {
		return false
	}

	// Must have positive median return (real stability)
	if cpcv.MedianReturn <= 0 {
		return false
	}

	// Must have minimum stability (at least 2/3 folds profitable)
	// minStability is passed from caller (0.66 for normal, 0.30 for unblock mode)
	if cpcv.Stability < minStability {
		return false
	}

	// Worst fold shouldn't be catastrophic
	if cpcv.WorstScore < minMeanScore*0.7 {
		return false
	}

	// FINAL-MODE STRICT: Ensure trades meet minimum threshold
	// Count folds with minimum trades (at least half of folds should have >= 50 trades)
	minTradesFolds := 0
	minTradesRequired := 50
	for _, trades := range cpcv.Trades {
		if trades >= minTradesRequired {
			minTradesFolds++
		}
	}
	// Require at least half of folds to have sufficient trades
	if minTradesFolds < len(cpcv.Trades)/2 {
		return false
	}

	return true
}

// CPCVSortResults sorts strategies by CPCV score (mean - penalty for instability)
func SortCPCVResults(results []CPCVResult) []int {
	// Create indices and sort by composite score
	indices := make([]int, len(results))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		// Composite: mean score - penalty for high variance
		// Penalty: (1 - stability) * 2.0
		scoreI := results[i].MeanScore - (1.0-results[i].Stability)*2.0
		scoreJ := results[j].MeanScore - (1.0-results[j].Stability)*2.0
		return scoreI > scoreJ
	})

	return indices
}
