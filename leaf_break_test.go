package main

import (
	"fmt"
	"testing"
)

// TestBreakUpAtT0 verifies guard check at t=0
func TestBreakUpAtT0(t *testing.T) {
	featureA := []float32{10, 20}
	featureB := []float32{15, 25}
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakUp, A: 0, B: 1}
	result := evaluateLeaf(&leaf, features, 0)

	if result {
		t.Errorf("FAIL: BreakUp at t=0 should return false (t<1 guard)")
	} else {
		fmt.Println("PASS: BreakUp at t=0 correctly returns false")
	}
}

// TestBreakDownAtT0 verifies guard check at t=0
func TestBreakDownAtT0(t *testing.T) {
	featureA := []float32{30, 20}
	featureB := []float32{15, 25}
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakDown, A: 0, B: 1}
	result := evaluateLeaf(&leaf, features, 0)

	if result {
		t.Errorf("FAIL: BreakDown at t=0 should return false (t<1 guard)")
	} else {
		fmt.Println("PASS: BreakDown at t=0 correctly returns false")
	}
}

// TestBreakUpLogic verifies A[t-1] <= B[t-1] && A[t] > B[t]
func TestBreakUpLogic(t *testing.T) {
	featureA := []float32{10, 10, 20, 30} // A: stays below, then breaks above
	featureB := []float32{15, 15, 15, 15}  // B: constant level
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakUp, A: 0, B: 1}

	// t=1: A[t-1]=10 <= B[t-1]=15, A[t]=10 <= B[t]=15 - NO BREAK
	if evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakUp at t=1 should return false (no break)")
	} else {
		fmt.Println("PASS: BreakUp at t=1 correctly returns false")
	}

	// t=2: A[t-1]=10 <= B[t-1]=15, A[t]=20 > B[t]=15 - BREAK!
	if !evaluateLeaf(&leaf, features, 2) {
		t.Errorf("FAIL: BreakUp at t=2 should return true (breakout)")
	} else {
		fmt.Println("PASS: BreakUp at t=2 correctly returns true")
	}
}

// TestBreakUpWithHH_prev validates breakout with Close vs HH_50_prev
func TestBreakUpWithHHPrev(t *testing.T) {
	// Simulate Close breaking above HH_20_prev
	closeSeries := []float32{100, 105, 110, 115, 120, 118}
	hh20Series := []float32{0, 0, 0, 0, 110, 115} // HH from bars [0-3], [1-4]

	features := [][]float32{closeSeries, hh20Series}
	leaf := Leaf{Kind: LeafBreakUp, A: 0, B: 1} // Close > HH_20_prev

	// t=4: Close[t-1]=115 <= HH[t-1]=110 (NO), Close[t]=120 > HH[t]=110 - but prevA > prevB
	result := evaluateLeaf(&leaf, features, 4)
	if result {
		t.Errorf("FAIL: BreakUp at t=4 should return false (prevA > prevB)")
	} else {
		fmt.Println("PASS: BreakUp at t=4 correctly returns false (prevA > prevB)")
	}

	// t=5: Close[t-1]=120 > HH[t-1]=110, Close[t]=118 > HH[t]=115 - but prevA > prevB
	result = evaluateLeaf(&leaf, features, 5)
	if result {
		t.Errorf("FAIL: BreakUp at t=5 should return false (prevA > prevB)")
	} else {
		fmt.Println("PASS: BreakUp at t=5 correctly returns false (prevA > prevB)")
	}
}

// TestBreakUpVsCrossUp verifies BreakUp doesn't require movement
func TestBreakUpVsCrossUp(t *testing.T) {
	// Feature A: constant
	// Feature B: moves up, crossing A
	featureA := []float32{10, 10, 10}
	featureB := []float32{5, 15, 25}
	features := [][]float32{featureA, featureB}

	// BreakUp: B[t-1]=5 <= A[t-1]=10, B[t]=15 > A[t]=10 - TRUE (only B moves)
	breakUpLeaf := Leaf{Kind: LeafBreakUp, A: 1, B: 0}
	if !evaluateLeaf(&breakUpLeaf, features, 1) {
		t.Errorf("FAIL: BreakUp should return true (B breaks above constant A)")
	} else {
		fmt.Println("PASS: BreakUp works when only B moves")
	}

	// CrossUp: Should FAIL because A doesn't move (eps guard)
	crossUpLeaf := Leaf{Kind: LeafCrossUp, A: 1, B: 0}
	if evaluateLeaf(&crossUpLeaf, features, 1) {
		t.Errorf("FAIL: CrossUp should return false (A doesn't move)")
	} else {
		fmt.Println("PASS: CrossUp blocked by eps guard")
	}
}

// TestBreakDownLogic verifies A[t-1] >= B[t-1] && A[t] < B[t]
func TestBreakDownLogic(t *testing.T) {
	featureA := []float32{30, 30, 20, 10} // A: stays above, then breaks below
	featureB := []float32{15, 15, 15, 15}  // B: constant level
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakDown, A: 0, B: 1}

	// t=1: A[t-1]=30 >= B[t-1]=15, A[t]=30 >= B[t]=15 - NO BREAK
	if evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakDown at t=1 should return false (no break)")
	} else {
		fmt.Println("PASS: BreakDown at t=1 correctly returns false")
	}

	// t=2: A[t-1]=30 >= B[t-1]=15, A[t]=20 > B[t]=15 - NO BREAK (not <)
	if evaluateLeaf(&leaf, features, 2) {
		t.Errorf("FAIL: BreakDown at t=2 should return false (not below)")
	} else {
		fmt.Println("PASS: BreakDown at t=2 correctly returns false")
	}

	// t=3: A[t-1]=20 >= B[t-1]=15, A[t]=10 < B[t]=15 - BREAK!
	if !evaluateLeaf(&leaf, features, 3) {
		t.Errorf("FAIL: BreakDown at t=3 should return true (breakdown)")
	} else {
		fmt.Println("PASS: BreakDown at t=3 correctly returns true")
	}
}

// TestBreakDownWithLLPrev validates breakdown with Close vs LL_20_prev
func TestBreakDownWithLLPrev(t *testing.T) {
	// Simulate Close breaking below LL_20_prev
	// Need: A[t-1] >= B[t-1] && A[t] < B[t] for BreakDown
	closeSeries := []float32{100, 95, 90, 92, 88, 84, 82}
	ll20Series := []float32{0, 0, 0, 0, 85, 85, 87} // LL from bars [0-3], [1-4], [2-5]

	features := [][]float32{closeSeries, ll20Series}
	leaf := Leaf{Kind: LeafBreakDown, A: 0, B: 1} // Close < LL_20_prev

	// t=5: Close[t-1]=88 >= LL[t-1]=85, Close[t]=84 < LL[t]=85 - true breakdown
	result := evaluateLeaf(&leaf, features, 5)
	if !result {
		t.Errorf("FAIL: BreakDown at t=5 should return true (breakdown)")
	} else {
		fmt.Println("PASS: BreakDown at t=5 correctly returns true")
	}

	// t=6: Close[t-1]=84 < LL[t-1]=85, Close[t]=82 < LL[t]=87 - NOT a breakdown (wasn't above)
	result = evaluateLeaf(&leaf, features, 6)
	if result {
		t.Errorf("FAIL: BreakDown at t=6 should return false (wasn't above)")
	} else {
		fmt.Println("PASS: BreakDown at t=6 correctly returns false (wasn't above)")
	}
}

// TestBreakUpEdgeCases tests edge cases for BreakUp
func TestBreakUpEdgeCases(t *testing.T) {
	// Test with equal values (boundary condition)
	featureA := []float32{10, 10, 10, 11} // A: stays at 10, then moves to 11
	featureB := []float32{10, 10, 10, 10}  // B: constant at 10
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakUp, A: 0, B: 1}

	// t=1: A[t-1]=10 <= B[t-1]=10 (equal), A[t]=10 <= B[t]=10 (equal) - NO BREAK
	if evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakUp at t=1 should return false (equal values)")
	} else {
		fmt.Println("PASS: BreakUp at t=1 correctly returns false (equal values)")
	}

	// t=3: A[t-1]=10 <= B[t-1]=10 (equal), A[t]=11 > B[t]=10 - BREAK!
	if !evaluateLeaf(&leaf, features, 3) {
		t.Errorf("FAIL: BreakUp at t=3 should return true (breakout from equality)")
	} else {
		fmt.Println("PASS: BreakUp at t=3 correctly returns true (breakout from equality)")
	}
}

// TestBreakDownEdgeCases tests edge cases for BreakDown
func TestBreakDownEdgeCases(t *testing.T) {
	// Test with equal values (boundary condition)
	featureA := []float32{10, 10, 10, 9} // A: stays at 10, then moves to 9
	featureB := []float32{10, 10, 10, 10} // B: constant at 10
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakDown, A: 0, B: 1}

	// t=1: A[t-1]=10 >= B[t-1]=10 (equal), A[t]=10 >= B[t]=10 (equal) - NO BREAK
	if evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakDown at t=1 should return false (equal values)")
	} else {
		fmt.Println("PASS: BreakDown at t=1 correctly returns false (equal values)")
	}

	// t=3: A[t-1]=10 >= B[t-1]=10 (equal), A[t]=9 < B[t]=10 - BREAK!
	if !evaluateLeaf(&leaf, features, 3) {
		t.Errorf("FAIL: BreakDown at t=3 should return true (breakdown from equality)")
	} else {
		fmt.Println("PASS: BreakDown at t=3 correctly returns true (breakdown from equality)")
	}
}

// TestBreakUpWithProof verifies BreakUp evaluation with proof
func TestBreakUpWithProof(t *testing.T) {
	featureA := []float32{10, 10, 20}
	featureB := []float32{15, 15, 15}
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakUp, A: 0, B: 1}
	result, proof := evaluateLeafWithProof(&leaf, features, nil, 2)

	if !result {
		t.Errorf("FAIL: BreakUp at t=2 should return true")
	}
	if proof.Kind != "BreakUp" {
		t.Errorf("FAIL: Proof kind should be BreakUp, got %s", proof.Kind)
	}
	if proof.Operator != "BreakUp" {
		t.Errorf("FAIL: Proof operator should be BreakUp, got %s", proof.Operator)
	}
	if !proof.Result {
		t.Errorf("FAIL: Proof result should match return value")
	}

	// Verify the proof contains the expected comparisons
	hasPrevComparison := false
	hasCurrentComparison := false
	for _, comp := range proof.Comparisons {
		if comp == "A[t-1] <= B[t-1]: 10.00 <= 15.00: true" {
			hasPrevComparison = true
		}
		if comp == "A[t] > B[t]: 20.00 > 15.00: true" {
			hasCurrentComparison = true
		}
	}
	if !hasPrevComparison {
		t.Errorf("FAIL: Proof should contain prev comparison, got: %v", proof.Comparisons)
	}
	if !hasCurrentComparison {
		t.Errorf("FAIL: Proof should contain current comparison, got: %v", proof.Comparisons)
	}

	fmt.Println("PASS: BreakUp with proof returns correct proof structure")
}

// TestBreakDownWithProof verifies BreakDown evaluation with proof
func TestBreakDownWithProof(t *testing.T) {
	featureA := []float32{30, 20, 10}
	featureB := []float32{15, 15, 15}
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakDown, A: 0, B: 1}
	result, proof := evaluateLeafWithProof(&leaf, features, nil, 2)

	if !result {
		t.Errorf("FAIL: BreakDown at t=2 should return true")
	}
	if proof.Kind != "BreakDown" {
		t.Errorf("FAIL: Proof kind should be BreakDown, got %s", proof.Kind)
	}
	if proof.Operator != "BreakDown" {
		t.Errorf("FAIL: Proof operator should be BreakDown, got %s", proof.Operator)
	}
	if !proof.Result {
		t.Errorf("FAIL: Proof result should match return value")
	}

	// Verify the proof contains the expected comparisons
	hasPrevComparison := false
	hasCurrentComparison := false
	for _, comp := range proof.Comparisons {
		if comp == "A[t-1] >= B[t-1]: 20.00 >= 15.00: true" {
			hasPrevComparison = true
		}
		if comp == "A[t] < B[t]: 10.00 < 15.00: true" {
			hasCurrentComparison = true
		}
	}
	if !hasPrevComparison {
		t.Errorf("FAIL: Proof should contain prev comparison, got: %v", proof.Comparisons)
	}
	if !hasCurrentComparison {
		t.Errorf("FAIL: Proof should contain current comparison, got: %v", proof.Comparisons)
	}

	fmt.Println("PASS: BreakDown with proof returns correct proof structure")
}

// TestBreakUpNoMovementRequired verifies BreakUp works without movement requirement
func TestBreakUpNoMovementRequired(t *testing.T) {
	// Even if A doesn't move, BreakUp should work if B moves
	featureA := []float32{10, 10, 10} // A: constant
	featureB := []float32{5, 15, 25}   // B: moves up
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakUp, A: 1, B: 0} // B > A

	// t=1: B[t-1]=5 <= A[t-1]=10, B[t]=15 > A[t]=10 - TRUE (B breaks above A)
	if !evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakUp should return true when B moves and A doesn't")
	} else {
		fmt.Println("PASS: BreakUp works when B moves and A doesn't")
	}

	// t=2: B[t-1]=15 > A[t-1]=10, B[t]=25 > A[t]=10 - FALSE (prevB > prevA)
	if evaluateLeaf(&leaf, features, 2) {
		t.Errorf("FAIL: BreakUp at t=2 should return false (prevB > prevA)")
	} else {
		fmt.Println("PASS: BreakUp at t=2 correctly returns false (prevB > prevA)")
	}
}

// TestBreakDownNoMovementRequired verifies BreakDown works without movement requirement
func TestBreakDownNoMovementRequired(t *testing.T) {
	// Even if A doesn't move, BreakDown should work if B moves
	featureA := []float32{10, 10, 10} // A: constant
	featureB := []float32{15, 5, 0}    // B: moves down
	features := [][]float32{featureA, featureB}

	leaf := Leaf{Kind: LeafBreakDown, A: 1, B: 0} // B < A

	// t=1: B[t-1]=15 >= A[t-1]=10, B[t]=5 < A[t]=10 - TRUE (B breaks below A)
	if !evaluateLeaf(&leaf, features, 1) {
		t.Errorf("FAIL: BreakDown should return true when B moves and A doesn't")
	} else {
		fmt.Println("PASS: BreakDown works when B moves and A doesn't")
	}

	// t=2: B[t-1]=5 < A[t-1]=10, B[t]=0 < A[t]=10 - FALSE (prevB < prevA)
	if evaluateLeaf(&leaf, features, 2) {
		t.Errorf("FAIL: BreakDown at t=2 should return false (prevB < prevA)")
	} else {
		fmt.Println("PASS: BreakDown at t=2 correctly returns false (prevB < prevA)")
	}
}

// TestSweepFlagsAreBinary verifies sweep features are 0/1 and never NaN/Inf
func TestSweepFlagsAreBinary(t *testing.T) {
	s := Series{
		High:  []float32{100, 110, 120, 115, 125},
		Low:   []float32{95, 105, 115, 110, 120},
		Open:  []float32{98, 108, 118, 113, 123},
		Close: []float32{108, 118, 110, 118, 122},
		Volume: []float32{1000, 1500, 2000, 1200, 1800},
		TakerBuyBase: []float32{500, 750, 1000, 600, 900},
		Trades: []int32{100, 150, 200, 120, 180},
		T:     5,
	}

	f := computeAllFeatures(s)

	// Check all sweep features
	sweepFeatures := []string{"SweepUp_20", "SweepDown_20", "SweepUp_50", "SweepDown_50"}
	for _, featName := range sweepFeatures {
		if idx, ok := f.Index[featName]; ok {
			arr := f.F[idx]
			for i, val := range arr {
				// Must be 0 or 1
				if val != 0 && val != 1 {
					t.Errorf("FAIL: %s[%d] = %.2f (not 0/1)", featName, i, val)
				}
			}
			fmt.Printf("PASS: %s is binary (0/1)\n", featName)
		}
	}
}

// TestCandleAnatomyNormalized verifies candle anatomy features are 0-1
func TestCandleAnatomyNormalized(t *testing.T) {
	s := Series{
		High:  []float32{110, 120},
		Low:   []float32{100, 110},
		Open:  []float32{105, 115},
		Close: []float32{115, 118},
		Volume: []float32{1000, 1500},
		TakerBuyBase: []float32{500, 750},
		Trades: []int32{100, 150},
		T:     2,
	}

	f := computeAllFeatures(s)

	candleFeatures := []string{"BodyPct", "WickUpPct", "WickDownPct", "ClosePos"}
	for _, featName := range candleFeatures {
		if idx, ok := f.Index[featName]; ok {
			arr := f.F[idx]
			for i, val := range arr {
				// Must be in [0, 1]
				if val < 0 || val > 1 {
					t.Errorf("FAIL: %s[%d] = %.4f (not in [0,1])", featName, i, val)
				}
			}
			fmt.Printf("PASS: %s is normalized [0,1]\n", featName)
		}
	}
}
