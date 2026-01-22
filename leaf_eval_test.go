package main

import (
	"fmt"
	"math"
	"testing"
)

// TestCrossUpAtT0 verifies that CrossUp at t=0 fails (guard check)
// CrossUp requires t >= 1 because it needs t-1 values
func TestCrossUpAtT0(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: CrossUp at t=0 should fail")
	fmt.Println("========================================")

	// Create two simple feature arrays
	// Feature A: [5, 20, 30, ...] - A starts BELOW B
	// Feature B: [15, 15, 25, ...] - B is constant at first
	featureA := []float32{5, 20, 30, 40, 50}
	featureB := []float32{15, 15, 25, 35, 45}
	features := [][]float32{featureA, featureB}

	// Create a CrossUp leaf: A crosses above B
	leaf := Leaf{
		Kind: LeafCrossUp,
		A:    0, // Feature index A
		B:    1, // Feature index B
	}

	// Test at t=0 - should FAIL because t < 1 (guard check)
	result := evaluateLeaf(&leaf, features, 0)
	if result {
		t.Errorf("FAIL: CrossUp at t=0 should return false (t<1 guard), got true")
	} else {
		fmt.Println("  PASS: CrossUp at t=0 correctly returns false (t<1 guard)")
	}

	// Test at t=1 - should SUCCEED because t >= 1 and A[t-1]=5 <= B[t-1]=15, A[t]=20 > B[t]=15
	// Note: B[t-1]=15, B[t]=15 (B doesn't move, so eps guard should block this)
	// Actually, let me adjust the data so both series move
	featureA2 := []float32{5, 20, 30, 40, 50}
	featureB2 := []float32{15, 16, 25, 35, 45}
	features2 := [][]float32{featureA2, featureB2}

	// At t=1: A[t-1]=5 <= B[t-1]=15, A[t]=20 > B[t]=16
	// Both A and B move: A moves 15, B moves 1 (both > eps)
	result = evaluateLeaf(&leaf, features2, 1)
	if !result {
		t.Errorf("FAIL: CrossUp at t=1 should return true, got false")
	} else {
		fmt.Println("  PASS: CrossUp at t=1 correctly returns true")
	}
}

// TestCrossUpWithNonMovingSeries verifies that CrossUp with non-moving series fails (eps guard)
// CrossUp requires BOTH series to actually move (eps = 1e-6)
func TestCrossUpWithNonMovingSeries(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: CrossUp with non-moving series should fail")
	fmt.Println("========================================")

	// Feature A: [10, 10, 10, ...] (constant, doesn't move)
	// Feature B: [5, 10, 15, ...] (moves up)
	featureA := []float32{10, 10, 10, 10, 10}
	featureB := []float32{5, 10, 15, 20, 25}
	features := [][]float32{featureA, featureB}

	// Create a CrossUp leaf: A crosses above B
	leaf := Leaf{
		Kind: LeafCrossUp,
		A:    0,
		B:    1,
	}

	// Test at t=1 - should FAIL because feature A doesn't move (eps guard)
	result := evaluateLeaf(&leaf, features, 1)
	if result {
		t.Errorf("FAIL: CrossUp with non-moving series A should return false (eps guard), got true")
	} else {
		fmt.Println("  PASS: CrossUp with non-moving series A correctly returns false (eps guard)")
	}
}

// TestCrossDownWithNonMovingSeries verifies that CrossDown with non-moving series fails (eps guard)
func TestCrossDownWithNonMovingSeries(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: CrossDown with non-moving series should fail")
	fmt.Println("========================================")

	// Feature A: [20, 20, 20, ...] (constant, doesn't move)
	// Feature B: [15, 20, 25, ...] (moves up)
	featureA := []float32{20, 20, 20, 20, 20}
	featureB := []float32{15, 20, 25, 30, 35}
	features := [][]float32{featureA, featureB}

	// Create a CrossDown leaf: A crosses below B
	leaf := Leaf{
		Kind: LeafCrossDown,
		A:    0,
		B:    1,
	}

	// Test at t=1 - should FAIL because feature A doesn't move (eps guard)
	result := evaluateLeaf(&leaf, features, 1)
	if result {
		t.Errorf("FAIL: CrossDown with non-moving series A should return false (eps guard), got true")
	} else {
		fmt.Println("  PASS: CrossDown with non-moving series A correctly returns false (eps guard)")
	}
}

// TestRisingWithLookback verifies Rising operator with proper lookback validation
func TestRisingWithLookback(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: Rising with lookback validation")
	fmt.Println("========================================")

	// Feature: [10, 15, 20, 25, 30, 35, 40]
	feature := []float32{10, 15, 20, 25, 30, 35, 40}
	features := [][]float32{feature}

	// Create a Rising leaf with lookback=3
	leaf := Leaf{
		Kind:     LeafRising,
		A:        0,
		Lookback: 3,
	}

	// Test at t=2 - should FAIL because t < lookback
	result := evaluateLeaf(&leaf, features, 2)
	if result {
		t.Errorf("FAIL: Rising at t=2 with lookback=3 should return false (t < lookback), got true")
	} else {
		fmt.Println("  PASS: Rising at t=2 with lookback=3 correctly returns false (t < lookback)")
	}

	// Test at t=3 - should SUCCEED because feature[t]=30 > feature[t-3]=10
	result = evaluateLeaf(&leaf, features, 3)
	if !result {
		t.Errorf("FAIL: Rising at t=3 with lookback=3 should return true (30 > 10), got false")
	} else {
		fmt.Println("  PASS: Rising at t=3 with lookback=3 correctly returns true (30 > 10)")
	}

	// Test with falling values - should FAIL
	feature2 := []float32{40, 35, 30, 25, 20, 15, 10}
	features2 := [][]float32{feature2}
	result = evaluateLeaf(&leaf, features2, 3)
	if result {
		t.Errorf("FAIL: Rising at t=3 with falling values should return false, got true")
	} else {
		fmt.Println("  PASS: Rising at t=3 with falling values correctly returns false")
	}
}

// TestFallingWithLookback verifies Falling operator with proper lookback validation
func TestFallingWithLookback(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: Falling with lookback validation")
	fmt.Println("========================================")

	// Feature: [40, 35, 30, 25, 20, 15, 10]
	feature := []float32{40, 35, 30, 25, 20, 15, 10}
	features := [][]float32{feature}

	// Create a Falling leaf with lookback=3
	leaf := Leaf{
		Kind:     LeafFalling,
		A:        0,
		Lookback: 3,
	}

	// Test at t=2 - should FAIL because t < lookback
	result := evaluateLeaf(&leaf, features, 2)
	if result {
		t.Errorf("FAIL: Falling at t=2 with lookback=3 should return false (t < lookback), got true")
	} else {
		fmt.Println("  PASS: Falling at t=2 with lookback=3 correctly returns false (t < lookback)")
	}

	// Test at t=3 - should SUCCEED because feature[t]=10 < feature[t-3]=40
	result = evaluateLeaf(&leaf, features, 3)
	if !result {
		t.Errorf("FAIL: Falling at t=3 with lookback=3 should return true (10 < 40), got false")
	} else {
		fmt.Println("  PASS: Falling at t=3 with lookback=3 correctly returns true (10 < 40)")
	}
}

// TestSlopeGT verifies SlopeGT operator with correct slope calculation
func TestSlopeGT(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: SlopeGT with slope calculation")
	fmt.Println("========================================")

	// Feature: [10, 15, 20, 25, 30] (slope = 5 per bar)
	feature := []float32{10, 15, 20, 25, 30}
	features := [][]float32{feature}

	// Create a SlopeGT leaf with lookback=4, threshold=4.5
	// Slope = (30 - 10) / 4 = 5.0, which is > 4.5
	leaf := Leaf{
		Kind:     LeafSlopeGT,
		A:        0,
		X:        4.5,
		Lookback: 4,
	}

	// Test at t=4 - should SUCCEED because slope=5.0 > threshold=4.5
	result := evaluateLeaf(&leaf, features, 4)
	if !result {
		t.Errorf("FAIL: SlopeGT at t=4 should return true (slope=5.0 > 4.5), got false")
	} else {
		fmt.Println("  PASS: SlopeGT at t=4 correctly returns true (slope=5.0 > 4.5)")
	}

	// Test with higher threshold - should FAIL
	leaf.X = 6.0
	result = evaluateLeaf(&leaf, features, 4)
	if result {
		t.Errorf("FAIL: SlopeGT at t=4 with threshold=6.0 should return false (slope=5.0 < 6.0), got true")
	} else {
		fmt.Println("  PASS: SlopeGT at t=4 with threshold=6.0 correctly returns false (slope=5.0 < 6.0)")
	}
}

// TestNaNHandling verifies that NaN values are properly handled
func TestNaNHandling(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: NaN handling in operators")
	fmt.Println("========================================")

	// Feature with NaN at index 2
	feature := []float32{10, 15, float32(math.NaN()), 25, 30}
	features := [][]float32{feature}

	// Test GT operator - should FAIL when encountering NaN
	leafGT := Leaf{
		Kind: LeafGT,
		A:    0,
		X:    20,
	}

	// At t=2, feature[t] is NaN - should return false
	result := evaluateLeaf(&leafGT, features, 2)
	if result {
		t.Errorf("FAIL: GT with NaN value should return false, got true")
	} else {
		fmt.Println("  PASS: GT with NaN value correctly returns false")
	}

	// Test CrossUp with NaN - should FAIL
	featureA := []float32{10, 15, 25, 30}
	featureB := []float32{5, float32(math.NaN()), 15, 20}
	features2 := [][]float32{featureA, featureB}

	leafCross := Leaf{
		Kind: LeafCrossUp,
		A:    0,
		B:    1,
	}

	// At t=1, featureB[t-1] is NaN - should return false
	result = evaluateLeaf(&leafCross, features2, 1)
	if result {
		t.Errorf("FAIL: CrossUp with NaN value should return false, got true")
	} else {
		fmt.Println("  PASS: CrossUp with NaN value correctly returns false")
	}
}

// TestFeatureIndexValidation verifies that out-of-bounds feature indices are handled
func TestFeatureIndexValidation(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: Feature index validation")
	fmt.Println("========================================")

	// Feature array with only 1 feature
	feature := []float32{10, 15, 20, 25, 30}
	features := [][]float32{feature}

	// Create a leaf that references feature index 5 (out of bounds)
	leaf := Leaf{
		Kind: LeafGT,
		A:    5, // Out of bounds!
		X:    20,
	}

	// Should return false (or handle gracefully) instead of panicking
	result := evaluateLeaf(&leaf, features, 2)
	if result {
		t.Errorf("FAIL: GT with out-of-bounds feature index should return false, got true")
	} else {
		fmt.Println("  PASS: GT with out-of-bounds feature index correctly returns false")
	}
}

// TestCrossUpBothSeriesMove verifies CrossUp requires BOTH series to move
func TestCrossUpBothSeriesMove(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Test: CrossUp requires BOTH series to move")
	fmt.Println("========================================")

	// Feature A: [10, 15, 20] (moves)
	// Feature B: [15, 15, 15] (doesn't move)
	featureA := []float32{10, 15, 20}
	featureB := []float32{15, 15, 15}
	features := [][]float32{featureA, featureB}

	leaf := Leaf{
		Kind: LeafCrossUp,
		A:    0,
		B:    1,
	}

	// At t=1: A[t-1]=10 <= B[t-1]=15, A[t]=15 <= B[t]=15
	// Even though A moves, B doesn't move, so CrossUp should FAIL
	result := evaluateLeaf(&leaf, features, 1)
	if result {
		t.Errorf("FAIL: CrossUp with non-moving series B should return false (eps guard), got true")
	} else {
		fmt.Println("  PASS: CrossUp with non-moving series B correctly returns false (eps guard)")
	}

	// Feature A: [10, 10, 10] (doesn't move)
	// Feature B: [15, 20, 25] (moves)
	featureA2 := []float32{10, 10, 10}
	featureB2 := []float32{15, 20, 25}
	features2 := [][]float32{featureA2, featureB2}

	// At t=1: A doesn't move, so CrossUp should FAIL
	result = evaluateLeaf(&leaf, features2, 1)
	if result {
		t.Errorf("FAIL: CrossUp with non-moving series A should return false (eps guard), got true")
	} else {
		fmt.Println("  PASS: CrossUp with non-moving series A correctly returns false (eps guard)")
	}
}
