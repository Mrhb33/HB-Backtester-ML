package main

import (
	"fmt"
	"log"
	"math"
)

// FeatureGateConfig holds configuration for feature validation
type FeatureGateConfig struct {
	CheckBarOwnership  bool   // Verify features correspond to correct bar
	CheckEMAScale      bool   // Verify EMA values are in valid range
	CheckNaNInf        bool   // Check for NaN/Inf values
	RandomSampleCount  int    // Number of random indices to check
	SampleIndices      []int  // Specific indices to check
}

// FeatureValidationResult holds validation results
type FeatureValidationResult struct {
	Passed         bool
	FailedChecks   []string
	OwnershipFailures []OwnershipFailure
}

// OwnershipFailure represents a bar ownership check failure
type OwnershipFailure struct {
	Index       int
	FeatureName string
	Expected    float32
	Actual      float32
	Reason      string
}

// ValidateFeatures performs comprehensive feature validation
// If any check fails, it logs.Fatal() to stop the run (HARD STOP)
func ValidateFeatures(s Series, f Features, warmup int, cfg FeatureGateConfig) FeatureValidationResult {
	result := FeatureValidationResult{
		Passed:       true,
		FailedChecks: []string{},
		OwnershipFailures: []OwnershipFailure{},
	}

	fmt.Println("\n=== FEATURE GATE: Starting Validation ===")

	// Check 1: Length check
	fmt.Println("\n[CHECK 1] Feature length validation...")
	if !checkFeatureLengths(s, f, &result) {
		result.FailedChecks = append(result.FailedChecks, "Feature length mismatch")
		result.Passed = false
	}

	// Check 2: Close feature check (if exists)
	fmt.Println("[CHECK 2] Close feature validation...")
	if !checkCloseFeature(s, f, &result) {
		result.FailedChecks = append(result.FailedChecks, "Close feature mismatch")
		result.Passed = false
	}

	// Check 3: Bar ownership check (NEW - catches feature offset bugs)
	if cfg.CheckBarOwnership {
		fmt.Println("[CHECK 3] Bar ownership validation...")
		if !checkBarOwnership(s, f, warmup, cfg, &result) {
			result.FailedChecks = append(result.FailedChecks, "Bar ownership check failed")
			result.Passed = false
		}
	}

	// Check 4: EMA scale check
	if cfg.CheckEMAScale {
		fmt.Println("[CHECK 4] EMA scale validation...")
		if !checkEMAScale(s, f, warmup, cfg, &result) {
			result.FailedChecks = append(result.FailedChecks, "EMA scale check failed")
			result.Passed = false
		}
	}

	// Check 5: NaN/Inf check
	if cfg.CheckNaNInf {
		fmt.Println("[CHECK 5] NaN/Inf validation...")
		if !checkNaNInf(f, warmup, &result) {
			result.FailedChecks = append(result.FailedChecks, "NaN/Inf check failed")
			result.Passed = false
		}
	}

	// Print summary
	fmt.Println("\n=== FEATURE GATE: Validation Summary ===")
	if result.Passed {
		fmt.Println("Status: ALL CHECKS PASSED ✓")
	} else {
		fmt.Println("Status: VALIDATION FAILED ✗")
		fmt.Println("Failed checks:")
		for _, check := range result.FailedChecks {
			fmt.Printf("  - %s\n", check)
		}
		if len(result.OwnershipFailures) > 0 {
			fmt.Println("\nBar ownership failures:")
			for _, fail := range result.OwnershipFailures {
				fmt.Printf("  [Index %d] %s: expected=%.4f, actual=%.4f - %s\n",
					fail.Index, fail.FeatureName, fail.Expected, fail.Actual, fail.Reason)
			}
		}

		// HARD STOP - log.Fatal to prevent continuing with invalid data
		log.Fatal("Feature gate failed: Halting execution due to validation errors")
	}

	return result
}

// checkFeatureLengths verifies all features have the same length as OHLC data
func checkFeatureLengths(s Series, f Features, result *FeatureValidationResult) bool {
	allPassed := true

	for i, feat := range f.F {
		if len(feat) != len(s.Close) {
			fmt.Printf("  ✗ FAIL: Feature[%d] (%s) length=%d, expected=%d\n",
				i, f.Names[i], len(feat), len(s.Close))
			allPassed = false
		}
	}

	if allPassed {
		fmt.Printf("  ✓ PASS: All %d features have length=%d\n", len(f.F), len(s.Close))
	}

	return allPassed
}

// checkCloseFeature verifies that if a "Close" feature exists, it matches series.Close
func checkCloseFeature(s Series, f Features, result *FeatureValidationResult) bool {
	closeIdx, ok := f.Index["Close"]
	if !ok {
		fmt.Println("  ⊘ SKIP: No 'Close' feature found")
		return true
	}

	closeFeat := f.F[closeIdx]
	mismatches := 0
	sampleSize := min(100, len(s.Close))

	for i := 0; i < sampleSize; i++ {
		idx := len(s.Close) - 1 - i // Check from end
		if closeFeat[idx] != s.Close[idx] {
			mismatches++
			if mismatches <= 5 { // Only print first 5
				fmt.Printf("  ✗ FAIL: Close[%d] mismatch: feature=%.4f, series=%.4f\n",
					idx, closeFeat[idx], s.Close[idx])
			}
		}
	}

	if mismatches > 0 {
		fmt.Printf("  ✗ FAIL: %d mismatches found in Close feature (checked %d samples)\n",
			mismatches, sampleSize)
		return false
	}

	fmt.Println("  ✓ PASS: Close feature matches series.Close")
	return true
}

// checkBarOwnership verifies that features at index t actually correspond to bar t
// This is CRITICAL for catching "feature stored at t but corresponds to t-25" bugs
func checkBarOwnership(s Series, f Features, warmup int, cfg FeatureGateConfig, result *FeatureValidationResult) bool {
	indicesToCheck := cfg.SampleIndices

	// If no specific indices provided, use random sampling
	if len(indicesToCheck) == 0 {
		count := cfg.RandomSampleCount
		if count == 0 {
			count = 10 // Default to 10 random checks
		}

		// Sample indices after warmup, spread across the dataset
		step := max(1, (len(s.Close)-warmup)/count)
		for i := 0; i < count; i++ {
			idx := warmup + i*step
			if idx < len(s.Close) {
				indicesToCheck = append(indicesToCheck, idx)
			}
		}
	}

	allPassed := true

	for _, t := range indicesToCheck {
		if t < warmup || t >= len(s.Close) {
			continue
		}

		fmt.Printf("\n  Index t=%d:\n", t)

		// Check Close feature (if exists)
		if closeIdx, ok := f.Index["Close"]; ok {
			expected := s.Close[t]
			actual := f.F[closeIdx][t]
			if diff := absFeature(expected - actual); diff > 0.001 {
				fmt.Printf("    ✗ Close: series=%.4f, feature=%.4f, diff=%.4f\n",
					expected, actual, diff)
				result.OwnershipFailures = append(result.OwnershipFailures, OwnershipFailure{
					Index:       t,
					FeatureName: "Close",
					Expected:    expected,
					Actual:      actual,
					Reason:      "Close feature doesn't match series.Close",
				})
				allPassed = false
			} else {
				fmt.Printf("    ✓ Close[t]=%.4f\n", actual)
			}
		}

		// Check EMA features (should be close to price scale)
		for _, name := range f.Names {
			if len(name) >= 3 && name[0:3] == "EMA" {
				idx := f.Index[name]
				val := f.F[idx][t]
				price := s.Close[t]

				// EMA should be within reasonable range of price (ratio 0.5-1.5)
				ratio := float64(0.0)
				if price != 0 {
					ratio = float64(val) / float64(price)
				}

				if ratio < 0.5 || ratio > 1.5 {
					fmt.Printf("    ✗ %s[t]=%.4f, price=%.4f, ratio=%.3f (out of range)\n",
						name, val, price, ratio)
					result.OwnershipFailures = append(result.OwnershipFailures, OwnershipFailure{
						Index:       t,
						FeatureName: name,
						Expected:    price,
						Actual:      val,
						Reason:      fmt.Sprintf("EMA/price ratio %.3f out of range [0.5, 1.5]", ratio),
					})
					allPassed = false
				} else {
					fmt.Printf("    ✓ %s[t]=%.4f (ratio=%.3f)\n", name, val, ratio)
				}
			}

			// Check RSI features (should be 0-100)
			if len(name) >= 3 && name[0:3] == "RSI" {
				idx := f.Index[name]
				val := f.F[idx][t]

				if val < 0 || val > 100 {
					fmt.Printf("    ✗ %s[t]=%.4f (out of range [0, 100])\n", name, val)
					result.OwnershipFailures = append(result.OwnershipFailures, OwnershipFailure{
						Index:       t,
						FeatureName: name,
						Expected:    50,
						Actual:      val,
						Reason:      "RSI out of range [0, 100]",
					})
					allPassed = false
				} else {
					fmt.Printf("    ✓ %s[t]=%.4f (valid range)\n", name, val)
				}
			}
		}
	}

	if allPassed {
		fmt.Println("\n  ✓ PASS: All bar ownership checks passed")
	} else {
		fmt.Printf("\n  ✗ FAIL: %d bar ownership failures detected\n", len(result.OwnershipFailures))
	}

	return allPassed
}

// checkEMAScale verifies EMA values are at reasonable scale compared to price
func checkEMAScale(s Series, f Features, warmup int, cfg FeatureGateConfig, result *FeatureValidationResult) bool {
	indicesToCheck := cfg.SampleIndices

	if len(indicesToCheck) == 0 {
		count := cfg.RandomSampleCount
		if count == 0 {
			count = 10
		}
		step := max(1, (len(s.Close)-warmup)/count)
		for i := 0; i < count; i++ {
			idx := warmup + i*step
			if idx < len(s.Close) {
				indicesToCheck = append(indicesToCheck, idx)
			}
		}
	}

	allPassed := true
	for _, name := range f.Names {
		if len(name) >= 3 && name[0:3] == "EMA" {
			idx := f.Index[name]
			for _, t := range indicesToCheck {
				if t >= len(s.Close) {
					continue
				}
				val := f.F[idx][t]
				price := s.Close[t]

				ratio := float64(0.0)
				if price != 0 {
					ratio = float64(val) / float64(price)
				}

				if ratio < 0.5 || ratio > 1.5 {
					fmt.Printf("  ✗ FAIL: %s[%d]=%.4f, price=%.4f, ratio=%.3f\n",
						name, t, val, price, ratio)
					allPassed = false
				}
			}
		}
	}

	if allPassed {
		fmt.Println("  ✓ PASS: All EMA values at valid scale")
	}

	return allPassed
}

// checkNaNInf verifies no NaN or Inf values after warmup
func checkNaNInf(f Features, warmup int, result *FeatureValidationResult) bool {
	foundNaN := false
	foundInf := false

	for i, feat := range f.F {
		for t := warmup; t < len(feat); t++ {
			val := feat[t]
			if math.IsNaN(float64(val)) {
				fmt.Printf("  ✗ FAIL: Feature[%d] (%s) has NaN at index %d\n",
					i, f.Names[i], t)
				foundNaN = true
			}
			if math.IsInf(float64(val), 0) {
				fmt.Printf("  ✗ FAIL: Feature[%d] (%s) has Inf at index %d\n",
					i, f.Names[i], t)
				foundInf = true
			}
		}
	}

	if !foundNaN && !foundInf {
		fmt.Println("  ✓ PASS: No NaN or Inf values found")
		return true
	}

	return false
}

// Helper functions
func absFeature(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
