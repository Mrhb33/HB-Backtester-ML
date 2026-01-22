package main

import (
	"fmt"
	"os"
)

// RunValidation executes the feature gate validation
// This can be called from main.go with -mode=validate
func RunValidation(dataPath string) {
	fmt.Println("\n=== FEATURE GATE VALIDATION ===")
	fmt.Println("Loading data from:", dataPath)

	// Load data
	s, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		os.Exit(1)
	}

	// Compute features
	f := computeAllFeatures(s)

	// Configure validation
	cfg := FeatureGateConfig{
		CheckBarOwnership: true,
		CheckEMAScale:     true,
		CheckNaNInf:       true,
		RandomSampleCount: 10,
	}

	// Run validation
	const warmup = 200
	result := ValidateFeatures(s, f, warmup, cfg)

	// Print summary
	if result.Passed {
		fmt.Println("\n=== VALIDATION SUMMARY ===")
		fmt.Println("Status: ALL CHECKS PASSED ✓")
		fmt.Printf("Series: %d bars\n", len(s.Close))
		fmt.Printf("Features: %d computed\n", len(f.F))
		fmt.Println("Feature names:")
		for i, name := range f.Names {
			fmt.Printf("  [%d] %s (length=%d)\n", i, name, len(f.F[i]))
		}
	} else {
		fmt.Println("\n=== VALIDATION FAILED ===")
		fmt.Println("Failed checks:", result.FailedChecks)
		os.Exit(1)
	}
}
