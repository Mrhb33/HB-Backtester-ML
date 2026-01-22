package main

import (
	"fmt"
	"testing"
)

// TestDeterministicCross tests cross operator bytecode evaluation
// Compares manual cross detection vs bytecode cross detection
func TestDeterministicCross(t *testing.T) {
	fmt.Println("\n=== DETERMINISTIC TEST: Cross Operator ===")

	// Load data
	s, f := loadTestData()
	if len(s.Close) == 0 {
		t.Skip("No test data available")
	}

	const warmup = 200

	// Check if we have EMA20 and EMA50 features
	ema20Idx, hasEMA20 := f.Index["EMA20"]
	ema50Idx, hasEMA50 := f.Index["EMA50"]

	if !hasEMA20 || !hasEMA50 {
		t.Skip("EMA20 or EMA50 feature not found")
	}

	// Create manual EMA20x50 strategy (using existing function from main.go)
	st := createTestEMAStrategy(f, 30, 8)

	// Compare manual vs bytecode for all indices after warmup
	mismatches := 0
	maxMismatchesToPrint := 10

	for i := warmup; i < len(s.Close); i++ {
		// Compute cross via raw arrays (manual formula)
		ema20 := f.F[ema20Idx]
		ema50 := f.F[ema50Idx]

		if i < 1 {
			continue
		}

		manualCross := (ema20[i-1] <= ema50[i-1]) && (ema20[i] > ema50[i])

		// Compare to bytecode cross
		bytecodeCross := evaluateCompiled(st.EntryCompiled.Code, f.F, i)

		if manualCross != bytecodeCross {
			mismatches++
			if mismatches <= maxMismatchesToPrint {
				t.Logf("CROSS MISMATCH at i=%d: manual=%v, bytecode=%v", i, manualCross, bytecodeCross)
				t.Logf("  ema20[i-1]=%.2f ema50[i-1]=%.2f ema20[i]=%.2f ema50[i]=%.2f",
					ema20[i-1], ema50[i-1], ema20[i], ema50[i])
			}
		}
	}

	fmt.Printf("Checked %d bars, found %d mismatches\n", len(s.Close)-warmup, mismatches)

	if mismatches > 0 {
		t.Fatalf("Cross operator test FAILED: %d mismatches found", mismatches)
	}

	fmt.Println("✓ PASS: Cross operator test PASSED")
}

// TestDeterministicThreshold tests threshold (GT/LT) operator bytecode evaluation
// Separates "cross operator bug" from "general feature alignment bug"
// NOTE: GT/LT compare against a fixed threshold value (X), not another feature
func TestDeterministicThreshold(t *testing.T) {
	fmt.Println("\n=== DETERMINISTIC TEST: Threshold Operator ===")

	// Load data
	s, f := loadTestData()
	if len(s.Close) == 0 {
		t.Skip("No test data available")
	}

	const warmup = 200

	// Check if we have EMA20 feature
	ema20Idx, hasEMA20 := f.Index["EMA20"]

	if !hasEMA20 {
		t.Skip("EMA20 feature not found")
	}

	// Get average price to set a reasonable threshold
	var avgPrice float32
	for i := warmup; i < warmup+1000; i++ {
		avgPrice += s.Close[i]
	}
	avgPrice /= 1000

	// Create threshold strategy: EMA20 > avgPrice (no cross, just threshold)
	st := buildFixedThresholdStrategy("EMA20", avgPrice, LeafGT, f)

	// Compare manual vs bytecode for all indices after warmup
	mismatches := 0
	maxMismatchesToPrint := 10

	for i := warmup; i < len(s.Close); i++ {
		ema20 := f.F[ema20Idx]
		manualResult := ema20[i] > avgPrice

		// Compare to bytecode
		bytecodeResult := evaluateCompiled(st.EntryCompiled.Code, f.F, i)

		if manualResult != bytecodeResult {
			mismatches++
			if mismatches <= maxMismatchesToPrint {
				t.Logf("THRESHOLD MISMATCH at i=%d: manual=%v, bytecode=%v", i, manualResult, bytecodeResult)
				t.Logf("  ema20[i]=%.2f threshold=%.2f", ema20[i], avgPrice)
			}
		}
	}

	fmt.Printf("Checked %d bars, found %d mismatches\n", len(s.Close)-warmup, mismatches)

	if mismatches > 0 {
		t.Fatalf("Threshold operator test FAILED: %d mismatches found", mismatches)
	}

	fmt.Println("✓ PASS: Threshold operator test PASSED")
}

// TestDeterministicRSI tests RSI threshold bytecode evaluation
// Non-price feature test (separates feature computation bug from operator bug)
func TestDeterministicRSI(t *testing.T) {
	fmt.Println("\n=== DETERMINISTIC TEST: RSI Threshold ===")

	// Load data
	s, f := loadTestData()
	if len(s.Close) == 0 {
		t.Skip("No test data available")
	}

	const warmup = 200

	// Check if we have RSI14 feature
	rsiIdx, hasRSI := f.Index["RSI14"]

	if !hasRSI {
		t.Skip("RSI14 feature not found")
	}

	// Create RSI strategy: RSI14 < 30
	st := buildRSIStrategy(14, 30, LeafLT, f)

	// Compare manual vs bytecode for all indices after warmup
	mismatches := 0
	maxMismatchesToPrint := 10

	for i := warmup; i < len(s.Close); i++ {
		rsi14 := f.F[rsiIdx]
		manualResult := rsi14[i] < 30

		// Compare to bytecode
		bytecodeResult := evaluateCompiled(st.EntryCompiled.Code, f.F, i)

		if manualResult != bytecodeResult {
			mismatches++
			if mismatches <= maxMismatchesToPrint {
				t.Logf("RSI MISMATCH at i=%d: manual=%v, bytecode=%v", i, manualResult, bytecodeResult)
				t.Logf("  rsi14[i]=%.2f", rsi14[i])
			}
		}
	}

	fmt.Printf("Checked %d bars, found %d mismatches\n", len(s.Close)-warmup, mismatches)

	if mismatches > 0 {
		t.Fatalf("RSI threshold test FAILED: %d mismatches found", mismatches)
	}

	fmt.Println("✓ PASS: RSI threshold test PASSED")
}

// TestDeterministicRising tests rising/falling operator bytecode evaluation
func TestDeterministicRising(t *testing.T) {
	fmt.Println("\n=== DETERMINISTIC TEST: Rising Operator ===")

	// Load data
	s, f := loadTestData()
	if len(s.Close) == 0 {
		t.Skip("No test data available")
	}

	const warmup = 200

	// Check if we have EMA20 feature
	ema20Idx, hasEMA20 := f.Index["EMA20"]

	if !hasEMA20 {
		t.Skip("EMA20 feature not found")
	}

	// Create rising strategy: EMA20 rising over 5 bars
	st := buildRisingStrategy("EMA20", 5, LeafRising, f)

	// Compare manual vs bytecode for all indices after warmup
	mismatches := 0
	maxMismatchesToPrint := 10
	lookback := 5

	for i := warmup + lookback; i < len(s.Close); i++ {
		ema20 := f.F[ema20Idx]
		manualResult := ema20[i] > ema20[i-lookback]

		// Compare to bytecode
		bytecodeResult := evaluateCompiled(st.EntryCompiled.Code, f.F, i)

		if manualResult != bytecodeResult {
			mismatches++
			if mismatches <= maxMismatchesToPrint {
				t.Logf("RISING MISMATCH at i=%d: manual=%v, bytecode=%v", i, manualResult, bytecodeResult)
				t.Logf("  ema20[i]=%.2f ema20[i-%d]=%.2f", ema20[i], lookback, ema20[i-lookback])
			}
		}
	}

	fmt.Printf("Checked %d bars, found %d mismatches\n", len(s.Close)-warmup-lookback, mismatches)

	if mismatches > 0 {
		t.Fatalf("Rising operator test FAILED: %d mismatches found", mismatches)
	}

	fmt.Println("✓ PASS: Rising operator test PASSED")
}

// Helper: loadTestData loads test data for deterministic tests
func loadTestData() (Series, Features) {
	// Try to load from default path
	s, err := LoadBinanceKlinesCSV("btc_5min_data.csv")
	if err != nil {
		fmt.Printf("Warning: Could not load test data: %v\n", err)
		return Series{}, Features{}
	}

	// Compute features
	f := computeAllFeatures(s)

	return s, f
}

// Helper: createTestEMAStrategy creates a simple EMA crossover strategy for testing
// This is a simplified version that avoids conflicts with main.go's buildManualEMA20x50
func createTestEMAStrategy(f Features, feeBps, slipBps float32) Strategy {
	// Check if EMA20 and EMA50 exist
	_, hasEMA20 := f.Index["EMA20"]
	_, hasEMA50 := f.Index["EMA50"]

	if !hasEMA20 || !hasEMA50 {
		// Return a minimal valid strategy if features don't exist
		return Strategy{
			Seed:        12345,
			FeeBps:      feeBps,
			SlippageBps: slipBps,
			RiskPct:     0.01,
			Direction:   1,
			EntryRule:   RuleTree{Root: nil},
			EntryCompiled: compileRuleTree(nil),
			ExitRule:    RuleTree{Root: nil},
			ExitCompiled: compileRuleTree(nil),
			StopLoss: StopModel{Kind: "fixed", Value: 2.0},
			TakeProfit: TPModel{Kind: "fixed", Value: 4.0},
			Trail:       TrailModel{Kind: "none", Active: false},
			RegimeFilter: RuleTree{Root: nil},
			RegimeCompiled: compileRuleTree(nil),
			MaxHoldBars:     150,
			MaxConsecLosses: 20,
			CooldownBars:    200,
		}
	}

	// Create entry rule: CrossUp(EMA20, EMA50)
	entryNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafCrossUp,
			A:    f.Index["EMA20"],
			B:    f.Index["EMA50"],
		},
	}

	// Create exit rule: CrossDown(EMA20, EMA50)
	exitNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafCrossDown,
			A:    f.Index["EMA20"],
			B:    f.Index["EMA50"],
		},
	}

	return Strategy{
		Seed:        12345,
		FeeBps:      feeBps,
		SlippageBps: slipBps,
		RiskPct:     0.01,
		Direction:   1, // Long
		EntryRule:   RuleTree{Root: entryNode},
		EntryCompiled: compileRuleTree(entryNode),
		ExitRule:    RuleTree{Root: exitNode},
		ExitCompiled: compileRuleTree(exitNode),
		StopLoss: StopModel{
			Kind:    "atr",
			ATRMult: 2.0,
		},
		TakeProfit: TPModel{
			Kind:    "atr",
			ATRMult: 4.0,
		},
		Trail: TrailModel{
			Kind:    "none",
			ATRMult: 0,
			Active:  false,
		},
		RegimeFilter:    RuleTree{Root: nil},
		RegimeCompiled:  compileRuleTree(nil),
		MaxHoldBars:     150,
		MaxConsecLosses: 20,
		CooldownBars:    200,
	}
}

// Helper: buildFixedThresholdStrategy creates a threshold strategy comparing against a fixed value
func buildFixedThresholdStrategy(featureA string, threshold float32, kind LeafKind, f Features) Strategy {
	aIdx, okA := f.Index[featureA]
	if !okA {
		// Return minimal strategy if feature not found
		return Strategy{Seed: 0, FeeBps: 0, SlippageBps: 0}
	}

	entryNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: kind,
			A:    aIdx,
			X:    threshold, // Fixed threshold value
			B:    0,         // Unused for GT/LT
		},
	}

	return Strategy{
		Seed:        12346,
		FeeBps:      30,
		SlippageBps: 8,
		RiskPct:     0.01,
		Direction:   1,
		EntryRule:   RuleTree{Root: entryNode},
		EntryCompiled: compileRuleTree(entryNode),
		ExitRule:    RuleTree{Root: nil},
		ExitCompiled: compileRuleTree(nil),
		StopLoss: StopModel{
			Kind:    "fixed",
			Value:   2.0,
		},
		TakeProfit: TPModel{
			Kind:  "fixed",
			Value: 4.0,
		},
		Trail:           TrailModel{Kind: "none", Active: false},
		RegimeFilter:    RuleTree{Root: nil},
		RegimeCompiled:  compileRuleTree(nil),
		MaxHoldBars:     150,
		MaxConsecLosses: 20,
		CooldownBars:    200,
	}
}

// Helper: buildRSIStrategy creates an RSI threshold strategy
func buildRSIStrategy(period int, threshold float32, kind LeafKind, f Features) Strategy {
	rsiName := fmt.Sprintf("RSI%d", period)

	entryNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: kind,
			A:    f.Index[rsiName],
			X:    threshold,
		},
	}

	return Strategy{
		Seed:        12347,
		FeeBps:      30,
		SlippageBps: 8,
		RiskPct:     0.01,
		Direction:   1,
		EntryRule:   RuleTree{Root: entryNode},
		EntryCompiled: compileRuleTree(entryNode),
		ExitRule:    RuleTree{Root: nil},
		ExitCompiled: compileRuleTree(nil),
		StopLoss: StopModel{
			Kind:    "fixed",
			Value:   1.5,
		},
		TakeProfit: TPModel{
			Kind:  "fixed",
			Value: 3.0,
		},
		Trail:           TrailModel{Kind: "none", Active: false},
		RegimeFilter:    RuleTree{Root: nil},
		RegimeCompiled:  compileRuleTree(nil),
		MaxHoldBars:     100,
		MaxConsecLosses: 20,
		CooldownBars:    200,
	}
}

// Helper: buildRisingStrategy creates a rising/falling strategy
func buildRisingStrategy(feature string, lookback int, kind LeafKind, f Features) Strategy {
	entryNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind:     kind,
			A:        f.Index[feature],
			Lookback: lookback,
		},
	}

	return Strategy{
		Seed:        12348,
		FeeBps:      30,
		SlippageBps: 8,
		RiskPct:     0.01,
		Direction:   1,
		EntryRule:   RuleTree{Root: entryNode},
		EntryCompiled: compileRuleTree(entryNode),
		ExitRule:    RuleTree{Root: nil},
		ExitCompiled: compileRuleTree(nil),
		StopLoss: StopModel{
			Kind:    "fixed",
			Value:   2.0,
		},
		TakeProfit: TPModel{
			Kind:  "fixed",
			Value: 4.0,
		},
		Trail:           TrailModel{Kind: "none", Active: false},
		RegimeFilter:    RuleTree{Root: nil},
		RegimeCompiled:  compileRuleTree(nil),
		MaxHoldBars:     150,
		MaxConsecLosses: 20,
		CooldownBars:    200,
	}
}
