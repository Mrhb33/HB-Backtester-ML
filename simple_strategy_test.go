package main

import (
	"fmt"
	"testing"
)

// TestSimpleStrategyEMACross tests the EMA crossover strategy
// Run with: go test -v -run TestSimpleStrategyEMACross
func TestSimpleStrategyEMACross(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Simple Strategy Backtest - EMA Cross")
	fmt.Println("========================================")

	// Load data (using the same data as the full backtest)
	dataPath := "btc_5min_data.csv"
	series, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d candles from %s\n\n", series.T, dataPath)

	// Compute features (using the same indicators as the full backtest)
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.Names))
	fmt.Printf("Features: %v\n\n", feats.Names)

	// Create strategy using same parameters as backtest
	// EMA20 crosses above EMA50, long only
	strategy := createEMACrossStrategy(20, 50, 30, 8)
	fmt.Printf("Strategy: %s\n", strategy.Name)
	fmt.Printf("Entry: EMA20 crosses above EMA50\n")
	fmt.Printf("Exit: EMA20 crosses below EMA50\n")
	fmt.Printf("Stop Loss: %.2f%%, Take Profit: %.2f%%\n\n", strategy.StopLossPct, strategy.TakeProfitPct)

	// Run strategy and output CSV
	csvPath := "ema20x50_simple_states.csv"
	trades, err := runSimpleStrategy(series, feats, strategy, csvPath)
	if err != nil {
		t.Fatalf("Failed to run strategy: %v", err)
	}

	fmt.Printf("CSV states written to: %s\n\n", csvPath)

	// Print results
	printSimpleStrategyResults(trades, strategy)
}

// TestSimpleStrategyRSI tests the RSI oversold strategy
// Run with: go test -v -run TestSimpleStrategyRSI
func TestSimpleStrategyRSI(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Simple Strategy Backtest - RSI + EMA")
	fmt.Println("========================================")

	dataPath := "btc_5min_data.csv"
	series, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d candles from %s\n\n", series.T, dataPath)

	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n\n", len(feats.Names))

	// RSI < 30 (oversold) + EMA20 rising for 5 bars
	strategy := createRSIEMAStrategy(14, 30, 20, 30, 8)
	fmt.Printf("Strategy: %s\n", strategy.Name)
	fmt.Printf("Entry: RSI14 < 30 AND EMA20 rising (5 bars)\n")
	fmt.Printf("Exit: RSI14 > 70\n")

	csvPath := "rsi14_ema20_simple_states.csv"
	trades, err := runSimpleStrategy(series, feats, strategy, csvPath)
	if err != nil {
		t.Fatalf("Failed to run strategy: %v", err)
	}

	fmt.Printf("CSV states written to: %s\n\n", csvPath)

	printSimpleStrategyResults(trades, strategy)
}

// TestSimpleStrategyMACD tests the MACD crossover strategy
// Run with: go test -v -run TestSimpleStrategyMACD
func TestSimpleStrategyMACD(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Simple Strategy Backtest - MACD Cross")
	fmt.Println("========================================")

	dataPath := "btc_5min_data.csv"
	series, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d candles from %s\n\n", series.T, dataPath)

	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n\n", len(feats.Names))

	// MACD crosses above signal line
	strategy := createMACDCrossStrategy(30, 8)
	fmt.Printf("Strategy: %s\n", strategy.Name)
	fmt.Printf("Entry: MACD crosses above MACD_Signal\n")
	fmt.Printf("Exit: MACD crosses below MACD_Signal\n\n")

	csvPath := "macd_simple_states.csv"
	trades, err := runSimpleStrategy(series, feats, strategy, csvPath)
	if err != nil {
		t.Fatalf("Failed to run strategy: %v", err)
	}

	fmt.Printf("CSV states written to: %s\n\n", csvPath)

	printSimpleStrategyResults(trades, strategy)
}

// TestListAllFeatures lists all available features with their indices
// Run with: go test -v -run TestListAllFeatures
func TestListAllFeatures(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Available Features")
	fmt.Println("========================================")

	dataPath := "btc_5min_data.csv"
	series, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}

	feats := computeAllFeatures(series)

	fmt.Println("Index | Feature Name")
	fmt.Println("------|-------------")
	for i, name := range feats.Names {
		fmt.Printf("%5d | %s\n", i, name)
	}
	fmt.Printf("\nTotal: %d features\n", len(feats.Names))
}

// Example custom strategy: RSI oversold with price below EMA50
// Run with: go test -v -run TestCustomStrategy
func TestCustomStrategy(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("Custom Strategy - RSI Oversold + Price")
	fmt.Println("========================================")

	dataPath := "btc_5min_data.csv"
	series, err := LoadBinanceKlinesCSV(dataPath)
	if err != nil {
		t.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d candles from %s\n\n", series.T, dataPath)

	feats := computeAllFeatures(series)

	// Custom strategy: RSI14 < 25 (very oversold) AND Price < EMA50 (trend following value)
	strategy := SimpleStrategy{
		Name:         "RSI14_Oversold_Price_Below_EMA50",
		Direction:    1, // Long
		FeeBps:       30,
		SlippageBps:  8,
		StopLossPct:  2.0,
		TakeProfitPct: 6.0, // 3:1 reward:risk
		MaxHoldBars:  100,
		EntryRule: SimpleRule{
			Type: "and",
			LHS: &SimpleRule{
				Type: "lt",
				A:    "RSI14",
				X:    25, // Very oversold
			},
			RHS: &SimpleRule{
				Type: "lt",
				A:    "Close",
				X:    0, // Placeholder - will be compared to EMA50
			},
		},
		ExitRule: SimpleRule{
			Type: "gt",
			A:    "RSI14",
			X:    65, // Exit when RSI recovers
		},
	}

	// Note: For comparing Close to EMA50, we'd need to add special handling
	// For now, let's use a simpler entry: RSI14 < 25
	strategy.EntryRule = SimpleRule{
		Type: "lt",
		A:    "RSI14",
		X:    25,
	}

	fmt.Printf("Strategy: %s\n", strategy.Name)
	fmt.Printf("Entry: RSI14 < 25 (very oversold)\n")
	fmt.Printf("Exit: RSI14 > 65\n")
	fmt.Printf("Stop Loss: %.2f%%, Take Profit: %.2f%%\n\n", strategy.StopLossPct, strategy.TakeProfitPct)

	csvPath := "custom_rsi25_simple_states.csv"
	trades, err := runSimpleStrategy(series, feats, strategy, csvPath)
	if err != nil {
		t.Fatalf("Failed to run strategy: %v", err)
	}

	fmt.Printf("CSV states written to: %s\n\n", csvPath)

	printSimpleStrategyResults(trades, strategy)
}
