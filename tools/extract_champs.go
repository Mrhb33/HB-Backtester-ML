package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"
)

type TestResult struct {
	Seed         uint64  `json:"seed"`
	FeeBps       float32 `json:"fee_bps"`
	SlippageBps  float32 `json:"slippage_bps"`
	Direction    int     `json:"direction"`
	StopLoss     string  `json:"stop_loss"`
	TakeProfit   string  `json:"take_profit"`
	Trail        string  `json:"trail"`
	EntryRule    string  `json:"entry_rule"`
	ExitRule     string  `json:"exit_rule"`
	RegimeFilter string  `json:"regime_filter"`
	TrainScore   float32 `json:"train_score"`
	TrainReturn  float32 `json:"train_return"`
	TrainMaxDD   float32 `json:"train_max_dd"`
	TrainWinRate float32 `json:"train_win_rate"`
	TrainTrades  int     `json:"train_trades"`
	ValScore     float32 `json:"val_score"`
	ValReturn    float32 `json:"val_return"`
	ValMaxDD     float32 `json:"val_max_dd"`
	ValWinRate   float32 `json:"val_win_rate"`
	ValTrades    int     `json:"val_trades"`
	TestScore    float32 `json:"test_score"`
	TestReturn   float32 `json:"test_return"`
	TestMaxDD    float32 `json:"test_max_dd"`
	TestWinRate  float32 `json:"test_win_rate"`
	TestTrades   int     `json:"test_trades"`
}

func main() {
	data, err := ioutil.ReadFile("winners_tested.jsonl")
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	lines := strings.Split(string(data), "\n")

	var results []TestResult
	var validLines []string

	for i, line := range lines {
		if line == "" {
			continue
		}
		var r TestResult
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			fmt.Printf("Error parsing line %d: %v\n", i, err)
			continue
		}
		results = append(results, r)
		validLines = append(validLines, line)
	}

	// Sort by test_score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].TestScore > results[j].TestScore
	})

	// Create champions.jsonl with top 5
	f, err := os.Create("champions.jsonl")
	if err != nil {
		fmt.Printf("Error creating champions.jsonl: %v\n", err)
		return
	}
	defer f.Close()

	// Write top 5 original lines
	for i := 0; i < 5 && i < len(results); i++ {
		// Find the original line that corresponds to this result
		for j, line := range validLines {
			var r TestResult
			json.Unmarshal([]byte(line), &r)
			if r.TestScore == results[i].TestScore &&
				r.TestReturn == results[i].TestReturn &&
				r.TestTrades == results[i].TestTrades {
				f.WriteString(line + "\n")
				// Remove from validLines to avoid duplicates
				validLines = append(validLines[:j], validLines[j+1:]...)
				break
			}
		}
	}

	fmt.Printf("Successfully wrote top %d strategies to champions.jsonl\n", min(5, len(results)))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
