package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"time"
)

// SlimElite is a minimal serializable version of Elite for checkpoints
// It stores rule trees as strings instead of compiled bytecode
type SlimElite struct {
	Seed         uint64  `json:"seed"`
	FeeBps       float32 `json:"fee_bps"`
	SlippageBps  float32 `json:"slippage_bps"`
	Direction    int     `json:"direction"`
	RiskPct      float32 `json:"risk_pct"`
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

	ValScore   float32 `json:"val_score"`
	ValReturn  float32 `json:"val_return"`
	ValMaxDD   float32 `json:"val_max_dd"`
	ValWinRate float32 `json:"val_win_rate"`
	ValTrades  int     `json:"val_trades"`
}

type Checkpoint struct {
	Version     int   `json:"version"`
	SavedAtUnix int64 `json:"saved_at_unix"`
	Seed        int64 `json:"seed"`

	PassedCount     int64   `json:"passed_count"`
	ValidatedLabels int64   `json:"validated_labels"`
	BestValSeen     float32 `json:"best_val_seen"`

	HOFElites        []SlimElite `json:"hof_elites"`
	ArchiveElites    []SlimElite `json:"archive_elites"` // Archive elites for diversity preservation
	SeenFingerprints []string    `json:"seen_fingerprints"`
}

// Convert Elite to SlimElite (strip compiled bytecode)
func eliteToSlim(e Elite) SlimElite {
	return SlimElite{
		Seed:         e.Strat.Seed,
		FeeBps:       e.Strat.FeeBps,
		SlippageBps:  e.Strat.SlippageBps,
		Direction:    e.Strat.Direction,
		RiskPct:      e.Strat.RiskPct,
		StopLoss:     stopModelToString(e.Strat.StopLoss),
		TakeProfit:   tpModelToString(e.Strat.TakeProfit),
		Trail:        trailModelToString(e.Strat.Trail),
		EntryRule:    ruleTreeToString(e.Strat.EntryRule.Root),
		ExitRule:     ruleTreeToString(e.Strat.ExitRule.Root),
		RegimeFilter: ruleTreeToString(e.Strat.RegimeFilter.Root),

		TrainScore:   e.Train.Score,
		TrainReturn:  e.Train.Return,
		TrainMaxDD:   e.Train.MaxDD,
		TrainWinRate: e.Train.WinRate,
		TrainTrades:  e.Train.Trades,

		ValScore:   e.Val.Score,
		ValReturn:  e.Val.Return,
		ValMaxDD:   e.Val.MaxDD,
		ValWinRate: e.Val.WinRate,
		ValTrades:  e.Val.Trades,
	}
}

// Convert SlimElite to Elite (recompile rules)
func slimToElite(se SlimElite, rng *rand.Rand) Elite {
	s := Strategy{
		Seed:        se.Seed,
		FeeBps:      se.FeeBps,
		SlippageBps: se.SlippageBps,
		RiskPct:     se.RiskPct,
		Direction:   se.Direction,
		EntryRule: RuleTree{
			Root: parseRuleTree(se.EntryRule),
		},
		ExitRule: RuleTree{
			Root: parseRuleTree(se.ExitRule),
		},
		RegimeFilter: RuleTree{
			Root: parseRuleTree(se.RegimeFilter),
		},
		StopLoss:   parseStopModel(se.StopLoss),
		TakeProfit: parseTPModel(se.TakeProfit),
		Trail:      parseTrailModel(se.Trail),
	}
	// Recompile rules
	s.EntryCompiled = compileRuleTree(s.EntryRule.Root)
	s.ExitCompiled = compileRuleTree(s.ExitRule.Root)
	s.RegimeCompiled = compileRuleTree(s.RegimeFilter.Root)

	return Elite{
		Strat: s,
		Train: Result{
			Score:    se.TrainScore,
			Return:   se.TrainReturn,
			MaxDD:    se.TrainMaxDD,
			WinRate:  se.TrainWinRate,
			Trades:   se.TrainTrades,
			Strategy: s,
		},
		Val: Result{
			Score:    se.ValScore,
			Return:   se.ValReturn,
			MaxDD:    se.ValMaxDD,
			WinRate:  se.ValWinRate,
			Trades:   se.ValTrades,
			Strategy: s,
		},
		ValScore: se.ValScore,
	}
}

func SaveCheckpoint(path string, cp Checkpoint) error {
	cp.Version = 1
	cp.SavedAtUnix = time.Now().Unix()

	tmp := path + ".tmp"
	b, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path) // atomic replace
}

func LoadCheckpoint(path string) (Checkpoint, error) {
	var cp Checkpoint
	b, err := os.ReadFile(path)
	if err != nil {
		return cp, err
	}
	err = json.Unmarshal(b, &cp)
	return cp, err
}
