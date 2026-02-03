package main

import (
	"fmt"
	"time"
)

// SendProgressUpdate broadcasts progress to web dashboard
func SendProgressUpdate(start time.Time, generated, tested, elites int64, rejectedSur, rejectedSeen, rejectedNovelty int64) {
	elapsed := time.Since(start)
	timeElapsed := formatDuration(elapsed)

	var stratPerSec float64
	if elapsed > 0 {
		stratPerSec = float64(tested) / elapsed.Seconds()
	}

	SendProgress(generated, tested, rejectedSur, rejectedSeen, rejectedNovelty, int(elites), timeElapsed, stratPerSec)

	// Also print to terminal
	fmt.Printf("[PROGRESS] Gen=%d Tested=%d Elites=%d RejSur=%d RejSeen=%d RejNov=%d Time=%s %.1f strat/sec\n",
		generated, tested, elites, rejectedSur, rejectedSeen, rejectedNovelty, timeElapsed, stratPerSec)
}

// SendMetricsUpdate broadcasts backtest metrics to web dashboard
func SendMetricsUpdate(result Result) {
	SendMetrics(
		result.Score,
		result.Return,
		result.MaxDD,
		result.WinRate,
		result.Expectancy,
		result.ProfitFactor,
		result.Trades,
		0, // totalHoldBars not in Result
	)

	// Also print to terminal
	fmt.Printf("[METRICS] Score=%s Return=%.2f%% MaxDD=%.2f%% WR=%.1f%% Exp=%.4f PF=%.2f Trades=%d\n",
		CleanScoreForDisplay(result.Score), result.Return*100, result.MaxDD*100, result.WinRate*100,
		result.Expectancy, result.ProfitFactor, result.Trades)
}

// SendOOSUpdate broadcasts OOS stats to web dashboard
func SendOOSUpdate(stats OOSStats) {
	SendOOSStats(stats)

	// Also print to terminal
	if stats.Rejected {
		fmt.Printf("[OOS] REJECTED: %s (code=%d)\n", stats.RejectReason, stats.RejectCode)
	} else {
		fmt.Printf("[OOS] PASS: GeoAvg=%.2f%%, Median=%.2f%%, Months=%d, Trades=%d\n",
			stats.GeoAvgMonthly*100, stats.MedianMonthly*100, stats.TotalMonths, stats.TotalTrades)
	}

	// Send monthly returns
	if len(stats.MonthlyReturns) > 0 {
		SendMonthlyReturns(stats.MonthlyReturns)
	}
}

// SendEliteUpdate broadcasts elite discovery to web dashboard
func SendEliteUpdate(rank int, elite Elite) {
	trainScore := elite.Train.Score
	valScore := elite.ValScore

	data := EliteLogData{
		Rank:      rank,
		Score:      trainScore,
		ValScore:  valScore,
		Return:    elite.Train.Return,
		MaxDD:     elite.Train.MaxDD,
		WinRate:   elite.Train.WinRate,
		Trades:    elite.Train.Trades,
		IsPreElite: elite.IsPreElite,
		Timestamp:  time.Now().Format("2006-01-02 15:04:05"),
	}

	Broadcast(MsgTypeEliteLog, data)

	// Also print to terminal
	fmt.Printf("[ELITE] Rank=%d TrainScore=%s ValScore=%s Ret=%.2f%% DD=%.2f%% WR=%.1f%% Trades=%d%s\n",
		rank, CleanScoreForDisplay(trainScore), CleanScoreForDisplay(valScore), elite.Train.Return*100, elite.Train.MaxDD*100,
		elite.Train.WinRate*100, elite.Train.Trades,
		func() string {
			if elite.IsPreElite {
				return " (PRE-ELITE)"
			}
			return ""
		}(),
	)
}

// SendHallOfFameUpdate broadcasts Hall of Fame state
func SendHallOfFameUpdate(hof *HallOfFame) {
	snapshot := hof.snapshot.Load()
	if snapshot == nil {
		return
	}

	elites := snapshot.([]Elite)
	eliteData := make([]EliteLogData, len(elites))
	for i, elite := range elites {
		trainScore := elite.Train.Score
		valScore := elite.ValScore

		eliteData[i] = EliteLogData{
			Rank:      i + 1,
			Score:      trainScore,
			ValScore:  valScore,
			Return:    elite.Train.Return,
			MaxDD:     elite.Train.MaxDD,
			WinRate:   elite.Train.WinRate,
			Trades:    elite.Train.Trades,
			IsPreElite: elite.IsPreElite,
			Timestamp:  time.Now().Format("2006-01-02 15:04:05"),
		}
	}

	data := HallOfFameData{
		K:      hof.K,
		Elites: eliteData,
	}

	Broadcast(MsgTypeHallOfFame, data)
}

// SendExitReasonsUpdate broadcasts exit reasons distribution
func SendExitReasonsUpdate(exitReasons map[string]int) {
	Broadcast(MsgTypeExitReasons, exitReasons)
}

// formatDuration formats a duration in a human-readable way
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	if minutes > 0 {
		return fmt.Sprintf("%dh%dm", hours, minutes)
	}
	return fmt.Sprintf("%dh", hours)
}
