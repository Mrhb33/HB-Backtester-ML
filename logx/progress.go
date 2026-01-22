package logx

import (
	"fmt"
	"time"
)

// LogProgress - single-line progress log
// tested: total strategies tested
// rate: strategies per second
// bestVal: best validation score seen
// elites: number of elites in HOF
// rejSeen: percentage rejected by seen/dedup
// rejNovelty: percentage rejected by novelty pressure (too similar)
// rejSur: percentage rejected by surrogate
// gen: generation/batch count
func LogProgress(tested int64, rate float64, bestVal float64, elites int, rejSeen float64, rejNovelty float64, rejSur float64, gen int64) {
	fmt.Printf("%s  %s  tested=%d  rate=%.1f/s  bestVal=%.4f  elites=%d  rej_seen=%s  rej_novelty=%s  rej_sur=%s  gen=%d\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		tested, rate, bestVal, elites,
		ColorPercent(rejSeen), ColorPercent(rejNovelty), ColorPercent(rejSur),
		gen,
	)
}

// ColorPercent returns a color-coded percentage string
// Low (<10%) is green, medium (10-30%) is yellow, high (>30%) is red
func ColorPercent(pct float64) string {
	if pct < 10 {
		return Success(fmt.Sprintf("%.1f%%", pct))
	}
	if pct < 30 {
		return Warn(fmt.Sprintf("%.1f%%", pct))
	}
	return Error(fmt.Sprintf("%.1f%%", pct))
}

// LogGenerator - generator stats
func LogGenerator(generated, rejectedSur, rejectedSeen, sentToJobs int64) {
	total := generated
	if total == 0 {
		total = 1
	}
	surPct := 100.0 * float64(rejectedSur) / float64(total)
	seenPct := 100.0 * float64(rejectedSeen) / float64(total)
	sentPct := 100.0 * float64(sentToJobs) / float64(total)

	fmt.Printf("%s  %s  gen=%d  rej_sur=%d(%.1f%%)  rej_seen=%d(%.1f%%)  sent=%d(%.1f%%)\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("GEN "),
		generated, rejectedSur, surPct, rejectedSeen, seenPct, sentToJobs, sentPct,
	)
}

// LogGenTypes - generation type statistics
func LogGenTypes(immigrant, heavyMut, cross, normalMut int64) {
	total := immigrant + heavyMut + cross + normalMut
	if total == 0 {
		return
	}
	immPct := 100.0 * float64(immigrant) / float64(total)
	hmPct := 100.0 * float64(heavyMut) / float64(total)
	crPct := 100.0 * float64(cross) / float64(total)
	nmPct := 100.0 * float64(normalMut) / float64(total)

	fmt.Printf("%s  %s  immigrant=%d(%.1f%%)  heavyMut=%d(%.1f%%)  crossover=%d(%.1f%%)  normalMut=%d(%.1f%%)  total=%d\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("GEN "),
		immigrant, immPct, heavyMut, hmPct, cross, crPct, normalMut, nmPct, total,
	)
}

// LogBatchProgress - batch completion progress
func LogBatchProgress(batchID int64, tested uint64, trainScore, valScore float32, trainReturn, valReturn, winRate float32,
	trades int, rate float64, elapsed time.Duration, fingerprint string) {
	fmt.Printf("%s  %s  Batch %d: Tested %d | Train: %s | Val: %s | Ret: %s | WR: %s | Trds: %d | Rate: %.0f/s | Runtime: %s | fp: %s\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		batchID, tested,
		ScoreColor(trainScore), ScoreColor(valScore),
		ReturnColor(valReturn), WinRateColor(winRate),
		trades, rate, formatDuration(elapsed), fingerprint,
	)
}

// formatDuration formats a duration in a human-readable way
// Shows hours, minutes, and seconds (e.g., "1h23m45s" or "45m32s" or "23s")
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	hours := int(d.Hours())
	minutes := int(d.Minutes()) % 60
	return fmt.Sprintf("%dh%dm", hours, minutes)
}

// LogCheckpoint - checkpoint saved message
func LogCheckpoint(path string, batches int64, bestVal float32, elitesCount int, elapsed time.Duration) {
	fmt.Printf("%s  %s  CHECKPOINT saved: %s (batches=%d, bestVal=%.4f, elites=%d, runtime=%s)\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		path, batches, bestVal, elitesCount, formatDuration(elapsed),
	)
}

// LogCheckpointLoad - checkpoint loaded message
func LogCheckpointLoad(path string, batches int64, bestVal float32, elitesCount int) {
	fmt.Printf("%s  %s  CHECKPOINT loaded: %s (batches=%d, bestVal=%.4f, elites=%d)\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		path, batches, bestVal, elitesCount,
	)
}

// LogScreenRelaxChange - screen relax level change
func LogScreenRelaxChange(oldLevel, newLevel int) {
	fmt.Printf("%s  %s  SCREEN relax level: %d -> %d\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		oldLevel, newLevel,
	)
}

// LogMetaStagnate - meta controller stagnation adjustment
func LogMetaStagnate(batchesNoImprove int64, oldRadical, newRadical, oldSurExplore, newSurExplore float32, passRate float32) {
	fmt.Printf("%s  %s  META-STAGNATE(%d): radicalP %.2f->%.2f, surExploreP %.2f->%.2f, passRate=%.2f%%\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		batchesNoImprove, oldRadical, newRadical, oldSurExplore, newSurExplore, passRate*100,
	)
}

// LogBootstrapComplete - bootstrap mode complete message
func LogBootstrapComplete(elitesCount int) {
	fmt.Printf("%s  %s  BOOTSTRAP COMPLETE: Elites=%d, exiting bootstrap mode\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		elitesCount,
	)
}
