package logx

import (
	"fmt"
	"strings"
	"time"
)

// LogProgress - single line progress log
// tested: total strategies tested
// rate: strategies per second
// bestValScore: best validation score seen (global)
// elites: number of elites in HOF
// rejSeen: percentage rejected by seen/dedup (unused in simplified output)
// rejNovelty: percentage rejected by novelty pressure (unused in simplified output)
// rejSur: percentage rejected by surrogate (unused in simplified output)
// gen: generation/batch count
func LogProgress(tested int64, rate float64, bestValScore float64, elites int, _, _, _ float64, gen int64) {
	fmt.Printf("Tested: %s | Rate: %.0f/s | Best: %.4f | Elites: %d | Gen: %s\n",
		formatNumber(int(tested)), rate, bestValScore, elites, formatNumber(int(gen)))
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

// LogGenTypes - generation type statistics (simplified single line)
func LogGenTypes(immigrant, heavyMut, cross, normalMut int64) {
	total := immigrant + heavyMut + cross + normalMut
	if total == 0 {
		return
	}
	immPct := 100.0 * float64(immigrant) / float64(total)

	// Show total with dominant type (typically immigrant)
	genType := ""
	if immPct > 90 {
		genType = fmt.Sprintf(" (imm: %s%%", formatNumber(int(immPct)))
	} else if heavyMut > 0 {
		genType = " (heavy_mut)"
	} else if cross > 0 {
		genType = " (crossover)"
	} else if normalMut > 0 {
		genType = " (mut)"
	}
	if genType != "" {
		genType += ")"
	}

	fmt.Printf("Gen: %s%s\n", formatNumber(int(total)), genType)
}

// LogBatchProgress - batch completion progress
func LogBatchProgress(batchID int64, tested uint64, trainScore, valScore float32, trainReturn, valReturn, winRate float32,
	trades int, rate float64, elapsed time.Duration, fingerprint string) {
	fmt.Printf("%s  %s  Batch %d: Tested %d | Train: %s | batchValScore: %s | Ret: %s | WR: %s | Trds: %d | Rate: %.0f/s | Runtime: %s | fp: %s\n",
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

// LogCheckpoint - checkpoint saved message (simplified)
func LogCheckpoint(path string, batches int64, bestValScore float32, elitesCount int, elapsed time.Duration) {
	fmt.Printf("Checkpoint saved (runtime: %s)\n", formatDuration(elapsed))
}

// LogCheckpointLoad - checkpoint loaded message
func LogCheckpointLoad(path string, batches int64, bestValScore float32, elitesCount int) {
	fmt.Printf("%s  %s  CHECKPOINT loaded: %s (batches=%d, bestValScore=%.4f, elites=%d)\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("PROG"),
		path, batches, bestValScore, elitesCount,
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

// Box formatting helpers for compact display

// BoxHeader creates a top border for a boxed section with title
func BoxHeader(title string, width int) string {
	if width < 20 {
		width = 50
	}
	// Create border like: ┌─ TITLE ─────────────────┐
	padding := width - len(title) - 6
	if padding < 2 {
		padding = 2
	}
	return fmt.Sprintf("┌─ %s %s┐\n", C(bold, title), C(gray, strings.Repeat("─", padding)+"─"))
}

// BoxFooter creates a bottom border for a boxed section
func BoxFooter(width int) string {
	if width < 20 {
		width = 50
	}
	return C(gray, "└"+strings.Repeat("─", width-2)+"┘") + "\n"
}

// BoxRow creates a content row for a boxed section (auto-pads to width)
func BoxRow(content string, width int) string {
	if width < 20 {
		width = 50
	}
	padding := width - len(content) - 4 // -4 for "│ " and " │"
	if padding < 0 {
		padding = 0
	}
	return fmt.Sprintf("│ %s%s │\n", content, C(gray, strings.Repeat(" ", padding)))
}

// formatNumber formats a number with thousands separators (e.g., 12,345)
func formatNumber(n int) string {
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	var result []string
	for i := len(s); i > 0; i -= 3 {
		start := i - 3
		if start < 0 {
			start = 0
		}
		result = append([]string{s[start:i]}, result...)
	}
	return strings.Join(result, ",")
}

// FormatNumberSimple formats a number with thousands separators (exported version)
func FormatNumberSimple(n int) string {
	return formatNumber(n)
}

// MetricsSnapshot holds comprehensive metrics for dashboard display
type MetricsSnapshot struct {
	// Performance metrics
	BestScore      float32
	BestReturn     float32
	BestWinRate    float32
	BestProfitFact float32

	// Progress metrics
	Tested       int64
	Rate         float64
	Elites       int
	Generation   int64

	// Rejection metrics (last N strategies)
	RejSeen      float64
	RejNovelty   float64
	RejSur       float64

	// Runtime
	Elapsed      time.Duration
}

// LogMetricsDashboard - comprehensive metrics summary dashboard
// Displays key performance, progress, and rejection metrics in a scannable format
func LogMetricsDashboard(m MetricsSnapshot) {
	fmt.Printf("\n%s  %s\n", C(gray, time.Now().UTC().Format("15:04:05Z")), Channel("PROG"))
	fmt.Printf("%s", BoxHeader("DASHBOARD", 54))

	// PERFORMANCE row
	fmt.Printf("│ %s PERFORMANCE %s\n",
		C(bold, "│"), C(bold, "                             │"))
	fmt.Printf("│ Score: %s │ Ret: %s │ WR: %s │ PF: %.2f │\n",
		ScoreColor(m.BestScore),
		ReturnColor(m.BestReturn),
		WinRateColor(m.BestWinRate),
		m.BestProfitFact,
	)

	// PROGRESS row
	fmt.Printf("│ Tested: %s │ Rate: %d/s │ Elites: %d │ Gen: %s │\n",
		formatNumber(int(m.Tested)),
		int(m.Rate),
		m.Elites,
		formatNumber(int(m.Generation)),
	)

	// REJECTIONS row
	totalRej := m.RejSeen + m.RejNovelty + m.RejSur
	fmt.Printf("│ %s Rej: %s │ Seen: %s │ Nov: %s │ Sur: %s │\n",
		Icon("reject"),
		ColorPercent(totalRej),
		ColorPercent(m.RejSeen),
		ColorPercent(m.RejNovelty),
		ColorPercent(m.RejSur),
	)

	// RUNTIME row
	fmt.Printf("│ Runtime: %s │\n",
		C(gray, FormatDuration(m.Elapsed)),
	)

	fmt.Printf("%s\n", BoxFooter(54))
}
