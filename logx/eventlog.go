package logx

import (
	"fmt"
	"time"

	"hb_bactest_checker/tui"
)

// Convenience functions that forward to TUI

func LogEliteAdded(score float32, elitesCount int) {
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "ELITE",
		Severity:  "info",
		Message:   fmt.Sprintf("New elite added (score=%.4f, total=%d)", score, elitesCount),
	}
	tui.PushEvent(event)
}

func LogNewBestScore(oldScore, newScore float32) {
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "BEST",
		Severity:  "info",
		Message:   fmt.Sprintf("Best score improved: %.4f → %.4f", oldScore, newScore),
	}
	tui.PushEvent(event)
}

func LogGateLoosened(paramName string, oldVal, newVal float32) {
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "GATE",
		Severity:  "warning",
		Message:   fmt.Sprintf("%s loosened: %.4f → %.4f", paramName, oldVal, newVal),
	}
	tui.PushEvent(event)
}

func LogStagnation(batchesNoImprove int64) {
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "STAGNATION",
		Severity:  "warning",
		Message:   fmt.Sprintf("No improvement for %d batches", batchesNoImprove),
	}
	tui.PushEvent(event)
}

func LogRecoveryMode(enabled bool) {
	msg := "Recovery mode DISABLED (normal gates)"
	severity := "info"
	if enabled {
		msg = "Recovery mode ENABLED (relaxed gates)"
		severity = "warning"
	}
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "RECOVERY",
		Severity:  severity,
		Message:   msg,
	}
	tui.PushEvent(event)
}

func LogBugWarning(message string) {
	event := tui.Event{
		Timestamp: time.Now(),
		Type:      "BUG",
		Severity:  "error",
		Message:   message,
	}
	tui.PushEvent(event)
}
