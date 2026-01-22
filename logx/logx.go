package logx

import (
	"fmt"
	"os"
	"time"

	"golang.org/x/term"
)

const (
	reset   = "\x1b[0m"
	bold    = "\x1b[1m"
	gray    = "\x1b[90m"
	cyan    = "\x1b[36m"
	blue    = "\x1b[34m"
	yellow  = "\x1b[33m"
	green   = "\x1b[32m"
	magenta = "\x1b[35m"
	red     = "\x1b[31m"
	white   = "\x1b[37m"
)

var enableColor = true

func init() {
	// Disable color if NO_COLOR is set or stdout is not a terminal
	if os.Getenv("NO_COLOR") != "" {
		enableColor = false
	}
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		enableColor = false
	}
}

// C returns a color-coded string (or plain string if color disabled)
func C(color, s string) string {
	if !enableColor {
		return s
	}
	return color + s + reset
}

// Cf returns a color-coded formatted string
func Cf(color, format string, args ...any) string {
	return C(color, fmt.Sprintf(format, args...))
}

// Channel returns a consistently-padded colored channel tag
// All channels are 6 chars: [PROG] [GEN ] [SIG ] [ENT ] [EXT ] [VAL ]
// IMPORTANT: Pass 4-char channel names: "PROG", "GEN ", "SIG ", "ENT ", "EXT ", "VAL "
// (Note: GEN, SIG, ENT, EXT, VAL have trailing space for padding)
func Channel(ch string) string {
	// Map by channel name (ch), then build padded label
	color := map[string]string{
		"PROG": cyan,
		"GEN ":  blue,
		"SIG ":  yellow,
		"ENT ":  green,
		"EXT ":  magenta,
		"VAL ":  red,
	}[ch]

	// Create padded label [XXXX] - left-justify in 4-char width
	label := fmt.Sprintf("[%-4s]", ch)
	return C(color, label)
}

// TS returns a gray UTC timestamp (caller controls time value)
func TS(ts string) string {
	return C(gray, ts)
}

// Colored output helpers for common use cases

// Success returns a green success message (for ✓, PASS, etc.)
func Success(s string) string {
	return C(green, s)
}

// Successf returns a formatted green success message
func Successf(format string, args ...any) string {
	return C(green, fmt.Sprintf(format, args...))
}

// Error returns a red error message (for ✗, FAIL, etc.)
func Error(s string) string {
	return C(red, s)
}

// Errorf returns a formatted red error message
func Errorf(format string, args ...any) string {
	return C(red, fmt.Sprintf(format, args...))
}

// Warn returns a yellow warning message (for ⚠, WARN, etc.)
func Warn(s string) string {
	return C(yellow, s)
}

// Warnf returns a formatted yellow warning message
func Warnf(format string, args ...any) string {
	return C(yellow, fmt.Sprintf(format, args...))
}

// Info returns a cyan info message
func Info(s string) string {
	return C(cyan, s)
}

// Infof returns a formatted cyan info message
func Infof(format string, args ...any) string {
	return C(cyan, fmt.Sprintf(format, args...))
}

// Highlight returns a bold highlighted message
func Highlight(s string) string {
	return C(bold, s)
}

// Highlightf returns a formatted bold highlighted message
func Highlightf(format string, args ...any) string {
	return C(bold, fmt.Sprintf(format, args...))
}

// Dim returns a gray dimmed message (for less important info)
func Dim(s string) string {
	return C(gray, s)
}

// Dimf returns a formatted gray dimmed message
func Dimf(format string, args ...any) string {
	return C(gray, fmt.Sprintf(format, args...))
}

// Checkmark returns a colored checkmark (green) or X (red)
func Checkmark(passed bool) string {
	if passed {
		return Success("✓")
	}
	return Error("✗")
}

// ScoreColor returns color-coded score based on value
// Positive scores are green, negative are red
func ScoreColor(score float32) string {
	if score > 0 {
		return Success(fmt.Sprintf("%.4f", score))
	}
	return Error(fmt.Sprintf("%.4f", score))
}

// ReturnColor returns color-coded return based on value
// Positive returns are green, negative are red
func ReturnColor(ret float32) string {
	if ret > 0 {
		return Success(fmt.Sprintf("%.2f%%", ret*100))
	}
	return Error(fmt.Sprintf("%.2f%%", ret*100))
}

// DDColor returns color-coded drawdown based on severity
// Low DD (<0.10) is green, medium (<0.20) is yellow, high is red
func DDColor(dd float32) string {
	if dd < 0.10 {
		return Success(fmt.Sprintf("%.2f%%", dd*100))
	}
	if dd < 0.20 {
		return Warn(fmt.Sprintf("%.2f%%", dd*100))
	}
	return Error(fmt.Sprintf("%.2f%%", dd*100))
}

// WinRateColor returns color-coded win rate
// High win rate (>0.50) is green, medium (>0.40) is yellow, low is red
func WinRateColor(wr float32) string {
	if wr > 0.50 {
		return Success(fmt.Sprintf("%.1f%%", wr*100))
	}
	if wr > 0.40 {
		return Warn(fmt.Sprintf("%.1f%%", wr*100))
	}
	return Error(fmt.Sprintf("%.1f%%", wr*100))
}

// FormatDuration formats a duration in a human-readable way
// Shows hours, minutes, and seconds (e.g., "1h23m" or "45m" or "23s")
func FormatDuration(d time.Duration) string {
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
