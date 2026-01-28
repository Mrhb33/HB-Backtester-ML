package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// Styles (defined at package init for reuse)
var (
	// Color styles
	styleGreen  = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	styleYellow = lipgloss.NewStyle().Foreground(lipgloss.Color("226"))
	styleRed    = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	styleCyan   = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
	styleGray   = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	styleDim    = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))

	// Panel styles
	stylePanel = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)

	styleHeader = lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("212")).
		Padding(0, 1)

	styleEventInfo  = lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
	styleEventWarn  = lipgloss.NewStyle().Foreground(lipgloss.Color("226"))
	styleEventError = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
)

// View renders the UI
func (m Model) View() string {
	if !m.ready {
		return "Initializing..."
	}

	// Build panels
	header := m.renderHeader()
	progress := m.renderProgress()
	stats := m.renderStats()
	meta := m.renderMeta()
	rejections := m.renderRejections()
	candidate := m.renderCandidate()
	events := m.renderEvents()
	footer := m.renderFooter()

	// Stack panels vertically
	body := lipgloss.JoinVertical(lipgloss.Left,
		header,
		progress,
		lipgloss.JoinHorizontal(lipgloss.Top, stats, meta),
		rejections,
		candidate,
		events,
		footer,
	)

	return body
}

func (m Model) renderHeader() string {
	runtime := time.Since(m.snapshot.StartTime)
	return styleHeader.Render(fmt.Sprintf(
		"%s │ mode=%s │ data=%s │ runtime=%s",
		m.snapshot.ProjectName,
		m.snapshot.Mode,
		m.snapshot.Dataset,
		FormatDuration(runtime),
	))
}

func (m Model) renderProgress() string {
	// Progress panel removed - simplified output
	return ""
}

func (m Model) renderStats() string {
	passRateColor := m.passRateColor(m.snapshot.PassRate)
	bestScoreColor := m.scoreChangeColor(m.snapshot.BestValScore)

	return stylePanel.Width(50).Render(fmt.Sprintf(
		"Stats: passRate=%s │ elites=%d │ bestValScore=%s",
		passRateColor,
		m.snapshot.ElitesCount,
		bestScoreColor,
	))
}

func (m Model) renderMeta() string {
	return stylePanel.Width(50).Render(fmt.Sprintf(
		"Meta: radicalP=%.2f │ surExploreP=%.2f │ stagnation=%d",
		m.snapshot.RadicalP,
		m.snapshot.SurExploreP,
		m.snapshot.StagnationCnt,
	))
}

func (m Model) renderRejections() string {
	total := m.snapshot.RejectedSeen + m.snapshot.RejectedSur + m.snapshot.RejectedNovelty
	if total == 0 {
		total = 1
	}

	seenPct := 100.0 * float64(m.snapshot.RejectedSeen) / float64(total)
	surPct := 100.0 * float64(m.snapshot.RejectedSur) / float64(total)
	noveltyPct := 100.0 * float64(m.snapshot.RejectedNovelty) / float64(total)

	return stylePanel.Render(fmt.Sprintf(
		"Rejections: seen=%s │ sur=%s │ novelty=%s",
		m.percentColor(seenPct),
		m.percentColor(surPct),
		m.percentColor(noveltyPct),
	))
}

func (m Model) renderCandidate() string {
	c := m.snapshot.CurrentCandidate

	// Check if candidate is stale (> 5 seconds) or never set
	if c.Timestamp.IsZero() || time.Since(c.Timestamp) > 5*time.Second {
		return stylePanel.Render(fmt.Sprintf(
			"Candidate: %s", styleDim.Render("(idle)"),
		))
	}

	ddColor := m.ddColor(c.DD)
	reasonColor := styleYellow
	if c.RejectionReason == "ACCEPTED" {
		reasonColor = styleGreen
	} else if strings.Contains(c.RejectionReason, "BUG") {
		reasonColor = styleRed
	}

	return stylePanel.Render(fmt.Sprintf(
		"Candidate: score=%.4f │ DD=%s │ trades=%d │ reason=%s",
		c.ValScore,
		ddColor,
		c.Trades,
		reasonColor.Render(c.RejectionReason),
	))
}

func (m Model) renderEvents() string {
	// viewport.Model is a struct, not a pointer - never nil
	// Content is updated in Update() on MsgEvent, not here
	if !m.ready || m.width == 0 {
		return stylePanel.Render("Events: initializing...")
	}
	return stylePanel.Render("Events (scroll):") + "\n" + m.viewport.View()
}

func (m Model) renderFooter() string {
	hints := []string{"q: quit", "p: pause", "d: debug"}
	if m.paused {
		hints = append(hints, "(PAUSED)")
	}

	hintStrings := make([]string, len(hints))
	for i, h := range hints {
		hintStrings[i] = styleDim.Render(h)
	}

	return styleGray.Render("│ " + strings.Join(hintStrings, " │ ") + " │")
}

// Color helper functions
// NOTE: passRate is stored as percent (0.85 = 0.85%, not 0.0085)
func (m Model) passRateColor(passRate float64) string {
	if passRate >= 1.0 {
		return styleGreen.Render(fmt.Sprintf("%.2f%%", passRate))
	}
	if passRate >= 0.2 {
		return styleYellow.Render(fmt.Sprintf("%.2f%%", passRate))
	}
	return styleRed.Render(fmt.Sprintf("%.2f%%", passRate))
}

func (m Model) scoreChangeColor(score float64) string {
	// Compare with previous best score
	if score > m.prevBest {
		return styleGreen.Render(fmt.Sprintf("%.4f ↑", score))
	}
	if score < m.prevBest {
		return styleRed.Render(fmt.Sprintf("%.4f ↓", score))
	}
	return styleDim.Render(fmt.Sprintf("%.4f =", score))
}

func (m Model) percentColor(pct float64) string {
	if pct < 10 {
		return styleGreen.Render(fmt.Sprintf("%.1f%%", pct))
	}
	if pct < 30 {
		return styleYellow.Render(fmt.Sprintf("%.1f%%", pct))
	}
	return styleRed.Render(fmt.Sprintf("%.1f%%", pct))
}

func (m Model) ddColor(dd float32) string {
	maxDD := float32(0.50) // Should come from meta
	threshold := maxDD * 1.10

	if dd <= maxDD {
		return styleGreen.Render(fmt.Sprintf("%.2f%%", dd*100))
	}
	if dd <= threshold {
		return styleYellow.Render(fmt.Sprintf("%.2f%%", dd*100))
	}
	return styleRed.Render(fmt.Sprintf("%.2f%%", dd*100))
}

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
