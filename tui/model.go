package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/viewport"
)

// StateSnapshot represents the complete backtest state at a point in time
type StateSnapshot struct {
	ProjectName string
	Mode        string
	Dataset     string
	StartTime   time.Time

	TestedCount int64
	TargetCount int64  // 0 = infinite mode
	RatePerSec  float64
	ETA         time.Duration

	PassRate     float64
	BestValScore float64
	ElitesCount  int

	RadicalP      float32
	SurExploreP   float32
	StagnationCnt int64

	RejectedSeen     int64
	RejectedSur      int64
	RejectedNovelty  int64

	CurrentCandidate CandidateInfo
}

type CandidateInfo struct {
	ValScore        float32
	DD              float32
	Trades          int
	RejectionReason string
	Timestamp       time.Time
}

// Event represents a significant event
type Event struct {
	Timestamp time.Time
	Type      string // "ELITE", "BEST", "GATE", "STAGNATION", "BUG", etc.
	Severity  string // "info", "warning", "error"
	Message   string
}

type (
	MsgStateSnapshot StateSnapshot
	MsgEvent         Event
	MsgShutdown      struct{}
	MsgTick          time.Time
)

type Model struct {
	snapshot  StateSnapshot
	events    []Event // Ring buffer, max 1000
	paused    bool
	debugMode bool

	width  int
	height int
	ready  bool

	progress progress.Model // NOT a pointer
	viewport viewport.Model // NOT a pointer

	// Track previous best to show ↑ ↓
	prevBest float64
}

func NewModel() Model {
	return Model{
		snapshot: StateSnapshot{StartTime: time.Now()},
		events:   make([]Event, 0, 1000),
		progress: progress.New(progress.WithWidth(40)),
		viewport: viewport.New(0, 10),
	}
}

func (m Model) Init() tea.Cmd {
	return tea.Tick(250*time.Millisecond, func(t time.Time) tea.Msg {
		return MsgTick(t)
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Handle custom keys first (q=quit, p=pause, d=debug)
		var cmd tea.Cmd
		m2, keyCmd := m.handleKey(msg)
		m = m2.(Model)

		// Then pass to viewport for scrolling
		m.viewport, cmd = m.viewport.Update(msg)
		return m, tea.Batch(cmd, keyCmd)

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true
		// Set viewport dimensions
		m.viewport.Width = m.width - 4
		m.viewport.Height = 10
		return m, nil

	case MsgStateSnapshot:
		// Explicit cast needed (MsgStateSnapshot is a distinct type)
		s := StateSnapshot(msg)
		m.prevBest = m.snapshot.BestValScore
		m.snapshot = s
		return m, nil

	case MsgEvent:
		// Explicit cast needed
		e := Event(msg)
		m.addEvent(e)
		// Update viewport content and auto-scroll to bottom
		m.updateViewportContent()
		m.viewport.GotoBottom()
		return m, nil

	case MsgTick:
		return m, tea.Tick(250*time.Millisecond, func(t time.Time) tea.Msg {
			return MsgTick(t)
		})

	case MsgShutdown:
		return m, tea.Quit
	}
	return m, nil
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q", "ctrl+c":
		return m, tea.Quit
	case "p":
		m.paused = !m.paused
		return m, nil
	case "d":
		m.debugMode = !m.debugMode
		return m, nil
	}
	return m, nil
}

func (m *Model) addEvent(e Event) {
	m.events = append(m.events, e)
	if len(m.events) > 1000 {
		m.events = m.events[1:]
	}
}

// updateViewportContent rebuilds events content for viewport
// Call this only when events change (on MsgEvent), not every render
func (m *Model) updateViewportContent() {
	var eventStrings []string
	for _, e := range m.events {
		style := styleEventInfo
		if e.Severity == "warning" {
			style = styleEventWarn
		} else if e.Severity == "error" {
			style = styleEventError
		}

		icon := "•"
		if e.Type == "ELITE" {
			icon = "✓"
		} else if e.Type == "BEST" {
			icon = "↗"
		} else if e.Severity == "warning" {
			icon = "⚠"
		} else if e.Severity == "error" {
			icon = "✗"
		}

		eventStrings = append(eventStrings, style.Render(
			fmt.Sprintf("[%s] %s %s", e.Timestamp.Format("15:04:05"), icon, e.Message),
		))
	}
	m.viewport.SetContent(strings.Join(eventStrings, "\n"))
}
