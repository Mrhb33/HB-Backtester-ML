package tui

import (
	"context"
	"fmt"
	"os"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
	"golang.org/x/term"
)

type TUIConfig struct {
	Title   string
	Mode    string
	Dataset string
}

var (
	mu      sync.RWMutex
	program *tea.Program
)

// Start initializes and starts the TUI
// Returns nil if TUI started successfully, error if disabled (non-TTY, TERM=dumb, etc.)
func Start(ctx context.Context, cfg TUIConfig) error {
	// Check if stdout is a terminal
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		return fmt.Errorf("TUI disabled (not a TTY)")
	}

	// Check for TERM=dumb
	if os.Getenv("TERM") == "dumb" {
		return fmt.Errorf("TUI disabled (TERM=dumb)")
	}

	m := NewModel()
	m.snapshot.ProjectName = cfg.Title
	m.snapshot.Mode = cfg.Mode
	m.snapshot.Dataset = cfg.Dataset

	p := tea.NewProgram(m, tea.WithContext(ctx))

	mu.Lock()
	program = p
	mu.Unlock()

	// Run in background (don't wait for it, let it run until ctx is cancelled)
	go func() {
		_, _ = p.Run()
	}()

	return nil
}

// Stop gracefully shuts down the TUI
func Stop() {
	mu.RLock()
	p := program
	mu.RUnlock()
	if p != nil {
		p.Send(MsgShutdown{})
		// Don't call ReleaseTerminal() - Bubble Tea handles cleanup on Quit
	}
}

// PushState sends a state snapshot to the TUI (thread-safe)
func PushState(s StateSnapshot) {
	mu.RLock()
	p := program
	mu.RUnlock()
	if p != nil {
		p.Send(MsgStateSnapshot(s))
	}
}

// PushEvent sends an event to the TUI (thread-safe)
func PushEvent(e Event) {
	mu.RLock()
	p := program
	mu.RUnlock()
	if p != nil {
		p.Send(MsgEvent(e))
	}
}
