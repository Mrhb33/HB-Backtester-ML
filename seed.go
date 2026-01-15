package main

import (
	"bufio"
	"encoding/json"
	"os"
)

// Load last N lines from winners.jsonl (simple + safe).
func loadRecentElites(path string, limit int) ([]EliteLog, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Ring buffer of last N lines
	ring := make([]string, 0, limit)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		if len(ring) < limit {
			ring = append(ring, line)
		} else {
			// rotate
			copy(ring, ring[1:])
			ring[len(ring)-1] = line
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	out := make([]EliteLog, 0, len(ring))
	for _, line := range ring {
		var e EliteLog
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			// don't crash; just skip bad lines
			continue
		}
		out = append(out, e)
	}
	return out, nil
}

// Convert loaded logs to runnable strategies (compile bytecode etc.).
func rebuildStrategy(s Strategy) (Strategy, error) {
	// Compile rule bytecode for speed
	s.EntryCompiled = compileRuleTree(s.EntryRule.Root)
	s.ExitCompiled = compileRuleTree(s.ExitRule.Root)
	s.RegimeCompiled = compileRuleTree(s.RegimeFilter.Root)

	return s, nil
}
