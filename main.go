package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Global verbose flag - controls debug output across all packages
var Verbose bool = false

// Generator loop counters (atomic for thread safety)
var (
	genGenerated   int64 // Total strategies generated
	genRejectedSur int64 // Rejected by surrogate
	genRejectedSeen  int64 // Rejected by markSeen (duplicate)
	genSentToJobs  int64 // Sent to worker jobs
)

// getGeneratorStats returns formatted generator loop statistics
func getGeneratorStats() string {
	generated := atomic.LoadInt64(&genGenerated)
	rejectedSur := atomic.LoadInt64(&genRejectedSur)
	rejectedSeen := atomic.LoadInt64(&genRejectedSeen)
	sentToJobs := atomic.LoadInt64(&genSentToJobs)

	totalGenerated := generated
	if totalGenerated == 0 {
		totalGenerated = 1 // avoid division by zero
	}

	surPct := 100.0 * float64(rejectedSur) / float64(totalGenerated)
	seenPct := 100.0 * float64(rejectedSeen) / float64(totalGenerated)
	sentPct := 100.0 * float64(sentToJobs) / float64(totalGenerated)

	return fmt.Sprintf("gen=%d rej_sur=%d(%.1f%%) rej_seen=%d(%.1f%%) sent=%d(%.1f%%)",
		generated, rejectedSur, surPct, rejectedSeen, seenPct, sentToJobs, sentPct)
}

type BatchMsg struct {
	TopN  []Result // top N results from this batch
	Count int64
}

type EliteLog struct {
	Seed         int64   `json:"seed"`
	FeeBps       float32 `json:"fee_bps"`
	SlippageBps  float32 `json:"slippage_bps"`
	Direction    int     `json:"direction"`
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
	ValScore     float32 `json:"val_score"`
	ValReturn    float32 `json:"val_return"`
	ValMaxDD     float32 `json:"val_max_dd"`
	ValWinRate   float32 `json:"val_win_rate"`
	ValTrades    int     `json:"val_trades"`
}

type ReportLine struct {
	Batch          int64   `json:"batch"`
	Tested         int64   `json:"tested"`
	Score          float32 `json:"score"`
	Return         float32 `json:"return"`
	MaxDD          float32 `json:"max_dd"`
	WinRate        float32 `json:"win_rate"`
	Expectancy     float32 `json:"expectancy"`
	ProfitFactor   float32 `json:"profit_factor"`
	Trades         int     `json:"trades"`
	FeeBps         float32 `json:"fee_bps"`
	SlippageBps    float32 `json:"slippage_bps"`
	Direction      int     `json:"direction"`
	Seed           int64   `json:"seed"`
	EntryRuleDesc  string  `json:"entry_rule"`
	ExitRuleDesc   string  `json:"exit_rule"`
	StopLossDesc   string  `json:"stop_loss"`
	TakeProfitDesc string  `json:"take_profit"`
	TrailDesc      string  `json:"trail"`
	Passed         bool    `json:"passed"`
	// Validation metrics
	ValScore   float32 `json:"val_score,omitempty"`
	ValReturn  float32 `json:"val_return,omitempty"`
	ValMaxDD   float32 `json:"val_max_dd,omitempty"`
	ValWinRate float32 `json:"val_win_rate,omitempty"`
	ValTrades  int     `json:"val_trades,omitempty"`
}

type MetaState struct {
	// Search knobs
	RadicalP     float32 `json:"radical_p"`
	SurExploreP  float32 `json:"sur_explore_p"`
	SurThreshold float32 `json:"sur_threshold"`

	// Validation gates (the "what counts as good")
	MaxValDD     float32 `json:"max_val_dd"`
	MinValReturn float32 `json:"min_val_return"`
	MinValPF     float32 `json:"min_val_pf"`
	MinValExpect float32 `json:"min_val_expect"`
	MinValTrades int     `json:"min_val_trades"`

	// Multi-fidelity gate control (0=strict, 1=normal, 2=relaxed, 3=very_relaxed)
	ScreenRelaxLevel int `json:"screen_relax_level"`

	// Telemetry
	Batches              int64   `json:"batches"`
	BestVal              float32 `json:"best_val"`
	LastImprovementBatch int64   `json:"last_improvement_batch"`

	// Mutex for thread-safe access (Bug #4 fix)
	mu sync.RWMutex
}

func clampf(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func main() {
	fmt.Println("HB Backtest Strategy Search Engine")
	fmt.Println("====================================")

	mode := flag.String("mode", "search", "search|test|golden|trace|validate")
	verbose := flag.Bool("verbose", false, "enable verbose debug logs")
	scoringMode := flag.String("scoring", "balanced", "scoring mode: balanced or aggressive")
	seedFlag := flag.Int64("seed", 0, "random seed (0 = time-based, nonzero = reproducible)")
	resumePath := flag.String("resume", "", "checkpoint file to resume from (ex: checkpoint.json)")
	checkpointPath := flag.String("checkpoint", "checkpoint.json", "checkpoint output path")
	checkpointEverySec := flag.Int("checkpoint_every", 60, "TEMP warm-start: auto-save checkpoint every N seconds")
	// Cost override flags for production-level realism
	feeBpsFlag := flag.Float64("fee_bps", 30, "transaction fee in basis points (0.01% per bps, default 30 = 0.3%)")
	slipBpsFlag := flag.Float64("slip_bps", 8, "slippage in basis points (default 8 = 0.08%)")
	// Golden mode flags
	goldenSeed := flag.Int64("golden_seed", 0, "seed of winner strategy to run in golden mode")
	goldenN := flag.Int("golden_print_trades", 10, "how many trades to print in golden mode")
	// Trace mode flags
	traceSeed := flag.Int64("trace_seed", 0, "seed of strategy to trace (outputs per-bar states)")
	traceCSVPath := flag.String("trace_csv", "trace.csv", "output CSV path for trace mode")
	traceManual := flag.String("trace_manual", "", "manual debug strategy (ema20x50, etc.)")
	traceOpenCSV := flag.Bool("trace_open", false, "open CSV after trace (Windows)")
	traceWindow := flag.String("trace_window", "test", "trace window: train | val | test")
	flag.Parse()

	// Set global verbose flag from command line
	Verbose = *verbose

	// Convert flag values to float32
	feeBps := float32(*feeBpsFlag)
	slipBps := float32(*slipBpsFlag)

	// Define scoring thresholds based on mode
	var maxValDD, minValReturn, minValPF, minValExpect float32
	var minValTrades int
	if *scoringMode == "aggressive" {
		// Aggressive mode: stricter profit gates, looser DD to encourage exploration
		maxValDD = 0.60       // Allow higher drawdown
		minValReturn = 0.0    // TEMP warm-start: allow any return
		minValPF = 1.0        // TEMP warm-start: relaxed profit factor
		minValExpect = 0.0    // TEMP warm-start: allow any expectancy
		minValTrades = 20     // TEMP warm-start: 20 trades until elites exist, then 30
		fmt.Printf("Scoring mode: AGGRESSIVE (DD<%.2f, ret>%.1f%%, pf>%.2f, exp>%.4f) [WARM-START]\n", maxValDD, minValReturn*100, minValPF, minValExpect)
	} else {
		// Balanced mode (default): balanced gates for stability and exploration
		maxValDD = 0.45        // Tighter drawdown limit
		minValReturn = 0.0     // TEMP warm-start: allow any return
		minValPF = 1.0         // TEMP warm-start: relaxed profit factor
		minValExpect = 0.0     // TEMP warm-start: allow any expectancy
		minValTrades = 20      // TEMP warm-start: 20 trades until elites exist, then 30
		fmt.Printf("Scoring mode: BALANCED (DD<%.2f, ret>%.1f%%, pf>%.2f, exp>%.4f) [WARM-START]\n", maxValDD, minValReturn*100, minValPF, minValExpect)
	}

	// Initialize MetaState for self-improving search
	meta := MetaState{
		RadicalP:         0.10,
		SurExploreP:      0.10,
		SurThreshold:     0.0,
		MaxValDD:         maxValDD,
		MinValReturn:     minValReturn,
		MinValPF:         minValPF,
		MinValExpect:     minValExpect,
		MinValTrades:     minValTrades,
		ScreenRelaxLevel: 3, // Default to very relaxed (unblock mode)
	}

	// Mode dispatch
	switch *mode {
	case "trace":
		runTraceMode(*traceSeed, *traceCSVPath, *traceManual, *traceOpenCSV, *feeBpsFlag, *slipBpsFlag, *traceWindow)
		return
	case "golden":
		runGoldenMode(*goldenSeed, *goldenN, *feeBpsFlag, *slipBpsFlag)
		return
	case "test":
		runTestMode(feeBps, slipBps)
		return
	case "validate":
		RunValidation("btc_5min_data.csv")
		return
	default: // search mode
		// Continue to search logic
	}

	// Define print helper at the top to avoid using builtin print
	var printMu sync.Mutex
	print := func(format string, args ...interface{}) {
		printMu.Lock()
		fmt.Printf(format, args...)
		printMu.Unlock()
	}

	// Helper function for min
	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}

	ncpu := runtime.NumCPU()
	workers := int(float64(ncpu) * 0.40)
	if workers < 1 {
		workers = 1
	}
	runtime.GOMAXPROCS(workers)

	print("CPU Cores: %d, Using: %d workers (~40%%)\n", ncpu, workers)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var saveNow func()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		print("\n\nReceived stop signal. Shutting down gracefully...\n")
		cancel()
		if saveNow != nil {
			saveNow()
		}
	}()

	print("Loading data...")
	startTime := time.Now()

	series, err := LoadBinanceKlinesCSV("btc_5min_data.csv")
	if err != nil {
		fmt.Printf("Error loading CSV: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", series.T)

	// Check if we have enough data for backtesting
	const minRequiredCandles = 1000 // minimum for meaningful backtesting
	if series.T < minRequiredCandles {
		fmt.Printf("WARNING: Only %d candles loaded. At least %d candles recommended for backtesting.\n", series.T, minRequiredCandles)
		fmt.Printf("Results may be poor or zero trades.\n\n")
	}

	fmt.Println("Computing features...")
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.F))

	// Split data into train/validation/test windows
	trainStart, trainEnd, valEnd := GetSplitIndices(series.OpenTimeMs)
	trainW, valW, testW := GetSplitWindows(trainStart, trainEnd, valEnd, series.T, 500)

	// Compute screen window (last 6 months of train for fast screening)
	screenW := getScreenWindow(trainW)
	fmt.Printf("Screen window: %d candles (%d -> %d)\n", screenW.End-screenW.Start, screenW.Start, screenW.End)

	// Recompute feature stats on train window only (no future leak!)
	computeStatsOnWindow(&feats, trainStart, trainEnd)
	fmt.Printf("Feature stats computed on train window (%d candles)\n", trainEnd-trainStart)

	fmt.Printf("\nData split:\n")
	fmt.Printf("  Train:  %d candles (%d -> %d)\n", trainEnd-trainStart, trainStart, trainEnd)
	fmt.Printf("  Val:    %d candles (%d -> %d)\n", valEnd-trainEnd, trainEnd, valEnd)
	fmt.Printf("  Test:   %d candles (%d -> %d)\n", series.T-valEnd, valEnd, series.T)
	fmt.Printf("  Total:  %d candles\n\n", series.T)

	loadTime := time.Since(startTime)
	fmt.Printf("Data load time: %v\n\n", loadTime)

	// Track top strategies for final test evaluation
	topK := make([]Result, 0, 20)
	const minTopScore = 0.65 // Lowered to allow tracking of realistic scores

	// Initialize Hall of Fame for evolution
	hof := NewHallOfFame(50) // keep top 50 by validation score (reduced from 200 to force exploration)

	// Initialize MAP-Elites archive for diversity preservation
	archive := NewArchive()

	// Helpers to snapshot/restore Hall of Fame (using slim elites to avoid compiled bytecode)
	snapshotHOFSlim := func(h *HallOfFame) []SlimElite {
		h.mu.RLock()
		defer h.mu.RUnlock()
		out := make([]SlimElite, len(h.Elites))
		for i, e := range h.Elites {
			out[i] = eliteToSlim(e)
		}
		return out
	}

	restoreHOFSlim := func(h *HallOfFame, slimElites []SlimElite, rng *rand.Rand) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.Elites = make([]Elite, len(slimElites))
		for i, se := range slimElites {
			h.Elites[i] = slimToElite(se, rng)
		}
	}

	// Track passed validations for adaptive criteria
	var passedCount int64
	var validatedLabels int64    // Track number of validation labels for surrogate training
	var bestValSeen float32 = -1e30 // Track best validation score seen (init to very low for bootstrap)
	var bestValSeenMu sync.Mutex // Protect bestValSeen for thread-safe access

	// Anti-stagnation tracking (thread-safe)
	var radicalP float32 = 0.10    // base radical mutation probability
	var surExploreP float32 = 0.10 // base surrogate exploration
	var surThreshold float64 = 0.0 // surrogate filtering threshold
	var lastBestValScore float32 = -1e30
	var batchesSinceLastImprovement int
	var stagnationMu sync.RWMutex // protects the above anti-stagnation variables

	// Load meta.json if exists and apply to runtime knobs (must be after vars declared)
	if b, err := os.ReadFile("meta.json"); err == nil {
		if err := json.Unmarshal(b, &meta); err == nil {
			fmt.Printf("Loaded meta.json: Batches=%d, BestVal=%.4f, RadicalP=%.2f, SurExploreP=%.2f, ScreenRelaxLevel=%d\n",
				meta.Batches, meta.BestVal, meta.RadicalP, meta.SurExploreP, meta.ScreenRelaxLevel)
			// Apply loaded meta into runtime knobs with TRUE CLAMPS (not reset)
			// Clamp prevents oscillation, doesn't hard-reset to defaults
			meta.MaxValDD = clampf(meta.MaxValDD, 0.25, 0.60)
			meta.MinValReturn = clampf(meta.MinValReturn, 0.02, 0.15)
			meta.MinValPF = clampf(meta.MinValPF, 1.05, 1.50)
			// Allow MinValExpect to go negative but clamp extreme values
			meta.MinValExpect = clampf(meta.MinValExpect, -0.001, 0.01)
			// MinValTrades has reasonable bounds
			if meta.MinValTrades < 10 {
				meta.MinValTrades = 10
			} else if meta.MinValTrades > 100 {
				meta.MinValTrades = 100
			}
			// ScreenRelaxLevel must be 0-3
			if meta.ScreenRelaxLevel < 0 {
				meta.ScreenRelaxLevel = 0
			} else if meta.ScreenRelaxLevel > 3 {
				meta.ScreenRelaxLevel = 3
			}

			maxValDD, minValReturn, minValPF, minValExpect = meta.MaxValDD, meta.MinValReturn, meta.MinValPF, meta.MinValExpect
			minValTrades = meta.MinValTrades
			surThreshold = float64(meta.SurThreshold)
			stagnationMu.Lock()
			radicalP, surExploreP = meta.RadicalP, meta.SurExploreP
			stagnationMu.Unlock()
			// Sync global screen relax level
			setScreenRelaxLevel(meta.ScreenRelaxLevel)
		}
	} else {
		// Initialize global screen relax level to default
		setScreenRelaxLevel(meta.ScreenRelaxLevel)
	}

	// Track seen fingerprints to avoid retesting strategies
	// Use SkeletonFingerprint for structure-only deduplication (less aggressive)
	// Full fingerprint is still used for leaderboard uniqueness via globalFingerprints
	seen := make(map[string]struct{})
	seenMu := sync.Mutex{}

	markSeen := func(s Strategy) bool {
		fp := s.SkeletonFingerprint() // Use structure-only fingerprint for seen tracking
		seenMu.Lock()
		defer seenMu.Unlock()
		if _, ok := seen[fp]; ok {
			return false
		}
		seen[fp] = struct{}{}
		return true
	}

	// Initialize RNG for seeding (needs to be before seedFromWinners)
	var seed int64
	if *seedFlag == 0 {
		seed = time.Now().UnixNano()
		fmt.Printf("Seed: %d (time-based)\n", seed)
	} else {
		seed = *seedFlag
		print("Seed: %d (user-provided)\n", seed)
	}
	rng := rand.New(rand.NewSource(seed))

	// Load checkpoint if provided (must be after HOF and RNG initialization)
	if *resumePath != "" {
		cp, err := LoadCheckpoint(*resumePath)
		if err != nil {
			fmt.Printf("WARN: failed to load checkpoint: %v\n", err)
		} else {
			fmt.Printf("Resuming from checkpoint: %s (saved %d)\n", *resumePath, cp.SavedAtUnix)

			// restore hof (from slim elites)
			restoreHOFSlim(hof, cp.HOFElites, rng)

			// restore archive (from slim elites)
			for _, se := range cp.ArchiveElites {
				e := slimToElite(se, rng)
				archive.Add(e.Val, e)
			}

			// restore counters
			atomic.StoreInt64(&passedCount, cp.PassedCount)
			atomic.StoreInt64(&validatedLabels, cp.ValidatedLabels)
			bestValSeenMu.Lock()
			bestValSeen = cp.BestValSeen
			bestValSeenMu.Unlock()
			// Initialize testedAtLastCheckpoint for resume
			// Will be set in aggregator goroutine at first checkpoint

			// restore seen
			seenMu.Lock()
			for _, fp := range cp.SeenFingerprints {
				seen[fp] = struct{}{}
			}
			seenMu.Unlock()

			// restore seed if user didn't provide one
			if *seedFlag == 0 && cp.Seed != 0 {
				seed = cp.Seed
				rng = rand.New(rand.NewSource(seed))
				print("Seed overridden by checkpoint: %d\n", seed)
			}
		}
	}

	// Initialize surrogate model for filtering
	sur := NewSurrogateWithSeed(SurDim, seed)
	if err := sur.Load("surrogate.json"); err == nil {
		print("Loaded surrogate weights from surrogate.json\n")
	} else {
		print("No surrogate.json found, starting with fresh model\n")
	}
	// surThreshold is now declared globally above

	// Winner logger for persistent memory
	winnerLog := make(chan EliteLog, 1024)
	go func() {
		winnerFile, err := os.OpenFile("winners.jsonl", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			print("ERROR: Failed to open winners.jsonl for writing: %v\n", err)
			return
		}
		defer winnerFile.Close()
		w := bufio.NewWriterSize(winnerFile, 1<<20)
		defer w.Flush()
		lineCount := 0

		// Track recent fingerprints for deduplication (keep last 1000 unique fingerprints)
		const maxRecentFingerprints = 1000
		recentFingerprints := make(map[string]bool)

		for log := range winnerLog {
			// Build fingerprint for deduplication
			fingerprint := log.EntryRule + "|" + log.ExitRule + "|" + log.RegimeFilter + "|" +
				log.StopLoss + "|" + log.TakeProfit + "|" + log.Trail

			// Skip if we've seen this fingerprint recently
			if recentFingerprints[fingerprint] {
				continue
			}

			// Add to recent set
			recentFingerprints[fingerprint] = true

			// Prune map if it gets too large (simple eviction)
			if len(recentFingerprints) > maxRecentFingerprints {
				// Clear 25% of entries periodically
				if lineCount%100 == 0 {
					newMap := make(map[string]bool)
					count := 0
					for fp := range recentFingerprints {
						if count < maxRecentFingerprints*3/4 {
							newMap[fp] = true
							count++
						}
					}
					recentFingerprints = newMap
				}
			}

			data, err := json.Marshal(log)
			if err == nil {
				w.Write(data)
				w.WriteString("\n")
				lineCount++
				// Flush every 100 lines instead of every line (much faster)
				if lineCount%100 == 0 {
					w.Flush()
				}
			}
		}
	}()

	// Function to save meta state
	saveMeta := func() {
		// Apply clamps before saving to prevent drift
		meta.MaxValDD = clampf(maxValDD, 0.25, 0.60)
		meta.MinValReturn = clampf(minValReturn, 0.02, 0.15)
		meta.MinValPF = clampf(minValPF, 1.05, 1.50)
		meta.MinValExpect = clampf(minValExpect, -0.001, 0.01)
		meta.MinValTrades = minValTrades
		meta.SurThreshold = float32(surThreshold)
		stagnationMu.RLock()
		meta.RadicalP = radicalP
		meta.SurExploreP = surExploreP
		stagnationMu.RUnlock()

		b, _ := json.MarshalIndent(meta, "", "  ")
		if err := os.WriteFile("meta.json", b, 0644); err != nil {
			fmt.Printf("WARN: meta.json save failed: %v\n", err)
		}
	}

	// Function to save checkpoint
	saveNow = func() {
		// cap seen size so file doesn't explode
		const maxSeen = 200000

		seenMu.Lock()
		seenList := make([]string, 0, len(seen))
		for fp := range seen {
			seenList = append(seenList, fp)
			if len(seenList) >= maxSeen {
				break
			}
		}
		seenMu.Unlock()

		// Snapshot archive elites (limit to 200 for checkpoint size)
		const maxArchiveElites = 200
		archiveSlim := make([]SlimElite, 0, maxArchiveElites)
		archive.mu.RLock()
		for _, elite := range archive.cells {
			if len(archiveSlim) >= maxArchiveElites {
				break
			}
			archiveSlim = append(archiveSlim, eliteToSlim(elite))
		}
		archive.mu.RUnlock()

		cp := Checkpoint{
			Seed:             seed,
			PassedCount:      atomic.LoadInt64(&passedCount),
			ValidatedLabels:  atomic.LoadInt64(&validatedLabels),
			BestValSeen:      func() float32 { bestValSeenMu.Lock(); defer bestValSeenMu.Unlock(); return bestValSeen }(),
			HOFElites:        snapshotHOFSlim(hof),
			ArchiveElites:    archiveSlim,
			SeenFingerprints: seenList,
		}
		// Print rejection stats before checkpointing
		printRejectionStats()

		if err := SaveCheckpoint(*checkpointPath, cp); err != nil {
			fmt.Printf("WARN: checkpoint save failed: %v\n", err)
		} else {
			fmt.Printf("\nCheckpoint saved: %s\n", *checkpointPath)
		}
		// Also save meta state whenever we checkpoint
		saveMeta()
	}

	// Seed initial population from winners.jsonl if it exists
	seedFromWinners := func() []Strategy {
		f, err := os.Open("winners.jsonl")
		if err != nil {
			return nil
		}
		defer f.Close()

		// Use a ring buffer to keep only the last N lines (no need to read entire file)
		const maxLines = 1000 // Keep last 1000 lines in ring buffer
		ringBuffer := make([]string, maxLines)
		pos := 0
		lineCount := 0

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			ringBuffer[pos] = scanner.Text()
			pos = (pos + 1) % maxLines
			lineCount++
		}
		if err := scanner.Err(); err != nil {
			return nil
		}

		var strategies []Strategy
		var archiveElites []Elite // Also collect elites for archive rebuilding
		maxToLoad := 100          // Load up to 100 previous winners
		loaded := 0

		// Iterate through ring buffer in correct order
		for i := 0; i < maxLines && loaded < maxToLoad; i++ {
			// Skip empty slots if we haven't filled the buffer
			if lineCount < maxLines && i >= lineCount {
				break
			}

			// Process in reverse order (newest first)
			actualIdx := (pos - 1 - i + maxLines) % maxLines
			if lineCount < maxLines {
				actualIdx = lineCount - 1 - i
				if actualIdx < 0 {
					break
				}
			}

			line := ringBuffer[actualIdx]
			if line == "" {
				continue
			}

			var log EliteLog
			if err := json.Unmarshal([]byte(line), &log); err != nil {
				continue
			}

			// Only load high-quality winners (relaxed threshold for cross-run learning)
			// Dynamic threshold: accept within 0.03 of best validation score seen
			// This ensures cross-restart learning works even when best scores evolve
			var loadMinScore float32 = 0.65  // baseline minimum
			var loadMinReturn float32 = 0.02 // +2% return gate for warm-start (same as validation)
			bestValSeenMu.Lock()
			currentBest := bestValSeen
			bestValSeenMu.Unlock()
			if currentBest > 0.65 {
				loadMinScore = currentBest - 0.03 // dynamic threshold
				if loadMinScore < 0.65 {
					loadMinScore = 0.65 // never go below baseline
				}
			}
			// Use meta gates with slight relaxation for warm-start (Bug #5 fix)
			loadMaxDD := maxValDD + 0.05 // relax meta gate by 5%
			loadMinTrades := minValTrades
			if loadMinTrades > 15 {
				loadMinTrades = 15 // minimum floor even if meta gate is higher
			}
			if log.ValScore >= loadMinScore && log.ValMaxDD < loadMaxDD && log.ValTrades >= loadMinTrades && log.ValReturn >= loadMinReturn {
				s := Strategy{
					Seed:        rng.Int63(),
					FeeBps:      log.FeeBps,
					SlippageBps: log.SlippageBps,
					RiskPct:     0.01,
					Direction:   log.Direction, // Preserve original direction for consistency
					EntryRule: RuleTree{
						Root: parseRuleTree(log.EntryRule),
					},
					ExitRule: RuleTree{
						Root: parseRuleTree(log.ExitRule),
					},
					RegimeFilter: RuleTree{
						Root: parseRuleTree(log.RegimeFilter),
					},
					StopLoss:   parseStopModel(log.StopLoss),
					TakeProfit: parseTPModel(log.TakeProfit),
					Trail:      parseTrailModel(log.Trail),
				}

				// Recompile rules
				s.EntryCompiled = compileRuleTree(s.EntryRule.Root)
				s.ExitCompiled = compileRuleTree(s.ExitRule.Root)
				s.RegimeCompiled = compileRuleTree(s.RegimeFilter.Root)

				strategies = append(strategies, s)

				// Also create elite for archive rebuilding
				archiveElites = append(archiveElites, Elite{
					Strat: s,
					Train: Result{
						Score:    log.TrainScore,
						Return:   log.TrainReturn,
						MaxDD:    log.TrainMaxDD,
						WinRate:  log.TrainWinRate,
						Trades:   log.TrainTrades,
						Strategy: s,
					},
					Val: Result{
						Score:    log.ValScore,
						Return:   log.ValReturn,
						MaxDD:    log.ValMaxDD,
						WinRate:  log.ValWinRate,
						Trades:   log.ValTrades,
						Strategy: s,
					},
					ValScore: log.ValScore,
				})

				loaded++
			}
		}

		// Rebuild archive from loaded elites (only after HOF is restored)
		go func() {
			for _, e := range archiveElites {
				archive.Add(e.Val, e)
			}
		}()

		return strategies
	}

	jobs := make(chan Strategy, 8192)
	batchResults := make(chan BatchMsg, workers*4)

	var tested uint64

	// Track generation types for diversity monitoring
	var immigrantCount, heavyMutCount, crossCount, normalMutCount int64

	f, err := os.OpenFile("best_every_10000.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Printf("Error opening output file: %v\n", err)
		return
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	defer w.Flush()

	doneAgg := make(chan struct{})
	go func() {
		defer close(doneAgg)

		// Define validation stability threshold once (not duplicated in loop)
		const stabilityThreshold = 0.25 // TEMP warm-start: val must be at least 25% of train (relaxed from 40%)

		batchSize := int64(10000)
		var batchID int64 = 0
		var batchBest Result
		batchBest.Score = -1e30
		lastReport := time.Now()
		nextCheckpoint := uint64(2000) // TEMP warm-start: checkpoint sooner to trigger validation

		// Track tested count for accurate pass-rate calculation (especially after resume)
		var testedAtLastCheckpoint int64
		// Initialize testedAtLastCheckpoint with current tested count on resume
		// This prevents the first testedThisCheckpoint from being massive
		if *resumePath != "" {
			testedAtLastCheckpoint = int64(atomic.LoadUint64(&tested))
		}

		// Global top candidates since last checkpoint (fixes validation bias)
		const maxGlobalCandidates = 100
		globalTopCandidates := make([]Result, 0, maxGlobalCandidates)
		globalFingerprints := make(map[string]bool) // Track fingerprints for deduplication

		// Self-improvement meta-update function
		updateMeta := func(improved bool, bestVal float32, passedThisBatch int64, testedThisBatch int64) {
			meta.mu.Lock()
			meta.Batches++
			currentBatches := meta.Batches
			currentBestVal := meta.BestVal
			meta.mu.Unlock()

			passRate := float32(0)
			if testedThisBatch > 0 {
				passRate = float32(passedThisBatch) / float32(testedThisBatch)
			}

			// 1) If we improved: exploit a bit (narrow search) + tighten gate slowly
			if improved {
				if bestVal > currentBestVal {
					meta.mu.Lock()
					meta.BestVal = bestVal
					meta.LastImprovementBatch = currentBatches
					meta.mu.Unlock()
					print(" [META-UPDATE: New best=%.4f at batch=%d]", bestVal, currentBatches)
				}

				oldRadical := radicalP
				oldSurExplore := surExploreP
				stagnationMu.Lock()
				// reduce chaos slightly
				radicalP = clampf(radicalP*0.85, 0.05, 0.60)
				surExploreP = clampf(surExploreP*0.90, 0.05, 0.60)
				stagnationMu.Unlock()

				if oldRadical != radicalP || oldSurExplore != surExploreP {
					print(" [META: radicalP %.2f->%.2f, surExploreP %.2f->%.2f]",
						oldRadical, radicalP, oldSurExplore, surExploreP)
				}

				// tighten gently if pass rate is too high (means your gate is too easy)
				if passRate > 0.03 { // 3% passing is "too easy" in big search
					oldReturn := minValReturn
					oldPF := minValPF
					oldDD := maxValDD
					minValReturn = clampf(minValReturn+0.005, 0.02, 0.30) // +0.5%
					minValPF = clampf(minValPF+0.02, 1.05, 1.60)
					maxValDD = clampf(maxValDD-0.01, 0.10, 0.80)
					if oldReturn != minValReturn || oldPF != minValPF || oldDD != maxValDD {
						print(" [META: Gate tightened: ret %.1f%%->%.1f%%, PF %.2f->%.2f, DD %.2f->%.2f]",
							oldReturn*100, minValReturn*100, oldPF, minValPF, oldDD, maxValDD)
					}
				}
				return
			}

			// 2) If no improvement: explore harder + loosen gate slightly (to collect signal)
			meta.mu.RLock()
			batchesNoImprove := currentBatches - meta.LastImprovementBatch
			meta.mu.RUnlock()

			if batchesNoImprove >= 5 {
				oldRadical := radicalP
				oldSurExplore := surExploreP
				stagnationMu.Lock()
				radicalP = clampf(radicalP*1.20, 0.10, 0.70)
				surExploreP = clampf(surExploreP*1.20, 0.10, 0.70)
				stagnationMu.Unlock()
				print(" [META-STAGNATE(%d): radicalP %.2f->%.2f, surExploreP %.2f->%.2f, passRate=%.2f%%]",
					batchesNoImprove, oldRadical, radicalP, oldSurExplore, surExploreP, passRate*100)

				// If pass rate is basically zero, gate is too strict -> loosen a bit
				if passRate < 0.002 { // <0.2%
					oldReturn := minValReturn
					oldPF := minValPF
					oldDD := maxValDD
					minValReturn = clampf(minValReturn-0.005, 0.00, 0.30)
					minValPF = clampf(minValPF-0.02, 1.00, 1.60)
					maxValDD = clampf(maxValDD+0.01, 0.10, 0.80)
					print(" [META: Gate loosened: ret %.1f%%->%.1f%%, PF %.2f->%.2f, DD %.2f->%.2f]",
						oldReturn*100, minValReturn*100, oldPF, minValPF, oldDD, maxValDD)
				}
			}
		}

		for {
			select {
			case <-ctx.Done():
				// Final test evaluation on shutdown - test top 20 strategies from Hall of Fame
				hof.mu.RLock()
				numToTest := 20
				if len(hof.Elites) < numToTest {
					numToTest = len(hof.Elites)
				}
				elitesToTest := make([]Elite, numToTest)
				copy(elitesToTest, hof.Elites[:numToTest])
				hof.mu.RUnlock()

				if numToTest == 0 {
					print("\n\nNo strategies in Hall of Fame to test.\n")
				} else {
					print("\n\nRunning final test evaluation on top %d strategies from Hall of Fame...\n", numToTest)
					print(strings.Repeat("=", 80))
					print("\n")

					w.WriteString("\n// FINAL TEST RESULTS (Top %d from HOF)\n" + fmt.Sprint(numToTest))
					for i, elite := range elitesToTest {
						testR := evaluateStrategyWindow(series, feats, elite.Strat, testW)
						print("Test %d/%d: Score=%.4f Return=%.2f%% WinRate=%.1f%% Trades=%d\n",
							i+1, numToTest, testR.Score, testR.Return*100, testR.WinRate*100, testR.Trades)

						testLine := ReportLine{
							Batch:          batchID + 1000 + int64(i), // Use batch ID > 1000 for test results
							Tested:         int64(atomic.LoadUint64(&tested)),
							Score:          elite.Train.Score,
							Return:         elite.Train.Return,
							MaxDD:          elite.Train.MaxDD,
							WinRate:        elite.Train.WinRate,
							Expectancy:     elite.Train.Expectancy,
							ProfitFactor:   elite.Train.ProfitFactor,
							Trades:         elite.Train.Trades,
							FeeBps:         elite.Strat.FeeBps,
							SlippageBps:    elite.Strat.SlippageBps,
							Direction:      elite.Strat.Direction,
							Seed:           elite.Strat.Seed,
							EntryRuleDesc:  ruleTreeToString(elite.Strat.EntryRule.Root),
							ExitRuleDesc:   ruleTreeToString(elite.Strat.ExitRule.Root),
							StopLossDesc:   stopModelToString(elite.Strat.StopLoss),
							TakeProfitDesc: tpModelToString(elite.Strat.TakeProfit),
							TrailDesc:      trailModelToString(elite.Strat.Trail),
							// Test metrics
							ValScore:   testR.Score,
							ValReturn:  testR.Return,
							ValMaxDD:   testR.MaxDD,
							ValWinRate: testR.WinRate,
							ValTrades:  testR.Trades,
						}
						b, _ := json.Marshal(testLine)
						w.Write(b)
						w.WriteString("\n")
					}
					w.Flush()
				}
				return
			case msg, ok := <-batchResults:
				if !ok {
					return
				}
				atomic.AddUint64(&tested, uint64(msg.Count))

				// Merge incoming TopN into global candidates IMMEDIATELY (fixes validation bias)
				// This ensures we build globalTopCandidates across the whole 10k window, not just last message
				for _, r := range msg.TopN {
					// Skip candidates with no trades or very low scores
					if r.Trades == 0 || r.Score <= -1e20 {
						continue
					}
					// Deduplicate: skip if we already have this strategy fingerprint
					fingerprint := r.Strategy.Fingerprint()
					if globalFingerprints[fingerprint] {
						continue // Skip duplicate
					}
					globalFingerprints[fingerprint] = true

					// Find insertion position in sorted order (highest score first)
					insertPos := len(globalTopCandidates)
					for i, candidate := range globalTopCandidates {
						if r.Score > candidate.Score {
							insertPos = i
							break
						}
					}

					// Insert at position using classic insertion approach
					if insertPos >= len(globalTopCandidates) {
						// Append at end
						if len(globalTopCandidates) < maxGlobalCandidates {
							globalTopCandidates = append(globalTopCandidates, r)
						}
					} else {
						// Insert in middle or replace
						if len(globalTopCandidates) < maxGlobalCandidates {
							// Make room by appending one element
							globalTopCandidates = append(globalTopCandidates, Result{})
							// Shift elements down
							for j := len(globalTopCandidates) - 1; j > insertPos; j-- {
								globalTopCandidates[j] = globalTopCandidates[j-1]
							}
							// Insert at position
							globalTopCandidates[insertPos] = r
						} else {
							// Full buffer - insert if better than or equal to last element
							if insertPos < len(globalTopCandidates) {
								// Shift elements down to make room
								for j := len(globalTopCandidates) - 1; j > insertPos; j-- {
									globalTopCandidates[j] = globalTopCandidates[j-1]
								}
								// Insert at position
								globalTopCandidates[insertPos] = r
							}
						}
					}
				}

				if len(msg.TopN) > 0 && msg.TopN[0].Score > batchBest.Score {
					batchBest = msg.TopN[0]
				}

				// Threshold crossing logic for checkpoints
				for atomic.LoadUint64(&tested) >= nextCheckpoint {
					// We've reached or passed a checkpoint, process it
					batchID++
					line := ReportLine{
						Batch:          batchID,
						Tested:         int64(atomic.LoadUint64(&tested)),
						Score:          batchBest.Score,
						Return:         batchBest.Return,
						MaxDD:          batchBest.MaxDD,
						WinRate:        batchBest.WinRate,
						Expectancy:     batchBest.Expectancy,
						ProfitFactor:   batchBest.ProfitFactor,
						Trades:         batchBest.Trades,
						FeeBps:         batchBest.Strategy.FeeBps,
						SlippageBps:    batchBest.Strategy.SlippageBps,
						Direction:      batchBest.Strategy.Direction,
						Seed:           batchBest.Strategy.Seed,
						EntryRuleDesc:  ruleTreeToString(batchBest.Strategy.EntryRule.Root),
						ExitRuleDesc:   ruleTreeToString(batchBest.Strategy.ExitRule.Root),
						StopLossDesc:   stopModelToString(batchBest.Strategy.StopLoss),
						TakeProfitDesc: tpModelToString(batchBest.Strategy.TakeProfit),
						TrailDesc:      trailModelToString(batchBest.Strategy.Trail),
					}

					// Validate top N strategies from global candidates with diversity (not just this batch)
					var bestValR Result
					bestValR.Score = -1e30
					var bestTrainResult Result
					var bestPassesValidation bool
					var bestQuickTestR Result // Track best quick test result for reporting

					// Diversity selection: top 10 by score + 10 diverse fingerprints
					// This reduces overfitting to one "family" of strategies
					type candidateWithFingerprint struct {
						candidate   Result
						fingerprint string
					}

					topN := 20
					if len(globalTopCandidates) < topN {
						topN = len(globalTopCandidates)
					}

					// Select candidates to validate
					candidatesToValidate := make([]Result, 0, topN)
					selectedFingerprints := make(map[string]bool)

					// First: select top 10 by score (if available)
					topScoreCount := 10
					if topScoreCount > len(globalTopCandidates) {
						topScoreCount = len(globalTopCandidates)
					}
					for i := 0; i < topScoreCount; i++ {
						candidate := globalTopCandidates[i]
						fp := candidate.Strategy.Fingerprint()
						if !selectedFingerprints[fp] {
							candidatesToValidate = append(candidatesToValidate, candidate)
							selectedFingerprints[fp] = true
						}
					}

					// Second: select 10 diverse candidates (different fingerprints)
					diverseCount := 10
					if diverseCount > len(globalTopCandidates)-topScoreCount {
						diverseCount = len(globalTopCandidates) - topScoreCount
					}
					for i := topScoreCount; i < len(globalTopCandidates) && len(candidatesToValidate) < topN; i++ {
						candidate := globalTopCandidates[i]
						fp := candidate.Strategy.Fingerprint()
						if !selectedFingerprints[fp] {
							candidatesToValidate = append(candidatesToValidate, candidate)
							selectedFingerprints[fp] = true
						}
					}

					numToValidate := len(candidatesToValidate)

					// Guard: skip validation if no candidates to validate
					if numToValidate == 0 {
						print("No valid candidates this checkpoint\n")
						printRejectionStats() // Print rejection stats to diagnose why
						// nothing to validate this checkpoint; keep moving
						batchBest = Result{Score: -1e30}
						globalTopCandidates = globalTopCandidates[:0]
						globalFingerprints = make(map[string]bool)
						nextCheckpoint += uint64(batchSize)
						continue
					}

					// Fixed validation threshold: rely on profit sanity gates instead of dynamic score
					// Use score only for ranking, not for pass/fail, until scale stabilizes
					// TEMP warm-start: relax score threshold until elites exist
					minValScore := float32(-5.0) // Warm-start: allow negative scores
					// Thresholds are set based on scoring mode at top of main()

					// Track stagnation per checkpoint, not per candidate
					improved := false
					passedThisBatch := int64(0) // Track how many passed validation this batch

					for i := 0; i < numToValidate; i++ {
						candidate := candidatesToValidate[i]

						// Use pre-computed validation result from multi-fidelity
						valR := Result{}
						if candidate.ValResult != nil {
							valR = *candidate.ValResult
						} else {
							// Fallback: compute val result if not available (shouldn't happen)
							valR = evaluateStrategyWindow(series, feats, candidate.Strategy, valW)
						}

						// Apply DSR-lite deflation: recompute scores with actual tested count
						// This prevents "celebrating lucky finds" as search grows
						// CRITICAL: Use computeScoreWithSmoothness to select for smooth equity curves
						testedNow := int64(atomic.LoadUint64(&tested))
						valR.Score = computeScoreWithSmoothness(valR.Return, valR.MaxDD, valR.Expectancy, valR.SmoothVol, valR.DownsideVol, valR.Trades, testedNow)
						// Also deflate train scores for fair comparison
						candidate.Score = computeScoreWithSmoothness(candidate.Return, candidate.MaxDD, candidate.Expectancy, candidate.SmoothVol, candidate.DownsideVol, candidate.Trades, testedNow)

						// Anti-stagnation: track if any candidate improved this batch
						if valR.Score > lastBestValScore {
							lastBestValScore = valR.Score
							improved = true
						}

						// Update surrogate's exploration probability
						stagnationMu.RLock()
						currentSurExploreP := surExploreP
						stagnationMu.RUnlock()
						sur.mu.Lock()
						sur.exploreP = float64(currentSurExploreP)
						sur.mu.Unlock()

						// Train surrogate using validation results
						surFeatures := ExtractSurFeatures(candidate.Strategy)
						sur.Update(surFeatures, float64(valR.Score))
						atomic.AddInt64(&validatedLabels, 1) // Track total validation labels - atomic update

						// Track best validation result (primary metric)
						if valR.Score > bestValR.Score {
							bestValR = valR
							bestTrainResult = candidate
						}

						// CRITICAL FIX #2: Track best EVALUATED strategy (not just passed)
						// This prevents bestVal from staying stuck at 0.0000 when nothing passes
						// Update bestValSeen for ALL evaluated strategies with minimum trade threshold
						bestValSeenMu.Lock()
						if valR.Trades >= 15 && valR.Score > bestValSeen {
							bestValSeen = valR.Score
						}
						bestValSeenMu.Unlock()

						// ---- Profit sanity gate (after costs) ----
						// Note: use valR (current candidate) NOT bestValR (best so far)
						// CRITICAL FIX: Bootstrap elites by ranking, NOT profitOK
						// During warm-start (elites < 10), accept top K by Val score even if negative
						// This unlocks evolution instead of infinite immigrants
						elitesCount := len(hof.Elites)

						// CRITICAL FIX: Exit "very relaxed" screening mode once elites exist
						// This forces the pipeline to stop admitting junk once the gene pool exists
						if elitesCount > 0 && getScreenRelaxLevel() > 1 {
							setScreenRelaxLevel(1) // Switch to normal screening mode
							fmt.Printf("[SCREEN] Elites detected (%d), switching to normal screening mode\n", elitesCount)
						}

						// BOOTSTRAP LADDER: Relax validation gates when elite pool is small
						// This allows population to grow before tightening back to real gates
						var effectiveMaxDD, effectiveMinReturn, effectiveMinPF, effectiveMinExpect, effectiveMinScore float32
						var effectiveMinTrades int
						var skipCPCV bool

						if elitesCount < 10 {
							// CRITICAL FIX #3: TRUE bootstrap gates for first 10 elites
							// Accept "least bad" candidates that aren't completely broken
							// No profit requirement yet - just survival constraints
							effectiveMaxDD = 0.65          // Allow higher DD during bootstrap
							effectiveMinReturn = -0.02      // Allow up to -2% loss (not 0%!)
							effectiveMinPF = 0.95           // Allow near-breakeven PF (not 1.0!)
							effectiveMinExpect = -0.0005    // Allow slightly negative expectancy
							effectiveMinScore = -50.0       // Allow very low scores during bootstrap
							effectiveMinTrades = 20         // Minimum 20 trades
							skipCPCV = true                 // Skip CPCV during bootstrap
						} else {
							// Use standard gates once we have enough elites
							effectiveMaxDD = maxValDD
							effectiveMinReturn = minValReturn
							effectiveMinPF = minValPF
							effectiveMinExpect = minValExpect
							effectiveMinScore = minValScore
							effectiveMinTrades = minValTrades
							skipCPCV = false
						}

						// Basic sanity gates (use effective gates from bootstrap ladder)
						basicGatesOK := valR.MaxDD < effectiveMaxDD && valR.Trades >= effectiveMinTrades

						// Profit sanity gate (use effective gates from bootstrap ladder)
						// During bootstrap: allow ret>=-2%, pf>=0.95, exp>=-0.0005, score>=-50
						// After bootstrap: use normal profit gates
						profitOK := valR.Return >= effectiveMinReturn &&
							valR.Expectancy > effectiveMinExpect &&
							valR.ProfitFactor >= effectiveMinPF &&
							valR.Score >= effectiveMinScore

						// Add to Hall of Fame if passes validation + CPCV
						// Stability check: val_score must be decent fraction of train_score to reduce overfitting
						stableEnough := true
						if candidate.Score > 0 {
							stableEnough = valR.Score >= stabilityThreshold*candidate.Score
						}
						passesValidation := basicGatesOK && stableEnough && profitOK

						// CRITICAL FIX #4: VAL rejection histogram
						// Log ALL candidates that fail validation with current effective gates
						// This helps identify the dominant blocker during bootstrap
						if !passesValidation {
							reasons := []string{}
							if valR.MaxDD >= effectiveMaxDD {
								reasons = append(reasons, fmt.Sprintf("DD>%.2f", effectiveMaxDD))
							}
							if valR.Trades < effectiveMinTrades {
								reasons = append(reasons, fmt.Sprintf("Trds<%d", effectiveMinTrades))
							}
							if !stableEnough {
								reasons = append(reasons, "stab")
							}
							// Always log profit gate failures with effective gates (not bootstrap-specific)
							if valR.Return < effectiveMinReturn {
								reasons = append(reasons, fmt.Sprintf("Ret<%.1f%%", effectiveMinReturn*100))
							}
							if valR.Expectancy <= effectiveMinExpect {
								reasons = append(reasons, fmt.Sprintf("Exp<%.4f", effectiveMinExpect))
							}
							if valR.ProfitFactor < effectiveMinPF {
								reasons = append(reasons, fmt.Sprintf("PF<%.2f", effectiveMinPF))
							}
							if valR.Score < effectiveMinScore {
								reasons = append(reasons, fmt.Sprintf("Score<%.1f", effectiveMinScore))
							}
							reasonStr := strings.Join(reasons, ",")
							fmt.Printf("[VAL-REJECT] score=%.4f ret=%.2f%% pf=%.2f exp=%.5f dd=%.3f trds=%d [%s]\n",
								valR.Score, valR.Return*100, valR.ProfitFactor, valR.Expectancy, valR.MaxDD, valR.Trades, reasonStr)
						}

						if passesValidation {
							// Track best validation score seen ONLY from fully validated strategies
							// This prevents bestValSeen from tracking failing candidates
							bestValSeenMu.Lock()
							if valR.Trades >= minValTrades && valR.Score > -1e20 && valR.Score > bestValSeen {
								bestValSeen = valR.Score
							}
							bestValSeenMu.Unlock()

							// Run CPCV to check stability across multiple folds BEFORE adding to HOF/archive
							// CRITICAL FIX: Make CPCV forward-looking by covering TRAIN + VAL (up to valW.End)
							// This prevents "VAL-only illusions" where strategies fail in future market regimes
							// Skip CPCV during bootstrap mode (elites < 10) to allow population growth
							if !skipCPCV {
								cpcv := evaluateCPCV(series, feats, candidate.Strategy, trainW.Start, valW.End,
									int64(atomic.LoadUint64(&tested)), minValScore)
								// TEMP warm-start: allow slightly negative fold scores
								if cpcv.MinFoldScore < -0.2 {
									continue // reject truly unstable "lucky" strategy
								}
								// Dynamic stability threshold based on screen relax level (unblock mode support)
								// Level 3 (unblock): 0.30, Level 0-2: 0.66
								minStability := float32(0.66)
								if getScreenRelaxLevel() >= 3 {
									minStability = 0.30 // Relax during unblock mode
								}
								if !CPCVPassCriteria(cpcv, minValScore, minStability) {
									continue // reject unstable "lucky" strategy
								}
							}

							// CRITICAL FIX: Disable QuickTest entirely until elites > 0
							// QuickTest with Trds=0 is common early and shouldn't veto candidates
							quickTestProfitOK := true // Assume passes during bootstrap
							if elitesCount > 0 {
								// Only run QuickTest after we have at least one elite
								// Use first 25% of test window as "quick test" to filter out obvious failures
								quickTestEnd := testW.Start + (testW.End-testW.Start)/4
								if quickTestEnd > testW.End {
									quickTestEnd = testW.End
								}
								quickTestW := Window{
									Start:  testW.Start,
									End:    quickTestEnd,
									Warmup: testW.Warmup,
								}
								quickTestR := evaluateStrategyWindow(series, feats, candidate.Strategy, quickTestW)

								// Track best quick test result for reporting (best score among all candidates)
								if quickTestR.Score > bestQuickTestR.Score {
									bestQuickTestR = quickTestR
								}

								// BOOTSTRAP LADDER: Relax QuickTest during bootstrap (elites < 10)
								// When elite pool is small, allow candidates with weaker QuickTest results
								var minQuickTestTrades int
								var minQuickTestReturn float32
								if elitesCount < 10 {
									// Relaxed QuickTest gates during bootstrap
									minQuickTestTrades = 5  // Lower trade threshold for small test window
									minQuickTestReturn = -0.05 // Allow small loss during bootstrap
								} else {
									// Standard QuickTest gates once we have enough elites
									minQuickTestTrades = 15
									minQuickTestReturn = 0.0
								}

								// Skip QuickTest profit check if trades == 0
								// Small test window may have 0 trades even if full validation has trades
								if quickTestR.Trades > 0 {
									// CRITICAL FIX: Require positive edge explicitly, not relative to minValReturn
									// After elites exist, minValReturn could be 0.0, which doesn't protect against losing strategies
									quickTestProfitOK = quickTestR.Return >= minQuickTestReturn && // Use bootstrap-adjusted threshold
										quickTestR.ProfitFactor >= 1.0 && // Must be profitable after costs
										quickTestR.Trades >= minQuickTestTrades && // Use bootstrap-adjusted threshold
										quickTestR.MaxDD < maxValDD // Drawdown check
								}
								// If quickTestR.Trades == 0, skip the profit check (treat as "unknown")

								// NEAR-MISS LOGGING: Log QuickTest failures with reasons
								if !quickTestProfitOK && quickTestR.Trades > 0 {
									reasons := []string{}
									if quickTestR.Return < minQuickTestReturn {
										reasons = append(reasons, fmt.Sprintf("QT_ret<%.1f%%", minQuickTestReturn*100))
									}
									if quickTestR.ProfitFactor < 1.0 {
										reasons = append(reasons, fmt.Sprintf("QT_pf<%.2f", quickTestR.ProfitFactor))
									}
									if quickTestR.Trades < minQuickTestTrades {
										reasons = append(reasons, fmt.Sprintf("QT_trds<%d", minQuickTestTrades))
									}
									if quickTestR.MaxDD >= maxValDD {
										reasons = append(reasons, fmt.Sprintf("QT_dd>%.2f", maxValDD))
									}
									reasonStr := strings.Join(reasons, ",")
									fmt.Printf("[NEAR-MISS-QT] val_score=%.4f val_ret=%.2f%% QT_ret=%.2f%% QT_pf=%.2f QT_trds=%d QT_dd=%.3f reasons=%s\n",
										valR.Score, valR.Return*100, quickTestR.Return*100, quickTestR.ProfitFactor, quickTestR.Trades, quickTestR.MaxDD, reasonStr)
								}
							}

							if !quickTestProfitOK {
								continue // reject strategy that fails quick test gate
							}

							// Overtrading check: reject strategies with excessive trades that might be overfitted
							// Estimate trades per year based on val window (assuming 5min data = 105120 candles/year)
							valCandles := valW.End - valW.Start
							tradesPerYear := float32(valR.Trades) * (105120.0 / float32(valCandles))
							// BOOTSTRAP LADDER: Relax overtrading check during bootstrap (elites < 10)
							var maxTradesPerYear float32
							if elitesCount < 10 {
								maxTradesPerYear = 1000 // Allow more trades during bootstrap
							} else {
								maxTradesPerYear = 500 // Standard cap at 500 trades/year (~1 per day)
							}
							if tradesPerYear > maxTradesPerYear {
								// NEAR-MISS LOGGING: Log overtrading rejections
								fmt.Printf("[NEAR-MISS-OVER] val_score=%.4f trades/year=%.0f >%.0f val_trds=%d\n",
									valR.Score, tradesPerYear, maxTradesPerYear, valR.Trades)
								continue // reject overtrading strategies
							}

							passedThisBatch++                // Count passed strategies this batch for meta-update
							atomic.AddInt64(&passedCount, 1) // Track adaptive criteria
							elite := Elite{
								Strat:    candidate.Strategy,
								Train:    candidate,
								Val:      valR, // Use current candidate's validation result
								ValScore: valR.Score,
							}
							hof.Add(elite)
							archive.Add(valR, elite) // Use current candidate's validation result

							// CRITICAL FIX #2 & #3: Exit bootstrap mode once we have enough elites
							// This enables normal cooldown (200) and MaxHoldBars (150-329) values
							// and enables strict profit gates for elite acceptance
							const bootstrapEliteThreshold = 10
							if hof.Len() >= bootstrapEliteThreshold && isBootstrapMode() {
								setBootstrapMode(false)
								print("\n[BOOTSTRAP COMPLETE: Elites=%d, exiting bootstrap mode]\n", hof.Len())
								print("[Cooldown: 0-50 -> 200, MaxHoldBars: 50-150 -> 150-329, Profit gates: ENABLED]\n\n")
							}
							atomic.AddInt64(&strategiesPassed, 1) // Count strategies that truly pass all validation gates

							// Also log to persistent file
							log := EliteLog{
								Seed:         candidate.Strategy.Seed,
								FeeBps:       candidate.Strategy.FeeBps,
								SlippageBps:  candidate.Strategy.SlippageBps,
								Direction:    candidate.Strategy.Direction,
								StopLoss:     stopModelToString(candidate.Strategy.StopLoss),
								TakeProfit:   tpModelToString(candidate.Strategy.TakeProfit),
								Trail:        trailModelToString(candidate.Strategy.Trail),
								EntryRule:    ruleTreeToString(candidate.Strategy.EntryRule.Root),
								ExitRule:     ruleTreeToString(candidate.Strategy.ExitRule.Root),
								RegimeFilter: ruleTreeToString(candidate.Strategy.RegimeFilter.Root),
								TrainScore:   candidate.Score,
								TrainReturn:  candidate.Return,
								TrainMaxDD:   candidate.MaxDD,
								TrainWinRate: candidate.WinRate,
								TrainTrades:  candidate.Trades,
								ValScore:     valR.Score,
								ValReturn:    valR.Return,
								ValMaxDD:     valR.MaxDD,
								ValWinRate:   valR.WinRate,
								ValTrades:    valR.Trades,
							}
							select {
							case winnerLog <- log:
							default:
							}
						}
					}

					// After validation loop: run self-improvement meta-update
					// Compute actual tested since last checkpoint for correct pass-rate
					testedNow := int64(atomic.LoadUint64(&tested))
					testedThisCheckpoint := testedNow - testedAtLastCheckpoint
					if testedThisCheckpoint == 0 {
						testedThisCheckpoint = batchSize // fallback to default
					}
					updateMeta(improved, bestValR.Score, passedThisBatch, testedThisCheckpoint)
					testedAtLastCheckpoint = testedNow

					// Adaptive surrogate threshold: adjust based on validated labels and improvement
					if atomic.LoadInt64(&validatedLabels) > 5000 && improved {
						surThreshold += 0.01
					}
					if !improved {
						surThreshold -= 0.01
					}
					surThreshold = float64(clampf(float32(surThreshold), -0.20, 0.50))

					// Update stagnation counter for backward compatibility
					if improved {
						batchesSinceLastImprovement = 0
					} else {
						batchesSinceLastImprovement++
					}

					// Legacy adaptive exploration: keep as backup (meta-update now handles this)
					// Made stronger with reduced elite count (50 instead of 200)
					if batchesSinceLastImprovement >= 10 {
						if batchesSinceLastImprovement%10 == 0 {
							print(" [STAGNATION DETECTED: aggressive exploration enabled]")
						}
					}

					// Use best validated candidate for reporting (primary metric is val score)
					// Update line to show validation metrics as primary
					line.Score = bestValR.Score               // Report val score as primary
					line.Return = bestValR.Return             // Report val return
					line.MaxDD = bestValR.MaxDD               // Report val maxDD
					line.WinRate = bestValR.WinRate           // Report val winrate
					line.Expectancy = bestValR.Expectancy     // Report val expectancy
					line.ProfitFactor = bestValR.ProfitFactor // Report val profitfactor
					line.Trades = bestValR.Trades             // Report val trades
					// Keep batchBest as train result for reference but use val metrics for reporting
					batchBest = bestTrainResult
					line.ValScore = bestValR.Score
					line.ValReturn = bestValR.Return
					line.ValMaxDD = bestValR.MaxDD
					line.ValWinRate = bestValR.WinRate
					line.ValTrades = bestValR.Trades

					// Re-check stability for final pass/fail marker
					stableEnoughForFinal := true
					if bestTrainResult.Score > 0 {
						stableEnoughForFinal = bestValR.Score >= stabilityThreshold*bestTrainResult.Score
					}

					// CRITICAL FIX #3: Bootstrap elite acceptance path
					// When few elites exist (hof.Len() < 10), accept "least bad" strategies
					// that meet basic constraints (trades, DD) but NOT strict profit gates
					// This unblocks evolution: no elites -> accept least bad -> evolve -> then demand profit
					const bootstrapEliteThreshold = 10 // Number of elites to exit bootstrap mode
					isBootstrapPhase := hof.Len() < bootstrapEliteThreshold

					if isBootstrapPhase {
						// Bootstrap mode: Only require basic survival constraints
						// Accept top by ValScore that aren't completely broken
						bestPassesValidation = bestValR.MaxDD < maxValDD &&
							bestValR.Trades >= minValTrades &&
							bestValR.Return >= -0.02 && // Allow up to -2% loss (not -14%)
							stableEnoughForFinal // Keep stability check
						// Note: NO profit gate checks (ret>0, pf>1, exp>0) during bootstrap
					} else {
						// Normal mode: Require all profit gates
						bestPassesValidation = bestValR.Score > minValScore &&
							bestValR.MaxDD < maxValDD &&
							bestValR.Trades >= minValTrades &&
							stableEnoughForFinal &&
							bestValR.Return >= minValReturn &&
							bestValR.Expectancy > minValExpect &&
							bestValR.ProfitFactor >= minValPF
					}
					line.Passed = bestPassesValidation

					// Always write to file (with pass/fail marker)
					b, _ := json.Marshal(line)
					w.Write(b)
					w.WriteString("\n")
					w.Flush()

					now := time.Now()
					elapsed := now.Sub(lastReport).Seconds()
					rate := float64(batchSize) / elapsed
					lastReport = now

					// Print batch report with pass/fail indicator and fingerprint
					// Guard: only print if we have valid results
					if bestTrainResult.Trades > 0 {
						checkmark := map[bool]string{true: "✓", false: "✗"}[bestPassesValidation]
						fingerprint := bestTrainResult.Strategy.Fingerprint()

						// Build rejection reason string if failed
						reasonParts := []string{}
						if bestValR.Score <= minValScore {
							reasonParts = append(reasonParts, "score")
						}
						if bestValR.MaxDD >= maxValDD {
							reasonParts = append(reasonParts, "dd")
						}
						if bestValR.Trades < minValTrades {
							reasonParts = append(reasonParts, "trds")
						}
						if bestValR.Return < minValReturn {
							reasonParts = append(reasonParts, "ret")
						}
						if bestValR.Expectancy <= minValExpect {
							reasonParts = append(reasonParts, "exp")
						}
						if bestValR.ProfitFactor < minValPF {
							reasonParts = append(reasonParts, "pf")
						}
						if bestTrainResult.Score > 0 {
							stabilityRatio := float32(0)
							if bestTrainResult.Score > 0 {
								stabilityRatio = bestValR.Score / bestTrainResult.Score
							}
							if stabilityRatio < stabilityThreshold {
								reasonParts = append(reasonParts, "stab")
							}
						}

						// Format reason string
						reasonStr := ""
						if len(reasonParts) > 0 {
							reasonStr = " reason=" + strings.Join(reasonParts, ",")
						}

						// Check overtrading for reporting
						valCandles := valW.End - valW.Start
						tradesPerYear := float32(bestValR.Trades) * (105120.0 / float32(valCandles))
						maxTradesPerYear := float32(500)
						overtrading := tradesPerYear > maxTradesPerYear
						if overtrading && len(reasonParts) > 0 {
							reasonStr += ",ovtrd"
						} else if overtrading {
							reasonStr = " reason=ovtrd"
						}

						// Reject overtrading strategies from passing validation
						bestPassesValidation = bestPassesValidation && !overtrading

						// Build human-readable entry/exit story with feature names
						entryRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.EntryRule.Root, feats.Names)
						exitRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.ExitRule.Root, feats.Names)
						regimeRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.RegimeFilter.Root, feats.Names)

						// Print summary line
						print("Batch %d: Tested %d | Train: %.4f | Val: %.4f | Ret: %.2f%% | WR: %.1f%% | Trds: %d | Rate: %.0f/s %s [fp: %s]\n",
							batchID, atomic.LoadUint64(&tested), bestTrainResult.Score, bestValR.Score, bestValR.Return*100, bestValR.WinRate*100, bestValR.Trades, rate, checkmark, fingerprint)

						// Print criteria summary
						print("  [crit: score>%.2f, trds>=%d, DD<%.2f, ret>%.1f%%, exp>%.4f, pf>%.2f, stab>%.0f%%]\n",
							minValScore, minValTrades, maxValDD, minValReturn*100, minValExpect, minValPF, stabilityThreshold*100)

						// Print validation details with quick test
						print("  [val: score=%.4f dd=%.3f trds=%d%s]\n",
							bestValR.Score, bestValR.MaxDD, bestValR.Trades, reasonStr)
						print("  [QuickTest: Score=%.4f Ret=%.2f%% DD=%.3f Trds=%d]\n",
							bestQuickTestR.Score, bestQuickTestR.Return*100, bestQuickTestR.MaxDD, bestQuickTestR.Trades)

						// Print entry/exit story with feature names (indented)
						print("  [ENTRY STORY]\n")
						if regimeRuleNamed != "" && regimeRuleNamed != "(NOT )" {
							print("    Regime Filter (must be true): %s\n", formatIndentedRule(regimeRuleNamed, "    "))
						} else {
							print("    Regime Filter: (Always Active - No Filter)\n")
						}
						print("    Entry Signal (candle t close): %s\n", formatIndentedRule(entryRuleNamed, "    "))
						print("    Entry Execution (candle t+1 open): pending entry enters at next bar open\n")
						print("    Exit Signal: %s\n", formatIndentedRule(exitRuleNamed, "    "))

						// Print risk management
						print("  [RISK MANAGEMENT]\n")
						print("    StopLoss: %s\n", stopModelToString(bestTrainResult.Strategy.StopLoss))
						print("    TakeProfit: %s\n", tpModelToString(bestTrainResult.Strategy.TakeProfit))
						print("    Trail: %s\n", trailModelToString(bestTrainResult.Strategy.Trail))
						print("\n")
					}

					// Track in topK based on validation score (only if passes)
					if bestPassesValidation && bestValR.Score >= minTopScore {
						if len(topK) < cap(topK) {
							topK = append(topK, bestValR)
						} else {
							// Replace worst if better
							worstIdx := -1
							worstScore := bestValR.Score
							for i, r := range topK {
								if r.Score < worstScore {
									worstScore = r.Score
									worstIdx = i
								}
							}
							if worstIdx >= 0 && bestValR.Score > topK[worstIdx].Score {
								topK[worstIdx] = bestValR
							}
						}
					}

					batchBest = Result{Score: -1e30}

					// Auto-adjust screen relax level based on candidate count
					// CRITICAL FIX: Only enforce profit gates AFTER we have enough elites to support evolution
					// This prevents getting stuck in "valley" where gates are too strict for early evolution
					elitesCount := len(hof.Elites)
					const minElitesForProfitGates = 10 // Wait for at least 10 elites before enforcing strict gates
					if elitesCount > 0 {
						const minCandidatesPerCheckpoint = 5     // Minimum candidates before tightening
						const maxCandidatesBeforeTightening = 50 // Maximum candidates before forcing tightening
						candidateCount := len(globalTopCandidates)
						currentRelaxLevel := meta.ScreenRelaxLevel

						// Raise minValTrades from 20 to 30 once we have elites
						// CRITICAL FIX: Also update the local variable used in validation gates
						if meta.MinValTrades < 30 {
							meta.mu.Lock()
							meta.MinValTrades = 30
							newMinTrades := meta.MinValTrades
							meta.mu.Unlock()
							print(" [AUTO-ADJUST: Elites detected, MinValTrades=20->%d]\n", newMinTrades)
							minValTrades = newMinTrades // FIX: Update local variable too!
						}

						// CRITICAL FIX: Only enforce profit gates AFTER we have enough elites
						// This allows evolution to climb out of the valley before strict gates kick in
						if elitesCount >= minElitesForProfitGates && minValReturn < 0.02 {
							oldReturn := minValReturn
							minValReturn = 0.02
							print(" [AUTO-ADJUST: Enforcing profit gates, minValReturn=%.4f->%.2f%%]\n", oldReturn, minValReturn*100)
						}
						if elitesCount >= minElitesForProfitGates && minValPF < 1.05 {
							oldPF := minValPF
							minValPF = 1.05
							print(" [AUTO-ADJUST: Enforcing profit gates, minValPF=%.2f->%.2f]\n", oldPF, minValPF)
						}
						if elitesCount >= minElitesForProfitGates && maxValDD > 0.35 {
							oldDD := maxValDD
							maxValDD = 0.35
							print(" [AUTO-ADJUST: Enforcing profit gates, maxValDD=%.2f->%.2f]\n", oldDD, maxValDD)
						}
						if elitesCount >= minElitesForProfitGates && minValExpect < 0.0001 {
							oldExp := minValExpect
							minValExpect = 0.0001
							print(" [AUTO-ADJUST: Enforcing profit gates, minValExpect=%.4f->%.4f]\n", oldExp, minValExpect)
						}

						if candidateCount < minCandidatesPerCheckpoint && currentRelaxLevel < 3 {
							// Too few candidates - loosen gates
							meta.mu.Lock()
							meta.ScreenRelaxLevel = currentRelaxLevel + 1
							newLevel := meta.ScreenRelaxLevel
							meta.mu.Unlock()
							setScreenRelaxLevel(newLevel)
							print(" [AUTO-ADJUST: Too few candidates (%d), ScreenRelaxLevel=%d->%d]\n",
								candidateCount, currentRelaxLevel, newLevel)
						} else if candidateCount > maxCandidatesBeforeTightening && currentRelaxLevel > 0 {
							// Plenty of candidates - tighten gates gradually
							meta.mu.Lock()
							meta.ScreenRelaxLevel = currentRelaxLevel - 1
							newLevel := meta.ScreenRelaxLevel
							meta.mu.Unlock()
							setScreenRelaxLevel(newLevel)
							print(" [AUTO-ADJUST: Many candidates (%d), ScreenRelaxLevel=%d->%d]\n",
								candidateCount, currentRelaxLevel, newLevel)
						}
					}

					globalTopCandidates = globalTopCandidates[:0] // Reset to empty slice (keep capacity)
					globalFingerprints = make(map[string]bool)    // Reset fingerprints map
					nextCheckpoint += uint64(batchSize)
				}
			}
		}
	}()

	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func(id int) {
			defer wg.Done()
			batchSize := 32
			localBatch := make([]Strategy, 0, batchSize)
			localResults := make([]Result, 0, batchSize)

			for {
				select {
				case <-ctx.Done():
					if len(localBatch) > 0 {
						// Evaluate remaining strategies using multi-fidelity pipeline
						testedNow := int64(atomic.LoadUint64(&tested))
						for _, s := range localBatch {
							// Use multi-fidelity: screen -> train -> val
							passedScreen, passedFull, _, trainR, valR, _ := evaluateMultiFidelity(series, feats, s, screenW, trainW, valW, testedNow)
							if !passedScreen {
								continue // failed fast screen
							}
							if !passedFull {
								continue // failed train phase
							}
							// TEMP: Allow zero-trade strategies through for debugging
							if !allowZeroTrades && trainR.Trades == 0 {
								continue // no trades
							}
							// Store train result with val metrics embedded
							trainR.ValResult = &valR
							localResults = append(localResults, trainR)
						}
						// Sort by score and send top 20
						sort.Slice(localResults, func(i, j int) bool { return localResults[i].Score > localResults[j].Score })
						topN := min(20, len(localResults))
						// Copy the top results before sending to avoid race condition
						topCopy := make([]Result, topN)
						copy(topCopy, localResults[:topN])
						batchResults <- BatchMsg{TopN: topCopy, Count: int64(len(localBatch))}
					}
					return
				case s, ok := <-jobs:
					if !ok {
						return
					}
					localBatch = append(localBatch, s)
					if len(localBatch) >= batchSize {
						// Log batch start for debugging
						batchStart := time.Now()
						if Verbose {
							fmt.Printf("[Worker %d] Batch start: evaluating %d strategies\n", id, len(localBatch))
						}

						// Evaluate all strategies using multi-fidelity pipeline
						localResults = localResults[:0]
						testedNow := int64(atomic.LoadUint64(&tested))
						for _, strat := range localBatch {
							// Use multi-fidelity: screen -> train -> val
							passedScreen, passedFull, _, trainR, valR, _ := evaluateMultiFidelity(series, feats, strat, screenW, trainW, valW, testedNow)
							if !passedScreen {
								continue // failed fast screen
							}
							if !passedFull {
								continue // failed train phase
							}
							// TEMP: Allow zero-trade strategies through for debugging
							if !allowZeroTrades && trainR.Trades == 0 {
								continue // no trades
							}
							// Store train result with val metrics embedded
							// We'll use the train result for ranking, but validation will use valR
							trainR.ValResult = &valR // Store val result reference
							localResults = append(localResults, trainR)
						}
						// Sort by score and send top 20
						sort.Slice(localResults, func(i, j int) bool { return localResults[i].Score > localResults[j].Score })
						topN := min(20, len(localResults))

						// Log batch completion with timing
						batchDuration := time.Since(batchStart)
						if Verbose {
							fmt.Printf("[Worker %d] Batch done in %.2fs, kept %d results\n", id, batchDuration.Seconds(), len(localResults))
						}

						// Copy the top results before sending to avoid race condition
						topCopy := make([]Result, topN)
						copy(topCopy, localResults[:topN])
						batchResults <- BatchMsg{TopN: topCopy, Count: int64(batchSize)}
						localBatch = localBatch[:0]
					}
				}
			}
		}(i)
	}

	go func() {
		defer close(jobs)
		// Don't close winnerLog here - aggregator writes to it

		// Seed initial population from previous winners
		seeded := seedFromWinners()
		if len(seeded) > 0 {
			print("Seeding initial population with %d strategies from winners.jsonl...\n", len(seeded))
			for _, s := range seeded {
				if !markSeen(s) {
					continue
				}
				select {
				case <-ctx.Done():
					return
				case jobs <- s:
				}
			}
		}

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			var s Strategy

			// ratios for diversity injection (random immigrants approach)
			// Dynamic immigration based on stagnation:
			// - Base: 10% when improving
			// - Increase to 25% when stagnating (keeps exploration when needed)
			// adaptive: heavy mutations (big jumps to escape local maxima)
			// 20%: crossover (mix two parents)
			// remaining: normal mutation (small threshold tweaks)

			// Dynamic immigration based on stagnation
			meta.mu.RLock()
			stagnationCount := meta.Batches - meta.LastImprovementBatch
			meta.mu.RUnlock()

			immigrantP := float32(0.10) // Base: 10% when improving
			if stagnationCount > 3 {
				immigrantP = 0.25 // Increase to 25% when stagnating
			}

			// Get adaptive parameters (thread-safe)
			stagnationMu.RLock()
			currentRadicalP := radicalP
			stagnationMu.RUnlock()

			heavyMutP := immigrantP + 0.50*currentRadicalP // was + currentRadicalP
			crossP := heavyMutP + 0.20
			if crossP > 0.95 {
				crossP = 0.95
			} // keep at least 5% normal mutation
			// normal mutation = 1.0 - crossP

			x := rng.Float32()

			// Random immigrants: dynamic (10-25%) completely fresh strategies
			if hof.Len() == 0 || x < immigrantP {
				s = randomStrategyWithCosts(rng, feats, feeBps, slipBps)
				atomic.AddInt64(&immigrantCount, 1)
			} else if x < heavyMutP && hof.Len() > 0 {
				// Heavy mutation: big jump to escape local maxima (adaptive %)
				// Use archive sampling for diversity (30% chance)
				var parent Strategy
				if archive.Len() > 0 && rng.Float32() < 0.30 {
					archiveParent, _ := archive.Sample(rng)
					parent = archiveParent.Strat
				} else {
					hofParent, _ := hof.Sample(rng)
					parent = hofParent.Strat
				}
				s = bigMutation(rng, parent, feats)
				atomic.AddInt64(&heavyMutCount, 1)
			} else if x < crossP && hof.Len() >= 2 {
				// Crossover: mix two parents (20% total)
				var a, b Strategy
				// 40% chance to use archive for one parent to increase diversity
				if archive.Len() > 0 && rng.Float32() < 0.40 {
					archiveA, _ := archive.Sample(rng)
					a = archiveA.Strat
					hofB, _ := hof.Sample(rng)
					b = hofB.Strat
				} else {
					hofA, _ := hof.Sample(rng)
					hofB, _ := hof.Sample(rng)
					a = hofA.Strat
					b = hofB.Strat
				}
				s = crossover(rng, a, b)
				// small mutation after crossover helps
				if rng.Float32() < 0.7 {
					s = mutateStrategy(rng, s, feats)
				}
				atomic.AddInt64(&crossCount, 1)
			} else {
				// Normal mutation: small tweaks (remaining %)
				// Use uniform bin sampling for diversity
				useArchiveP := float32(0.30) // 30% of parents from archive
				if rng.Float32() < useArchiveP && archive.Len() > 0 {
					parent, _ := archive.Sample(rng)
					s = mutateStrategy(rng, parent.Strat, feats)
				} else {
					parent, ok := SampleUniformFromHOF(hof, archive, rng)
					if !ok {
						s = randomStrategyWithCosts(rng, feats, feeBps, slipBps)
					} else {
						s = mutateStrategy(rng, parent.Strat, feats)
					}
				}
				atomic.AddInt64(&normalMutCount, 1)
			}

			// COUNTER: Increment generated counter
			atomic.AddInt64(&genGenerated, 1)

			// Use surrogate to filter out obviously bad strategies
			// Meta-controller (in checkpoint logic) adjusts surThreshold based on improvement
			// No schedule-based computation - let meta drive it

			// Surrogate model filtering - CRITICAL for performance
			// The surrogate model filters out obviously bad strategies before full backtesting
			// Without this, the search must test every strategy with full backtests (10-100x slower)
			const useSurrogate = true

			// Warm-start: don't let surrogate block exploration until we have stable elites
			// When elites=0, the model can become an "everything is bad" bouncer
			// CRITICAL FIX: Delay surrogate until elites >= 20 to avoid premature filtering
			hof.mu.RLock()
			haveElites := len(hof.Elites) >= 20
			hof.mu.RUnlock()

			if useSurrogate && haveElites {
				surFeatures := ExtractSurFeatures(s)
				// Read current threshold (controlled by meta updates at checkpoints)
				stagnationMu.RLock()
				currentThreshold := surThreshold
				stagnationMu.RUnlock()

				if !sur.Accept(surFeatures, currentThreshold) {
					// COUNTER: Surrogate rejected
					atomic.AddInt64(&genRejectedSur, 1)
					continue // skip junk quickly
				}
			}

			// FAST FIX: Temporarily bypass markSeen filter to confirm it's not the killer
			// Check if we've already seen this strategy fingerprint
			// if !markSeen(s) {
			// 	continue
			// }
			isNew := markSeen(s)
			if !isNew {
				// COUNTER: markSeen rejected (duplicate)
				atomic.AddInt64(&genRejectedSeen, 1)
			}

			select {
			case <-ctx.Done():
				return
			case jobs <- s:
				// COUNTER: Sent to jobs
				atomic.AddInt64(&genSentToJobs, 1)
			}
		}
	}()

	print("Starting strategy search (press Ctrl+C to stop)...")
	print("")

	start := time.Now()
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()

		var last uint64
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				cur := atomic.LoadUint64(&tested)
				dt := cur - last
				last = cur

				// Calculate rate
				rate := float64(dt) / 10.0 // per second over 10s

				// Get current best validation score and HOF size
				hof.mu.RLock()
				currentBestVal := float32(0.0)
				if len(hof.Elites) > 0 {
					currentBestVal = hof.Elites[0].ValScore
				}
				elitesCount := len(hof.Elites)
				hof.mu.RUnlock()

				// Track best seen globally (for progress reporting) - thread-safe
				bestValSeenMu.Lock()
				if currentBestVal > bestValSeen {
					bestValSeen = currentBestVal
				}
				reportBestVal := bestValSeen
				bestValSeenMu.Unlock()

				stagnationMu.RLock()
				currentRadicalP := radicalP
				currentSurExploreP := surExploreP
				stagnationMu.RUnlock()

				meta.mu.RLock()
				currentBatches := meta.Batches
				meta.mu.RUnlock()

				// Get generator loop stats
				genStats := getGeneratorStats()

				print("[progress] tested=%d  rate=%.1f/s  elapsed=%s  bestVal=%.4f  elites=%d  [meta: radicalP=%.2f surExploreP=%.2f batches=%d]\n",
					cur, rate, time.Since(start).Truncate(time.Second), reportBestVal, elitesCount, currentRadicalP, currentSurExploreP, currentBatches)
				print("[generator] %s", genStats)
			}
		}
	}()

	// Generation type stats ticker (every 30s)
	go func() {
		t := time.NewTicker(30 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				imm := atomic.LoadInt64(&immigrantCount)
				hm := atomic.LoadInt64(&heavyMutCount)
				cr := atomic.LoadInt64(&crossCount)
				nm := atomic.LoadInt64(&normalMutCount)
				total := imm + hm + cr + nm
				if total > 0 {
					print("[gen-types] immigrant=%d(%.1f%%) heavyMut=%d(%.1f%%) crossover=%d(%.1f%%) normalMut=%d(%.1f%%) total=%d\n",
						imm, 100.0*float64(imm)/float64(total),
						hm, 100.0*float64(hm)/float64(total),
						cr, 100.0*float64(cr)/float64(total),
						nm, 100.0*float64(nm)/float64(total),
						total)
				}
			}
		}
	}()

	// Save checkpoint periodically
	go func() {
		t := time.NewTicker(time.Duration(*checkpointEverySec) * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				saveNow()
			}
		}
	}()

	// Save surrogate model periodically
	go func() {
		t := time.NewTicker(1 * time.Minute)
		defer t.Stop()

		for {
			select {
			case <-ctx.Done():
				// Save on shutdown
				if err := sur.Save("surrogate.json"); err != nil {
					print("Error saving surrogate: %v\n", err)
				}
				return
			case <-t.C:
				// Save every minute
				if err := sur.Save("surrogate.json"); err != nil {
					print("Error saving surrogate: %v\n", err)
				}
			}
		}
	}()

	// Shutdown sequence:
	// 1. wg.Wait() - wait for all worker goroutines to finish
	// 2. close(batchResults) - signal aggregator that no more results will come
	// 3. <-doneAgg - wait for aggregator goroutine to finish (it closes doneAgg)
	// 4. close(winnerLog) - close the output file channel
	wg.Wait()
	close(batchResults)
	<-doneAgg
	// Close winnerLog after aggregator is fully done
	close(winnerLog)

	print("\n\nStopped. Total tested: %d\n", atomic.LoadUint64(&tested))
	print("Results saved to: best_every_10000.txt")
}

func runTestMode(feeBps, slipBps float32) {
	fmt.Println("Running in TEST mode - evaluating saved winners on test data")
	fmt.Println("=============================================================")
	fmt.Printf("Cost overrides: Fee=%.1f bps (%.2f%%), Slippage=%.1f bps (%.2f%%)\n", feeBps, feeBps/100, slipBps, slipBps/100)
	fmt.Println()

	// Load data (same as search mode)
	fmt.Println("Loading data...")
	startTime := time.Now()

	series, err := LoadBinanceKlinesCSV("btc_5min_data.csv")
	if err != nil {
		fmt.Printf("Error loading CSV: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", series.T)

	fmt.Println("Computing features...")
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.F))

	// Split data into train/validation/test windows
	trainStart, trainEnd, valEnd := GetSplitIndices(series.OpenTimeMs)
	_, _, testW := GetSplitWindows(trainStart, trainEnd, valEnd, series.T, 500)

	// Recompute feature stats on train window only
	computeStatsOnWindow(&feats, trainStart, trainEnd)
	fmt.Printf("Feature stats computed on train window (%d candles)\n", trainEnd-trainStart)

	loadTime := time.Since(startTime)
	fmt.Printf("Data load time: %v\n\n", loadTime)

	// Load winners from winners.jsonl
	fmt.Println("Loading winners from winners.jsonl...")
	loadedLogs, err := loadRecentElites("winners.jsonl", 10000)
	if err != nil {
		fmt.Printf("Error loading winners: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d strategies from winners.jsonl\n\n", len(loadedLogs))

	// Test output file
	outFile, err := os.OpenFile("winners_tested.jsonl", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		fmt.Printf("Error opening output file: %v\n", err)
		return
	}
	defer outFile.Close()
	w := bufio.NewWriterSize(outFile, 1<<20)
	defer w.Flush()

	// Rebuild and test each strategy
	type TestResult struct {
		EliteLog
		TestScore    float32         `json:"test_score"`
		TestReturn   float32         `json:"test_return"`
		TestMaxDD    float32         `json:"test_max_dd"`
		TestWinRate  float32         `json:"test_win_rate"`
		TestTrades   int             `json:"test_trades"`
		TestExitReasons map[string]int `json:"test_exit_reasons"`
	}

	// Print test window info
	fmt.Printf("Test window: %d candles (indices %d -> %d)\n\n", testW.End-testW.Start, testW.Start, testW.End)

	var allResults []TestResult
	var parseErrors int
	var noTradeStrategies int

	// STRICT TEST FILTERS - These are the gates for passing test mode
	// Match production reality: Return>0, MaxDD<=0.35, Trades>=50
	minTrades := 50
	maxDD := float32(0.35)
	minReturn := float32(0.0)

	fmt.Printf("STRICT TEST GATES: Trades>=%d, MaxDD<=%.2f, Return>%.1f%%\n", minTrades, maxDD, minReturn*100)
	fmt.Println()

	fmt.Println("Evaluating strategies on test window...")
	for i, log := range loadedLogs {
		// Rebuild strategy from saved log
		var strategy Strategy
		var parseErr error

		// Use recover to catch any parsing panics
		func() {
			defer func() {
				if r := recover(); r != nil {
					parseErr = fmt.Errorf("parse panic: %v", r)
				}
			}()

			strategy = Strategy{
				Seed:        log.Seed,
				FeeBps:      feeBps,  // OVERRIDE: Use specified production fee for testing
				SlippageBps: slipBps, // OVERRIDE: Use specified production slippage for testing
				RiskPct:     0.01,
				Direction:   log.Direction,
				EntryRule: RuleTree{
					Root: parseRuleTree(log.EntryRule),
				},
				ExitRule: RuleTree{
					Root: parseRuleTree(log.ExitRule),
				},
				RegimeFilter: RuleTree{
					Root: parseRuleTree(log.RegimeFilter),
				},
				StopLoss:   parseStopModel(log.StopLoss),
				TakeProfit: parseTPModel(log.TakeProfit),
				Trail:      parseTrailModel(log.Trail),
			}
		}()

		// Skip strategies with parse errors
		if parseErr != nil || strategy.EntryRule.Root == nil || strategy.ExitRule.Root == nil {
			parseErrors++
			if parseErrors <= 5 {
				if parseErr != nil {
					fmt.Printf("Warning: Skipping strategy #%d due to error: %v\n", i+1, parseErr)
				} else {
					fmt.Printf("Warning: Skipping strategy #%d due to parse error\n", i+1)
				}
			}
			continue
		}

		// Compile bytecode (critical step!)
		strategy.EntryCompiled = compileRuleTree(strategy.EntryRule.Root)
		strategy.ExitCompiled = compileRuleTree(strategy.ExitRule.Root)
		strategy.RegimeCompiled = compileRuleTree(strategy.RegimeFilter.Root)

		// Evaluate on test window
		testR := evaluateStrategyWindow(series, feats, strategy, testW)

		// Track strategies with no trades
		if testR.Trades == 0 {
			noTradeStrategies++
		}

		// Create test result
		result := TestResult{
			EliteLog:       log,
			TestScore:      testR.Score,
			TestReturn:     testR.Return,
			TestMaxDD:      testR.MaxDD,
			TestWinRate:    testR.WinRate,
			TestTrades:     testR.Trades,
			TestExitReasons: testR.ExitReasons,
		}

		allResults = append(allResults, result)

		// Write to output file
		data, err := json.Marshal(result)
		if err == nil {
			w.Write(data)
			w.WriteString("\n")
		}

		if (i+1)%10 == 0 || i == len(loadedLogs)-1 {
			fmt.Printf("Progress: %d/%d strategies tested\n", i+1, len(loadedLogs))
		}
	}

	w.Flush()

	// Apply STRICT TEST FILTERS - Only pass strategies that meet production gates
	var passed []TestResult
	for _, r := range allResults {
		if r.TestTrades < minTrades {
			continue
		}
		if r.TestMaxDD > maxDD {
			continue
		}
		if r.TestReturn <= minReturn {
			continue
		}
		passed = append(passed, r)
	}

	// Rank PASSED strategies by TEST performance (not validation)
	// Primary: TestScore desc, Secondary: TestMaxDD asc, Tertiary: TestReturn desc
	sort.Slice(passed, func(i, j int) bool {
		if passed[i].TestScore != passed[j].TestScore {
			return passed[i].TestScore > passed[j].TestScore
		}
		if passed[i].TestMaxDD != passed[j].TestMaxDD {
			return passed[i].TestMaxDD < passed[j].TestMaxDD
		}
		return passed[i].TestReturn > passed[j].TestReturn
	})

	// ALSO print top 20 by TEST score even if they FAIL gates (debug visibility)
	// This helps identify best strategies even when passed=0
	// CRITICAL: Sort by Return+DD, NOT TestScore (TestScore is -1e30 sentinel for failed strategies)
	sort.Slice(allResults, func(i, j int) bool {
		// Primary sort: higher TestReturn
		if allResults[i].TestReturn != allResults[j].TestReturn {
			return allResults[i].TestReturn > allResults[j].TestReturn
		}
		// Secondary sort: lower TestMaxDD
		if allResults[i].TestMaxDD != allResults[j].TestMaxDD {
			return allResults[i].TestMaxDD < allResults[j].TestMaxDD
		}
		// Tertiary sort: higher TestTrades
		return allResults[i].TestTrades > allResults[j].TestTrades
	})

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TOP 20 BY TEST RETURN (NO GATES) — DEBUG VISIBILITY")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Shows best strategies on TEST even if they fail validation gates")
	fmt.Println("Sorted by: Return (desc) → DD (asc) → Trades (desc)")
	fmt.Println()

	numToPrintDebug := 20
	if len(allResults) < numToPrintDebug {
		numToPrintDebug = len(allResults)
	}

	if numToPrintDebug == 0 {
		fmt.Println("No strategies evaluated on test data")
	} else {
		for i := 0; i < numToPrintDebug; i++ {
			r := allResults[i]
			fmt.Printf("#%2d | TestScore: %.4f | Ret: %6.2f%% | DD: %5.1f%% | Trades: %4d | ValScore: %.4f\n",
				i+1, r.TestScore, r.TestReturn*100, r.TestMaxDD*100, r.TestTrades, r.ValScore)
		}
	}

	// Print top 20 leaderboard (PASSED strategies only)
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TOP 20 STRATEGIES BY TEST SCORE (STRICT FILTERS APPLIED)")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("Gates: Trades>=%d, MaxDD<=%.2f, Return>%.1f%%\n", minTrades, maxDD, minReturn*100)
	fmt.Println()

	numToPrint := 20
	if len(passed) < numToPrint {
		numToPrint = len(passed)
	}

	if numToPrint == 0 {
		fmt.Println("NO STRATEGIES PASSED THE STRICT TEST GATES!")
		fmt.Println()
		fmt.Println("Try adjusting gates: -mode test -fee_bps=30 -slip_bps=8")
	} else {
		for i := 0; i < numToPrint; i++ {
			r := passed[i]
			fmt.Printf("#%2d | Test Score: %.4f | Test Return: %6.2f%% | Test WinRate: %5.1f%% | Test Trades: %4d | Test DD: %.2f%% | Val Score: %.4f\n",
				i+1, r.TestScore, r.TestReturn*100, r.TestWinRate*100, r.TestTrades, r.TestMaxDD*100, r.ValScore)
		}
	}

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Printf("Total strategies loaded:     %d\n", len(loadedLogs))
	fmt.Printf("Tested successfully:         %d\n", len(allResults)-parseErrors)
	fmt.Printf("Skipped parse errors:        %d\n", parseErrors)
	fmt.Printf("No-trade on test:            %d\n", noTradeStrategies)
	fmt.Printf("Strategies with trades:      %d\n", len(allResults))
	fmt.Printf("PASSED strict test gates:    %d (%.1f%%)\n", len(passed), float32(len(passed))*100/float32(len(allResults)))
	fmt.Println(strings.Repeat("-", 80))
	fmt.Println("Results saved to: winners_tested.jsonl")

	// Aggregate exit reasons across all passing strategies
	if len(passed) > 0 {
		fmt.Println("\n" + strings.Repeat("=", 80))
		fmt.Println("AGGREGATE EXIT REASONS (Passing Strategies)")
		fmt.Println(strings.Repeat("=", 80))

		totalExitReasons := make(map[string]int)
		for _, r := range passed {
			for reason, count := range r.TestExitReasons {
				totalExitReasons[reason] += count
			}
		}

		// Print exit reason breakdown
		totalTrades := 0
		for _, count := range totalExitReasons {
			totalTrades += count
		}

		if totalTrades > 0 {
			fmt.Println("\nExit Reason Breakdown:")
			// Sort by count descending
			type reasonCount struct {
				reason string
				count  int
			}
			var sorted []reasonCount
			for reason, count := range totalExitReasons {
				sorted = append(sorted, reasonCount{reason, count})
			}
			sort.Slice(sorted, func(i, j int) bool {
				return sorted[i].count > sorted[j].count
			})
			for _, rc := range sorted {
				fmt.Printf("  %s: %d (%.1f%%)\n", rc.reason, rc.count, float32(rc.count)*100/float32(totalTrades))
			}
			fmt.Printf("\nTotal Trades: %d\n", totalTrades)
		}
		fmt.Println(strings.Repeat("=", 80))
	}

	// Create detailed text files for TOP 20 passing strategies only
	fmt.Println("\nCreating detailed strategy files...")
	os.MkdirAll("test_results", 0755)
	numFiles := 20
	if len(passed) < numFiles {
		numFiles = len(passed)
	}
	for i := 0; i < numFiles; i++ {
		r := passed[i]
		createStrategyDetailsFile(i+1, &r.EliteLog, r.TestScore, r.TestReturn, r.TestMaxDD, r.TestWinRate, r.TestTrades, r.ValScore, r.ValReturn, r.TrainScore, r.TrainReturn, feats.Names, series, feeBps, slipBps, r.Direction)
	}
	fmt.Printf("Created %d detailed strategy files in test_results/ folder (top %d only)\n", numFiles, numFiles)
}

// createStrategyDetailsFile creates a detailed text file for a single strategy
func createStrategyDetailsFile(rank int, log *EliteLog, testScore, testReturn, testMaxDD, testWinRate float32, testTrades int, valScore, valReturn, trainScore, trainReturn float32, featureNames []string, series Series, feeBps, slipBps float32, direction int) {
	var buf strings.Builder

	// Header
	buf.WriteString("=" + strings.Repeat("=", 78) + "\n")
	buf.WriteString(fmt.Sprintf("STRATEGY #%d - TEST RESULTS\n", rank))
	buf.WriteString("=" + strings.Repeat("=", 78) + "\n\n")

	// Test Results Summary
	buf.WriteString("TEST PERFORMANCE METRICS\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString(fmt.Sprintf("  Test Score:      %.4f\n", testScore))
	buf.WriteString(fmt.Sprintf("  Test Return:     %.2f%%\n", testReturn*100))
	buf.WriteString(fmt.Sprintf("  Test Win Rate:   %.1f%%\n", testWinRate*100))
	buf.WriteString(fmt.Sprintf("  Test Trades:     %d\n", testTrades))
	buf.WriteString(fmt.Sprintf("  Test Max DD:     %.2f%%\n", testMaxDD*100))
	buf.WriteString(fmt.Sprintf("  Validation Score: %.4f\n", valScore))
	buf.WriteString(fmt.Sprintf("  Validation Ret:   %.2f%%\n", valReturn*100))
	buf.WriteString(fmt.Sprintf("  Train Score:      %.4f\n", trainScore))
	buf.WriteString(fmt.Sprintf("  Train Return:     %.2f%%\n", trainReturn*100))
	buf.WriteString("\n")

	// Strategy Parameters
	buf.WriteString("STRATEGY PARAMETERS\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString(fmt.Sprintf("  Direction:       %s\n", directionToString(direction)))
	buf.WriteString(fmt.Sprintf("  Fee BPS:         %.1f (%.3f%%)\n", feeBps, feeBps/100))
	buf.WriteString(fmt.Sprintf("  Slippage BPS:    %.1f (%.3f%%)\n", slipBps, slipBps/100))
	buf.WriteString(fmt.Sprintf("  Risk Per Trade:  %.1f%%\n", 1.0))
	buf.WriteString(fmt.Sprintf("  Seed:            %d\n", log.Seed))
	buf.WriteString("\n")

	// Risk Management
	buf.WriteString("RISK MANAGEMENT\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString(fmt.Sprintf("  Stop Loss:       %s\n", log.StopLoss))
	buf.WriteString(fmt.Sprintf("  Take Profit:     %s\n", log.TakeProfit))
	buf.WriteString(fmt.Sprintf("  Trailing Stop:   %s\n", formatTrail(log.Trail)))
	buf.WriteString("\n")

	// Entry Rule
	buf.WriteString("ENTRY CONDITIONS\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString("Entry Rule:\n")
	entryRule := formatRuleWithNames(log.EntryRule, featureNames)
	buf.WriteString(formatIndentedRule(entryRule, "  "))
	buf.WriteString("\n")

	// Exit Rule
	buf.WriteString("EXIT CONDITIONS\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString("Exit Rule:\n")
	exitRule := formatRuleWithNames(log.ExitRule, featureNames)
	buf.WriteString(formatIndentedRule(exitRule, "  "))
	buf.WriteString("\n")

	// Regime Filter
	buf.WriteString("REGIME FILTER\n")
	buf.WriteString("-" + strings.Repeat("-", 78) + "\n")
	buf.WriteString("Regime Filter:\n")
	regimeRule := formatRuleWithNames(log.RegimeFilter, featureNames)
	buf.WriteString(formatIndentedRule(regimeRule, "  "))
	buf.WriteString("\n")

	// Footer
	buf.WriteString("=" + strings.Repeat("=", 78) + "\n")
	buf.WriteString("Generated by HB Backtest Strategy Search Engine\n")
	buf.WriteString(fmt.Sprintf("Data: %d candles from %s to %s\n", series.T,
		formatTimestamp(series.OpenTimeMs[0]), formatTimestamp(series.OpenTimeMs[series.T-1])))
	buf.WriteString("=" + strings.Repeat("=", 78) + "\n")

	// Write to file
	filename := fmt.Sprintf("test_results/strategy_%03d_rank_%.0f.txt", rank, testScore)
	err := os.WriteFile(filename, []byte(buf.String()), 0644)
	if err != nil {
		fmt.Printf("Warning: Could not write file %s: %v\n", filename, err)
	}
}

// Helper functions for formatting
func directionToString(d int) string {
	if d == 1 {
		return "LONG ONLY"
	}
	return "SHORT ONLY"
}

func formatTrail(trail string) string {
	if trail == "" || trail == "none" {
		return "Disabled"
	}
	return trail
}

func formatRuleWithNames(rule string, featureNames []string) string {
	// Replace feature indices with names
	result := rule
	for i, name := range featureNames {
		// Replace F[i] with the actual feature name
		result = strings.ReplaceAll(result, fmt.Sprintf("F[%d]", i), name)
	}
	return result
}

func formatIndentedRule(rule string, indent string) string {
	if rule == "" || rule == "nil" {
		return indent + "(Always Active - No Filter)"
	}
	lines := strings.Split(rule, "\n")
	for i, line := range lines {
		lines[i] = indent + line
	}
	return strings.Join(lines, "\n")
}

func formatTimestamp(ms int64) string {
	t := time.Unix(ms/1000, 0)
	return t.Format("2006-01-02 15:04:05")
}

func ruleTreeToString(node *RuleNode) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		return leafToString(&node.Leaf)
	}

	leftStr := ruleTreeToString(node.L)
	rightStr := ruleTreeToString(node.R)

	switch node.Op {
	case OpAnd:
		return fmt.Sprintf("(AND %s %s)", leftStr, rightStr)
	case OpOr:
		return fmt.Sprintf("(OR %s %s)", leftStr, rightStr)
	case OpNot:
		return fmt.Sprintf("(NOT %s)", leftStr)
	default:
		return ""
	}
}

func leafToString(leaf *Leaf) string {
	kindNames := map[LeafKind]string{
		LeafGT:        "GT",
		LeafLT:        "LT",
		LeafCrossUp:   "CrossUp",
		LeafCrossDown: "CrossDown",
		LeafRising:    "Rising",
		LeafFalling:   "Falling",
		LeafBetween:   "Between",
		LeafAbsGT:     "AbsGT",
		LeafAbsLT:     "AbsLT",
		LeafSlopeGT:   "SlopeGT",
		LeafSlopeLT:   "SlopeLT",
	}
	kindName, ok := kindNames[leaf.Kind]
	if !ok {
		kindName = "UNKNOWN"
	}

	switch leaf.Kind {
	case LeafGT, LeafLT:
		return fmt.Sprintf("(%s F[%d] %.2f)", kindName, leaf.A, leaf.X)
	case LeafBetween:
		return fmt.Sprintf("(%s F[%d] %.2f %.2f)", kindName, leaf.A, leaf.X, leaf.Y)
	case LeafAbsGT, LeafAbsLT:
		return fmt.Sprintf("(%s F[%d] %.2f)", kindName, leaf.A, leaf.X)
	case LeafSlopeGT, LeafSlopeLT:
		return fmt.Sprintf("(%s F[%d] %.2f %d)", kindName, leaf.A, leaf.X, leaf.Lookback)
	case LeafCrossUp, LeafCrossDown:
		return fmt.Sprintf("(%s F[%d] F[%d])", kindName, leaf.A, leaf.B)
	case LeafRising, LeafFalling:
		return fmt.Sprintf("(%s F[%d] %d)", kindName, leaf.A, leaf.Lookback)
	default:
		return fmt.Sprintf("(%s F[%d])", kindName, leaf.A)
	}
}

// leafToStringNamed returns a string representation of a leaf with feature names
func leafToStringNamed(leaf *Leaf, names []string) string {
	kindNames := map[LeafKind]string{
		LeafGT:        "GT",
		LeafLT:        "LT",
		LeafCrossUp:   "CrossUp",
		LeafCrossDown: "CrossDown",
		LeafRising:    "Rising",
		LeafFalling:   "Falling",
		LeafBetween:   "Between",
		LeafAbsGT:     "AbsGT",
		LeafAbsLT:     "AbsLT",
		LeafSlopeGT:   "SlopeGT",
		LeafSlopeLT:   "SlopeLT",
	}
	kindName, ok := kindNames[leaf.Kind]
	if !ok {
		kindName = "UNKNOWN"
	}

	nameA := fmt.Sprintf("F[%d]", leaf.A)
	if leaf.A >= 0 && leaf.A < len(names) {
		nameA = fmt.Sprintf("%s(F[%d])", names[leaf.A], leaf.A)
	}

	nameB := fmt.Sprintf("F[%d]", leaf.B)
	if leaf.B >= 0 && leaf.B < len(names) {
		nameB = fmt.Sprintf("%s(F[%d])", names[leaf.B], leaf.B)
	}

	switch leaf.Kind {
	case LeafGT, LeafLT:
		return fmt.Sprintf("(%s %s %.2f)", kindName, nameA, leaf.X)
	case LeafBetween:
		return fmt.Sprintf("(%s %s %.2f %.2f)", kindName, nameA, leaf.X, leaf.Y)
	case LeafAbsGT, LeafAbsLT:
		return fmt.Sprintf("(%s %s %.2f)", kindName, nameA, leaf.X)
	case LeafSlopeGT, LeafSlopeLT:
		return fmt.Sprintf("(%s %s %.2f %d)", kindName, nameA, leaf.X, leaf.Lookback)
	case LeafCrossUp, LeafCrossDown:
		return fmt.Sprintf("(%s %s %s)", kindName, nameA, nameB)
	case LeafRising, LeafFalling:
		return fmt.Sprintf("(%s %s %d)", kindName, nameA, leaf.Lookback)
	default:
		return fmt.Sprintf("(%s %s)", kindName, nameA)
	}
}

// ruleTreeToStringNamed returns a string representation of a rule tree with feature names
func ruleTreeToStringNamed(node *RuleNode, names []string) string {
	if node == nil {
		return ""
	}

	if node.Op == OpLeaf {
		return leafToStringNamed(&node.Leaf, names)
	}

	leftStr := ruleTreeToStringNamed(node.L, names)
	rightStr := ruleTreeToStringNamed(node.R, names)

	switch node.Op {
	case OpAnd:
		return fmt.Sprintf("(AND %s %s)", leftStr, rightStr)
	case OpOr:
		return fmt.Sprintf("(OR %s %s)", leftStr, rightStr)
	case OpNot:
		return fmt.Sprintf("(NOT %s)", leftStr)
	default:
		return ""
	}
}

func stopModelToString(sm StopModel) string {
	switch sm.Kind {
	case "fixed":
		return fmt.Sprintf("Fixed %.2f%%", sm.Value)
	case "atr":
		return fmt.Sprintf("ATR*%.2f", sm.ATRMult)
	case "swing":
		return fmt.Sprintf("Swing[%d]", sm.SwingIdx)
	default:
		return sm.Kind
	}
}

func tpModelToString(tp TPModel) string {
	switch tp.Kind {
	case "fixed":
		return fmt.Sprintf("Fixed %.2f%%", tp.Value)
	case "atr":
		return fmt.Sprintf("ATR*%.2f", tp.ATRMult)
	default:
		return tp.Kind
	}
}

func trailModelToString(tm TrailModel) string {
	if !tm.Active {
		return "none"
	}
	switch tm.Kind {
	case "atr":
		return fmt.Sprintf("ATR*%.2f", tm.ATRMult)
	case "swing":
		return "swing"
	default:
		return tm.Kind
	}
}

func parseRuleTree(s string) *RuleNode {
	s = strings.TrimSpace(s)
	if s == "" || s == "(NOT )" {
		return nil
	}

	// Check for leaf pattern: (KIND F[idx] value) or (KIND F[idx] F[idx2]) or (KIND F[idx] lookback)
	if !strings.HasPrefix(s, "(AND ") && !strings.HasPrefix(s, "(OR ") && !strings.HasPrefix(s, "(NOT ") {
		// It's a leaf
		return parseLeaf(s)
	}

	// It's a composite node
	if strings.HasPrefix(s, "(NOT ") {
		// Safety: ensure string is long enough
		if len(s) < 7 {
			return nil
		}
		innerStr := strings.TrimSpace(s[5 : len(s)-1])
		return &RuleNode{Op: OpNot, L: parseRuleTree(innerStr), R: nil}
	}

	spaceAfterOp := strings.Index(s[1:], " ") + 1
	if spaceAfterOp <= 1 {
		return nil
	}
	opStr := s[1:spaceAfterOp]

	// Find balanced parentheses for left subtree
	leftStart := spaceAfterOp + 1
	leftEnd := findMatchingParen(s, leftStart)
	if leftEnd < 0 {
		return nil // Guard: if no matching paren found, parsing failed
	}

	// Safety: clamp indices to string bounds
	if leftStart < 0 {
		leftStart = 0
	}
	if leftEnd > len(s) {
		leftEnd = len(s)
	}
	if leftStart > leftEnd {
		return nil
	}
	leftStr := strings.TrimSpace(s[leftStart : leftEnd+1])

	// Right subtree is from leftEnd+1 to len(s)-1
	var rightStr string
	rightStart := leftEnd + 1
	rightEnd := len(s) - 1
	if rightStart < 0 {
		rightStart = 0
	}
	if rightEnd > len(s) {
		rightEnd = len(s)
	}
	if rightStart > rightEnd {
		rightStr = ""
	} else {
		rightStr = strings.TrimSpace(s[rightStart:rightEnd])
	}

	var op Op
	if opStr == "AND" {
		op = OpAnd
	} else if opStr == "OR" {
		op = OpOr
	} else {
		return nil
	}

	return &RuleNode{
		Op: op,
		L:  parseRuleTree(leftStr),
		R:  parseRuleTree(rightStr),
	}
}

func findMatchingParen(s string, start int) int {
	count := 0 // Start at 0 since we begin scanning after the opening paren of left subtree
	for i := start; i < len(s); i++ {
		if s[i] == '(' {
			count++
		} else if s[i] == ')' {
			count--
			if count == 0 {
				return i
			}
		}
	}
	return -1 // Guard: return -1 if no matching paren found
}

func parseLeaf(s string) *RuleNode {
	s = strings.TrimSpace(s[1 : len(s)-1]) // Remove outer parens

	// Split by space, but only first 2-4 parts matter
	parts := strings.Fields(s)
	if len(parts) < 2 {
		return nil
	}

	kindStr := parts[0]

	var kind LeafKind
	switch kindStr {
	case "GT":
		kind = LeafGT
	case "LT":
		kind = LeafLT
	case "Between":
		kind = LeafBetween
	case "AbsGT":
		kind = LeafAbsGT
	case "AbsLT":
		kind = LeafAbsLT
	case "SlopeGT":
		kind = LeafSlopeGT
	case "SlopeLT":
		kind = LeafSlopeLT
	case "CrossUp":
		kind = LeafCrossUp
	case "CrossDown":
		kind = LeafCrossDown
	case "Rising":
		kind = LeafRising
	case "Falling":
		kind = LeafFalling
	default:
		return nil
	}

	// Parse feature indices
	featureA := -1
	featureB := -1
	threshold := float32(0)
	leafY := float32(0)
	lookback := 0

	// Parse F[idx] from feature string
	if len(parts) >= 2 && strings.HasPrefix(parts[1], "F[") {
		fmt.Sscanf(parts[1], "F[%d]", &featureA)
	}

	// For GT/LT/AbsGT/AbsLT, we have a threshold
	if (kind == LeafGT || kind == LeafLT || kind == LeafAbsGT || kind == LeafAbsLT) && len(parts) >= 3 {
		fmt.Sscanf(parts[2], "%f", &threshold)
	}

	// For Between, we have two thresholds
	if kind == LeafBetween && len(parts) >= 4 {
		fmt.Sscanf(parts[2], "%f", &threshold)
		fmt.Sscanf(parts[3], "%f", &leafY)
	}

	// For SlopeGT/SlopeLT, we have threshold and lookback
	if (kind == LeafSlopeGT || kind == LeafSlopeLT) && len(parts) >= 4 {
		fmt.Sscanf(parts[2], "%f", &threshold)
		fmt.Sscanf(parts[3], "%d", &lookback)
	}

	// For CrossUp/CrossDown, we have two features
	if (kind == LeafCrossUp || kind == LeafCrossDown) && len(parts) >= 3 && strings.HasPrefix(parts[2], "F[") {
		fmt.Sscanf(parts[2], "F[%d]", &featureB)
	}

	// For Rising/Falling, we have lookback
	if (kind == LeafRising || kind == LeafFalling) && len(parts) >= 3 {
		fmt.Sscanf(parts[2], "%d", &lookback)
	}

	return &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind:     kind,
			A:        featureA,
			B:        featureB,
			X:        threshold,
			Y:        leafY,
			Lookback: lookback,
		},
	}
}

func parseStopModel(s string) StopModel {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "Fixed ") {
		var val float32
		fmt.Sscanf(s[6:], "%f", &val)
		return StopModel{Kind: "fixed", Value: val}
	}
	if strings.HasPrefix(s, "ATR*") {
		var mult float32
		fmt.Sscanf(s[4:], "%f", &mult)
		return StopModel{Kind: "atr", ATRMult: mult}
	}
	if strings.HasPrefix(s, "Swing[") {
		var idx int
		fmt.Sscanf(s[6:], "%d", &idx)
		return StopModel{Kind: "swing", SwingIdx: idx}
	}
	return StopModel{Kind: "fixed", Value: 2.0}
}

func parseTPModel(s string) TPModel {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "Fixed ") {
		var val float32
		fmt.Sscanf(s[6:], "%f", &val)
		return TPModel{Kind: "fixed", Value: val}
	}
	if strings.HasPrefix(s, "ATR*") {
		var mult float32
		fmt.Sscanf(s[4:], "%f", &mult)
		return TPModel{Kind: "atr", ATRMult: mult}
	}
	return TPModel{Kind: "fixed", Value: 4.0}
}

func parseTrailModel(s string) TrailModel {
	s = strings.TrimSpace(s)
	if s == "none" {
		return TrailModel{Active: false}
	}
	if strings.HasPrefix(s, "ATR*") {
		var mult float32
		fmt.Sscanf(s[4:], "%f", &mult)
		return TrailModel{Kind: "atr", ATRMult: mult, Active: true}
	}
	if s == "swing" {
		return TrailModel{Kind: "swing", Active: true}
	}
	return TrailModel{Active: false}
}

// WinnerRow represents a single winner from winners.jsonl or winners_tested.jsonl
type WinnerRow struct {
	Seed         int64   `json:"seed"`
	FeeBps       float32 `json:"fee_bps"`
	SlippageBps  float32 `json:"slippage_bps"`
	Direction    int     `json:"direction"`
	StopLoss     string  `json:"stop_loss"`
	TakeProfit   string  `json:"take_profit"`
	Trail        string  `json:"trail"`
	EntryRule    string  `json:"entry_rule"`
	ExitRule     string  `json:"exit_rule"`
	RegimeFilter string  `json:"regime_filter"`
	TrainScore   float32 `json:"train_score,omitempty"`
	ValScore     float32 `json:"val_score,omitempty"`
	TestScore    float32 `json:"test_score,omitempty"`
}

// loadWinnerBySeed loads a specific winner by seed from winners_tested.jsonl or winners.jsonl
func loadWinnerBySeed(path string, seed int64) (*WinnerRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var w WinnerRow
		if err := json.Unmarshal(sc.Bytes(), &w); err != nil {
			continue
		}
		if w.Seed == seed {
			return &w, nil
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("seed %d not found in %s", seed, path)
}

// loadFirstWinner loads the first (best) winner from the file
func loadFirstWinner(path string) (*WinnerRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	if sc.Scan() {
		var w WinnerRow
		if err := json.Unmarshal(sc.Bytes(), &w); err != nil {
			return nil, err
		}
		return &w, nil
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no winners found in %s", path)
}

// runGoldenMode runs a single winner strategy with detailed trade logging
func runGoldenMode(seed int64, printTrades int, feeBps, slipBps float64) {
	fmt.Println("Running in GOLDEN mode - single strategy with trade logging")
	fmt.Println("================================================================")
	fmt.Printf("Seed: %d, FeeBps: %.1f, SlippageBps: %.1f\n", seed, feeBps, slipBps)
	fmt.Println()

	// Load data (same as test mode)
	fmt.Println("Loading data...")
	series, err := LoadBinanceKlinesCSV("btc_5min_data.csv")
	if err != nil {
		fmt.Printf("Error loading CSV: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", series.T)

	fmt.Println("Computing features...")
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.F))

	// Split data into train/validation/test windows
	trainStart, trainEnd, valEnd := GetSplitIndices(series.OpenTimeMs)
	_, _, testW := GetSplitWindows(trainStart, trainEnd, valEnd, series.T, 500)

	// Recompute feature stats on train window only
	computeStatsOnWindow(&feats, trainStart, trainEnd)
	fmt.Printf("Feature stats computed on train window (%d candles)\n", trainEnd-trainStart)

	fmt.Printf("\nTest window: %d candles (indices %d -> %d)\n\n", testW.End-testW.Start, testW.Start, testW.End)

	// Load winner from winners_tested.jsonl or winners.jsonl
	var w *WinnerRow
	if seed > 0 {
		w, err = loadWinnerBySeed("winners_tested.jsonl", seed)
		if err != nil {
			// Try winners.jsonl as fallback
			w, err = loadWinnerBySeed("winners.jsonl", seed)
			if err != nil {
				fmt.Printf("Error loading winner with seed %d: %v\n", seed, err)
				return
			}
		}
		fmt.Printf("Loaded winner with seed %d from tested results\n", seed)
	} else {
		w, err = loadFirstWinner("winners_tested.jsonl")
		if err != nil {
			// Try winners.jsonl as fallback
			w, err = loadFirstWinner("winners.jsonl")
			if err != nil {
				fmt.Printf("Error loading first winner: %v\n", err)
				return
			}
		}
		fmt.Printf("Loaded first winner (seed=%d) from tested results\n", w.Seed)
		seed = w.Seed
	}

	// Rebuild strategy from winner
	st := Strategy{
		Seed:        w.Seed,
		FeeBps:      float32(feeBps),
		SlippageBps: float32(slipBps),
		RiskPct:     0.01,
		Direction:   w.Direction,
		EntryRule: RuleTree{
			Root: parseRuleTree(w.EntryRule),
		},
		ExitRule: RuleTree{
			Root: parseRuleTree(w.ExitRule),
		},
		RegimeFilter: RuleTree{
			Root: parseRuleTree(w.RegimeFilter),
		},
		StopLoss:   parseStopModel(w.StopLoss),
		TakeProfit: parseTPModel(w.TakeProfit),
		Trail:      parseTrailModel(w.Trail),
	}

	// Compile bytecode
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	// Run backtest with trade logging on TEST window
	fmt.Println("\nRunning backtest on TEST window...")
	result := evaluateStrategyWithTrades(series, feats, st, testW, false)

	// Write states to CSV for debugging
	fmt.Println("\nWriting per-bar states to states.csv...")
	if err := WriteStatesToCSV(result.States, "states.csv"); err != nil {
		fmt.Printf("Error writing states CSV: %v\n", err)
	} else {
		fmt.Printf("States written to states.csv (%d bars)\n", len(result.States))
	}

	// Print golden results
	printGoldenResult(result, printTrades)
}

// printGoldenResult prints detailed trade information for golden mode
func printGoldenResult(result GoldenResult, printTrades int) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("GOLDEN MODE RESULTS")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("\nStrategy Summary:\n")
	fmt.Printf("  Total Trades:     %d\n", result.TotalTrades)
	fmt.Printf("  Return (raw):     %.2f%%  [matches HB reports/winner files]\n", result.RawReturnPct*100)
	fmt.Printf("  Return (risk-adj): %.2f%%  [uses RiskPct=1%%]\n", result.ReturnPct*100)
	fmt.Printf("  Max DD:           %.2f%%  [mark-to-market, raw]\n", result.MaxDDPct*100)
	fmt.Printf("  Win Rate:         %.2f%%\n", result.WinRate*100)
	fmt.Printf("  Expectancy:       %.4f\n", result.Expectancy)
	fmt.Printf("  Profit Factor:    %.2f\n", result.ProfitFactor)

	if len(result.Trades) == 0 {
		fmt.Println("\nNo trades to display.")
		return
	}

	// Print first N trades
	n := printTrades
	if n > len(result.Trades) {
		n = len(result.Trades)
	}

	fmt.Printf("\nFirst %d trades:\n", n)
	fmt.Println(strings.Repeat("-", 80))
	for i := 0; i < n; i++ {
		tr := result.Trades[i]
		fmt.Printf("#%2d ", i+1)

		direction := "LONG"
		if tr.Direction == -1 {
			direction = "SHORT"
		}

		fmt.Printf("[%s] ", direction)

		entryTS := formatTimestamp(seriesTimeToMs(tr.EntryTime))
		exitTS := formatTimestamp(seriesTimeToMs(tr.ExitTime))

		// Print trade details with new fields
		fmt.Printf("Entry[%d %s] %.6f -> Exit[%d %s] %.6f ",
			tr.EntryIdx, entryTS, tr.EntryPrice,
			tr.ExitIdx, exitTS, tr.ExitPrice)
		fmt.Printf("| PnL=%.4f%% ", tr.PnL*100)
		if tr.PnL > 0 {
			fmt.Printf("WIN [%s]", tr.Reason)
		} else {
			fmt.Printf("LOSS [%s]", tr.Reason)
		}
		if tr.HoldBars > 0 {
			fmt.Printf(" | %d bars", tr.HoldBars)
		}
		fmt.Println()
		// Print detailed levels on next line
		fmt.Printf("    EntryPrice: %.6f | StopPrice: %.6f | TPPrice: %.6f | TrailActive: %v\n",
			tr.EntryPrice, tr.StopPrice, tr.TPPrice, tr.TrailActive)
		fmt.Printf("    ExitCandle: Open=%.6f High=%.6f Low=%.6f Close=%.6f\n",
			tr.ExitOpen, tr.ExitHigh, tr.ExitLow, tr.ExitClose)
		fmt.Println()
	}
	fmt.Println(strings.Repeat("=", 80))

	// Print trade statistics
	wins := 0
	losses := 0
	totalWinPnL := float32(0)
	totalLossPnL := float32(0)
	maxHoldBars := 0
	for _, tr := range result.Trades {
		if tr.PnL > 0 {
			wins++
			totalWinPnL += tr.PnL
		} else {
			losses++
			totalLossPnL += -tr.PnL
		}
		if tr.HoldBars > maxHoldBars {
			maxHoldBars = tr.HoldBars
		}
	}

	fmt.Println("\nTrade Statistics:")
	fmt.Printf("  Wins:             %d (%.1f%%)\n", wins, float32(wins)*100/float32(result.TotalTrades))
	fmt.Printf("  Losses:           %d (%.1f%%)\n", losses, float32(losses)*100/float32(result.TotalTrades))
	if wins > 0 {
		fmt.Printf("  Avg Win:          %.4f%%\n", totalWinPnL/float32(wins)*100)
	}
	if losses > 0 {
		fmt.Printf("  Avg Loss:         %.4f%%\n", totalLossPnL/float32(losses)*100)
	}
	fmt.Printf("  Max Hold Bars:    %d\n", maxHoldBars)
	if result.TotalTrades > 0 {
		fmt.Printf("  Avg Hold Bars:    %.1f\n", float32(result.TotalHoldBars)/float32(result.TotalTrades))
	}

	// Count exit reasons
	fmt.Println("\nExit Reason Breakdown:")
	for reason, count := range result.ExitReasons {
		fmt.Printf("  %s: %d (%.1f%%)\n", reason, count, float32(count)*100/float32(result.TotalTrades))
	}
}

// seriesTimeToMs converts a time.Time (seconds since epoch) to milliseconds
func seriesTimeToMs(t time.Time) int64 {
	return t.Unix() * 1000
}

// writeTraceCSV outputs per-bar states to CSV for a given strategy
// Now includes detailed debug info for signal detection (EMA values, cross logic)
func writeTraceCSV(s Series, trades []Trade, outPath string, feats Features, st Strategy, testW Window) error {
	// Calculate the offset: sliced series starts at (testW.Start - testW.Warmup)
	sliceStartIdx := testW.Start - testW.Warmup
	if sliceStartIdx < 0 {
		sliceStartIdx = 0
	}

	// Default state: FLAT for every bar
	states := make([]string, s.T)
	for i := range states {
		states[i] = "FLAT"
	}

	// Map to store signal details by bar index (in full series)
	type signalDetails struct {
		detailsStr string
	}
	signalDetailsMap := make(map[int]signalDetails)

	// Map to store exit details by bar index (in full series)
	type exitDetails struct {
		reason      string
		pnl         float32
		entryPrice  float32
		exitPrice   float32
		stopPrice   float32
		tpPrice     float32
	}
	exitDetailsMap := make(map[int]exitDetails)

	// Mark ENTRY EXEC, HOLDING and exits
	for _, tr := range trades {
		// Convert local trade index to full series index
		fullEntryIdx := sliceStartIdx + tr.EntryIdx
		fullExitIdx := sliceStartIdx + tr.ExitIdx
		sigIdx := fullEntryIdx - 1 // Signal is detected at entryIdx - 1

		// Mark ENTRY EXEC at entry bar (when position is actually taken)
		if fullEntryIdx >= 0 && fullEntryIdx < len(states) {
			states[fullEntryIdx] = "ENTRY EXEC"
		}

		// Mark holding from entry+1 to exit-1
		for t := fullEntryIdx + 1; t < fullExitIdx && t >= 0 && t < len(states); t++ {
			states[t] = "HOLDING"
		}

		// Mark exit reason on exit bar
		if fullExitIdx >= 0 && fullExitIdx < len(states) {
			var state string
			switch tr.Reason {
			case "TP":
				state = "TP-HIT"
			case "SL":
				state = "SL-HIT"
			case "TRAIL":
				state = "TRAIL-HIT"
			case "MAX_HOLD":
				state = "MAX_HOLD"
			default:
				state = "EXIT_RULE"
			}
			states[fullExitIdx] = state

			// Store exit details
			exitDetailsMap[fullExitIdx] = exitDetails{
				reason:     tr.Reason,
				pnl:        tr.PnL,
				entryPrice: tr.EntryPrice,
				exitPrice:  tr.ExitPrice,
				stopPrice:  tr.StopPrice,
				tpPrice:    tr.TPPrice,
			}
		}

		// Mark signal detect = entryIdx - 1 (because pendingEntry executes next bar open)
		if sigIdx >= 0 && sigIdx < len(states) {
			// Don't overwrite if something "stronger" is already there
			if states[sigIdx] == "FLAT" {
				states[sigIdx] = "SIGNAL DETECT"

				// FAST DEBUGGING: Log all required indicator values at signal detection
				closePrice := fmt.Sprintf("%.2f", s.Close[sigIdx])
				openPrice := fmt.Sprintf("%.2f", s.Open[sigIdx])

				// Build details string with all required indicators
				detailsStr := fmt.Sprintf("Close=%s Open=%s", closePrice, openPrice)

				// 1. EMA20, EMA50
				var ema20Cur, ema50Cur string
				if ema20Idx, ok20 := feats.Index["EMA20"]; ok20 && ema20Idx >= 0 && ema20Idx < len(feats.F) {
					ema20Cur = fmt.Sprintf("%.2f", feats.F[ema20Idx][sigIdx])
				}
				if ema50Idx, ok50 := feats.Index["EMA50"]; ok50 && ema50Idx >= 0 && ema50Idx < len(feats.F) {
					ema50Cur = fmt.Sprintf("%.2f", feats.F[ema50Idx][sigIdx])
				}
				if ema20Cur != "" && ema50Cur != "" {
					detailsStr += fmt.Sprintf(" EMA20=%s EMA50=%s", ema20Cur, ema50Cur)
				}

				// 2. PlusDI (F[21])
				if plusDiIdx, ok := feats.Index["PlusDI"]; ok && plusDiIdx >= 0 && plusDiIdx < len(feats.F) {
					detailsStr += fmt.Sprintf(" PlusDI=%.2f", feats.F[plusDiIdx][sigIdx])
				}

				// 3. RSI7 (F[5])
				if rsi7Idx, ok := feats.Index["RSI7"]; ok && rsi7Idx >= 0 && rsi7Idx < len(feats.F) {
					detailsStr += fmt.Sprintf(" RSI7=%.2f", feats.F[rsi7Idx][sigIdx])
				}

				// 4. VolZ20 (F[30]), VolZ50 (F[31])
				if volZ20Idx, ok := feats.Index["VolZ20"]; ok && volZ20Idx >= 0 && volZ20Idx < len(feats.F) {
					detailsStr += fmt.Sprintf(" VolZ20=%.2f", feats.F[volZ20Idx][sigIdx])
				}
				if volZ50Idx, ok := feats.Index["VolZ50"]; ok && volZ50Idx >= 0 && volZ50Idx < len(feats.F) {
					detailsStr += fmt.Sprintf(" VolZ50=%.2f", feats.F[volZ50Idx][sigIdx])
				}

				// 5. Body (F[37])
				if bodyIdx, ok := feats.Index["Body"]; ok && bodyIdx >= 0 && bodyIdx < len(feats.F) {
					detailsStr += fmt.Sprintf(" Body=%.2f", feats.F[bodyIdx][sigIdx])
				}

				// 6. BB_Width50 (F[15])
				if bbWidthIdx, ok := feats.Index["BB_Width50"]; ok && bbWidthIdx >= 0 && bbWidthIdx < len(feats.F) {
					detailsStr += fmt.Sprintf(" BB_Width50=%.2f", feats.F[bbWidthIdx][sigIdx])
				}

				// 7. Cross debug info (optional)
				crossInfos := evaluateCrossDebug(st.EntryCompiled.Code, feats.F, feats.Names, sigIdx)
				for _, ci := range crossInfos {
					if ci.Result {
						if ci.Kind == "CrossUp" {
							crossStr := fmt.Sprintf(" CrossUp: prevA(%.2f)<=prevB(%.2f) curA(%.2f)>curB(%.2f)",
								ci.PrevA, ci.PrevB, ci.CurA, ci.CurB)
							detailsStr += crossStr
						} else if ci.Kind == "CrossDown" {
							crossStr := fmt.Sprintf(" CrossDown: prevA(%.2f)>=prevB(%.2f) curA(%.2f)<curB(%.2f)",
								ci.PrevA, ci.PrevB, ci.CurA, ci.CurB)
							detailsStr += crossStr
						}
					}
				}

				signalDetailsMap[sigIdx] = signalDetails{
					detailsStr: detailsStr,
				}
			}
		}
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// No header (matches your example)
	for i := 0; i < s.T; i++ {
		ts := time.Unix(int64(s.OpenTimeMs[i])/1000, 0).UTC().Format(time.RFC3339)

		// If this is a SIGNAL DETECT bar, include the details
		if states[i] == "SIGNAL DETECT" {
			if details, ok := signalDetailsMap[i]; ok {
				row := []string{
					fmt.Sprintf("%d", i),
					ts,
					states[i],
					details.detailsStr,
				}
				if err := w.Write(row); err != nil {
					return err
				}
				continue
			}
		}

		// If this is an exit bar, include exit details with clear return percentage
		if exit, ok := exitDetailsMap[i]; ok {
			// Format return as clear percentage (e.g., "+1.23%" or "-1.23%")
			returnPct := exit.pnl * 100 // Convert to percentage
			returnStr := fmt.Sprintf("%.2f%%", returnPct)
			if returnPct >= 0 {
				returnStr = "+" + returnStr // Add + sign for positive returns
			}

			row := []string{
				fmt.Sprintf("%d", i),
				ts,
				states[i],
				exit.reason,
				returnStr,                          // Return percentage (e.g., "+1.23%" or "-1.23%")
				fmt.Sprintf("%.2f", exit.entryPrice),
				fmt.Sprintf("%.2f", exit.exitPrice), // Added exit price
				fmt.Sprintf("%.2f", exit.stopPrice),
				fmt.Sprintf("%.2f", exit.tpPrice),
			}
			if err := w.Write(row); err != nil {
				return err
			}
			continue
		}

		// Normal row (no signal details)
		if err := w.Write([]string{fmt.Sprintf("%d", i), ts, states[i]}); err != nil {
			return err
		}
	}
	return w.Error()
}

// buildManualEMA20x50 builds a deterministic EMA20x50 crossover strategy for manual debugging
func buildManualEMA20x50(feats Features, feeBps, slipBps float32) Strategy {
	// Find feature indices for EMA20 and EMA50
	ema20Idx, ok20 := feats.Index["EMA20"]
	ema50Idx, ok50 := feats.Index["EMA50"]

	if !ok20 || !ok50 {
		// Fall back to first two features if EMAs not found (should not happen)
		fmt.Printf("Warning: EMA20/EMA50 not found, using first two features\n")
		ema20Idx = 0
		ema50Idx = 1
	}

	// Create CrossUp rule: EMA20 crosses above EMA50
	entryRoot := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafCrossUp,
			A:    ema20Idx,
			B:    ema50Idx,
		},
	}

	// Empty exit rule (no exit signal - relies on SL/TP)
	// Use LeafLT with impossible threshold to ensure it's never true
	exitRoot := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafLT,
			A:    0,
			X:    -1e9, // Never true - price will never be < -1 billion
		},
	}

	// Empty regime filter (always active)
	// Use LeafLT with very large threshold - price < 1e9 is always true for BTC
	regimeRoot := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafLT,
			A:    0,
			X:    1e9, // Always true - price will never exceed 1 billion
		},
	}

	// Fixed SL = 1%, TP = 2%, Trail off
	st := Strategy{
		Seed:            0, // Deterministic seed for manual strategy
		FeeBps:          feeBps,
		SlippageBps:     slipBps,
		RiskPct:         0.01,
		Direction:       1, // Long only
		EntryRule:       RuleTree{Root: entryRoot},
		ExitRule:        RuleTree{Root: exitRoot},
		RegimeFilter:    RuleTree{Root: regimeRoot},
		StopLoss:        StopModel{Kind: "fixed", Value: 1.0},
		TakeProfit:      TPModel{Kind: "fixed", Value: 2.0},
		Trail:           TrailModel{Active: false},
		MaxHoldBars:     150,
		MaxConsecLosses: 0,
		CooldownBars:    0,
	}

	// Compile rules
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	return st
}

// runTraceMode runs a single strategy and outputs per-bar state CSV
func runTraceMode(seed int64, csvPath, manual string, openCSV bool, feeBps, slipBps float64, traceWindow string) {
	fmt.Println("Running in TRACE mode - per-bar state output")
	fmt.Println("============================================")
	fmt.Printf("Seed: %d, CSV: %s, Manual: %s, FeeBps: %.1f, SlippageBps: %.1f\n", seed, csvPath, manual, feeBps, slipBps)
	fmt.Println()

	// Load data
	fmt.Println("Loading data...")
	series, err := LoadBinanceKlinesCSV("btc_5min_data.csv")
	if err != nil {
		fmt.Printf("Error loading CSV: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", series.T)

	fmt.Println("Computing features...")
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.F))

	// Split data into train/validation/test windows
	trainStart, trainEnd, valEnd := GetSplitIndices(series.OpenTimeMs)
	trainW, valW, testW := GetSplitWindows(trainStart, trainEnd, valEnd, series.T, 500)

	// Select window based on trace_window flag
	var w Window
	switch traceWindow {
	case "train":
		w = trainW
	case "val":
		w = valW
	case "test":
		w = testW
	default:
		fmt.Printf("Error: invalid trace_window '%s'. Must be 'train', 'val', or 'test'\n", traceWindow)
		return
	}

	// Recompute feature stats on train window only
	computeStatsOnWindow(&feats, trainStart, trainEnd)
	fmt.Printf("Feature stats computed on train window (%d candles)\n", trainEnd-trainStart)

	fmt.Printf("\n%s window: %d candles (indices %d -> %d)\n\n", strings.ToUpper(traceWindow), w.End-w.Start, w.Start, w.End)

	// Load or build strategy
	var st Strategy
	if manual != "" {
		// Only accept "ema20x50" as manual strategy
		if manual != "ema20x50" {
			fmt.Printf("Error: -trace_manual must be 'ema20x50', got '%s'\n", manual)
			return
		}
		fmt.Printf("Using manual EMA20x50 strategy\n")
		st = buildManualEMA20x50(feats, float32(feeBps), float32(slipBps))
	} else {
		// Load winner from winners_tested.jsonl or winners.jsonl
		var w *WinnerRow
		if seed > 0 {
			w, err = loadWinnerBySeed("winners_tested.jsonl", seed)
			if err != nil {
				// Try winners.jsonl as fallback
				w, err = loadWinnerBySeed("winners.jsonl", seed)
				if err != nil {
					fmt.Printf("Error loading winner with seed %d: %v\n", seed, err)
					return
				}
			}
			fmt.Printf("Loaded winner with seed %d\n", seed)
		} else {
			w, err = loadFirstWinner("winners_tested.jsonl")
			if err != nil {
				// Try winners.jsonl as fallback
				w, err = loadFirstWinner("winners.jsonl")
				if err != nil {
					fmt.Printf("Error loading first winner: %v\n", err)
					return
				}
			}
			fmt.Printf("Loaded first winner (seed=%d)\n", w.Seed)
		}

		// Rebuild strategy from winner
		st = Strategy{
			Seed:        w.Seed,
			FeeBps:      float32(feeBps),
			SlippageBps: float32(slipBps),
			RiskPct:     0.01,
			Direction:   w.Direction,
			EntryRule: RuleTree{
				Root: parseRuleTree(w.EntryRule),
			},
			ExitRule: RuleTree{
				Root: parseRuleTree(w.ExitRule),
			},
			RegimeFilter: RuleTree{
				Root: parseRuleTree(w.RegimeFilter),
			},
			StopLoss:   parseStopModel(w.StopLoss),
			TakeProfit: parseTPModel(w.TakeProfit),
			Trail:      parseTrailModel(w.Trail),
		}
	}

	// Compile bytecode
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	// Run backtest with trade logging on TEST window
	fmt.Printf("Running backtest on %s window...\n", strings.ToUpper(traceWindow))
	result := evaluateStrategyWithTrades(series, feats, st, w, true)

	// Write trace CSV
	fmt.Printf("\nWriting trace CSV to %s...\n", csvPath)
	if err := writeTraceCSV(series, result.Trades, csvPath, feats, st, w); err != nil {
		fmt.Printf("Error writing trace CSV: %v\n", err)
		return
	}

	// Auto-open CSV on Windows if requested
	if openCSV {
		exec.Command("cmd", "/c", "start", csvPath).Start()
	}

	// Print summary
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TRACE MODE SUMMARY")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("\nCSV Output: %s\n", csvPath)
	fmt.Printf("Total Trades: %d\n", result.TotalTrades)
	fmt.Printf("Total Bars: %d\n", series.T)
	fmt.Printf("Test Window: %d -> %d\n", testW.Start, testW.End)
	fmt.Println("\nPer-bar states:")
	fmt.Println("  FLAT        - Not in position")
	fmt.Println("  SIGNAL DETECT- Entry rule became true (bar before entry)")
	fmt.Println("  HOLDING     - In position (from entry+1 until exit bar)")
	fmt.Println("  TP-HIT      - Take profit hit on this bar")
	fmt.Println("  SL-HIT      - Stop loss hit on this bar")
	fmt.Println("  EXIT_RULE   - Exit rule triggered")
	fmt.Println("  MAX_HOLD    - Max hold duration reached")
	fmt.Println("\nCSV format: barIndex,timestamp,state")
	fmt.Println("No header row (as requested)")
}
