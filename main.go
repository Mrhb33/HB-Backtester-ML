package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type BatchMsg struct {
	TopN  []Result // top N results from this batch
	Count int64
}

type EliteLog struct {
	Seed         uint64  `json:"seed"`
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
	Seed           uint64  `json:"seed"`
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

	mode := flag.String("mode", "search", "search or test")
	scoringMode := flag.String("scoring", "balanced", "scoring mode: balanced or aggressive")
	seedFlag := flag.Int64("seed", 0, "random seed (0 = time-based, nonzero = reproducible)")
	resumePath := flag.String("resume", "", "checkpoint file to resume from (ex: checkpoint.json)")
	checkpointPath := flag.String("checkpoint", "checkpoint.json", "checkpoint output path")
	checkpointEverySec := flag.Int("checkpoint_every", 300, "auto-save checkpoint every N seconds")
	flag.Parse()

	// Define scoring thresholds based on mode
	var maxValDD, minValReturn, minValPF, minValExpect float32
	var minValTrades int
	if *scoringMode == "aggressive" {
		// Aggressive mode: stricter profit gates, looser DD to encourage exploration
		maxValDD = 0.60       // Allow higher drawdown
		minValReturn = 0.10   // Require +10% return (stricter)
		minValPF = 1.10       // Higher profit factor requirement (stricter)
		minValExpect = 0.0001 // Require positive edge (stricter)
		minValTrades = 30
		fmt.Printf("Scoring mode: AGGRESSIVE (DD<%.2f, ret>%.1f%%, pf>%.2f, exp>%.4f)\n", maxValDD, minValReturn*100, minValPF, minValExpect)
	} else {
		// Balanced mode (default): balanced gates for stability and exploration
		maxValDD = 0.45        // Tighter drawdown limit
		minValReturn = 0.05    // +5% return gate (balanced)
		minValPF = 1.10        // Higher profit factor requirement (stricter)
		minValExpect = 0.00005 // Require positive expectancy (balanced)
		minValTrades = 30
		fmt.Printf("Scoring mode: BALANCED (DD<%.2f, ret>%.1f%%, pf>%.2f, exp>%.4f)\n", maxValDD, minValReturn*100, minValPF, minValExpect)
	}

	// Initialize MetaState for self-improving search
	meta := MetaState{
		RadicalP:     0.10,
		SurExploreP:  0.10,
		SurThreshold: 0.0,
		MaxValDD:     maxValDD,
		MinValReturn: minValReturn,
		MinValPF:     minValPF,
		MinValExpect: minValExpect,
		MinValTrades: minValTrades,
	}

	if *mode == "test" {
		runTestMode()
		return
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
	var bestValSeen float32      // Track best validation score seen (for adaptive bootstrap)
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
			fmt.Printf("Loaded meta.json: Batches=%d, BestVal=%.4f, RadicalP=%.2f, SurExploreP=%.2f\n",
				meta.Batches, meta.BestVal, meta.RadicalP, meta.SurExploreP)
			// Apply loaded meta into runtime knobs
			maxValDD, minValReturn, minValPF, minValExpect = meta.MaxValDD, meta.MinValReturn, meta.MinValPF, meta.MinValExpect
			minValTrades = meta.MinValTrades
			surThreshold = float64(meta.SurThreshold)
			stagnationMu.Lock()
			radicalP, surExploreP = meta.RadicalP, meta.SurExploreP
			stagnationMu.Unlock()
		}
	}

	// Track seen fingerprints to avoid retesting strategies
	seen := make(map[string]struct{})
	seenMu := sync.Mutex{}

	markSeen := func(s Strategy) bool {
		fp := s.Fingerprint()
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
		meta.MaxValDD = maxValDD
		meta.MinValReturn = minValReturn
		meta.MinValPF = minValPF
		meta.MinValExpect = minValExpect
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
					Seed:        uint64(rng.Int63()),
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
		const stabilityThreshold = 0.40 // val must be at least 40% of train

		batchSize := int64(10000)
		var batchID int64 = 0
		var batchBest Result
		batchBest.Score = -1e30
		lastReport := time.Now()
		nextCheckpoint := uint64(10000)

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
				// Final test evaluation on shutdown - only test winners from winners.jsonl
				loadedLogs, err := loadRecentElites("winners.jsonl", 1000)
				if err != nil || len(loadedLogs) == 0 {
					print("\n\nNo winners to test.\n")
				} else {
					print("\n\nRunning final test evaluation on %d winners from winners.jsonl...\n", len(loadedLogs))
					print(strings.Repeat("=", 80))
					print("\n")

					w.WriteString("\n// FINAL TEST RESULTS\n")
					for i, log := range loadedLogs {
						// Rebuild strategy from saved log
						strategy := Strategy{
							Seed:        log.Seed,
							FeeBps:      log.FeeBps,
							SlippageBps: log.SlippageBps,
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

						// Compile rules
						strategy.EntryCompiled = compileRuleTree(strategy.EntryRule.Root)
						strategy.ExitCompiled = compileRuleTree(strategy.ExitRule.Root)
						strategy.RegimeCompiled = compileRuleTree(strategy.RegimeFilter.Root)

						testR := evaluateStrategyWindow(series, feats, strategy, testW)
						print("Test %d/%d: Score=%.4f Return=%.2f%% WinRate=%.1f%% Trades=%d\n",
							i+1, len(loadedLogs), testR.Score, testR.Return*100, testR.WinRate*100, testR.Trades)

						testLine := ReportLine{
							Batch:          batchID + 1000 + int64(i), // Use batch ID > 1000 for test results
							Tested:         int64(atomic.LoadUint64(&tested)),
							Score:          log.TrainScore,
							Return:         log.TrainReturn,
							MaxDD:          log.TrainMaxDD,
							WinRate:        log.TrainWinRate,
							Expectancy:     0, // Not in EliteLog
							ProfitFactor:   0, // Not in EliteLog
							Trades:         log.TrainTrades,
							FeeBps:         log.FeeBps,
							SlippageBps:    log.SlippageBps,
							Direction:      log.Direction,
							Seed:           log.Seed,
							EntryRuleDesc:  log.EntryRule,
							ExitRuleDesc:   log.ExitRule,
							StopLossDesc:   log.StopLoss,
							TakeProfitDesc: log.TakeProfit,
							TrailDesc:      log.Trail,
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
						// nothing to validate this checkpoint; keep moving
						batchBest = Result{Score: -1e30}
						globalTopCandidates = globalTopCandidates[:0]
						globalFingerprints = make(map[string]bool)
						nextCheckpoint += uint64(batchSize)
						continue
					}

				// Fixed validation threshold: rely on profit sanity gates instead of dynamic score
				// Use score only for ranking, not for pass/fail, until scale stabilizes
				minValScore := float32(0.0) // Fixed threshold - accept all that pass profit gates
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
						testedNow := int64(atomic.LoadUint64(&tested))
						valR.Score = computeScore(valR.Return, valR.MaxDD, valR.Expectancy, valR.Trades, testedNow)
						// Also deflate train scores for fair comparison
						candidate.Score = computeScore(candidate.Return, candidate.MaxDD, candidate.Expectancy, candidate.Trades, testedNow)

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

						// ---- Profit sanity gate (after costs) ----
						// Note: use valR (current candidate) NOT bestValR (best so far)
						// Use meta gate variables for dynamic control (Bug #3 fix)
						minAvgPerTrade := float32(0.00003) // average net edge per trade

						avgPerTrade := float32(0)
						if valR.Trades > 0 {
							avgPerTrade = valR.Return / float32(valR.Trades)
						}

						// Enhanced profit sanity gate: use meta-controlled gates
						profitOK := valR.Return >= minValReturn && // Use meta gate (dynamic)
							valR.Expectancy > minValExpect && // Use meta gate (dynamic)
							valR.ProfitFactor >= minValPF && // Use meta gate (dynamic)
							valR.Trades >= minValTrades && // Use meta gate (dynamic)
							avgPerTrade >= minAvgPerTrade

						// Add to Hall of Fame if passes validation + CPCV
						// Stability check: val_score must be decent fraction of train_score to reduce overfitting
						stableEnough := true
						if candidate.Score > 0 {
							stableEnough = valR.Score >= stabilityThreshold*candidate.Score
						}
						passesValidation := valR.Score > minValScore && valR.MaxDD < maxValDD && valR.Trades >= minValTrades && stableEnough && profitOK
						if passesValidation {
							// Track best validation score seen ONLY from fully validated strategies
							// This prevents bestValSeen from tracking failing candidates
							bestValSeenMu.Lock()
							if valR.Trades >= minValTrades && valR.Score > -1e20 && valR.Score > bestValSeen {
								bestValSeen = valR.Score
							}
							bestValSeenMu.Unlock()

							// Run CPCV to check stability across multiple folds BEFORE adding to HOF/archive
							cpcv := evaluateCPCV(series, feats, candidate.Strategy, trainW.Start, trainW.End,
								int64(atomic.LoadUint64(&tested)), minValScore)
							if cpcv.MinFoldScore < 0.0 {
								continue // reject unstable "lucky" strategy
							}
							if !CPCVPassCriteria(cpcv, minValScore, 0.66) {
								continue // reject unstable "lucky" strategy
							}

							// Quick test gate: run on a shorter test slice before writing to winners.jsonl
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

							// Apply same profit gates to quick test (use slightly looser meta gates)
							quickTestProfitOK := quickTestR.Return >= minValReturn*0.8 && // 80% of meta gate
								quickTestR.Expectancy > minValExpect*0.5 && // 50% of meta gate
								quickTestR.ProfitFactor >= minValPF*0.95 && // 95% of meta gate
								quickTestR.Trades >= 15 && // Lower threshold for quick test
								quickTestR.MaxDD < maxValDD

							if !quickTestProfitOK {
								continue // reject strategy that fails quick test gate
							}

							// Overtrading check: reject strategies with excessive trades that might be overfitted
							// Estimate trades per year based on val window (assuming 5min data = 105120 candles/year)
							valCandles := valW.End - valW.Start
							tradesPerYear := float32(valR.Trades) * (105120.0 / float32(valCandles))
							maxTradesPerYear := float32(500) // Cap at 500 trades/year (~1 per day)
							if tradesPerYear > maxTradesPerYear {
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
					bestPassesValidation = bestValR.Score > minValScore &&
						bestValR.MaxDD < maxValDD &&
						bestValR.Trades >= minValTrades &&
						stableEnoughForFinal &&
						bestValR.Return >= minValReturn &&
						bestValR.Expectancy > minValExpect &&
						bestValR.ProfitFactor >= minValPF
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

						print("Batch %d: Tested %d | Train: %.4f | Val: %.4f | Ret: %.2f%% | WR: %.1f%% | Trds: %d | Rate: %.0f/s %s [fp: %s] [crit: score>%.2f, trds>=%d, DD<%.2f, ret>%.1f%%, exp>%.4f, pf>%.2f, stab>%.0f%%] [val: score=%.4f dd=%.3f trds=%d%s]\n",
							batchID, atomic.LoadUint64(&tested), bestTrainResult.Score, bestValR.Score, bestValR.Return*100, bestValR.WinRate*100, bestValR.Trades, rate, checkmark, fingerprint, minValScore, minValTrades, maxValDD, minValReturn*100, minValExpect, minValPF, stabilityThreshold*100,
							bestValR.Score, bestValR.MaxDD, bestValR.Trades, reasonStr)
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
			batchSize := 256
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
							passedScreen, passedFull, trainR, valR := evaluateMultiFidelity(series, feats, s, screenW, trainW, valW, testedNow)
							if !passedScreen {
								continue // failed fast screen
							}
							if !passedFull || trainR.Trades == 0 {
								continue // failed train phase or no trades
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
						// Evaluate all strategies using multi-fidelity pipeline
						localResults = localResults[:0]
						testedNow := int64(atomic.LoadUint64(&tested))
						for _, strat := range localBatch {
							// Use multi-fidelity: screen -> train -> val
							passedScreen, passedFull, trainR, valR := evaluateMultiFidelity(series, feats, strat, screenW, trainW, valW, testedNow)
							if !passedScreen {
								continue // failed fast screen
							}
							if !passedFull || trainR.Trades == 0 {
								continue // failed train phase or no trades
							}
							// Store train result with val metrics embedded
							// We'll use the train result for ranking, but validation will use valR
							trainR.ValResult = &valR // Store val result reference
							localResults = append(localResults, trainR)
						}
						// Sort by score and send top 20
						sort.Slice(localResults, func(i, j int) bool { return localResults[i].Score > localResults[j].Score })
						topN := min(20, len(localResults))
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
			// 30%: random immigrants (completely fresh strategies)
			// adaptive: heavy mutations (big jumps to escape local maxima)
			// 20%: crossover (mix two parents)
			// remaining: normal mutation (small threshold tweaks)
			const immigrantP = 0.30

			// Get adaptive parameters (thread-safe)
			stagnationMu.RLock()
			currentRadicalP := radicalP
			stagnationMu.RUnlock()

			heavyMutP := immigrantP + currentRadicalP // adaptive: 0.40 (base) to 0.60 (stagnation)
			crossP := heavyMutP + 0.20                // adaptive: 0.60 (base) to 0.80 (stagnation)
			// normal mutation = 1.0 - crossP (adaptive: 0.40 to 0.20)

			x := rng.Float32()

			// Random immigrants: 30% completely fresh strategies
			if hof.Len() == 0 || x < immigrantP {
				s = randomStrategy(rng, feats)
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
						s = randomStrategy(rng, feats)
					} else {
						s = mutateStrategy(rng, parent.Strat, feats)
					}
				}
			}

			// Use surrogate to filter out obviously bad strategies
			// Meta-controller (in checkpoint logic) adjusts surThreshold based on improvement
			// No schedule-based computation - let meta drive it
			surFeatures := ExtractSurFeatures(s)

			// Read current threshold (controlled by meta updates at checkpoints)
			stagnationMu.RLock()
			currentThreshold := surThreshold
			stagnationMu.RUnlock()

			if !sur.Accept(surFeatures, currentThreshold) {
				continue // skip junk quickly
			}

			// Check if we've already seen this strategy fingerprint
			if !markSeen(s) {
				continue
			}

			select {
			case <-ctx.Done():
				return
			case jobs <- s:
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

				print("[progress] tested=%d  rate=%.1f/s  elapsed=%s  bestVal=%.4f  elites=%d  [meta: radicalP=%.2f surExploreP=%.2f batches=%d]\n",
					cur, rate, time.Since(start).Truncate(time.Second), reportBestVal, elitesCount, currentRadicalP, currentSurExploreP, currentBatches)
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

	wg.Wait()
	close(batchResults)
	<-doneAgg
	// Close winnerLog after aggregator is fully done
	close(winnerLog)

	print("\n\nStopped. Total tested: %d\n", atomic.LoadUint64(&tested))
	print("Results saved to: best_every_10000.txt")
}

func runTestMode() {
	fmt.Println("Running in TEST mode - evaluating saved winners on test data")
	fmt.Println("=============================================================")

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
		TestScore   float32 `json:"test_score"`
		TestReturn  float32 `json:"test_return"`
		TestMaxDD   float32 `json:"test_max_dd"`
		TestWinRate float32 `json:"test_win_rate"`
		TestTrades  int     `json:"test_trades"`
	}

	// Print test window info
	fmt.Printf("Test window: %d candles (indices %d -> %d)\n\n", testW.End-testW.Start, testW.Start, testW.End)

	var allResults []TestResult
	var parseErrors int
	var noTradeStrategies int

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
				FeeBps:      log.FeeBps,      // Use original fee from winner log
				SlippageBps: log.SlippageBps, // Use original slippage from winner log
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
			EliteLog:    log,
			TestScore:   testR.Score,
			TestReturn:  testR.Return,
			TestMaxDD:   testR.MaxDD,
			TestWinRate: testR.WinRate,
			TestTrades:  testR.Trades,
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

	// Sort results by test score, but only include strategies with trades
	var rankedResults []TestResult
	for _, r := range allResults {
		if r.TestTrades > 0 {
			rankedResults = append(rankedResults, r)
		}
	}
	sort.Slice(rankedResults, func(i, j int) bool {
		return rankedResults[i].TestScore > rankedResults[j].TestScore
	})

	// Print top 20 leaderboard
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TOP 20 STRATEGIES BY TEST SCORE")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()

	numToPrint := 20
	if len(rankedResults) < numToPrint {
		numToPrint = len(rankedResults)
	}

	for i := 0; i < numToPrint; i++ {
		r := rankedResults[i]
		fmt.Printf("#%2d | Test Score: %.4f | Test Return: %6.2f%% | Test WinRate: %5.1f%% | Test Trades: %4d | Val Score: %.4f\n",
			i+1, r.TestScore, r.TestReturn*100, r.TestWinRate*100, r.TestTrades, r.ValScore)
	}

	fmt.Printf("\nTotal strategies loaded: %d\n", len(loadedLogs))
	fmt.Printf("Tested successfully: %d\n", len(allResults)-parseErrors)
	fmt.Printf("Skipped parse errors: %d\n", parseErrors)
	fmt.Printf("No-trade on test: %d\n", noTradeStrategies)
	fmt.Printf("Strategies with trades: %d\n", len(rankedResults))
	fmt.Println("Results saved to: winners_tested.jsonl")
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
	case LeafZScoreGT, LeafZScoreLT:
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
