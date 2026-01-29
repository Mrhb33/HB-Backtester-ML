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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"hb_bactest_checker/logx"
)

// Global verbose flag - controls debug output across all packages
var Verbose bool = false

// Global logTrades flag - enables detailed per-trade logging
var LogTrades bool = false

// Global recovery mode flag - thread-safe for concurrent worker access
var RecoveryMode atomic.Bool

// Global auto-recovery toggle flag - separate from manual RecoveryMode
// Automatically toggles based on elite count to adjust entry constraints
var recoveryModeActive atomic.Bool

// Global holdingHeartbeat interval - how often to print holding logs (in bars, 0 = disabled)
var HoldingHeartbeat int = 0

// Global verifyMode flag - enables bytecode vs AST verification
var VerifyMode bool = false

// Global debugPF flag - enables throttled PF debug logging
var DebugPF bool = false

// Global debugWalkForward flag - enables walk-forward debug logging
var DebugWalkForward bool = false

// Global walk-forward configuration
var wfConfig WFConfig

// Global timeframe in minutes (default 5 for 5-minute bars)
var globalTimeframeMinutes int32 = 5

// Generator loop counters (atomic for thread safety)
var (
	genGenerated       int64 // Total strategies generated
	genRejectedSur     int64 // Rejected by surrogate
	genRejectedSeen    int64 // Rejected by markSeen (duplicate) - DEPRECATED, use seenHits/seenNew instead
	genRejectedNovelty int64 // Rejected by novelty pressure (too similar to recent)
	genRerolled        int64 // Strategies that were rerolled from seen to unseen
	genSentToJobs      int64 // Sent to worker jobs
	// EXACT counters for seen tracking - call CheckAndSet exactly once per final candidate
	seenHits int64 // CheckAndSet returned false (already existed)
	seenNew  int64 // CheckAndSet returned true (new insert)
)

// getGeneratorStats returns formatted generator loop statistics
func getGeneratorStats() string {
	generated := atomic.LoadInt64(&genGenerated)
	rejectedSur := atomic.LoadInt64(&genRejectedSur)
	rejectedNovelty := atomic.LoadInt64(&genRejectedNovelty)
	rerolled := atomic.LoadInt64(&genRerolled)
	sentToJobs := atomic.LoadInt64(&genSentToJobs)
	// Use exact counters for seen tracking
	hits := atomic.LoadInt64(&seenHits)
	newFp := atomic.LoadInt64(&seenNew)

	totalGenerated := generated
	if totalGenerated == 0 {
		totalGenerated = 1 // avoid division by zero
	}

	// FIX: Add explicit bounds checking for percentages to handle potential
	// race conditions where numerators might exceed denominator temporarily.
	// Use math/min for clamping to ensure percentages stay in [0, 100] range.
	surPct := 100.0 * float64(rejectedSur) / float64(totalGenerated)
	if surPct < 0 {
		surPct = 0
	}
	if surPct > 100 {
		surPct = 100
	}

	// Use exact formula: seen_hits / (seen_hits + seen_new)
	seenTotal := hits + newFp
	if seenTotal == 0 {
		seenTotal = 1 // avoid division by zero
	}
	seenPct := 100.0 * float64(hits) / float64(seenTotal)
	if seenPct < 0 {
		seenPct = 0
	}
	if seenPct > 100 {
		seenPct = 100
	}

	noveltyPct := 100.0 * float64(rejectedNovelty) / float64(totalGenerated)
	if noveltyPct < 0 {
		noveltyPct = 0
	}
	if noveltyPct > 100 {
		noveltyPct = 100
	}

	rerolledPct := 100.0 * float64(rerolled) / float64(totalGenerated)
	if rerolledPct < 0 {
		rerolledPct = 0
	}
	if rerolledPct > 100 {
		rerolledPct = 100
	}

	sentPct := 100.0 * float64(sentToJobs) / float64(totalGenerated)
	if sentPct < 0 {
		sentPct = 0
	}
	if sentPct > 100 {
		sentPct = 100
	}

	return fmt.Sprintf("gen=%d rej_sur=%d(%.1f%%) rej_seen=%d/%d(%.1f%%) rerolled=%d(%.1f%%) rej_novelty=%d(%.1f%%) sent=%d(%.1f%%)",
		generated, rejectedSur, surPct, hits, seenTotal, seenPct, rerolled, rerolledPct, rejectedNovelty, noveltyPct, sentToJobs, sentPct)
}

type BatchMsg struct {
	TopN  []Result // top N results from this batch
	Count int64
}

type EliteLog struct {
	Seed           int64   `json:"seed"`
	FeeBps         float32 `json:"fee_bps"`
	SlippageBps    float32 `json:"slippage_bps"`
	Direction      int     `json:"direction"`
	StopLoss       string  `json:"stop_loss"`
	TakeProfit     string  `json:"take_profit"`
	Trail          string  `json:"trail"`
	EntryRule      string  `json:"entry_rule"`
	ExitRule       string  `json:"exit_rule"`
	RegimeFilter   string  `json:"regime_filter"`
	FeatureMapHash string  `json:"feature_map_hash,omitempty"` // Feature ordering fingerprint
	TrainScore     float32 `json:"train_score"`
	TrainReturn    float32 `json:"train_return"`
	TrainMaxDD     float32 `json:"train_max_dd"`
	TrainWinRate   float32 `json:"train_win_rate"`
	TrainTrades    int     `json:"train_trades"`
	ValScore       float32 `json:"val_score"`
	ValReturn      float32 `json:"val_return"`
	ValMaxDD       float32 `json:"val_max_dd"`
	ValWinRate     float32 `json:"val_win_rate"`
	ValTrades      int     `json:"val_trades"`
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

// parseTimeframe parses a timeframe string (e.g., "5m", "60m", "1h", "4h") and returns minutes
// Accepts both "5m"/"15m"/"60m" format and "1h"/"4h" format
func parseTimeframe(tf string) int32 {
	// Accept "5m", "15m", "60m" OR "1h", "4h"
	// Return minutes as int32
	tf = strings.ToLower(strings.TrimSpace(tf))
	if strings.HasSuffix(tf, "h") {
		// Parse "1h", "4h" as hours
		hoursStr := strings.TrimSuffix(tf, "h")
		if hours, err := strconv.Atoi(hoursStr); err == nil {
			return int32(hours * 60)
		}
	} else if strings.HasSuffix(tf, "m") {
		// Parse "5m", "15m", "60m" as minutes
		minStr := strings.TrimSuffix(tf, "m")
		if mins, err := strconv.Atoi(minStr); err == nil {
			return int32(mins)
		}
	}
	// Default to 5 minutes if parsing fails
	return 5
}

func main() {
	fmt.Println("HB Backtest Strategy Search Engine")
	fmt.Println("====================================")

	mode := flag.String("mode", "search", "search|test|golden|trace|validate|diagnose|verify")
	verbose := flag.Bool("verbose", false, "enable verbose debug logs")
	scoringMode := flag.String("scoring", "balanced", "scoring mode: balanced or aggressive")
	seedFlag := flag.Int64("seed", 0, "random seed (0 = time-based, nonzero = reproducible)")
	resumePath := flag.String("resume", "", "checkpoint file to resume from (ex: checkpoint.json)")
	checkpointPath := flag.String("checkpoint", "checkpoint.json", "checkpoint output path")
	checkpointEverySec := flag.Int("checkpoint_every", 60, "TEMP warm-start: auto-save checkpoint every N seconds")
	// Cost override flags for production-level realism
	feeBpsFlag := flag.Float64("fee_bps", 20, "transaction fee in basis points (0.01% per bps, default 20 = 0.2%)")
	slipBpsFlag := flag.Float64("slip_bps", 5, "slippage in basis points (default 5 = 0.05%)")
	// Golden mode flags
	goldenSeed := flag.Int64("golden_seed", 0, "seed of winner strategy to run in golden mode")
	goldenN := flag.Int("golden_print_trades", 10, "how many trades to print in golden mode")
	exportCSVPath := flag.String("export_csv", "", "export trades to CSV file (works with golden/test mode)")
	exportStatesPath := flag.String("export_states", "", "export states with proofs to CSV file (works with golden/test mode)")
	// Trace mode flags
	traceSeed := flag.Int64("trace_seed", 0, "seed of strategy to trace (outputs per-bar states)")
	traceCSVPath := flag.String("trace_csv", "trace.csv", "output CSV path for trace mode")
	traceManual := flag.String("trace_manual", "", "manual debug strategy (ema20x50, etc.)")
	traceOpenCSV := flag.Bool("trace_open", false, "open CSV after trace (Windows)")
	traceWindow := flag.String("trace_window", "test", "trace window: train | val | test")
	// Trade logging flags
	logTrades := flag.Bool("log_trades", false, "enable detailed per-trade logging (4 events: SIGNAL DETECT, ENTRY EXEC, HOLDING, EXIT)")
	holdingHeartbeat := flag.Int("holding_heartbeat", 0, "holding log interval in bars (0 = disabled, e.g., 50 = log every 50 bars)")
	// Debug flags
	debugPF := flag.Bool("debug_pf", false, "enable throttled PF debug logging")
	debugWalkForward := flag.Bool("debug_wf", false, "enable walk-forward debug logging")
	// Recovery mode flag
	recoveryMode := flag.Bool("recovery", false, "enable recovery mode (relaxed screening, expanded seeding)")
	// Verification flags
	verifyMode := flag.Bool("verify", false, "enable bytecode vs AST verification mode (reports discrepancies)")
	// Custom window replay flags (for consistency testing)
	fromIdx := flag.Int("from_idx", -1, "custom start bar index (-1 for full data)")
	toIdx := flag.Int("to_idx", -1, "custom end bar index (-1 for end)")
	fromTime := flag.String("from", "", "custom start time (YYYY-MM-DD HH:MM:SS)")
	toTime := flag.String("to", "", "custom end time (YYYY-MM-DD HH:MM:SS)")
	warmupBars := flag.Int("warmup_bars", 2000, "warmup buffer before custom window")
	// Data file selection
	dataFile := flag.String("data", "btc_5min_data.csv", "CSV data file path (e.g., btc_5min_data.csv, btc_data_15m.csv, btc_data_60m.csv)")
	timeframe := flag.String("timeframe", "5m", "Data timeframe (e.g., 5m, 15m, 60m, 1h, 4h)")
	// Walk-forward validation flags
	wfEnable := flag.Bool("wf", false, "enable walk-forward validation mode")
	wfTrainDays := flag.Int("wf_train_days", 360, "walk-forward training window size in days")
	wfTestDays := flag.Int("wf_test_days", 90, "walk-forward test window size in days")
	wfStepDays := flag.Int("wf_step_days", 60, "walk-forward step size in days")
	wfMinTradesOOS := flag.Int("wf_min_trades_oos", 30, "minimum total trades in OOS period")
	// QuickTest validation flags
	qtMinTrades := flag.Int("qt_min_trades", 5, "QuickTest minimum trades threshold")
	wfMaxDD := flag.Float64("wf_max_dd", 0.70, "maximum drawdown allowed (0-1)")
	wfMaxSparseMonthsRatio := flag.Float64("wf_max_sparse_months_ratio", 0.80, "maximum ratio of sparse months to total months")
	wfMinActiveMonthsRatio := flag.Float64("wf_min_active_months_ratio", 0.25, "minimum ratio of active months to total months")
	wfEnableMinMonth := flag.Bool("wf_enable_min_month", true, "enable minimum monthly return constraint")
	wfMinMonth := flag.Float64("wf_min_month", -0.35, "minimum monthly return threshold")
	wfMinMedianMonthly := flag.Float64("wf_min_median_monthly", 0.003, "minimum median monthly return")
	wfMinGeoMonthly := flag.Float64("wf_min_geo_monthly", 0.005, "minimum geometric average monthly return")
	// FIX PROBLEM B: Increased from 0.03 to 30.0 to prevent zero-trade OOS rejections
	// 0.03 was way too low and allowed "dead" strategies through that never trade in OOS
	// 30 edges/year = ~2.5 trades/month minimum for statistical significance
	wfMinEdgesPerYear := flag.Float64("wf_min_edges_per_year", 30.0, "minimum edges per year threshold (FIX: increased from 0.03 to reduce zero-trade OOS)")
	// FIX PROBLEM B: Enable per-fold edge gate with meaningful threshold
	// Was 0 (disabled), now set to 15.0 to ensure each fold has minimum trading activity
	// This prevents strategies that only trade in 1-2 folds from passing
	wfFoldMinEdgesPerYear := flag.Float64("wf_fold_min_edges_per_year", 15.0, "minimum edges per year per-fold threshold (FIX: increased from 0 to reduce sparse fold rejections)")
	wfDDLambda := flag.Float64("wf_dd_lambda", 0.008, "drawdown penalty lambda")
	wfVolLambda := flag.Float64("wf_vol_lambda", 0.004, "volatility penalty lambda")
	_ = flag.Float64("target_geo_monthly", 0.12, "target geometric monthly return for scoring (reserved for future use)")
	minValReturnFlag := flag.Float64("min_val_return", 0.01, "minimum validation return threshold")
	minValPFFlag := flag.Float64("min_val_pf", 1.01, "minimum validation portfolio factor")
	maxValDDFlag := flag.Float64("max_val_dd", 0.45, "maximum validation drawdown")
	autoAdjust := flag.Bool("auto_adjust", false, "Enable automatic tightening of validation gates during search")
	flag.Parse()

	// Parse timeframe (e.g., "5m", "15m", "60m" OR "1h", "4h")
	tfMinutes := parseTimeframe(*timeframe)
	atomic.StoreInt32(&globalTimeframeMinutes, tfMinutes)

	// Set global verbose flag from command line
	Verbose = *verbose
	LogTrades = *logTrades
	HoldingHeartbeat = *holdingHeartbeat
	DebugPF = *debugPF
	DebugWalkForward = *debugWalkForward
	RecoveryMode.Store(*recoveryMode)
	VerifyMode = *verifyMode

	// Set walk-forward configuration from flags
	wfConfig = WFConfig{
		Enable:                   *wfEnable,
		TrainDays:                *wfTrainDays,
		TestDays:                 *wfTestDays,
		StepDays:                 *wfStepDays,
		MinFolds:                 3,
		MinTradesPerFold:         10,
		MinMonths:                3,
		MinTotalTradesOOS:        *wfMinTradesOOS,
		MinTradesPerMonth:        2,
		MaxDrawdown:              *wfMaxDD,
		MinMonthReturn:           *wfMinMonth,
		EnableMinMonthConstraint: *wfEnableMinMonth,
		MinGeoMonthlyReturn:      *wfMinGeoMonthly,
		MinActiveMonthsRatio:     *wfMinActiveMonthsRatio,
		MaxSparseMonthsRatio:     *wfMaxSparseMonthsRatio,
		MinMedianMonthly:         *wfMinMedianMonthly,
		MaxStdMonth:              0.5,
		MonthlyVolLambda:         *wfVolLambda,
		DDPenaltyLambda:          *wfDDLambda,
		LambdaNodes:              0.001,
		LambdaFeats:              0.001,
		LambdaDepth:              0.001,
		MaxNodes:                 50,
		MaxFeatures:              12,
		MinEdgesPerYear:          *wfMinEdgesPerYear,
		FoldMinEdgesPerYear:      *wfFoldMinEdgesPerYear,
	}
	VerifyMode = *verifyMode

	// Convert flag values to float32
	feeBps := float32(*feeBpsFlag)
	slipBps := float32(*slipBpsFlag)

	// Track which flags were explicitly set for friction reduction
	feeBpsWasSet := false
	slipBpsWasSet := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "fee_bps" {
			feeBpsWasSet = true
		}
		if f.Name == "slip_bps" {
			slipBpsWasSet = true
		}
	})

	// Auto-reduce friction in recovery mode ONLY if not explicitly set
	if RecoveryMode.Load() {
		if !feeBpsWasSet && *feeBpsFlag == 30 { // Default value
			feeBps = float32(5)
			fmt.Println("[RECOVERY] Fee auto-reduced to 5 bps (was 30)")
			fmt.Println("[RECOVERY] Override with explicit -fee_bps flag")
		}
		if !slipBpsWasSet && *slipBpsFlag == 8 { // Default value
			slipBps = float32(5)
			fmt.Println("[RECOVERY] Slippage auto-reduced to 5 bps (was 8)")
			fmt.Println("[RECOVERY] Override with explicit -slip_bps flag")
		}
	}

	// STARTUP CONFIG: Print key configuration values for debugging
	fmt.Printf("=== STARTUP CONFIGURATION ===\n")
	fmt.Printf("Fee: %.2f bps (%.3f%%)\n", feeBps, feeBps/100.0)
	fmt.Printf("Slippage: %.2f bps (%.3f%%)\n", slipBps, slipBps/100.0)
	fmt.Printf("Warmup (custom): %d bars\n", *warmupBars)
	fmt.Printf("=============================\n\n")

	// Part C1: Log score formula at startup
	fmt.Printf("====================================================================================================\n")
	fmt.Printf("SCORING FORMULA: v3 (computeScore with smoothness)\n")
	fmt.Printf("  Components: logReturn*6.0 + calmar*2.0 + expectancy*200.0 + tradesReward*0.01 - ddPenalty*6.0\n")
	fmt.Printf("              + sortino*0.5 - smoothPenalty*0.5 - deflationPenalty\n")
	fmt.Printf("  Caps: calmar=[-inf,10], expectancy=[-5,5], sortino=[-10,20]\n")
	fmt.Printf("  Sentinel: -1e30 for hard rejection (trades==0 || dd>=0.45)\n")
	fmt.Printf("====================================================================================================\n\n")

	// Define scoring thresholds based on mode
	var maxValDD, minValReturn, minValPF, minValExpect float32
	var minValTrades int
	if *scoringMode == "aggressive" {
		// Aggressive mode: stricter profit gates, looser DD to encourage exploration
		maxValDD = float32(*maxValDDFlag)
		minValReturn = float32(*minValReturnFlag)
		minValPF = float32(*minValPFFlag)
		minValExpect = 0.0 // TEMP warm-start: allow any expectancy
		minValTrades = 20  // TEMP warm-start: 20 trades until elites exist, then 30
		fmt.Printf("Scoring mode: AGGRESSIVE (DD<%.2f, ret>%.1f%%, pf>%.2f, exp>%.4f) [WARM-START]\n", maxValDD, minValReturn*100, minValPF, minValExpect)
	} else {
		// Balanced mode (default): balanced gates for stability and exploration
		maxValDD = float32(*maxValDDFlag)
		minValReturn = float32(*minValReturnFlag)
		minValPF = float32(*minValPFFlag)
		minValExpect = 0.0 // TEMP warm-start: allow any expectancy
		minValTrades = 20  // TEMP warm-start: 20 trades until elites exist, then 30
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
		ScreenRelaxLevel: 1, // CRITICAL FIX #6: Default to normal screening (was 3)
		// The complexity rule in strategy.go prevents volume-only junk strategies,
		// so we can keep screening strict without needing desperate relaxation
	}

	// Part C2: Log gate thresholds at startup
	relaxLevel := meta.ScreenRelaxLevel
	relaxNames := []string{"Strict", "Normal", "Relaxed", "Very_Relaxed"}
	relaxName := "Unknown"
	if relaxLevel >= 0 && relaxLevel < len(relaxNames) {
		relaxName = relaxNames[relaxLevel]
	}
	fmt.Printf("GATE THRESHOLDS:\n")
	fmt.Printf("  Val:    minReturn=%.4f, minPF=%.2f, minExp=%.4f, maxDD=%.2f, minTrades=%d\n",
		meta.MinValReturn, meta.MinValPF, meta.MinValExpect, meta.MaxValDD, meta.MinValTrades)
	fmt.Printf("  RelaxLevel: %d (0=strict, 1=normal, 2=relaxed, 3=very-relaxed) [%s]\n", relaxLevel, relaxName)
	fmt.Printf("====================================================================================================\n\n")

	// Mode dispatch
	switch *mode {
	case "trace":
		runTraceMode(*dataFile, *traceSeed, *traceCSVPath, *traceManual, *traceOpenCSV, *feeBpsFlag, *slipBpsFlag, *traceWindow)
		return
	case "golden":
		runGoldenMode(*dataFile, *goldenSeed, *goldenN, *feeBpsFlag, *slipBpsFlag, *exportCSVPath, *exportStatesPath)
		return
	case "test":
		runTestMode(*dataFile, feeBps, slipBps, *fromIdx, *toIdx, *fromTime, *toTime, *warmupBars)
		return
	case "validate":
		RunValidation("btc_5min_data.csv")
		return
	case "diagnose":
		runDiagnosticMode(*dataFile)
		return
	case "verify":
		runVerifyMode(*dataFile)
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

	// Track search start time for runtime display
	// Initialized here (before saveNow) so it's accessible throughout
	var searchStartTime time.Time
	var searchStartTimeMu sync.Mutex

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

	series, err := LoadBinanceKlinesCSV(*dataFile)
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
	hof := NewHallOfFame(200) // keep top 200 by validation score (increased to preserve diversity)

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

	restoreHOFSlim := func(h *HallOfFame, slimElites []SlimElite, rng *rand.Rand, feats *Features) {
		h.mu.Lock()
		defer h.mu.Unlock()
		validElites := make([]Elite, 0, len(slimElites))
		for _, se := range slimElites {
			elite, err := slimToElite(se, rng, feats)
			if err != nil {
				// Silently skip invalid elites - will log summary below
				continue
			}
			validElites = append(validElites, elite)
		}
		h.Elites = validElites

		// Log summary using structured logging
		droppedCount := len(slimElites) - len(validElites)
		if droppedCount > 0 {
			logx.LogHOFRestore(len(validElites), len(slimElites), droppedCount)
		}
	}

	// Track passed validations for adaptive criteria
	var passedCount int64
	var validatedLabels int64       // Track number of validation labels for surrogate training
	var bestValSeen float32 = -1e30 // Track best validation score seen (init to very low for bootstrap)
	var bestValSeenMu sync.Mutex    // Protect bestValSeen for thread-safe access

	// Anti-stagnation tracking (thread-safe)
	var radicalP float32 = 0.10     // base radical mutation probability
	var surExploreP float32 = 0.10  // base surrogate exploration
	var surThreshold float64 = -0.5 // surrogate filtering threshold - start more aggressive to filter junk early
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
	// Use CoarseFingerprint for deduplication (threshold buckets)
	// This reduces duplicates more effectively than SkeletonFingerprint
	// Full fingerprint is still used for leaderboard uniqueness via globalFingerprints
	// OPTIMIZATION: Use sharded map to reduce mutex contention (3-5x speedup)
	seen := NewShardedSeenMap()

	// markSeen checks if a strategy has been seen before
	// When hof.Len() == 0: use full Fingerprint() for maximum exploration (exact match only)
	// When hof.Len() > 0: use CoarseFingerprint() for faster deduplication (threshold buckets)
	// Returns true if NEW (just inserted), false if ALREADY SEEN
	// Increments exact counters: seenNew when new, seenHits when already seen
	markSeen := func(s Strategy) bool {
		var fp string
		if hof.Len() == 0 {
			// Bootstrap mode: use exact fingerprint to maximize exploration
			fp = s.Fingerprint()
		} else {
			// Normal mode: use coarse fingerprint for speed
			fp = s.CoarseFingerprint()
		}
		isNew := seen.CheckAndSet(fp)
		if isNew {
			atomic.AddInt64(&seenNew, 1) // CheckAndSet returned true (new insert)
		} else {
			atomic.AddInt64(&seenHits, 1) // CheckAndSet returned false (already existed)
		}
		return isNew
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

	// rerollOnSeen attempts to mutate a seen strategy to find an unseen variant
	// Does NOT call CheckAndSet - caller must call markSeen after this returns
	// Performs multiple mutations until maxRetries, returns last mutation
	rerollOnSeen := func(st Strategy, fullF Features, maxRetries int) Strategy {
		current := st

		for attempt := 0; attempt < maxRetries; attempt++ {
			// Create a mutated version
			current = mutateStrategy(rng, current, fullF)

			// NOTE: We DON'T call CheckAndSet here - caller will do that
			// Just count this as a reroll attempt and continue to next iteration
			// Only count on first successful reroll (attempt == 0)
			if attempt == 0 {
				atomic.AddInt64(&genRerolled, 1)
			}
		}

		// Return the last mutated version
		return current
	}

	// Load checkpoint if provided (must be after HOF and RNG initialization)
	if *resumePath != "" {
		cp, err := LoadCheckpoint(*resumePath)
		if err != nil {
			fmt.Printf("WARN: failed to load checkpoint: %v\n", err)
		} else {
			fmt.Printf("Resuming from checkpoint: %s (saved %d)\n", *resumePath, cp.SavedAtUnix)

			// === FEATURE MAP SAFETY PRINT FOR CHECKPOINT LOADING ===
			runtimeHash := ComputeFeatureMapHash(feats)
			runtimeVersion := GetFeatureMapVersion(feats)
			fmt.Println("\n=== CHECKPOINT LOADED - Feature Map Info ===")
			fmt.Printf("FeatureMapHash: %s\n", runtimeHash)
			fmt.Printf("FeatureSetVersion: %s\n", runtimeVersion)
			fmt.Println("=============================================")

			// restore hof (from slim elites)
			restoreHOFSlim(hof, cp.HOFElites, rng, &feats)
			// OPTIMIZATION: Initialize HOF snapshot for lock-free reads
			hof.InitSnapshot()

			// restore archive (from slim elites)
			archiveLoaded := 0
			for _, se := range cp.ArchiveElites {
				e, err := slimToElite(se, rng, &feats)
				if err != nil {
					continue // Skip invalid elites
				}
				archive.Add(e.Val, e)
				archiveLoaded++
			}
			droppedArchive := len(cp.ArchiveElites) - archiveLoaded
			if droppedArchive > 0 {
				logx.LogArchiveRestore(archiveLoaded, len(cp.ArchiveElites), droppedArchive)
			}

			// Fallback: if no valid elites loaded, log clear message
			// The search will bootstrap fresh population automatically
			if hof.Len() == 0 {
				fmt.Println("[CHECKPOINT] Warning: No valid HOF elites found in checkpoint.")
				fmt.Println("[CHECKPOINT] Starting with empty population - will bootstrap fresh strategies.")
			}

			// === CHECKPOINT HASH VALIDATION ===
			if cp.FeatureMapHash != "" && cp.FeatureMapHash != runtimeHash {
				fmt.Printf("\n!!! CHECKPOINT FEATURE MAP MISMATCH DETECTED !!!\n")
				fmt.Printf("Checkpoint hash:  %s\n", cp.FeatureMapHash)
				fmt.Printf("Runtime hash:    %s\n\n", runtimeHash)
				fmt.Printf("Checkpoint version: %s\n\n", truncateVersion(cp.FeatureMapHash, runtimeVersion, 5))
				fmt.Printf("Runtime version:    %s\n", truncateVersion(runtimeHash, runtimeVersion, 5))
				fmt.Printf("\nWARNING: Checkpoint was saved with different feature ordering!\n")
				fmt.Printf("All elites from this checkpoint will use WRONG feature indices.\n")
				fmt.Printf("RECOMMENDATION: Start fresh or regenerate with current feature order.\n")
				fmt.Printf("====================================================\n\n")
			} else if cp.FeatureMapHash != "" {
				fmt.Printf("\n%s Checkpoint feature map hash validated: MATCH\n", logx.Success("✓"))
			} else {
				fmt.Printf("\n%s Legacy checkpoint (no feature_map_hash stored)\n", logx.Warn("⚠"))
				fmt.Printf("   Cannot validate feature ordering. May produce WRONG results.\n")
			}

			// restore counters
			atomic.StoreInt64(&passedCount, cp.PassedCount)
			atomic.StoreInt64(&validatedLabels, cp.ValidatedLabels)
			bestValSeenMu.Lock()
			bestValSeen = cp.BestValSeen
			bestValSeenMu.Unlock()
			// Initialize testedAtLastCheckpoint for resume
			// Will be set in aggregator goroutine at first checkpoint

			// restore seen - use sharded map's Restore method
			seen.Restore(cp.SeenFingerprints)

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

		// OPTIMIZATION: Use sharded map's Snapshot method (thread-safe)
		allFingerprints := seen.Snapshot()
		seenList := allFingerprints
		if len(seenList) > maxSeen {
			seenList = seenList[:maxSeen]
		}

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
			Seed:              seed,
			PassedCount:       atomic.LoadInt64(&passedCount),
			ValidatedLabels:   atomic.LoadInt64(&validatedLabels),
			BestValSeen:       func() float32 { bestValSeenMu.Lock(); defer bestValSeenMu.Unlock(); return bestValSeen }(),
			HOFElites:         snapshotHOFSlim(hof),
			ArchiveElites:     archiveSlim,
			SeenFingerprints:  seenList,
			FeatureMapHash:    ComputeFeatureMapHash(feats),
			FeatureMapVersion: GetFeatureMapVersion(feats),
		}
		// Print rejection stats before checkpointing
		printRejectionStats()

		// Calculate elapsed time since search started
		searchStartTimeMu.Lock()
		elapsed := time.Since(searchStartTime)
		searchStartTimeMu.Unlock()

		if err := SaveCheckpoint(*checkpointPath, cp); err != nil {
			fmt.Printf("WARN: checkpoint save failed: %v\n", err)
		} else {
			fmt.Printf("\n%s Checkpoint saved: %s (runtime: %s)\n", logx.Success("✓"), *checkpointPath, logx.FormatDuration(elapsed))
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
					RiskPct:     1.0,
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
					StopLoss:         parseStopModel(log.StopLoss),
					TakeProfit:       parseTPModel(log.TakeProfit),
					Trail:            parseTrailModel(log.Trail),
					VolatilityFilter: VolFilterModel{Enabled: false}, // Default disabled for old strategies
				}

				// SANITY CHECK: Validate cross operations when loading from disk
				// This prevents strategies with invalid CrossUp/CrossDown from being loaded
				{
					_, entryInvalid := validateCrossSanity(s.EntryRule.Root, feats)
					_, exitInvalid := validateCrossSanity(s.ExitRule.Root, feats)
					_, regimeInvalid := validateCrossSanity(s.RegimeFilter.Root, feats)

					totalInvalid := entryInvalid + exitInvalid + regimeInvalid
					if totalInvalid > 0 {
						// Reject this strategy due to invalid cross operations
						atomic.AddInt64(&rejectedCrossSanityLoad, 1)
						continue // Skip this strategy
					}
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

	// Track recent fingerprints for novelty pressure (penalize too-similar children)
	// This reduces the duplicate storm by rejecting strategies too similar to recent ones
	const recentFingerprintsWindow = 5000         // Track last 5000 generated fingerprints
	var recentFingerprints = make(map[string]int) // fingerprint -> count
	var recentFingerprintsMu sync.RWMutex
	var totalRecentFingerprints int64 = 0

	// Fingerprint frequency monitoring for canonicalization
	// Log top 20 most common fingerprints every 10 batches to detect stuck generators
	var fingerprintBatchCounter int64

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

		// Initialize search start time now that the search loop is starting
		searchStartTimeMu.Lock()
		searchStartTime = time.Now()
		searchStartTimeMu.Unlock()

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
				if *autoAdjust && passRate > 0.03 { // 3% passing is "too easy" in big search
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
				logx.LogMetaStagnate(batchesNoImprove, oldRadical, radicalP, oldSurExplore, surExploreP, passRate)

				// If pass rate is basically zero, gate is too strict -> loosen a bit (disable with -auto_adjust=false)
				if *autoAdjust && passRate < 0.002 { // <0.2%
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
						// Format score: show REJECTED for sentinel values, otherwise 3 decimals
						scoreStr := "REJECTED"
						if testR.Score > -1e20 {
							scoreStr = fmt.Sprintf("%.3f", testR.Score)
						}
						print("Test %d/%d: Score=%s Return=%.2f%% WinRate=%.1f%% Trades=%d\n",
							i+1, numToTest, scoreStr, testR.Return*100, testR.WinRate*100, testR.Trades)

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
					fingerprintBatchCounter++

					// CANONICALIZATION CHECK: Log fingerprint frequency every 10 batches
					// This detects if the generator is stuck producing the same patterns
					if fingerprintBatchCounter%10 == 0 {
						recentFingerprintsMu.RLock()
						// Build list of fingerprint counts for sorting
						type fpCount struct {
							fp    string
							count int
						}
						var fpList []fpCount
						for fp, count := range recentFingerprints {
							fpList = append(fpList, fpCount{fp: fp, count: count})
						}
						recentFingerprintsMu.RUnlock()

						// Sort by count (descending) to find most common fingerprints
						sort.Slice(fpList, func(i, j int) bool {
							return fpList[i].count > fpList[j].count
						})

						// Log top 10 fingerprints
						topN := 10
						if len(fpList) < topN {
							topN = len(fpList)
						}
						if topN > 0 && fpList[0].count > 10 { // Only log if there's significant repetition
							fmt.Printf("\n  [FINGERPRINT CANONICALIZATION] Top %d skeletons (last %d generated):\n",
								topN, totalRecentFingerprints)
							for i := 0; i < topN; i++ {
								fmt.Printf("    #%d: count=%d, fp=%s\n", i+1, fpList[i].count, fpList[i].fp[:80])
							}
							// Warn if generator is stuck
							if fpList[0].count > 50 {
								fmt.Printf("    %s WARNING: Generator stuck! Top fingerprint appears %d times\n",
									logx.Warn("⚠"), fpList[0].count)
							}
						}
					}

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
					bestQuickTestR.Score = -1e30

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
					var batchAccepted []Result  // Part D2: Track accepted candidates for emergency seeding
					// Part D2: Track TRAIN-passed candidates for recovery seeding (with memory cap)
					const maxTrainPassedCandidates = 200 // Cap memory growth - keep top 200 by score
					type trainCandidate struct {
						Strategy Strategy
						Result   Result
					}
					var trainPassedCandidates []trainCandidate

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

						// Update surrogate's exploration probability with adaptive scaling
						stagnationMu.RLock()
						currentSurExploreP := surExploreP
						stagnationMu.RUnlock()

						// Scale exploration probability based on hof size
						// When hof.Len() is 50-100: higher exploration (70-90%) to limit rejection to 10-30%
						// When hof.Len() >= 100: normal exploration (10%)
						hof.mu.RLock()
						hofLen := hof.Len()
						hof.mu.RUnlock()

						scaledExploreP := currentSurExploreP
						if hofLen >= 50 && hofLen < 100 {
							// Map 50->100 elites to exploreP 0.90->0.70
							// This gives rej_sur ~10-30% (vs current ~72%)
							t := float32(hofLen-50) / 50.0 // 0.0 at 50 elites, 1.0 at 100 elites
							scaledExploreP = 0.90 - t*0.20 // 0.90 at 50, 0.70 at 100
						} else if hofLen >= 100 {
							scaledExploreP = currentSurExploreP // Use meta-controlled value (default 0.10)
						}

						sur.mu.Lock()
						sur.exploreP = float64(scaledExploreP)
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

						// RECOVERY MODE: Auto-toggle minEdges based on elite count (hysteresis to prevent flapping)
						// Enable minEdges=2 when elites < 25, disable when elites >= 30
						if !recoveryModeActive.Load() && elitesCount < 25 {
							recoveryModeActive.Store(true)
							setEdgeMinMultiplier(2)
							currentMinEdges := getEdgeMinMultiplier()
							if currentMinEdges < 2 {
								currentMinEdges = 2
							}
							fmt.Printf("[AUTO-RECOVERY] ENABLED (elites=%d < 25), minEdges=%d\n", elitesCount, currentMinEdges)
						} else if recoveryModeActive.Load() && elitesCount >= 30 {
							recoveryModeActive.Store(false)
							setEdgeMinMultiplier(3)
							currentMinEdges := getEdgeMinMultiplier()
							if currentMinEdges < 2 {
								currentMinEdges = 2
							}
							fmt.Printf("[AUTO-RECOVERY] DISABLED (elites=%d >= 30), minEdges=%d\n", elitesCount, currentMinEdges)
						}

						// THREE-TIER LADDER: Bootstrap (0-9) → Recovery (10-19) → Standard (20+)
						const bootstrapThreshold = 10
						const recoveryThreshold = 20

						var effectiveMaxDD, effectiveMinReturn, effectiveMinPF, effectiveMinExpect, effectiveMinScore float32
						var effectiveMinTrades int
						var skipCPCV bool
						var passesValidation bool
						var isPreElite bool = false

						// Variables for rejection logging (must be in outer scope)
						var basicGatesOK, profitOK, trainOK, stableEnough bool

						if elitesCount < bootstrapThreshold {
							// BOOTSTRAP MODE (0-9 elites): Very relaxed VAL gates
							effectiveMaxDD = 0.65
							effectiveMinReturn = -0.02
							effectiveMinPF = 0.95
							effectiveMinExpect = -0.0005
							effectiveMinScore = -50.0
							effectiveMinTrades = 20
							skipCPCV = true

							// Basic sanity gates
							basicGatesOK = valR.MaxDD < effectiveMaxDD && valR.Trades >= effectiveMinTrades
							profitOK = valR.Return >= effectiveMinReturn &&
								valR.Expectancy > effectiveMinExpect &&
								valR.ProfitFactor >= effectiveMinPF &&
								valR.Score >= effectiveMinScore

							var minTrainScore, minTrainReturn float32
							minTrainScore = -10.0
							minTrainReturn = -0.05
							trainOK = candidate.Score > minTrainScore || candidate.Return > minTrainReturn

							stableEnough = true
							if candidate.Score > 0 {
								stableEnough = valR.Score >= stabilityThreshold*candidate.Score
							}
							passesValidation = basicGatesOK && stableEnough && profitOK && trainOK

						} else if elitesCount < recoveryThreshold {
							// RECOVERY MODE (10-19 elites): Accept based on train, bypass VAL profit gates
							// Check: VAL DD + VAL trades, train score + train return
							recoveryValOK := valR.MaxDD < 0.65 && valR.Trades >= 15                // Basic VAL sanity
							recoveryTrainOK := candidate.Score > -20.0 || candidate.Return > -0.10 // Train not terrible

							if recoveryValOK && recoveryTrainOK {
								passesValidation = false // Don't use normal validation path
								isPreElite = true        // Mark for special handling below
								skipCPCV = true
								fmt.Printf("[PRE-ELITE] train_score=%.2f train_ret=%.2f%% val_trades=%d val_dd=%.2f%%\n",
									candidate.Score, candidate.Return*100, valR.Trades, valR.MaxDD*100)
							}

							// Set default values for rejection logging (not used for pre-elite path)
							effectiveMaxDD = 0.65
							effectiveMinReturn = -0.02
							effectiveMinPF = 0.95
							effectiveMinExpect = -0.0005
							effectiveMinScore = -50.0
							effectiveMinTrades = 15
							basicGatesOK = recoveryValOK
							profitOK = true
							trainOK = recoveryTrainOK
							stableEnough = true

						} else {
							// STANDARD MODE (20+ elites): Full validation gates
							effectiveMaxDD = maxValDD
							effectiveMinReturn = minValReturn
							effectiveMinPF = minValPF
							effectiveMinExpect = minValExpect
							effectiveMinScore = minValScore
							effectiveMinTrades = minValTrades
							skipCPCV = false

							basicGatesOK = valR.MaxDD < effectiveMaxDD && valR.Trades >= effectiveMinTrades
							profitOK = valR.Return >= effectiveMinReturn &&
								valR.Expectancy > effectiveMinExpect &&
								valR.ProfitFactor >= effectiveMinPF &&
								valR.Score >= effectiveMinScore

							var minTrainScore, minTrainReturn float32
							minTrainScore = -5.0
							minTrainReturn = 0.0
							trainOK = candidate.Score > minTrainScore || candidate.Return > minTrainReturn

							stableEnough = true
							if candidate.Score > 0 {
								stableEnough = valR.Score >= stabilityThreshold*candidate.Score
							}
							passesValidation = basicGatesOK && stableEnough && profitOK && trainOK
						}

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
							// Anti-overfit: log train gate failures
							if !trainOK {
								reasons = append(reasons, fmt.Sprintf("train_bad(score=%.2f,ret=%.1f%%)", candidate.Score, candidate.Return*100))
							}
							reasonStr := strings.Join(reasons, ",")
							fmt.Printf("[VAL-REJECT] score=%.4f ret=%.2f%% pf=%.2f exp=%.5f dd=%.3f trds=%d [%s]\n",
								valR.Score, valR.Return*100, valR.ProfitFactor, valR.Expectancy, valR.MaxDD, valR.Trades, reasonStr)

							// EMERGENCY SEEDING: Always track candidates that passed TRAIN but failed VAL
							// This enables recovery seeding even when VAL never passes
							// Track candidates with reasonable trades that passed multi-fidelity
							if candidate.Trades >= 10 {
								// Track TRAIN-passed candidates for emergency seeding
								cand := trainCandidate{
									Strategy: candidate.Strategy,
									Result:   candidate, // Use train result
								}
								trainPassedCandidates = append(trainPassedCandidates, cand)

								// Cap memory: keep only top N by train score
								if len(trainPassedCandidates) > maxTrainPassedCandidates {
									sort.Slice(trainPassedCandidates, func(i, j int) bool {
										return trainPassedCandidates[i].Result.Score > trainPassedCandidates[j].Result.Score
									})
									trainPassedCandidates = trainPassedCandidates[:maxTrainPassedCandidates]
								}
							}
						}

						if passesValidation || isPreElite {
							// Track best validation score seen ONLY from fully validated strategies
							// This prevents bestValSeen from tracking failing candidates
							// PRE-ELITE: Don't update bestValSeen for recovery mode candidates
							if !isPreElite {
								bestValSeenMu.Lock()
								if valR.Trades >= minValTrades && valR.Score > -1e20 && valR.Score > bestValSeen {
									bestValSeen = valR.Score
								}
								bestValSeenMu.Unlock()
							}

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

							// CRITICAL FIX: Always compute QuickTest for reporting, but only enforce gates after elites exist
							// This avoids "REJECTED / 0 trades" display when QuickTest is intentionally skipped.
							quickTestProfitOK := true // Default: don't veto during bootstrap

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

							// DEBUG: Print window info before evaluation
							windowCandles := quickTestEnd - testW.Start
							firstCandleTime := int64(0)
							if testW.Start < len(series.OpenTimeMs) {
								firstCandleTime = series.OpenTimeMs[testW.Start]
							}
							fmt.Printf("[QuickTest-DEBUG] window=[%d:%d] candles=%d first_candle_ts=%d\n",
								testW.Start, quickTestEnd, windowCandles, firstCandleTime)

							quickTestR := evaluateStrategyWindow(series, feats, candidate.Strategy, quickTestW)

							// DEBUG: Print result after evaluation
							fmt.Printf("[QuickTest-DEBUG] result: trades=%d return=%.4f pf=%.2f score=%.4f\n",
								quickTestR.Trades, quickTestR.Return, quickTestR.ProfitFactor, quickTestR.Score)

							// Track best quick test result for reporting (best score among all candidates)
							if quickTestR.Score > bestQuickTestR.Score {
								bestQuickTestR = quickTestR
							}

							// Only ENFORCE QuickTest gates after elites exist
							if elitesCount > 0 {
								// If -qt_min_trades=0, treat QuickTest as informational only (never reject)
								if *qtMinTrades == 0 {
									// quickTestProfitOK stays true (default), skip all gate enforcement
								} else {
								// BOOTSTRAP LADDER: Relax QuickTest during bootstrap (elites < 10)
								// When elite pool is small, allow candidates with weaker QuickTest results
								var minQuickTestTrades int
								var minQuickTestReturn float32
								if elitesCount < 10 {
									// Relaxed QuickTest gates during bootstrap
									minQuickTestTrades = *qtMinTrades
									minQuickTestReturn = -0.05 // Allow small loss during bootstrap
								} else {
									// Standard QuickTest gates once we have enough elites
									minQuickTestTrades = *qtMinTrades
									minQuickTestReturn = 0.0
								}

								// Skip QuickTest profit check if trades == 0
								// Small test window may have 0 trades even if full validation has trades
								if quickTestR.Trades > 0 {
									// CRITICAL FIX: Require positive edge explicitly, not relative to minValReturn
									// After elites exist, minValReturn could be 0.0, which doesn't protect against losing strategies

									// FIX PROBLEM A: Reject "no-loss / one-trade" PF inflation
									// PF=999 with 1 trade is NOT a robust strategy - it's statistical noise
									// Require meaningful PF by checking both wins AND losses exist
									minWinsLossesForPF := 3 // Need at least 3 wins AND 3 losses for meaningful PF
									hasMeaningfulPF := true
									if quickTestR.ProfitFactor >= 999.0 {
										// Infinite PF cap hit - check if this is due to too few trades
										// Compute approximate wins/losses from winrate
										estimatedWins := int(float64(quickTestR.Trades) * float64(quickTestR.WinRate))
										estimatedLosses := quickTestR.Trades - estimatedWins
										if estimatedWins < minWinsLossesForPF || estimatedLosses < minWinsLossesForPF {
											hasMeaningfulPF = false // Reject: not enough data for meaningful PF
										}
									}

									quickTestProfitOK = quickTestR.Return >= minQuickTestReturn && // Use bootstrap-adjusted threshold
										quickTestR.ProfitFactor >= 1.0 && // Must be profitable after costs
										hasMeaningfulPF && // FIX A: Reject PF=999 with insufficient trades
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
									// FIX PROBLEM A: Add logging for PF inflation rejection
									if quickTestR.ProfitFactor >= 999.0 {
										estimatedWins := int(float64(quickTestR.Trades) * float64(quickTestR.WinRate))
										estimatedLosses := quickTestR.Trades - estimatedWins
										if estimatedWins < 3 || estimatedLosses < 3 {
											reasons = append(reasons, fmt.Sprintf("QT_pf_inflated_wins=%d_losses=%d", estimatedWins, estimatedLosses))
										}
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
								} // end of else block (QuickTest gate enforcement)
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

							// Part A1: Score sanity check BEFORE adding elite
							if !IsValidScore(valR.Score) {
								fmt.Printf("[INVALID-SCORE] Rejecting elite with invalid score=%.4f\n", valR.Score)
								continue
							}

							// Part D2: Track for emergency seeding
							batchAccepted = append(batchAccepted, candidate)

							elite := Elite{
								Strat:      candidate.Strategy,
								Train:      candidate,
								Val:        valR, // Use current candidate's validation result
								ValScore:   valR.Score,
								IsPreElite: isPreElite, // Mark recovery mode elites
							}
							hof.Add(elite)
							archive.Add(valR, elite) // Use current candidate's validation result

							// CRITICAL FIX #2 & #3: Exit bootstrap mode once we have enough elites
							// This enables normal cooldown (200) and MaxHoldBars (150-329) values
							// and enables strict profit gates for elite acceptance
							const bootstrapEliteThreshold = 8
							if hof.Len() >= bootstrapEliteThreshold && isBootstrapMode() {
								setBootstrapMode(false)
								logx.LogBootstrapComplete(hof.Len())
								print("[Cooldown: 0-50 -> 200, MaxHoldBars: 50-150 -> 150-329, Profit gates: ENABLED]\n\n")
							}
							atomic.AddInt64(&strategiesPassed, 1) // Count strategies that truly pass all validation gates

							// Compute feature map hash for this run
							featureHash := ComputeFeatureMapHash(feats)

							// Also log to persistent file
							log := EliteLog{
								Seed:           candidate.Strategy.Seed,
								FeeBps:         candidate.Strategy.FeeBps,
								SlippageBps:    candidate.Strategy.SlippageBps,
								Direction:      candidate.Strategy.Direction,
								StopLoss:       stopModelToString(candidate.Strategy.StopLoss),
								TakeProfit:     tpModelToString(candidate.Strategy.TakeProfit),
								Trail:          trailModelToString(candidate.Strategy.Trail),
								EntryRule:      ruleTreeToString(candidate.Strategy.EntryRule.Root),
								ExitRule:       ruleTreeToString(candidate.Strategy.ExitRule.Root),
								RegimeFilter:   ruleTreeToString(candidate.Strategy.RegimeFilter.Root),
								FeatureMapHash: featureHash,
								TrainScore:     candidate.Score,
								TrainReturn:    candidate.Return,
								TrainMaxDD:     candidate.MaxDD,
								TrainWinRate:   candidate.WinRate,
								TrainTrades:    candidate.Trades,
								ValScore:       valR.Score,
								ValReturn:      valR.Return,
								ValMaxDD:       valR.MaxDD,
								ValWinRate:     valR.WinRate,
								ValTrades:      valR.Trades,
							}
							select {
							case winnerLog <- log:
							default:
							}
						}
					}

					// Part D2: RECOVERY SEEDING - Seed elites when population is low
					// Priority: trainPassedCandidates (passed SCREEN+TRAIN but failed VAL profit gates)
					var seedCandidates []Elite

					// First: try trainPassedCandidates (they passed SCREEN+TRAIN)
					if len(trainPassedCandidates) > 0 {
						for _, c := range trainPassedCandidates {
							seedCandidates = append(seedCandidates, Elite{
								Strat:    c.Strategy,
								Train:    c.Result,
								Val:      c.Result, // Use train result as proxy
								ValScore: c.Result.Score,
							})
						}
					}

					// Second: fallback to batchAccepted (VAL-passed + all gates)
					if len(seedCandidates) == 0 && len(batchAccepted) > 0 {
						for _, r := range batchAccepted {
							seedCandidates = append(seedCandidates, Elite{
								Strat:    r.Strategy,
								Train:    r,
								Val:      r,
								ValScore: r.Score,
							})
						}
					}

					// Seeding logic (expanded in recovery mode)
					if hof.Len() == 0 && len(seedCandidates) > 0 {
						maxSeeds := 5
						if RecoveryMode.Load() {
							maxSeeds = 20 // Expanded seeding in recovery mode
							fmt.Println("[RECOVERY] Emergency seeding up to 20 elites from", len(seedCandidates), "candidates")
						}

						// CRITICAL: Use hof.Add() for each candidate to enforce fingerprint diversity
						// Do NOT use SeedFromCandidates which may bypass diversity rules
						seeded := 0
						for _, cand := range seedCandidates {
							if seeded >= maxSeeds {
								break
							}
							beforeLen := hof.Len()
							hof.Add(cand) // hof.Add enforces fingerprint diversity (max 2 per fingerprint family)
							if hof.Len() > beforeLen {
								seeded++
							}
						}
						fmt.Printf("[RECOVERY] Seeded %d/%d elites (HOF now has %d)\n", seeded, maxSeeds, hof.Len())
					}

					// Reset tracking slices
					trainPassedCandidates = nil
					batchAccepted = nil

					// After validation loop: run self-improvement meta-update
					// Compute actual tested since last checkpoint for correct pass-rate
					testedNow := int64(atomic.LoadUint64(&tested))
					testedThisCheckpoint := testedNow - testedAtLastCheckpoint
					if testedThisCheckpoint == 0 {
						testedThisCheckpoint = batchSize // fallback to default
					}
					updateMeta(improved, bestValR.Score, passedThisBatch, testedThisCheckpoint)
					testedAtLastCheckpoint = testedNow

					// Part D3: SELF-TERMINATING RECOVERY
					// Automatically exit recovery mode when population is healthy
					if RecoveryMode.Load() && hof.Len() >= 20 {
						RecoveryMode.Store(false)
						fmt.Println("[RECOVERY] Auto-disabled: elites =", hof.Len(), ">= 20 threshold")
						fmt.Println("[RECOVERY] Returning to normal screening and generation mode")
					}

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
						checkmark := logx.Checkmark(bestPassesValidation)
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

						// Format reason string with color
						reasonStr := ""
						if len(reasonParts) > 0 {
							reasonStr = " " + logx.Warn("reason="+strings.Join(reasonParts, ","))
						}

						// Check overtrading for reporting
						valCandles := valW.End - valW.Start
						tradesPerYear := float32(bestValR.Trades) * (105120.0 / float32(valCandles))
						maxTradesPerYear := float32(500)
						overtrading := tradesPerYear > maxTradesPerYear
						if overtrading && len(reasonParts) > 0 {
							reasonStr += " " + logx.Error(",ovtrd")
						} else if overtrading {
							reasonStr = " " + logx.Error("reason=ovtrd")
						}

						// Reject overtrading strategies from passing validation
						bestPassesValidation = bestPassesValidation && !overtrading

						// Build human-readable entry/exit story with feature names
						entryRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.EntryRule.Root, feats.Names)
						exitRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.ExitRule.Root, feats.Names)
						regimeRuleNamed := ruleTreeToStringNamed(bestTrainResult.Strategy.RegimeFilter.Root, feats.Names)

						// Print summary line using structured logging
						searchStartTimeMu.Lock()
						runtimeElapsed := now.Sub(searchStartTime)
						searchStartTimeMu.Unlock()
						logx.LogBatchProgress(batchID, atomic.LoadUint64(&tested), bestTrainResult.Score, bestValR.Score,
							bestTrainResult.Return, bestValR.Return, float32(bestValR.OOSGeoAvgMonthly), float32(bestValR.OOSMedianMonthly), bestValR.WinRate, bestValR.Trades, bestValR.OOSTotalMonths, rate, runtimeElapsed, fingerprint+" "+checkmark)

						// Always show real WF OOS stats when WF is enabled
						if wfConfig.Enable {
							fmt.Printf("  %s GeoAvg=%+.2f%%  Median=%+.2f%%  MinMo=%+.2f%%  Months=%d  Trades=%d  OOS_DD=%.2f%%\n",
								logx.Info("[OOS]"),
								bestValR.OOSGeoAvgMonthly*100,
								bestValR.OOSMedianMonthly*100,
								bestValR.OOSMinMonth*100,
								bestValR.OOSTotalMonths,
								bestValR.OOSTotalTrades,
								bestValR.OOSMaxDD*100,
							)
						}

						// Print criteria summary
						print("  %s\n",
							logx.Dimf("[crit: score>%.2f, trds>=%d, DD<%.2f, ret>%.1f%%, exp>%.4f, pf>%.2f, stab>%.0f%%]",
								minValScore, minValTrades, maxValDD, minValReturn*100, minValExpect, minValPF, stabilityThreshold*100))

						// Print validation details with quick test (colored)
						print("  [val: score=%s dd=%s trds=%d%s]\n",
							logx.ScoreColor(bestValR.Score), logx.DDColor(bestValR.MaxDD), bestValR.Trades, reasonStr)
						print("  [QuickTest: Score=%s Ret=%s DD=%s Trds=%d]\n",
							logx.ScoreColor(bestQuickTestR.Score), logx.ReturnColor(bestQuickTestR.Return),
							logx.DDColor(bestQuickTestR.MaxDD), bestQuickTestR.Trades)

						// Print entry/exit story with feature names (indented)
						print("  %s\n", logx.Info("[ENTRY STORY]"))
						if regimeRuleNamed != "" && regimeRuleNamed != "(NOT )" {
							print("    Regime Filter (must be true): %s\n", formatIndentedRule(regimeRuleNamed, "    "))
						} else {
							print("    Regime Filter: (Always Active - No Filter)\n")
						}
						print("    Entry Signal (candle t close): %s\n", formatIndentedRule(entryRuleNamed, "    "))
						print("    Entry Execution (candle t+1 open): pending entry enters at next bar open\n")
						print("    Exit Signal: %s\n", formatIndentedRule(exitRuleNamed, "    "))

						// Print risk management
						print("  %s\n", logx.Highlight("[RISK MANAGEMENT]"))
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

					// Auto-adjust enforcement: Only enforce profit gates AFTER we have enough elites to support evolution
					// CRITICAL FIX #6: Removed candidate-count-based relaxation logic
					// The complexity rule in strategy.go prevents volume-only junk strategies,
					// so screening can stay strict without needing desperate relaxation
					elitesCount := len(hof.Elites)
					const minElitesForProfitGates = 10 // Wait for at least 10 elites before enforcing strict gates
					if elitesCount > 0 {
						// Keep MinValTrades at 20 (do NOT auto-raise to 30)
						if meta.MinValTrades > 20 {
							meta.mu.Lock()
							meta.MinValTrades = 20
							meta.mu.Unlock()
						}
						minValTrades = 20

						// Auto-adjust profit gates (disable with -auto_adjust=false for tuning)
						if *autoAdjust {
							// CRITICAL FIX: Only enforce profit gates AFTER we have enough elites
							// This allows evolution to climb out of the valley before strict gates kick in
							if elitesCount >= minElitesForProfitGates && minValReturn < 0.02 {
								oldReturn := minValReturn
								minValReturn = 0.02
								logx.LogAutoAdjust("Enforcing profit gates, minValReturn", oldReturn, minValReturn)
							}
							if elitesCount >= minElitesForProfitGates && minValPF < 1.05 {
								oldPF := minValPF
								minValPF = 1.05
								logx.LogAutoAdjust("Enforcing profit gates, minValPF", oldPF, minValPF)
							}
							if elitesCount >= minElitesForProfitGates && maxValDD > 0.35 {
								oldDD := maxValDD
								maxValDD = 0.35
								logx.LogAutoAdjust("Enforcing profit gates, maxValDD", oldDD, maxValDD)
							}
							if elitesCount >= minElitesForProfitGates && minValExpect < 0.0001 {
								oldExp := minValExpect
								minValExpect = 0.0001
								logx.LogAutoAdjust("Enforcing profit gates, minValExpect", oldExp, minValExpect)
							}
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
			batchSize := 128 // OPTIMIZATION: Tuned from 32 for better amortization (1.2-1.5x speedup)
			localBatch := make([]Strategy, 0, batchSize)
			localResults := make([]Result, 0, batchSize)

			for {
				select {
				case <-ctx.Done():
					if len(localBatch) > 0 {
						// Evaluate remaining strategies using multi-fidelity pipeline
						testedNow := int64(atomic.LoadUint64(&tested))
						for _, s := range localBatch {
							// Use multi-fidelity: screen -> train -> val (or walk-forward if enabled)
							passedScreen, passedFull, _, trainR, valR, _ := evaluateWithWalkForward(series, feats, s, screenW, trainW, valW, testedNow, wfConfig)
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
							// IMPORTANT: allocate a distinct copy to avoid pointer aliasing across iterations
							trainR.ValResult = new(Result)
							*trainR.ValResult = valR
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
							// Use multi-fidelity: screen -> train -> val (or walk-forward if enabled)
							passedScreen, passedFull, _, trainR, valR, _ := evaluateWithWalkForward(series, feats, strat, screenW, trainW, valW, testedNow, wfConfig)
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
							// IMPORTANT: allocate a distinct copy to avoid pointer aliasing across iterations
							trainR.ValResult = new(Result)
							*trainR.ValResult = valR
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

			// FIX: Use normal generator proportions once we have >= 8 elites
			// Bootstrap mode (elites < 8): 100% immigrants to seed the population
			// Normal mode (elites >= 8): 25-40% immigrants, rest is evolution
			immigrantP := float32(0.25) // Base: 25% when improving and elites >= 8
			if hof.Len() < 8 {
				immigrantP = 1.00 // 100% immigrants when elites < 8 (bootstrap phase)
			} else if hof.Len() >= 50 {
				immigrantP = 0.40 // Increase to 40% when elites full (more exploration)
			}
			if stagnationCount > 3 {
				immigrantP += 0.08 // Additional boost when stagnating (reduced from 0.15 to prevent over-exploration)
			}
			// Cap at reasonable max
			if immigrantP > 0.50 {
				immigrantP = 0.50 // Max 50% immigrants
			}

			// Get adaptive parameters (thread-safe)
			stagnationMu.RLock()
			currentRadicalP := radicalP
			stagnationMu.RUnlock()

			heavyMutP := immigrantP + 0.50*currentRadicalP // was + currentRadicalP

			// ADAPTIVE CROSSOVER: Increase crossover rate as elite pool grows
			// More elites = more diverse gene pool = higher crossover value
			// Base 20%, ramp to 25% when elites >= 50 (increased from 10-15% to better utilize elite diversity)
			elitesCount := hof.Len()
			crossBonus := float32(0.0)
			if elitesCount >= 50 {
				crossBonus = 0.05 // Increase from 20% to 25% crossover
			} else if elitesCount >= 25 {
				crossBonus = 0.025 // Intermediate step: 22.5% crossover
			}
			crossP := heavyMutP + 0.20 + crossBonus // Adaptive: 20-25% crossover (increased from 0.10)
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
				// Crossover: mix two parents (adaptive 10-15% based on elite count)
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
				s = crossover(rng, a, b, feats)
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
			// CRITICAL FIX: Delay surrogate until elites >= 50 to avoid premature filtering
			hof.mu.RLock()
			haveElites := len(hof.Elites) >= 50
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

			// Check if we've already seen this strategy fingerprint
			// IMPORTANT: Instead of rejecting, try to reroll until unseen (max 10 retries)
			// OPTIMIZATION: Skip reroll when elites==0 to save compute
			if hof.Len() == 0 {
				// No elites: just reject without reroll (save compute)
				if !markSeen(s) {
					atomic.AddInt64(&genRejectedSeen, 1)
					continue
				}
				// Fingerprint was new - already inserted by markSeen
			} else {
				// Normal mode: reroll if seen
				if !markSeen(s) {
					// Instead of rejecting, try to reroll until unseen (max 10 retries)
					s = rerollOnSeen(s, feats, 10)

					// Check again if rerolled version is seen
					if !markSeen(s) {
						atomic.AddInt64(&genRejectedSeen, 1)
						continue // Still seen after retries - reject
					}
					// Reroll succeeded - s is now an unseen strategy, continue to novelty check
				}
			}

			// NOVELTY PRESSURE: Reject strategies too similar to recently generated ones
			// This reduces duplicate storm by penalizing children that are too close to parent shape
			// Use coarse fingerprint (threshold buckets) which captures structural + threshold similarity
			coarseFP := s.CoarseFingerprint()
			recentFingerprintsMu.Lock()
			count := recentFingerprints[coarseFP]
			// Reject if this coarse fingerprint appears too frequently in recent window
			// Threshold: reject if seen more than 5 times in last 5000 strategies
			// This catches generators that keep producing the same structural + threshold pattern
			if count > 5 {
				recentFingerprintsMu.Unlock()
				atomic.AddInt64(&genRejectedNovelty, 1) // Separate counter for novelty pressure
				continue
			}
			// Track this fingerprint
			recentFingerprints[coarseFP] = count + 1
			totalRecentFingerprints++
			// Periodically prune the fingerprint map to keep it bounded
			if totalRecentFingerprints > recentFingerprintsWindow*2 {
				// Prune: keep only recent entries by halving all counts
				for fp := range recentFingerprints {
					recentFingerprints[fp] = recentFingerprints[fp] / 2
					if recentFingerprints[fp] == 0 {
						delete(recentFingerprints, fp)
					}
				}
				totalRecentFingerprints = totalRecentFingerprints / 2
			}
			recentFingerprintsMu.Unlock()

			// CRITICAL: Ensure trigger leaf after ALL generation (mutations, crossovers, immigrants)
			// This prevents screen_entry_rate_dead rejections by guaranteeing entry capability
			if !hasTriggerLeaf(s.EntryRule.Root) {
				// Replace a random leaf with a trigger leaf
				ensureTriggerLeaf(s.EntryRule.Root, rng, feats)
				// Recompile since tree changed
				s.EntryCompiled = compileRuleTree(s.EntryRule.Root)
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

				meta.mu.RLock()
				currentBatches := meta.Batches
				meta.mu.RUnlock()

				// Get generator loop stats
				genStats := getGeneratorStats()

				// Calculate rejection percentages with clamping
				generated := atomic.LoadInt64(&genGenerated)
				if generated == 0 {
					generated = 1
				}
				rejSeen := 100.0 * float64(atomic.LoadInt64(&genRejectedSeen)) / float64(generated)
				if rejSeen < 0 {
					rejSeen = 0
				}
				if rejSeen > 100 {
					rejSeen = 100
				}
				rejNovelty := 100.0 * float64(atomic.LoadInt64(&genRejectedNovelty)) / float64(generated)
				if rejNovelty < 0 {
					rejNovelty = 0
				}
				if rejNovelty > 100 {
					rejNovelty = 100
				}
				rejSur := 100.0 * float64(atomic.LoadInt64(&genRejectedSur)) / float64(generated)
				if rejSur < 0 {
					rejSur = 0
				}
				if rejSur > 100 {
					rejSur = 100
				}

				// Use structured progress logging
				logx.LogProgress(int64(cur), rate, float64(reportBestVal), elitesCount, rejSeen, rejNovelty, rejSur, int64(currentBatches))

				// Print generator stats separately
				print("%s", genStats)
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
				if imm+hm+cr+nm > 0 {
					logx.LogGenTypes(imm, hm, cr, nm)
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

// runDiagnosticMode exports indicator values at specific timestamp for TradingView comparison
// Phase 1: Data Collection - Export local values at 2025-12-21 11:30:00
func runDiagnosticMode(dataFile string) {
	fmt.Println("Running in DIAGNOSE mode - exporting indicator values for TradingView comparison")
	fmt.Println("===============================================================================")
	fmt.Println()

	// Load data
	fmt.Println("Loading data...")
	s, err := LoadBinanceKlinesCSV(dataFile)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", s.T)

	// Compute features on FULL dataset
	fmt.Println("Computing features on FULL dataset...")
	f := computeAllFeatures(s)
	fmt.Printf("Computed %d features\n", len(f.F))

	// Find index for 2025-12-21 13:30:00 (matching actual data in CSV)
	targetTime, err := time.Parse("2006-01-02 15:04:05", "2025-12-21 13:30:00")
	if err != nil {
		fmt.Printf("Error parsing target time: %v\n", err)
		return
	}
	targetMs := targetTime.UnixMilli()

	idx := -1
	for i, t := range s.OpenTimeMs {
		if t == targetMs {
			idx = i
			break
		}
	}

	if idx == -1 {
		fmt.Printf("Timestamp 2025-12-21 13:30:00 not found in data.\n")
		fmt.Printf("Searching for nearby timestamps...\n")

		// Find the closest timestamp
		closestIdx := -1
		minDiff := int64(1 << 62)
		for i, t := range s.OpenTimeMs {
			diff := t - targetMs
			if diff < 0 {
				diff = -diff
			}
			if diff < minDiff {
				minDiff = diff
				closestIdx = i
			}
		}

		if closestIdx >= 0 {
			closestTime := time.Unix(s.OpenTimeMs[closestIdx]/1000, 0)
			fmt.Printf("Closest timestamp found at index %d: %s (diff: %d ms)\n",
				closestIdx, closestTime.Format("2006-01-02 15:04:05"), minDiff)
			fmt.Printf("Using index %d for diagnostic output.\n", closestIdx)
			idx = closestIdx
		} else {
			fmt.Printf("Could not find any timestamp in data.\n")
			return
		}
	}

	fmt.Printf("\n=== DIAGNOSTIC REPORT: %s (Index %d) ===\n",
		time.Unix(s.OpenTimeMs[idx]/1000, 0).Format("2006-01-02 15:04:05"), idx)

	// Raw OHLCV
	fmt.Printf("\nRAW DATA:\n")
	fmt.Printf("  Open:   %.2f\n", s.Open[idx])
	fmt.Printf("  High:   %.2f\n", s.High[idx])
	fmt.Printf("  Low:    %.2f\n", s.Low[idx])
	fmt.Printf("  Close:  %.2f\n", s.Close[idx])
	fmt.Printf("  Volume: %.2f\n", s.Volume[idx])

	// Previous bar (for EMA/RSI context)
	if idx > 0 {
		fmt.Printf("\nPREVIOUS BAR (Index %d):\n", idx-1)
		fmt.Printf("  Open:  %.2f\n", s.Open[idx-1])
		fmt.Printf("  High:  %.2f\n", s.High[idx-1])
		fmt.Printf("  Low:   %.2f\n", s.Low[idx-1])
		fmt.Printf("  Close: %.2f\n", s.Close[idx-1])
	}

	// Helper function to safely get feature value
	getFeat := func(name string) float32 {
		if i, ok := f.Index[name]; ok {
			if idx < len(f.F[i]) {
				return f.F[i][idx]
			}
		}
		return -1
	}

	// Helper to get previous value
	getFeatPrev := func(name string) float32 {
		if i, ok := f.Index[name]; ok {
			if idx-1 >= 0 && idx-1 < len(f.F[i]) {
				return f.F[i][idx-1]
			}
		}
		return -1
	}

	// Helper to get first value
	getFeatFirst := func(name string) float32 {
		if i, ok := f.Index[name]; ok {
			if len(f.F[i]) > 0 {
				return f.F[i][0]
			}
		}
		return -1
	}

	// EMA20
	ema20 := getFeat("EMA20")
	ema20Prev := getFeatPrev("EMA20")
	ema20First := getFeatFirst("EMA20")
	fmt.Printf("\nEMA20: %.2f\n", ema20)
	if ema20Prev >= 0 {
		fmt.Printf("  Previous EMA20: %.2f\n", ema20Prev)
	}
	if ema20First >= 0 {
		fmt.Printf("  First EMA20: %.2f (at index 0)\n", ema20First)
	}

	// RSI14
	rsi14 := getFeat("RSI14")
	fmt.Printf("\nRSI14: %.2f\n", rsi14)

	// MACD
	macd := getFeat("MACD")
	signal := getFeat("MACD_Signal")
	hist := getFeat("MACD_Hist")
	fmt.Printf("\nMACD Line: %.2f\n", macd)
	fmt.Printf("  Signal: %.2f\n", signal)
	fmt.Printf("  Histogram: %.2f\n", hist)

	// ADX
	adxDx := getFeat("ADX")
	adxFirst := getFeatFirst("ADX")
	fmt.Printf("\nADX: %.2f\n", adxDx)
	if adxFirst >= 0 {
		fmt.Printf("  First ADX: %.2f (at index 0)\n", adxFirst)
	}

	// PlusDI and MinusDI
	plusDI := getFeat("PlusDI")
	minusDI := getFeat("MinusDI")
	fmt.Printf("  PlusDI: %.2f\n", plusDI)
	fmt.Printf("  MinusDI: %.2f\n", minusDI)

	// ATR14
	atr14 := getFeat("ATR14")
	fmt.Printf("\nATR14: %.2f\n", atr14)

	// Bollinger Bands 20
	bbUpper := getFeat("BB_Upper20")
	bbLower := getFeat("BB_Lower20")
	bbWidth := getFeat("BB_Width20")
	fmt.Printf("\nBB Upper20: %.2f\n", bbUpper)
	fmt.Printf("BB Lower20: %.2f\n", bbLower)
	fmt.Printf("BB Width20: %.4f\n", bbWidth)

	// SMA20 (computed from BB_Upper20 basis - BB uses SMA)
	// BB_Upper = SMA + 2*Std, BB_Lower = SMA - 2*Std
	// So SMA = (BB_Upper + BB_Lower) / 2
	sma20 := (bbUpper + bbLower) / 2
	fmt.Printf("SMA20 (from BB): %.2f\n", sma20)

	// Export for comparison
	fmt.Printf("\n=== EXPORT FOR TRADINGVIEW COMPARISON ===\n")
	fmt.Printf("Timestamp: %s\n", time.Unix(s.OpenTimeMs[idx]/1000, 0).Format("2006-01-02 15:04:05"))
	fmt.Printf("Index: %d of %d\n", idx, s.T)
	fmt.Printf("Close: %.2f\n", s.Close[idx])
	if idx > 0 {
		fmt.Printf("Previous Close: %.2f\n", s.Close[idx-1])
	}
	fmt.Printf("EMA20: %.2f\n", ema20)
	fmt.Printf("RSI14: %.2f\n", rsi14)
	fmt.Printf("MACD: %.2f, Signal: %.2f, Hist: %.2f\n", macd, signal, hist)
	fmt.Printf("ADX: %.2f\n", adxDx)
	fmt.Printf("PlusDI: %.2f\n", plusDI)
	fmt.Printf("MinusDI: %.2f\n", minusDI)
	fmt.Printf("ATR14: %.2f\n", atr14)
	fmt.Printf("BB Upper20: %.2f\n", bbUpper)
	fmt.Printf("BB Lower20: %.2f\n", bbLower)

	// Also show values at different indices to check for warmup issues
	fmt.Printf("\n=== WARMUP ANALYSIS - Values at different indices ===\n")
	indicesToShow := []int{100, 500, 1000, 5000}
	for _, showIdx := range indicesToShow {
		if showIdx >= s.T {
			continue
		}
		fmt.Printf("\nAt index %d (%s):\n", showIdx, time.Unix(s.OpenTimeMs[showIdx]/1000, 0).Format("2006-01-02 15:04:05"))

		// Helper for this index
		getAtIdx := func(name string) float32 {
			if i, ok := f.Index[name]; ok {
				if showIdx < len(f.F[i]) {
					return f.F[i][showIdx]
				}
			}
			return -1
		}

		fmt.Printf("  Close: %.2f\n", s.Close[showIdx])
		fmt.Printf("  EMA20: %.2f\n", getAtIdx("EMA20"))
		fmt.Printf("  RSI14: %.2f\n", getAtIdx("RSI14"))
		fmt.Printf("  ADX: %.2f\n", getAtIdx("ADX"))
	}

	fmt.Printf("\n=== DIAGNOSTIC NOTES ===\n")
	fmt.Printf("1. Compare the values above with TradingView at the same timestamp\n")
	fmt.Printf("2. Note TradingView's EXACT settings:\n")
	fmt.Printf("   - EMA source: Close (default)\n")
	fmt.Printf("   - RSI length: 14 (default)\n")
	fmt.Printf("   - MACD params: 12/26/9 (default)\n")
	fmt.Printf("   - ADX length: 14 (default)\n")
	fmt.Printf("   - BB params: 20, 2.0 (default)\n")
	fmt.Printf("3. If ADX ≈ 1.4x TradingView value: Using EMA smoothing instead of Wilder\n")
	fmt.Printf("4. If early values wrong but later values match: Warmup/seed issue\n")
	fmt.Printf("5. If ALL values wrong: Data source or formula issue\n")

	// Export to CSV for easy comparison
	fmt.Printf("\n=== EXPORTING TO CSV ===\n")
	csvPath := "diagnostic_values.csv"
	csvFile, err := os.Create(csvPath)
	if err != nil {
		fmt.Printf("Error creating CSV file: %v\n", err)
		return
	}
	defer csvFile.Close()

	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	// Write header
	header := []string{"Timestamp", "Index", "Open", "High", "Low", "Close", "Volume",
		"EMA20", "RSI14", "MACD", "MACD_Signal", "MACD_Hist",
		"ADX", "PlusDI", "MinusDI", "ATR14", "BB_Upper20", "BB_Lower20", "BB_Width20", "SMA20"}
	if err := writer.Write(header); err != nil {
		fmt.Printf("Error writing CSV header: %v\n", err)
		return
	}

	// Helper to get value at specific index
	getValueAtIdx := func(name string, idx int) float32 {
		if i, ok := f.Index[name]; ok {
			if idx >= 0 && idx < len(f.F[i]) {
				return f.F[i][idx]
			}
		}
		return -1
	}

	// Export the target bar
	row := []string{
		time.Unix(s.OpenTimeMs[idx]/1000, 0).Format("2006-01-02 15:04:05"),
		fmt.Sprintf("%d", idx),
		fmt.Sprintf("%.2f", s.Open[idx]),
		fmt.Sprintf("%.2f", s.High[idx]),
		fmt.Sprintf("%.2f", s.Low[idx]),
		fmt.Sprintf("%.2f", s.Close[idx]),
		fmt.Sprintf("%.2f", s.Volume[idx]),
		fmt.Sprintf("%.2f", getFeat("EMA20")),
		fmt.Sprintf("%.2f", getFeat("RSI14")),
		fmt.Sprintf("%.2f", getFeat("MACD")),
		fmt.Sprintf("%.2f", getFeat("MACD_Signal")),
		fmt.Sprintf("%.2f", getFeat("MACD_Hist")),
		fmt.Sprintf("%.2f", getFeat("ADX")),
		fmt.Sprintf("%.2f", getFeat("PlusDI")),
		fmt.Sprintf("%.2f", getFeat("MinusDI")),
		fmt.Sprintf("%.2f", getFeat("ATR14")),
		fmt.Sprintf("%.2f", getFeat("BB_Upper20")),
		fmt.Sprintf("%.2f", getFeat("BB_Lower20")),
		fmt.Sprintf("%.4f", getFeat("BB_Width20")),
		fmt.Sprintf("%.2f", sma20),
	}
	if err := writer.Write(row); err != nil {
		fmt.Printf("Error writing CSV row: %v\n", err)
		return
	}

	// Also export warmup analysis bars
	warmupIndices := []int{100, 500, 1000, 5000}
	for _, showIdx := range warmupIndices {
		if showIdx >= s.T {
			continue
		}
		showSMA := (getValueAtIdx("BB_Upper20", showIdx) + getValueAtIdx("BB_Lower20", showIdx)) / 2
		row := []string{
			time.Unix(s.OpenTimeMs[showIdx]/1000, 0).Format("2006-01-02 15:04:05"),
			fmt.Sprintf("%d", showIdx),
			fmt.Sprintf("%.2f", s.Open[showIdx]),
			fmt.Sprintf("%.2f", s.High[showIdx]),
			fmt.Sprintf("%.2f", s.Low[showIdx]),
			fmt.Sprintf("%.2f", s.Close[showIdx]),
			fmt.Sprintf("%.2f", s.Volume[showIdx]),
			fmt.Sprintf("%.2f", getValueAtIdx("EMA20", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("RSI14", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("MACD", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("MACD_Signal", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("MACD_Hist", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("ADX", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("PlusDI", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("MinusDI", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("ATR14", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("BB_Upper20", showIdx)),
			fmt.Sprintf("%.2f", getValueAtIdx("BB_Lower20", showIdx)),
			fmt.Sprintf("%.4f", getValueAtIdx("BB_Width20", showIdx)),
			fmt.Sprintf("%.2f", showSMA),
		}
		if err := writer.Write(row); err != nil {
			fmt.Printf("Error writing CSV warmup row: %v\n", err)
			return
		}
	}

	fmt.Printf("CSV exported to: %s\n", csvPath)
	fmt.Printf("Open this file in Excel/Google Sheets to compare with TradingView values\n")
}

// runVerifyMode - Simple verification that indicators are computed from CSV OHLCV data
// Loads data from btc_5min_data.csv, computes indicators, shows the values
func runVerifyMode(dataFile string) {
	fmt.Println("Running in VERIFY mode - Computing indicators from CSV OHLCV data")
	fmt.Println("==================================================================")
	fmt.Println()

	// Load data from CSV
	fmt.Printf("Loading data from %s...\n", dataFile)
	s, err := LoadBinanceKlinesCSV(dataFile)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles from CSV\n", s.T)

	// Find the target timestamp: 2025-12-21 13:30:00
	targetTime, _ := time.Parse("2006-01-02 15:04:05", "2025-12-21 13:30:00")
	targetMs := targetTime.UnixMilli()

	idx := -1
	for i, t := range s.OpenTimeMs {
		if t == targetMs {
			idx = i
			break
		}
	}

	if idx == -1 {
		// Find closest
		minDiff := int64(1 << 62)
		for i, t := range s.OpenTimeMs {
			diff := t - targetMs
			if diff < 0 {
				diff = -diff
			}
			if diff < minDiff {
				minDiff = diff
				idx = i
			}
		}
	}

	fmt.Printf("\nTarget timestamp: %s (Index %d)\n",
		time.Unix(s.OpenTimeMs[idx]/1000, 0).Format("2006-01-02 15:04:05"), idx)

	// Show raw OHLCV from CSV
	fmt.Printf("\n=== RAW OHLCV DATA FROM CSV ===\n")
	fmt.Printf("  Open:   %.2f\n", s.Open[idx])
	fmt.Printf("  High:   %.2f\n", s.High[idx])
	fmt.Printf("  Low:    %.2f\n", s.Low[idx])
	fmt.Printf("  Close:  %.2f\n", s.Close[idx])
	fmt.Printf("  Volume: %.2f\n", s.Volume[idx])

	// Compute indicators on FULL dataset
	fmt.Printf("\nComputing indicators from CSV data...\n")
	f := computeAllFeatures(s)

	// Helper to get feature value
	getFeat := func(name string) float32 {
		if i, ok := f.Index[name]; ok {
			if idx < len(f.F[i]) {
				return f.F[i][idx]
			}
		}
		return -1
	}

	// Show computed indicators
	fmt.Printf("\n=== COMPUTED INDICATORS (from CSV OHLCV) ===\n")
	fmt.Printf("  EMA20:       %.2f\n", getFeat("EMA20"))
	fmt.Printf("  RSI14:       %.2f\n", getFeat("RSI14"))
	fmt.Printf("  MACD:        %.2f\n", getFeat("MACD"))
	fmt.Printf("  MACD Signal: %.2f\n", getFeat("MACD_Signal"))
	fmt.Printf("  MACD Hist:   %.2f\n", getFeat("MACD_Hist"))
	fmt.Printf("  ADX:         %.2f\n", getFeat("ADX"))
	fmt.Printf("  PlusDI:      %.2f\n", getFeat("PlusDI"))
	fmt.Printf("  MinusDI:     %.2f\n", getFeat("MinusDI"))
	fmt.Printf("  ATR14:       %.2f\n", getFeat("ATR14"))
	fmt.Printf("  BB_Upper20:  %.2f\n", getFeat("BB_Upper20"))
	fmt.Printf("  BB_Lower20:  %.2f\n", getFeat("BB_Lower20"))

	fmt.Printf("\n=== VERIFICATION RESULT ===\n")
	fmt.Printf("All indicators above are computed FROM the CSV OHLCV data.\n")
	fmt.Printf("Raw data source: btc_5min_data.csv\n")
	fmt.Printf("If these match TradingView at the same timestamp, indicators are CORRECT.\n")
}

func runTestMode(dataFile string, feeBps, slipBps float32, fromIdx, toIdx int, fromTime, toTime string, warmupBars int) {
	fmt.Println("Running in TEST mode - evaluating saved winners on test data")
	fmt.Println("=============================================================")
	fmt.Printf("Cost overrides: Fee=%.1f bps (%.2f%%), Slippage=%.1f bps (%.2f%%)\n", feeBps, feeBps/100, slipBps, slipBps/100)
	fmt.Println()

	// Load data (same as search mode)
	fmt.Println("Loading data...")
	startTime := time.Now()

	series, err := LoadBinanceKlinesCSV(dataFile)
	if err != nil {
		fmt.Printf("Error loading CSV: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d candles\n", series.T)

	fmt.Println("Computing features...")
	feats := computeAllFeatures(series)
	fmt.Printf("Computed %d features\n", len(feats.F))

	// Get custom window if specified, otherwise use default test window
	var testW Window
	if fromIdx >= 0 || toIdx >= 0 || fromTime != "" || toTime != "" {
		// Custom window mode
		fmt.Printf("Using custom window: from_idx=%d, to_idx=%d, from=\"%s\", to=\"%s\", warmup=%d\n",
			fromIdx, toIdx, fromTime, toTime, warmupBars)
		testW = GetCustomWindow(series.OpenTimeMs, fromIdx, toIdx, fromTime, toTime, warmupBars, series.T)
		fmt.Printf("Custom window: %d candles (%d -> %d)\n", testW.End-testW.Start, testW.Start, testW.End)

		// For custom windows, compute stats on the full dataset to avoid future leak
		computeStatsOnWindow(&feats, 0, series.T)
		fmt.Printf("Feature stats computed on full dataset (%d candles)\n", series.T)
	} else {
		// Default test mode
		trainStart, trainEnd, valEnd := GetSplitIndices(series.OpenTimeMs)
		_, _, testW = GetSplitWindows(trainStart, trainEnd, valEnd, series.T, 500)
		fmt.Printf("Using default test window: %d candles (%d -> %d)\n", testW.End-testW.Start, testW.Start, testW.End)

		// Recompute feature stats on train window only
		computeStatsOnWindow(&feats, trainStart, trainEnd)
		fmt.Printf("Feature stats computed on train window (%d candles)\n", trainEnd-trainStart)
	}

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
		TestScore       float32        `json:"test_score"`
		TestReturn      float32        `json:"test_return"`
		TestMaxDD       float32        `json:"test_max_dd"`
		TestWinRate     float32        `json:"test_win_rate"`
		TestTrades      int            `json:"test_trades"`
		TestExitReasons map[string]int `json:"test_exit_reasons"`
	}

	// Helper function: deduplicate TestResults by fingerprint
	// Uses EliteLog fields (EntryRule, ExitRule, etc.) to build fingerprint
	dedupeTestResults := func(results []TestResult) []TestResult {
		seen := make(map[string]bool)
		out := make([]TestResult, 0, len(results))

		for _, r := range results {
			// Build fingerprint from EliteLog fields (same as Strategy.Fingerprint())
			fp := r.EntryRule + "|" +
				r.ExitRule + "|" +
				r.RegimeFilter + "|" +
				r.StopLoss + "|" +
				r.TakeProfit + "|" +
				r.Trail + "|" +
				fmt.Sprintf("hold=%d|loss=%d|cd=%d", 0, 0, 0) // Not stored in EliteLog
			if seen[fp] {
				continue // Skip duplicate
			}
			seen[fp] = true
			out = append(out, r)
		}

		return out
	}

	// Helper function: short fingerprint for display
	shortFingerprint := func(fp string) string {
		const maxLen = 12
		if len(fp) <= maxLen {
			return fp
		}
		return fp[:maxLen]
	}

	// Print test window info
	fmt.Printf("Test window: %d candles (indices %d -> %d)\n\n", testW.End-testW.Start, testW.Start, testW.End)

	var allResults []TestResult
	var parseErrors int
	var invalidStrategyCount int
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
				RiskPct:     1.0,
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
				StopLoss:         parseStopModel(log.StopLoss),
				TakeProfit:       parseTPModel(log.TakeProfit),
				Trail:            parseTrailModel(log.Trail),
				VolatilityFilter: VolFilterModel{Enabled: false}, // Default disabled for old strategies
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

		// SANITY CHECK: Validate cross operations when loading for test mode
		// This prevents strategies with invalid CrossUp/CrossDown from being tested
		if err := validateLoadedStrategy(strategy, &feats); err != nil {
			// Use separate counter - validation failure is NOT a parse error
			invalidStrategyCount++
			if invalidStrategyCount <= 5 {
				fmt.Printf("Warning: Skipping strategy #%d: %v\n", i+1, err)
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
			EliteLog:        log,
			TestScore:       testR.Score,
			TestReturn:      testR.Return,
			TestMaxDD:       testR.MaxDD,
			TestWinRate:     testR.WinRate,
			TestTrades:      testR.Trades,
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

	// DEDUPLICATE by fingerprint before ranking (removes duplicate strategies)
	passed = dedupeTestResults(passed)
	allResults = dedupeTestResults(allResults)

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
	fmt.Println("(Deduplicated by fingerprint)")
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
			// Build fingerprint from EliteLog fields directly
			fp := r.EntryRule + "|" + r.ExitRule + "|" + r.RegimeFilter + "|" + r.StopLoss + "|" + r.TakeProfit + "|" + r.Trail
			fpShort := shortFingerprint(fp)
			// Format TestScore: show REJECTED for sentinel values
			testScoreStr := "REJECTED"
			if r.TestScore > -1e20 {
				testScoreStr = fmt.Sprintf("%.3f", r.TestScore)
			}
			fmt.Printf("#%2d | TestScore: %s | Ret: %6.2f%% | DD: %5.1f%% | Trades: %4d | ValScore: %.3f | FP: %s\n",
				i+1, testScoreStr, r.TestReturn*100, r.TestMaxDD*100, r.TestTrades, r.ValScore, fpShort)
		}
	}

	// Print top 20 leaderboard (PASSED strategies only)
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TOP 20 STRATEGIES BY TEST SCORE (STRICT FILTERS APPLIED)")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("Gates: Trades>=%d, MaxDD<=%.2f, Return>%.1f%%\n", minTrades, maxDD, minReturn*100)
	fmt.Println("(Deduplicated by fingerprint)")
	fmt.Println()

	numToPrint := 20
	if len(passed) < numToPrint {
		numToPrint = len(passed)
	}

	if numToPrint == 0 {
		fmt.Println(logx.Error("NO STRATEGIES PASSED THE STRICT TEST GATES!"))
		fmt.Println()
		fmt.Println("Try adjusting gates: -mode test -fee_bps=30 -slip_bps=8")
	} else {
		for i := 0; i < numToPrint; i++ {
			r := passed[i]
			// Build fingerprint from EliteLog fields directly
			fp := r.EntryRule + "|" + r.ExitRule + "|" + r.RegimeFilter + "|" + r.StopLoss + "|" + r.TakeProfit + "|" + r.Trail
			fpShort := shortFingerprint(fp)
			// Format TestScore: show REJECTED for sentinel values
			testScoreStr := "REJECTED"
			if r.TestScore > -1e20 {
				testScoreStr = fmt.Sprintf("%.3f", r.TestScore)
			}
			valScoreStr := "REJECTED"
			if r.ValScore > -1e20 {
				valScoreStr = fmt.Sprintf("%.3f", r.ValScore)
			}
			fmt.Printf("#%2d | Test Score: %s | Test Return: %6.2f%% | Test WinRate: %5.1f%% | Test Trades: %4d | Test DD: %.2f%% | Val Score: %s | FP: %s\n",
				i+1, testScoreStr, r.TestReturn*100, r.TestWinRate*100, r.TestTrades, r.TestMaxDD*100, valScoreStr, fpShort)
		}
	}

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Printf("Total strategies loaded:     %d\n", len(loadedLogs))
	fmt.Printf("Tested successfully:         %d\n", len(allResults)-parseErrors)
	fmt.Printf("Skipped parse errors:        %d\n", parseErrors)
	fmt.Printf("No-trade on test:            %d\n", noTradeStrategies)
	fmt.Printf("Strategies with trades:      %d (after deduplication)\n", len(allResults))
	passedPct := float32(len(passed)) * 100 / float32(len(allResults))
	fmt.Printf("PASSED strict test gates:    %d (%s)\n", len(passed), logx.WinRateColor(passedPct/100))
	fmt.Println(strings.Repeat("-", 80))
	fmt.Println("Results saved to: winners_tested.jsonl")
	fmt.Println("(Deduplicated by strategy fingerprint including all constants)")

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
	// Format scores: show REJECTED for sentinel values
	formatScore := func(s float32) string {
		if s < -1e20 {
			return "REJECTED"
		}
		return fmt.Sprintf("%.3f", s)
	}
	buf.WriteString(fmt.Sprintf("  Test Score:      %s\n", formatScore(testScore)))
	buf.WriteString(fmt.Sprintf("  Test Return:     %.2f%%\n", testReturn*100))
	buf.WriteString(fmt.Sprintf("  Test Win Rate:   %.1f%%\n", testWinRate*100))
	buf.WriteString(fmt.Sprintf("  Test Trades:     %d\n", testTrades))
	buf.WriteString(fmt.Sprintf("  Test Max DD:     %.2f%%\n", testMaxDD*100))
	buf.WriteString(fmt.Sprintf("  Validation Score: %s\n", formatScore(valScore)))
	buf.WriteString(fmt.Sprintf("  Validation Ret:   %.2f%%\n", valReturn*100))
	buf.WriteString(fmt.Sprintf("  Train Score:      %s\n", formatScore(trainScore)))
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

func volFilterToString(vf VolFilterModel) string {
	if !vf.Enabled {
		return "vol_off"
	}
	return fmt.Sprintf("vol_ATR%d>SMA%d*%.2f", vf.ATRPeriod, vf.SMAPeriod, vf.Threshold)
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
	Seed           int64   `json:"seed"`
	FeeBps         float32 `json:"fee_bps"`
	SlippageBps    float32 `json:"slippage_bps"`
	Direction      int     `json:"direction"`
	StopLoss       string  `json:"stop_loss"`
	TakeProfit     string  `json:"take_profit"`
	Trail          string  `json:"trail"`
	EntryRule      string  `json:"entry_rule"`
	ExitRule       string  `json:"exit_rule"`
	RegimeFilter   string  `json:"regime_filter"`
	FeatureMapHash string  `json:"feature_map_hash,omitempty"` // Feature ordering fingerprint
	TrainScore     float32 `json:"train_score,omitempty"`
	ValScore       float32 `json:"val_score,omitempty"`
	TestScore      float32 `json:"test_score,omitempty"`
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
func runGoldenMode(dataFile string, seed int64, printTrades int, feeBps, slipBps float64, exportCSV, exportStates string) {
	fmt.Println("Running in GOLDEN mode - single strategy with trade logging")
	fmt.Println("================================================================")
	fmt.Printf("Seed: %d, FeeBps: %.1f, SlippageBps: %.1f\n", seed, feeBps, slipBps)
	fmt.Println()

	// Load data (same as test mode)
	fmt.Println("Loading data...")
	series, err := LoadBinanceKlinesCSV(dataFile)
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
		RiskPct:     1.0,
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
		StopLoss:         parseStopModel(w.StopLoss),
		TakeProfit:       parseTPModel(w.TakeProfit),
		Trail:            parseTrailModel(w.Trail),
		VolatilityFilter: VolFilterModel{Enabled: false}, // Default disabled for old strategies
	}

	// SANITY CHECK: Validate cross operations in golden mode
	// Warn user if strategy has invalid cross operations (but don't fail, just sanitize)
	{
		_, entryInvalid := validateCrossSanity(st.EntryRule.Root, feats)
		_, exitInvalid := validateCrossSanity(st.ExitRule.Root, feats)
		_, regimeInvalid := validateCrossSanity(st.RegimeFilter.Root, feats)

		totalInvalid := entryInvalid + exitInvalid + regimeInvalid
		if totalInvalid > 0 {
			fmt.Printf("WARNING: Strategy has %d invalid CrossUp/CrossDown operations - sanitizing...\n", totalInvalid)
			// Sanitize the strategy
			sanitizeCrossOperations(rand.New(rand.NewSource(w.Seed)), st.EntryRule.Root, feats)
			sanitizeCrossOperations(rand.New(rand.NewSource(w.Seed)), st.ExitRule.Root, feats)
			sanitizeCrossOperations(rand.New(rand.NewSource(w.Seed)), st.RegimeFilter.Root, feats)
		}
	}

	// Compile bytecode
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	// === SAFETY PRINTS: Feature map fingerprint + rule resolution ===
	// Compute runtime feature map hash
	runtimeHash := ComputeFeatureMapHash(feats)
	runtimeVersion := GetFeatureMapVersion(feats)

	fmt.Println("\n=== STRATEGY LOADED - Feature Index Resolution ===")
	fmt.Printf("FeatureMapHash: %s\n", runtimeHash)
	fmt.Printf("FeatureSetVersion: %s\n", runtimeVersion)

	// Print raw AST (original rule text with indices)
	fmt.Printf("\n--- Entry Rule (raw AST with indices) ---\n")
	fmt.Printf("  %s\n", w.EntryRule)

	// Print resolved version (with feature names)
	fmt.Printf("\n--- Entry Rule (resolved with feature names) ---\n")
	fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.EntryRule.Root, feats))

	fmt.Printf("\n--- Regime Filter (resolved) ---\n")
	fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.RegimeFilter.Root, feats))

	fmt.Printf("\n--- Exit Rule (resolved) ---\n")
	fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.ExitRule.Root, feats))

	// Validate feature map hash if stored in winner file
	if w.FeatureMapHash != "" && w.FeatureMapHash != runtimeHash {
		fmt.Printf("\n!!! FEATURE MAP MISMATCH DETECTED - REJECTING STRATEGY !!!\n")
		fmt.Printf("Stored hash:  %s\n", w.FeatureMapHash)
		fmt.Printf("Runtime hash: %s\n\n", runtimeHash)
		fmt.Printf("Stored version (first 5 features): %s\n\n", truncateVersion(w.FeatureMapHash, runtimeVersion, 5))
		fmt.Printf("Runtime version (first 5 features): %s\n", truncateVersion(runtimeHash, runtimeVersion, 5))
		fmt.Printf("\nSTRATEGY REJECTED - Feature indices have changed since strategy was created.\n")
		fmt.Printf("This strategy will produce WRONG results and MUST be regenerated.\n")
		fmt.Printf("============================================================\n\n")
		return
	} else if w.FeatureMapHash != "" {
		fmt.Printf("\n%s Feature map hash validated: MATCH\n", logx.Success("✓"))
	} else {
		fmt.Printf("\n%s WARNING: No stored feature_map_hash (legacy winner from old version)\n", logx.Warn("⚠"))
		fmt.Printf("   Cannot validate feature indices. Strategy may produce WRONG results.\n")
		fmt.Printf("   Recommendation: Regenerate this strategy with current feature order.\n")
		fmt.Printf("   Proceeding in golden/trace mode only (NOT recommended for live trading)\n")
	}
	fmt.Println("==================================================")

	// Run backtest with trade logging on TEST window
	fmt.Println("\nRunning backtest on TEST window...")
	result := evaluateStrategyWithTrades(series, feats, st, testW, false)

	// Write states with proofs and indicators to CSV
	// Only ONE CSV file is generated per run
	if exportStates != "" {
		fmt.Printf("\nWriting states with proofs to %s...\n", exportStates)
		if err := WriteStatesWithProofsToCSV(result.States, series, st, feats, result.SignalProofs, result.WindowOffset, exportStates); err != nil {
			fmt.Printf("Error writing states CSV: %v\n", err)
		} else {
			fmt.Printf("States written to %s (%d bars)\n", exportStates, len(result.States))
		}
	} else {
		// Default to states.csv if no export path specified
		fmt.Println("\nWriting states to states.csv...")
		if err := WriteStatesWithProofsToCSV(result.States, series, st, feats, result.SignalProofs, result.WindowOffset, "states.csv"); err != nil {
			fmt.Printf("Error writing states CSV: %v\n", err)
		} else {
			fmt.Printf("States written to states.csv (%d bars)\n", len(result.States))
		}
	}

	// Write trades to CSV if export path is specified
	if exportCSV != "" {
		fmt.Printf("\nWriting trades to %s...\n", exportCSV)
		if err := WriteTradesToCSV(result.Trades, st, feats, result.WindowOffset, exportCSV); err != nil {
			fmt.Printf("Error writing trades CSV: %v\n", err)
		} else {
			fmt.Printf("Trades written to %s (%d trades)\n", exportCSV, len(result.Trades))
		}
	}

	// Print golden results
	printGoldenResult(result, printTrades)
}

// truncateVersion extracts first N features from version string for diff hint
func truncateVersion(storedHash, runtimeVersion string, n int) string {
	if storedHash == "" {
		return "(unknown - no stored hash)"
	}
	// Runtime version is "EMA10@F[0],EMA20@F[1],..." - extract first n features
	parts := strings.Split(runtimeVersion, ",")
	if len(parts) > n {
		parts = parts[:n]
	}
	return strings.Join(parts, ", ") + "..."
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

	// Map to store exit details by bar index (in full series)
	type exitDetails struct {
		reason     string
		pnl        float32
		entryPrice float32
		exitPrice  float32
		stopPrice  float32
		tpPrice    float32
	}
	exitDetailsMap := make(map[int]exitDetails)

	// Map to store signal boolean evaluations for SIGNAL DETECT bars
	type signalBooleans struct {
		regimeNotRisingVolPerTrade string
		regimeImbalanceGT          string
		regimeSwingHighGT          string
		entryRisingVolSMA20        string
		entryFallingMinusDI        string
		entryFallingEMA50          string
		entryCrossUpMACD           string
		entryCrossUpEMASwing       string
		entrySlopeLTEMA200         string
	}
	signalBoolsMap := make(map[int]signalBooleans)

	// Helper function to check if rising (current > value N bars ago)
	isRising := func(featIdx, lookback, t int) bool {
		if t < lookback || featIdx < 0 || featIdx >= len(feats.F) {
			return false
		}
		return feats.F[featIdx][t] > feats.F[featIdx][t-lookback]
	}

	// Helper function to check if falling (current < value N bars ago)
	isFalling := func(featIdx, lookback, t int) bool {
		if t < lookback || featIdx < 0 || featIdx >= len(feats.F) {
			return false
		}
		return feats.F[featIdx][t] < feats.F[featIdx][t-lookback]
	}

	// Helper function to check CrossUp
	isCrossUp := func(featA, featB, t int) (bool, float32, float32, float32, float32) {
		if t < 1 || featA < 0 || featA >= len(feats.F) || featB < 0 || featB >= len(feats.F) {
			return false, 0, 0, 0, 0
		}
		prevA := feats.F[featA][t-1]
		curA := feats.F[featA][t]
		prevB := feats.F[featB][t-1]
		curB := feats.F[featB][t]
		result := prevA <= prevB && curA > curB
		return result, prevA, prevB, curA, curB
	}

	// Helper function to check slope less than threshold
	slopeLT := func(featIdx, t int, threshold float32, lookback int) bool {
		if t < lookback || featIdx < 0 || featIdx >= len(feats.F) {
			return false
		}
		slope := (feats.F[featIdx][t] - feats.F[featIdx][t-lookback]) / float32(lookback)
		return slope < threshold
	}

	// Get feature indices for this specific strategy
	volPerTradeIdx := feats.Index["VolPerTrade"] // F[35]
	imbalanceIdx := feats.Index["Imbalance"]     // F[34]
	swingHighIdx := feats.Index["SwingHigh"]     // F[8] and F[40]
	volSMA20Idx := feats.Index["VolSMA20"]       // F[27]
	minusDIIdx := feats.Index["MinusDI"]         // F[23]
	ema50Idx := feats.Index["EMA50"]             // F[2]
	ema20Idx := feats.Index["EMA20"]             // F[1]
	bbWidth50Idx := feats.Index["BB_Width50"]    // F[13]
	macdIdx := feats.Index["MACD"]               // F[14]
	ema200Idx := feats.Index["EMA200"]           // F[4]

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
		// AND calculate all boolean evaluations for this signal
		if sigIdx >= 0 && sigIdx < len(states) {
			// Don't overwrite if something "stronger" is already there
			if states[sigIdx] == "FLAT" {
				states[sigIdx] = "SIGNAL DETECT"

				// === REGIME FILTER BOOLEANS ===

				// 1. NOT Rising(VolPerTrade, 19)
				risingVolPerTrade := isRising(volPerTradeIdx, 19, sigIdx)
				regimeNotRisingVolPerTrade := fmt.Sprintf("NOT_Rising(VolPerTrade,19)=%t", !risingVolPerTrade)

				// 2. Imbalance > 0.18
				imbalanceVal := float32(0)
				if imbalanceIdx >= 0 && imbalanceIdx < len(feats.F) && sigIdx < len(feats.F[imbalanceIdx]) {
					imbalanceVal = feats.F[imbalanceIdx][sigIdx]
				}
				regimeImbalanceGT := fmt.Sprintf("Imbalance>0.18=%t(value=%.4f)", imbalanceVal > 0.18, imbalanceVal)

				// 3. SwingHigh > 144.19
				swingHighVal := float32(0)
				if swingHighIdx >= 0 && swingHighIdx < len(feats.F) && sigIdx < len(feats.F[swingHighIdx]) {
					swingHighVal = feats.F[swingHighIdx][sigIdx]
				}
				regimeSwingHighGT := fmt.Sprintf("SwingHigh>144.19=%t(value=%.4f)", swingHighVal > 144.19, swingHighVal)

				// === ENTRY RULE BOOLEANS ===

				// 1. Rising(VolSMA20, 20)
				risingVolSMA20 := isRising(volSMA20Idx, 20, sigIdx)
				entryRisingVolSMA20 := fmt.Sprintf("Rising(VolSMA20,20)=%t", risingVolSMA20)

				// 2. Falling(MinusDI, 7)
				fallingMinusDI := isFalling(minusDIIdx, 7, sigIdx)
				entryFallingMinusDI := fmt.Sprintf("Falling(MinusDI,7)=%t", fallingMinusDI)

				// 3. Falling(EMA50, 16)
				fallingEMA50 := isFalling(ema50Idx, 16, sigIdx)
				entryFallingEMA50 := fmt.Sprintf("Falling(EMA50,16)=%t", fallingEMA50)

				// 4. CrossUp(BB_Width50, MACD) - F[16] crosses above F[17]
				crossMACD, prevA, prevB, curA, curB := isCrossUp(bbWidth50Idx, macdIdx, sigIdx)
				entryCrossUpMACD := fmt.Sprintf("CrossUp(BB_Width50,MACD)=%t(prev(%.2f<=%.2f)cur(%.2f>%.2f))",
					crossMACD, prevA, prevB, curA, curB)

				// 5. CrossUp(EMA20, SwingHigh) - F[1] crosses above F[40]
				crossEMASwing, prevA2, prevB2, curA2, curB2 := isCrossUp(ema20Idx, swingHighIdx, sigIdx)
				entryCrossUpEMASwing := fmt.Sprintf("CrossUp(EMA20,SwingHigh)=%t(prev(%.2f<=%.2f)cur(%.2f>%.2f))",
					crossEMASwing, prevA2, prevB2, curA2, curB2)

				// 6. SlopeLT(EMA200, 0.06, 6)
				slopeLTResult := slopeLT(ema200Idx, sigIdx, 0.06, 6)
				entrySlopeLTEMA200 := fmt.Sprintf("SlopeLT(EMA200,0.06,6)=%t", slopeLTResult)

				// Store all booleans for this signal
				signalBoolsMap[sigIdx] = signalBooleans{
					regimeNotRisingVolPerTrade: regimeNotRisingVolPerTrade,
					regimeImbalanceGT:          regimeImbalanceGT,
					regimeSwingHighGT:          regimeSwingHighGT,
					entryRisingVolSMA20:        entryRisingVolSMA20,
					entryFallingMinusDI:        entryFallingMinusDI,
					entryFallingEMA50:          entryFallingEMA50,
					entryCrossUpMACD:           entryCrossUpMACD,
					entryCrossUpEMASwing:       entryCrossUpEMASwing,
					entrySlopeLTEMA200:         entrySlopeLTEMA200,
				}

				// === EVALUATE ACTUAL ENTRY RULE FOR DEBUGGING ===
				// This ensures "the strategy you think" == "the strategy being executed"
				entryRuleFinalResult := evaluateRule(st.EntryRule.Root, feats.F, sigIdx)

				// Also print to console as requested
				fmt.Printf("\n=== SIGNAL DETECT at Bar %d ===\n", sigIdx)
				fmt.Printf("Entry Rule (resolved): %s\n", ruleTreeToStringWithNames(st.EntryRule.Root, feats))
				fmt.Printf("\nRegime Filter:\n")
				fmt.Printf("  %s\n", regimeNotRisingVolPerTrade)
				fmt.Printf("  %s\n", regimeImbalanceGT)
				fmt.Printf("  %s\n", regimeSwingHighGT)
				fmt.Printf("\nEntry Rule Leaf Evaluations:\n")
				fmt.Printf("  %s\n", entryRisingVolSMA20)
				fmt.Printf("  %s\n", entryFallingMinusDI)
				fmt.Printf("  %s\n", entryFallingEMA50)
				fmt.Printf("  %s\n", entryCrossUpMACD)
				fmt.Printf("  %s\n", entryCrossUpEMASwing)
				fmt.Printf("  %s\n", entrySlopeLTEMA200)
				fmt.Printf("\n  >>> entryRuleResult = %t <<<\n", entryRuleFinalResult)
				fmt.Printf("===========================\n\n")
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

	// Write header row with all feature names + boolean evaluation columns
	header := []string{"BarIndex", "Time", "State"}
	for _, name := range feats.Names {
		header = append(header, name)
	}
	// Add boolean evaluation columns for SIGNAL DETECT rows
	header = append(header,
		"Regime_NOT_Rising_VolPerTrade",
		"Regime_Imbalance_GT_0.18",
		"Regime_SwingHigh_GT_144.19",
		"Entry_Rising_VolSMA20",
		"Entry_Falling_MinusDI",
		"Entry_Falling_EMA50",
		"Entry_CrossUp_BBWidth50_MACD",
		"Entry_CrossUp_EMA10_SwingHigh",
		"Entry_SlopeLT_EMA200",
	)
	if err := w.Write(header); err != nil {
		return err
	}

	// Write data rows with all indicator values - INCLUDE FEATURE NAMES in each row
	for i := 0; i < s.T; i++ {
		ts := time.Unix(int64(s.OpenTimeMs[i])/1000, 0).UTC().Format(time.RFC3339)

		// Build row with bar index, timestamp, state - with labels
		row := []string{
			fmt.Sprintf("BarIndex=%d", i),
			fmt.Sprintf("Time=%s", ts),
			fmt.Sprintf("State=%s", states[i]),
		}

		// Add all feature values with labels for this bar
		for j := 0; j < len(feats.F); j++ {
			if i < len(feats.F[j]) {
				val := feats.F[j][i]
				// Format: "IndicatorName=Value"
				row = append(row, fmt.Sprintf("%s=%.4f", feats.Names[j], val))
			} else {
				row = append(row, fmt.Sprintf("%s=", feats.Names[j]))
			}
		}

		// Add boolean evaluations if this is a SIGNAL DETECT bar
		if bools, ok := signalBoolsMap[i]; ok {
			row = append(row,
				bools.regimeNotRisingVolPerTrade,
				bools.regimeImbalanceGT,
				bools.regimeSwingHighGT,
				bools.entryRisingVolSMA20,
				bools.entryFallingMinusDI,
				bools.entryFallingEMA50,
				bools.entryCrossUpMACD,
				bools.entryCrossUpEMASwing,
				bools.entrySlopeLTEMA200,
			)
		} else {
			// Add empty strings for non-signal rows
			row = append(row, "", "", "", "", "", "", "", "", "")
		}

		if err := w.Write(row); err != nil {
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
		Seed:             0, // Deterministic seed for manual strategy
		FeeBps:           feeBps,
		SlippageBps:      slipBps,
		RiskPct:          1.0,
		Direction:        1, // Long only
		EntryRule:        RuleTree{Root: entryRoot},
		ExitRule:         RuleTree{Root: exitRoot},
		RegimeFilter:     RuleTree{Root: regimeRoot},
		StopLoss:         StopModel{Kind: "fixed", Value: 1.0},
		TakeProfit:       TPModel{Kind: "fixed", Value: 2.0},
		Trail:            TrailModel{Active: false},
		VolatilityFilter: VolFilterModel{Enabled: false}, // Disabled for manual strategies
		MaxHoldBars:      150,
		MaxConsecLosses:  0,
		CooldownBars:     0,
	}

	// Compile rules
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	return st
}

// buildManualVolSMAEMAStrategy builds the VolSMA+EMA strategy from seed 6302889439695856639
// This replicates the original winning strategy with corrected feature indices
func buildManualVolSMAEMAStrategy(feats Features, feeBps, slipBps float32) Strategy {
	// Helper to safely get feature index
	getIdx := func(name string) int {
		if idx, ok := feats.Index[name]; ok {
			return idx
		}
		fmt.Printf("Warning: Feature '%s' not found, using -1\n", name)
		return -1
	}

	// Original strategy feature mapping (with corrections):
	// Rising F[27] 20  → Rising VolSMA20 20   (was F[27] originally, now at F[26])
	// Falling F[23] 7 → Falling MinusDI 7    (was F[23] originally, now at F[24])
	// Falling F[2] 16 → Falling EMA50 16     (still at F[2])
	// CrossUp F[16] F[17] → CrossUp BB_Width50 MACD (was F[16],F[17] originally, now at F[13],F[14])
	// CrossUp F[1] F[40] → CrossUp EMA20 SwingHigh (was F[1],F[40] originally, now at F[1],F[39])
	// SlopeLT F[4] 0.06 6 → SlopeLT EMA200 0.06 6 (still at F[4])

	volSMA20Idx := getIdx("VolSMA20")
	minusDIIIdx := getIdx("MinusDI")
	ema50Idx := getIdx("EMA50")
	bbWidth50Idx := getIdx("BB_Width50")
	macdIdx := getIdx("MACD")
	ema20Idx := getIdx("EMA20") // F[1] = EMA20 (both originally and now)
	swingHighIdx := getIdx("SwingHigh")
	ema200Idx := getIdx("EMA200")

	activeIdx := getIdx("Active")
	volPerTradeIdx := getIdx("VolPerTrade")
	bbUpper20Idx := getIdx("BB_Upper20")

	// Entry Rule: (AND (AND (AND (AND (Rising VolSMA20 20) (Falling MinusDI 7))
	//                            (OR (Falling EMA50 16) (CrossUp BB_Width50 MACD)))
	//                            (CrossUp EMA20 SwingHigh))
	//                            (SlopeLT EMA200 0.06 6))

	// Innermost: (Rising VolSMA20 20) AND (Falling MinusDI 7)
	inner1 := &RuleNode{
		Op: OpAnd,
		L: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind:     LeafRising,
				A:        volSMA20Idx,
				Lookback: 20,
			},
		},
		R: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind:     LeafFalling,
				A:        minusDIIIdx,
				Lookback: 7,
			},
		},
	}

	// (OR (Falling EMA50 16) (CrossUp BB_Width50 MACD))
	orNode := &RuleNode{
		Op: OpOr,
		L: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind:     LeafFalling,
				A:        ema50Idx,
				Lookback: 16,
			},
		},
		R: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind: LeafCrossUp,
				A:    bbWidth50Idx,
				B:    macdIdx,
			},
		},
	}

	// Combine: inner1 AND orNode
	inner2 := &RuleNode{
		Op: OpAnd,
		L:  inner1,
		R:  orNode,
	}

	// (CrossUp EMA20 SwingHigh)
	crossNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafCrossUp,
			A:    ema20Idx,
			B:    swingHighIdx,
		},
	}

	// Combine: inner2 AND crossNode
	inner3 := &RuleNode{
		Op: OpAnd,
		L:  inner2,
		R:  crossNode,
	}

	// (SlopeLT EMA200 0.06 6)
	slopeNode := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind:     LeafSlopeLT,
			A:        ema200Idx,
			X:        0.06,
			Lookback: 6,
		},
	}

	// Entry root: inner3 AND slopeNode
	entryRoot := &RuleNode{
		Op: OpAnd,
		L:  inner3,
		R:  slopeNode,
	}

	// Exit Rule: (GT ROC20 -345827.62) - essentially always false (no exit signal)
	roc20Idx := getIdx("ROC20")
	exitRoot := &RuleNode{
		Op: OpLeaf,
		Leaf: Leaf{
			Kind: LeafGT,
			A:    roc20Idx,
			X:    -345827.62,
		},
	}

	// Regime Filter: (AND (NOT (Rising Active 19)) (OR (GT VolPerTrade 0.18) (GT BB_Upper20 144.19)))
	notNode := &RuleNode{
		Op: OpNot,
		L: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind:     LeafRising,
				A:        activeIdx,
				Lookback: 19,
			},
		},
	}

	orRegimeNode := &RuleNode{
		Op: OpOr,
		L: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind: LeafGT,
				A:    volPerTradeIdx,
				X:    0.18,
			},
		},
		R: &RuleNode{
			Op: OpLeaf,
			Leaf: Leaf{
				Kind: LeafGT,
				A:    bbUpper20Idx,
				X:    144.19,
			},
		},
	}

	regimeRoot := &RuleNode{
		Op: OpAnd,
		L:  notNode,
		R:  orRegimeNode,
	}

	// Build strategy with original risk parameters
	st := Strategy{
		Seed:             6302889439695856639, // Original seed for reference
		FeeBps:           feeBps,
		SlippageBps:      slipBps,
		RiskPct:          1.0,
		Direction:        1, // Long only
		EntryRule:        RuleTree{Root: entryRoot},
		ExitRule:         RuleTree{Root: exitRoot},
		RegimeFilter:     RuleTree{Root: regimeRoot},
		StopLoss:         StopModel{Kind: "fixed", Value: 3.75},
		TakeProfit:       TPModel{Kind: "fixed", Value: 6.09},
		Trail:            TrailModel{Active: true, ATRMult: 2.50},
		VolatilityFilter: VolFilterModel{Enabled: false}, // Disabled for manual strategies
		MaxHoldBars:      150,
		MaxConsecLosses:  0,
		CooldownBars:     0,
	}

	// Debug: Print feature indices being used
	fmt.Println("\n=== VolSMA+EMA Strategy Feature Indices ===")
	fmt.Printf("VolSMA20: F[%d]\n", volSMA20Idx)
	fmt.Printf("MinusDI: F[%d]\n", minusDIIIdx)
	fmt.Printf("EMA50: F[%d]\n", ema50Idx)
	fmt.Printf("BB_Width50: F[%d]\n", bbWidth50Idx)
	fmt.Printf("MACD: F[%d]\n", macdIdx)
	fmt.Printf("EMA20: F[%d]\n", ema20Idx)
	fmt.Printf("SwingHigh: F[%d]\n", swingHighIdx)
	fmt.Printf("EMA200: F[%d]\n", ema200Idx)
	fmt.Printf("Active: F[%d]\n", activeIdx)
	fmt.Printf("VolPerTrade: F[%d]\n", volPerTradeIdx)
	fmt.Printf("BB_Upper20: F[%d]\n", bbUpper20Idx)
	fmt.Printf("ROC20: F[%d]\n", roc20Idx)
	fmt.Println("==============================================")

	// Debug: Print rule tree with feature names
	fmt.Println("=== Entry Rule (with feature names) ===")
	fmt.Println(ruleTreeToStringWithNames(entryRoot, feats))
	fmt.Println()

	fmt.Println("=== Regime Filter (with feature names) ===")
	fmt.Println(ruleTreeToStringWithNames(regimeRoot, feats))
	fmt.Println()

	// Compile rules
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	return st
}

// runTraceMode runs a single strategy and outputs per-bar state CSV
func runTraceMode(dataFile string, seed int64, csvPath, manual string, openCSV bool, feeBps, slipBps float64, traceWindow string) {
	fmt.Println("Running in TRACE mode - per-bar state output")
	fmt.Println("============================================")
	fmt.Printf("Seed: %d, CSV: %s, Manual: %s, FeeBps: %.1f, SlippageBps: %.1f\n", seed, csvPath, manual, feeBps, slipBps)
	fmt.Println()

	// Load data
	fmt.Println("Loading data...")
	series, err := LoadBinanceKlinesCSV(dataFile)
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
	var winnerRow *WinnerRow // Declare outside so it's in scope for safety prints
	if manual != "" {
		// Accept "ema20x50" or "volsma_ema" as manual strategies
		if manual == "ema20x50" {
			fmt.Printf("Using manual EMA20x50 strategy\n")
			st = buildManualEMA20x50(feats, float32(feeBps), float32(slipBps))
		} else if manual == "volsma_ema" {
			fmt.Printf("Using manual VolSMA+EMA strategy (seed 6302889439695856639)\n")
			st = buildManualVolSMAEMAStrategy(feats, float32(feeBps), float32(slipBps))
		} else {
			fmt.Printf("Error: -trace_manual must be 'ema20x50' or 'volsma_ema', got '%s'\n", manual)
			return
		}
	} else {
		// Load winner from winners_tested.jsonl or winners.jsonl
		winnerRow = new(WinnerRow)
		if seed > 0 {
			winnerRow, err = loadWinnerBySeed("winners_tested.jsonl", seed)
			if err != nil {
				// Try winners.jsonl as fallback
				winnerRow, err = loadWinnerBySeed("winners.jsonl", seed)
				if err != nil {
					fmt.Printf("Error loading winner with seed %d: %v\n", seed, err)
					return
				}
			}
			fmt.Printf("Loaded winner with seed %d\n", seed)
		} else {
			winnerRow, err = loadFirstWinner("winners_tested.jsonl")
			if err != nil {
				// Try winners.jsonl as fallback
				winnerRow, err = loadFirstWinner("winners.jsonl")
				if err != nil {
					fmt.Printf("Error loading first winner: %v\n", err)
					return
				}
			}
			fmt.Printf("Loaded first winner (seed=%d)\n", winnerRow.Seed)
		}

		// Rebuild strategy from winner
		st = Strategy{
			Seed:        winnerRow.Seed,
			FeeBps:      float32(feeBps),
			SlippageBps: float32(slipBps),
			RiskPct:     1.0,
			Direction:   winnerRow.Direction,
			EntryRule: RuleTree{
				Root: parseRuleTree(winnerRow.EntryRule),
			},
			ExitRule: RuleTree{
				Root: parseRuleTree(winnerRow.ExitRule),
			},
			RegimeFilter: RuleTree{
				Root: parseRuleTree(winnerRow.RegimeFilter),
			},
			StopLoss:         parseStopModel(winnerRow.StopLoss),
			TakeProfit:       parseTPModel(winnerRow.TakeProfit),
			Trail:            parseTrailModel(winnerRow.Trail),
			VolatilityFilter: VolFilterModel{Enabled: false}, // Default disabled for old strategies
		}

		// SANITY CHECK: Validate cross operations in trace mode
		// Warn user if strategy has invalid cross operations (but don't fail, just sanitize)
		{
			_, entryInvalid := validateCrossSanity(st.EntryRule.Root, feats)
			_, exitInvalid := validateCrossSanity(st.ExitRule.Root, feats)
			_, regimeInvalid := validateCrossSanity(st.RegimeFilter.Root, feats)

			totalInvalid := entryInvalid + exitInvalid + regimeInvalid
			if totalInvalid > 0 {
				fmt.Printf("WARNING: Strategy has %d invalid CrossUp/CrossDown operations - sanitizing...\n", totalInvalid)
				// Sanitize the strategy
				sanitizeCrossOperations(rand.New(rand.NewSource(winnerRow.Seed)), st.EntryRule.Root, feats)
				sanitizeCrossOperations(rand.New(rand.NewSource(winnerRow.Seed)), st.ExitRule.Root, feats)
				sanitizeCrossOperations(rand.New(rand.NewSource(winnerRow.Seed)), st.RegimeFilter.Root, feats)
			}
		}
	}

	// Compile bytecode
	st.EntryCompiled = compileRuleTree(st.EntryRule.Root)
	st.ExitCompiled = compileRuleTree(st.ExitRule.Root)
	st.RegimeCompiled = compileRuleTree(st.RegimeFilter.Root)

	// SAFETY PRINT: Show resolved feature names (only for loaded strategies, manual has its own debug)
	if manual == "" && winnerRow != nil {
		// Compute runtime feature map hash
		runtimeHash := ComputeFeatureMapHash(feats)
		runtimeVersion := GetFeatureMapVersion(feats)

		fmt.Println("\n=== STRATEGY LOADED - Feature Index Resolution ===")
		fmt.Printf("FeatureMapHash: %s\n", runtimeHash)
		fmt.Printf("FeatureSetVersion: %s\n", runtimeVersion)

		// Print raw AST (original rule text with indices)
		fmt.Printf("\n--- Entry Rule (raw AST with indices) ---\n")
		fmt.Printf("  %s\n", winnerRow.EntryRule)

		// Print resolved version (with feature names)
		fmt.Printf("\n--- Entry Rule (resolved with feature names) ---\n")
		fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.EntryRule.Root, feats))

		fmt.Printf("\n--- Regime Filter (resolved) ---\n")
		fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.RegimeFilter.Root, feats))

		fmt.Printf("\n--- Exit Rule (resolved) ---\n")
		fmt.Printf("  %s\n", ruleTreeToStringWithNames(st.ExitRule.Root, feats))

		// Validate feature map hash if stored in winner file
		if winnerRow.FeatureMapHash != "" && winnerRow.FeatureMapHash != runtimeHash {
			fmt.Printf("\n!!! FEATURE MAP MISMATCH DETECTED - REJECTING STRATEGY !!!\n")
			fmt.Printf("Stored hash:  %s\n", winnerRow.FeatureMapHash)
			fmt.Printf("Runtime hash: %s\n\n", runtimeHash)
			fmt.Printf("Stored version (first 5 features): %s\n\n", truncateVersion(winnerRow.FeatureMapHash, runtimeVersion, 5))
			fmt.Printf("Runtime version (first 5 features): %s\n", truncateVersion(runtimeHash, runtimeVersion, 5))
			fmt.Printf("\nSTRATEGY REJECTED - Feature indices have changed since strategy was created.\n")
			fmt.Printf("This strategy will produce WRONG results and MUST be regenerated.\n")
			fmt.Printf("============================================================\n\n")
			return
		} else if winnerRow.FeatureMapHash != "" {
			fmt.Printf("\n%s Feature map hash validated: MATCH\n", logx.Success("✓"))
		} else {
			fmt.Printf("\n%s WARNING: No stored feature_map_hash (legacy winner from old version)\n", logx.Warn("⚠"))
			fmt.Printf("   Cannot validate feature indices. Strategy may produce WRONG results.\n")
			fmt.Printf("   Recommendation: Regenerate this strategy with current feature order.\n")
			fmt.Printf("   Proceeding in trace mode for analysis only (NOT recommended for live trading)\n")
		}
		fmt.Println("==================================================")
	}

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
