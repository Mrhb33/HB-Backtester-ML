# HB Backtest Checker

## Table of Contents

1. [Overview](#overview)
2. [Project Purpose](#project-purpose)
3. [Core Concepts](#core-concepts)
4. [Architecture](#architecture)
5. [File-by-File Documentation](#file-by-file-documentation)
6. [Data Structures](#data-structures)
7. [Algorithms and Processing](#algorithms-and-processing)
8. [Configuration](#configuration)
9. [Usage](#usage)
10. [Key Features](#key-features)
11. [Technical Indicators](#technical-indicators)
12. [Validation Pipeline](#validation-pipeline)

---

## Overview

**HB Backtest Checker** is a state-of-the-art, evolutionary trading strategy search engine written in Go. It automatically discovers, validates, and optimizes trading strategies using genetic algorithms combined with rigorous out-of-sample (OOS) testing and walk-forward analysis.

The system is designed to find robust, profitable trading strategies across different timeframes and market conditions while preventing overfitting through multi-stage validation.

**Key Capabilities:**
- Automated strategy discovery using genetic algorithms
- 50+ technical indicators for rule generation
- Multi-fidelity evaluation pipeline (screen → train → validation → OOS)
- Walk-forward validation with monthly constraints
- Anti-lookahead bias enforcement with mathematical proofs
- Real-time monitoring via TUI and WebSocket dashboard
- Parallel evaluation using worker pools

---

## Project Purpose

The primary goal of this project is to **automate the discovery of profitable trading strategies** without manual intervention. Unlike traditional backtesting tools that require pre-defined strategies, HB Backtest Checker:

1. **Generates** millions of unique strategy combinations
2. **Tests** each strategy across multiple time periods
3. **Validates** using rigorous statistical methods
4. **Ranks** strategies by risk-adjusted returns
5. **Outputs** only the most robust, profitable strategies

The system prevents common quantitative trading pitfalls:
- **Overfitting** via walk-forward validation
- **Lookahead bias** via strict barrier rules
- **Data snooping** via DSR-lite deflation penalty
- **Curve fitting** via out-of-sample constraints

---

## Core Concepts

### Strategy Representation

A **Strategy** consists of:
- **Entry Rule**: When to open a position (logical tree of conditions)
- **Exit Rule**: When to close a position (signals + risk management)
- **Regime Filter**: Market conditions to trade in
- **Risk Management**: Stop loss, take profit, trailing stops
- **Direction**: Long (+1) or Short (-1)

### Rule Trees

Trading rules are represented as **logical trees** with operators:
- **AND**: Both conditions must be true
- **OR**: Either condition can be true
- **NOT**: Negate a condition
- **LEAF**: Primitive condition (e.g., `CrossUp(EMA10, EMA20)`)

Example entry rule:
```
(AND
  (OR
    (CrossUp EMA10 EMA20)
    (CrossUp EMA20 EMA50)
  )
  (GT RSI14 70)
)
```

### Features and Feature Types

The system uses **typed features** to prevent invalid operations:
- **PriceLevel**: EMAs, HMAs, Bollinger Bands, etc.
- **Oscillator**: RSI, MFI, ADX, Stochastic (0-100 bounded)
- **Momentum**: ROC, MACD (centered around 0)
- **VolumeRaw**: OBV, VolSMA (raw volume units)
- **VolumeZScore**: VolZ20, VolZ50 (standardized)
- **RangeNorm**: BB_Width, BodyPct (0-1 ratios)
- **EventFlag**: BOS, FVG, Sweep (binary events)

**Critical Rule**: Features can only be crossed with compatible types (e.g., you can't CrossUp a PriceLevel with an Oscillator).

### Multi-Fidelity Evaluation

Strategies progress through **stages of increasing rigor**:

1. **Screen Stage**: Quick evaluation on 6 months (filter obvious junk)
2. **Train Stage**: Full backtest on training data (4+ years)
3. **Validation Stage**: Evaluate on validation data (2 years)
4. **OOS Stage**: Walk-forward analysis with monthly constraints

Each stage has increasingly strict requirements for trades, drawdown, and returns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.go                              │
│  - Command-line parsing                                     │
│  - Worker pool orchestration                                │
│  - Checkpoint management                                    │
│  - Progress tracking                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   evolution.go                              │
│  - Hall of Fame (elite strategies)                          │
│  - Tournament selection                                     │
│  - Mutation operations                                      │
│  - Crossover (mating)                                       │
│  - Diversity tracking via fingerprints                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  multi_fidelity.go                           │
│  - Screen → Train → Validation → OOS pipeline               │
│  - Entry rate checking (prevent zero trades)                │
│  - Regime validation                                        │
│  - Walk-forward fold management                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    backtest.go                              │
│  - Core backtesting engine                                  │
│  - Position lifecycle (entry → hold → exit)                 │
│  - Fee/slippage simulation                                  │
│  - MTM equity tracking                                      │
│  - Performance metrics calculation                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   bytecode.go                               │
│  - Rule tree compilation                                    │
│  - Stack-based VM for fast evaluation                       │
│  - Verification against AST                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    features.go                              │
│  - 50+ technical indicator calculations                     │
│  - Feature type metadata                                    │
│  - Feature statistics (min/max/mean/std)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## File-by-File Documentation

### Core Application Files

#### `main.go`
**Purpose**: Application entry point and orchestration

**Key Functions**:
- `main()`: Parse flags, load data, initialize search loop
- `worker()`: Parallel strategy evaluation goroutine
- `searchMode()`: Main evolutionary search loop
- `testMode()`: Test a single strategy
- `goldenMode()`: Full backtest with trade logging

**Key Structures**:
- Command-line flags for configuration
- Checkpoint loading/saving
- Worker pool management
- Statistics aggregation

#### `backtest.go`
**Purpose**: Core backtesting engine

**Key Functions**:
- `evaluateStrategy()`: Run backtest and return results
- `evaluateStrategyWindow()`: Backtest on a specific time window
- `coreBacktest()`: Unified backtest engine (the heart of the system)
- `computeScore()`: Calculate fitness score from metrics
- `computeScoreWithSmoothness()`: Score with Sortino ratio

**Key Structures**:
- `Trade`: Single completed trade with entry/exit details
- `ActiveTrade`: Currently open position
- `PendingEntry`: Scheduled entry for next bar
- `Result`: Performance metrics (return, DD, win rate, etc.)

**Processing Flow**:
```
For each bar:
  1. Execute pending entry (if any)
  2. Check exits (TP/SL/Trail/MaxHold/ExitRule)
  3. Apply cooldown
  4. Evaluate entry signal
  5. Update mark-to-market equity
```

#### `strategy.go`
**Purpose**: Strategy representation and generation

**Key Functions**:
- `randomStrategy()`: Generate a new random strategy
- `randomRuleNode()`: Generate random rule tree
- `randomEntryLeaf()`: Generate random entry condition
- `mutateStrategy()`: Apply mutations to a strategy
- `crossover()`: Combine two parent strategies
- `CoarseFingerprint()`: Strategy diversity tracking

**Key Structures**:
- `Strategy`: Complete trading strategy definition
- `RuleTree`: Logical tree of trading rules
- `RuleNode`: Single node in rule tree (AND/OR/NOT/LEAF)
- `Leaf`: Primitive condition with features and thresholds

**Leaf Types**:
- `GT`/`LT`: Greater than / less than
- `CrossUp`/`CrossDown`: Series crossing
- `BreakUp`/`BreakDown`: Level breaking
- `Rising`/`Falling`: Momentum over lookback
- `Between`: Value in range
- `SlopeGT`/`SlopeLT`: Slope comparison

#### `evolution.go`
**Purpose**: Genetic algorithm implementation

**Key Functions**:
- `NewHallOfFame()`: Create elite strategy storage
- `Sample()`: Tournament selection from elites
- `Add()`: Add elite with diversity tracking
- `SeedFromCandidates()`: Emergency recovery mechanism
- `mutateRuleTree()`: Apply tree mutations
- `bigMutation()`: Radical mutation for escaping local maxima

**Key Structures**:
- `HallOfFame`: Thread-safe elite strategy storage
- `Elite`: Strategy with train/validation results

**Mutation Types**:
1. Leaf parameter mutations (threshold changes)
2. Feature changes (different indicators)
3. Operator flips (AND ↔ OR)
4. Subtree replacement
5. NOT insertion/removal
6. Branch swapping
7. Pruning
8. Simplification

#### `multi_fidelity.go`
**Purpose**: Multi-stage validation pipeline

**Key Functions**:
- `evaluateMultiFidelity()`: Screen → Train → Validation pipeline
- `evaluateWithWalkForward()`: Walk-forward OOS validation
- `checkEntryRate()`: Quick scan for entry signals
- `validateRegimeRate()`: Sanity check regime filter
- `checkEntryRegimeOverlap()`: Ensure entry/regime overlap

**Global Rejection Counters**:
- `screenFail*`: Strategies failing screen stage
- `trainFail*`: Strategies failing train stage
- `oosReject*`: Strategies failing OOS validation
- `strategiesPassed`: Total accepted strategies

**Screen Relax Levels**:
- `0` (Strict): High trade count requirements
- `1` (Normal): Balanced thresholds
- `2` (Relaxed): Lower trade counts allowed
- `3` (Very Relaxed): Minimum barriers

#### `features.go`
**Purpose**: Technical indicator calculations

**Key Functions**:
- `computeAllFeatures()`: Calculate all 50+ indicators
- `computeEMA()`, `computeSMA()`, `computeHMA()`: Moving averages
- `computeRSI()`, `computeMFI()`: Oscillators
- `computeMACD()`: Momentum indicator
- `computeBollinger()`: Volatility bands
- `computeATR()`, `computeADX()`: Volatility/trend
- `computeZScore()`: Standardization

**Feature Categories**:
1. **Moving Averages**: EMA10/20/50/100/200, HMA9/20/50/100/200
2. **Oscillators**: RSI7/14/21, MFI14, Stochastic, Williams %R
3. **Momentum**: ROC5/10, MACD, ForceIndex
4. **Volatility**: ATR7/14, Bollinger Bands, Keltner Channels
5. **Volume**: OBV, VolSMA/EMA, VolZ-score, BuyRatio, Imbalance
6. **Market Structure**: HH/LL, BOS, FVG, Sweep, SwingHigh/Low
7. **Candle Anatomy**: BodyPct, WickUpPct, WickDownPct, ClosePos

**Feature Type System**:
```go
type FeatureType uint8
const (
    FeatTypePriceLevel    // Same scale: EMAs, prices
    FeatTypeOscillator    // 0-100 bounded
    FeatTypeMomentum      // Centered at 0
    FeatTypeVolumeRaw     // Raw volume
    FeatTypeVolumeZScore  // Standardized
    FeatTypeRangeNorm     // 0-1 ratios
    FeatTypeEventFlag     // Binary events
)
```

#### `bytecode.go`
**Purpose**: Rule compilation and fast evaluation

**Key Functions**:
- `compileRuleTree()`: Convert AST to bytecode
- `evaluateCompiled()`: Stack-based VM execution
- `evaluateFastLeaf()`: Optimized single-leaf path
- `verifyBytecodeVsAST()`: Verification mode

**Bytecode Format**:
```go
type ByteCode struct {
    Op       OpKind    // AND, OR, NOT, LEAF
    Kind     uint8     // Leaf kind
    A, B     int16     // Feature indices
    X, Y     float32   // Thresholds
    Lookback uint8     // Lookback period
}
```

**Optimizations**:
- Fast path for simple single-leaf rules (1.2-1.5x speedup)
- Pooled boolean stack (reduces GC pressure)
- Epsilon checks for CrossUp/CrossDown (prevents fake signals)

#### `split.go`
**Purpose**: Data splitting utilities

**Key Functions**:
- `GetSplitIndices()`: Compute train/validation/test boundaries
- `GetSplitWindows()`: Create Window structs
- `GetCustomWindow()`: Create custom replay window
- `SliceSeries()`: Slice time series data
- `SliceFeatures()`: Slice feature arrays

**Window Structure**:
```go
type Window struct {
    Start  int  // Trade starts here (inclusive)
    End    int  // Trade ends here (exclusive)
    Warmup int  // History before Start
}
```

**Default Splits**:
- Train: 2017-08-17 to 2022-01-01 (~4.3 years)
- Validation: 2022-01-01 to 2024-01-01 (~2 years)
- Test: 2024-01-01 onwards

#### `oos_stats.go`
**Purpose**: Out-of-sample statistics and walk-forward analysis

**Key Functions**:
- `EvaluateWalkForward()`: Run OOS validation across folds
- `BuildWalkForwardFolds()`: Create train/test windows
- `ComputeOSEStatistics()`: Calculate OOS metrics
- `ComputeMonthlyReturns()`: Per-month performance breakdown

**Walk-Forward Parameters**:
- `TrainDays`: Training period size (default: 365)
- `TestDays`: Test period size (default: 90)
- `StepDays`: Step between folds (default: 30)
- `MinFolds`: Minimum folds required (default: 6)

**OOS Constraints**:
- Minimum monthly return threshold
- Maximum drawdown limit
- Minimum active months ratio
- Sparse month detection

#### `checkpoint.go`
**Purpose**: Persistence and resume capability

**Key Functions**:
- `SaveCheckpoint()`: Write progress to disk
- `LoadCheckpoint()`: Read progress from disk
- Checkpoint structure includes elites, fingerprints, metadata

**Checkpoint Contents**:
```json
{
  "version": 1,
  "elites": [...],
  "seen_fingerprints": [...],
  "tested_count": 10000,
  "start_time": "..."
}
```

#### `backtest_events.go`
**Purpose**: Event logging and tracing

**Key Functions**:
- `buildSubConditionProofsThrottled()`: Generate mathematical proofs
- `buildSubConditionString()`: Human-readable condition string

**Proof Contents**:
- Feature values at signal time
- Comparison results
- Guard checks for lookahead bias
- Computed slopes for slope comparisons

---

## Data Structures

### Strategy
```go
type Strategy struct {
    Seed          int64        // Random seed
    FeeBps        float32      // Trading fee in basis points
    SlippageBps   float32      // Slippage in basis points
    Direction     int          // 1=Long, -1=Short
    RiskPct       float32      // Risk per trade (0-1)
    MaxHoldBars   int          // Maximum hold duration
    CooldownBars  int          // Bars to wait after exit
    MaxConsecLosses int         // Bust protection threshold
    StopLoss      StopModel    // Stop loss configuration
    TakeProfit    TPModel      // Take profit configuration
    Trail         TrailModel   // Trailing stop configuration
    VolatilityFilter VolFilter // Volatility regime filter
    EntryRule     RuleTree     // Entry conditions
    ExitRule      RuleTree     // Exit conditions
    RegimeFilter  RuleTree     // Market regime filter
    EntryCompiled CompiledRule // Compiled entry bytecode
    ExitCompiled  CompiledRule // Compiled exit bytecode
    RegimeCompiled CompiledRule // Compiled regime bytecode
}
```

### Rule Tree
```go
type RuleNode struct {
    Op   OpKind     // AND, OR, NOT, LEAF
    Leaf Leaf       // For LEAF nodes
    L    *RuleNode  // Left child
    R    *RuleNode  // Right child
}

type Leaf struct {
    Kind     LeafKind // GT, LT, CrossUp, etc.
    A        int      // Feature A index
    B        int      // Feature B index (for cross)
    X        float32  // Threshold
    Y        float32  // High threshold (for Between)
    Lookback int      // Lookback period
}
```

### Result
```go
type Result struct {
    Strategy     Strategy
    Score        float32  // Fitness score
    Return       float32  // Total return
    MaxDD        float32  // Maximum drawdown
    WinRate      float32  // Win rate (0-1)
    Expectancy   float32  // Average PnL per trade
    ProfitFactor float32  // Gross wins / gross losses
    Trades       int      // Total trades
    SmoothVol    float32  // Equity volatility
    DownsideVol  float32  // Downside volatility
    // OOS statistics...
}
```

---

## Algorithms and Processing

### Genetic Algorithm Flow

```
1. INITIALIZATION
   ├── Load historical data
   ├── Compute all features
   ├── Create train/val/test windows
   └── Initialize empty Hall of Fame

2. MAIN SEARCH LOOP
   ├── For each iteration:
   │   ├── SELECT parents from Hall of Fame (tournament)
   │   ├── GENERATE child strategies:
   │   │   ├── Mutation (80% probability)
   │   │   └── Crossover (20% probability)
   │   ├── EVALUATE through multi-fidelity pipeline
   │   └── ADD to Hall of Fame if qualified
   │
   ├── Every N iterations:
   │   ├── SAVE checkpoint
   │   ├── PRINT statistics
   │   └── UPDATE dashboards

3. ELITE SELECTION
   ├── Must pass screen gates
   ├── Must pass train gates
   ├── Must pass validation gates
   └── Must pass OOS walk-forward
```

### Multi-Fidelity Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 0: Pre-check                                         │
│  - Empty/nil strategy rejection                             │
│  - Cross-operation validation                               │
│  - Entry rule compilation check                              │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Screen (6 months)                                  │
│  - Entry rate scanning (cheap)                              │
│  - Minimum trades: 5-30 (adaptive)                          │
│  - Maximum DD: 95% (permissive)                             │
│  - Purpose: Filter obvious junk quickly                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Train (4+ years)                                   │
│  - Full backtest on training data                           │
│  - Minimum trades: 15-80 (adaptive)                         │
│  - Maximum DD: 50-75% (timeframe-based)                     │
│  - Minimum return: -10% (floor)                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Validation (2 years)                               │
│  - Backtest on validation data                              │
│  - Compute DSR-lite score with deflation penalty            │
│  - Sortino ratio for downside risk focus                    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: OOS Walk-Forward (if enabled)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Fold 1: Train[0-365] → Test[365-455]                │   │
│  │ Fold 2: Train[30-395] → Test[395-485]               │   │
│  │ Fold 3: Train[60-425] → Test[425-515]               │   │
│  │ ...                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│  OOS Constraints:                                            │
│  - Minimum edges per fold: 30/year                          │
│  - Maximum DD per month: 10-15%                             │
│  - Minimum monthly return: -20% to +2%                      │
│  - Minimum active months: 50%                               │
└─────────────────────────────────────────────────────────────┘
```

### Backtest Processing Loop

```go
for t := 1; t < endLocal; t++ {
    // STEP 0: Compute mark-to-market equity
    markEquity = getCurrentEquity(activeTrade)

    // STEP A: Execute pending entry
    if pending != nil && pending.entryIdx == t {
        executeEntry(pending)
        pending = nil
    }

    // STEP B: Check exits (if in position)
    if activeTrade != nil {
        if gapOpenTP() || gapOpenSL() {
            closeTrade("gap_open")
            continue
        }
        if intrabarTP() || intrabarSL() {
            closeTrade("tp_hit" or "sl_hit")
            continue
        }
        if exitRule() || maxHold() {
            closeTrade("exit_rule")
            continue
        }
        updateTrailingStop()
    }

    // STEP C: Apply cooldown filter
    if t <= cooldownUntil { continue }

    // STEP D: Check for entry signal
    if entryRule() && regimeOK() && volOK() && !inPosition() {
        pending = scheduleEntry()
    }

    // STEP E: Update equity and statistics
    updatePeakAndDD()
    updateSmoothness()
}
```

---

## Configuration

### Command-Line Flags

```bash
# Data
-data string          # CSV file path
-timeframe int        # Bar timeframe in minutes (default: 5)

# Search Mode
-workers int          # Parallel workers (default: 4)
-iterations int      # Total search iterations (default: 100000)
-screen-relax int    # Screen relax level 0-3 (default: 1)
-checkpoint string   # Checkpoint file path

# Walk-Forward
-wf                   # Enable walk-forward validation
-wf-train-days int   # Training period (default: 365)
-wf-test-days int    # Test period (default: 90)
-wf-step-days int    # Step size (default: 30)
-wf-min-folds int    # Minimum folds (default: 6)

# Risk Management
-max-val-dd float     # Maximum validation DD (default: 0.55)
-min-val-return float # Minimum validation return (default: 0.02)
-min-val-trades int   # Minimum validation trades (default: 20)

# Output
-verbose              # Enable verbose logging
-log-trades           # Log all trades
-dashboard            # Enable TUI dashboard
-web-port int         # WebSocket dashboard port
```

### meta.json

```json
{
  "radical_p": 0.05,           # Probability of radical mutation
  "sur_explore_p": 0.059,      # Surrogate exploration probability
  "sur_threshold": -0.2,       # Surrogate model threshold
  "max_val_dd": 0.55,          # Maximum validation drawdown
  "min_val_return": 0.02,      # Minimum validation return
  "min_val_trades": 20,        # Minimum validation trades
  "screen_relax_level": 1      # Default screening strictness
}
```

### Data Format

**Input CSV** (Binance klines format):
```csv
timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote
```

**Winners JSONL** (elite strategies):
```json
{"strategy": {...}, "train_result": {...}, "val_result": {...}, "val_score": 1.23}
```

---

## Usage

### Basic Search

```bash
# Run search on 5-minute data
go run . -data btcusdt_5m.csv -timeframe 5 -workers 8

# Resume from checkpoint
go run . -data btcusdt_5m.csv -checkpoint checkpoint.json

# With walk-forward validation
go run . -data btcusdt_5m.csv -wf -wf-train-days 365 -wf-test-days 90
```

### Test a Strategy

```bash
# Test from winners file
go run . -mode test -winners winners.jsonl

# Test with full trade logging
go run . -mode test -winners winners.jsonl -log-trades
```

### Golden Mode (Full Export)

```bash
# Export all trades with proofs
go run . -mode golden -winners winners.jsonl -golden-trades.csv
```

### Monitoring

```bash
# Enable TUI dashboard
go run . -data btcusdt_5m.csv -dashboard

# Enable WebSocket dashboard
go run . -data btcusdt_5m.csv -web-port 8080
# Then open dashboard.html in browser
```

---

## Key Features

### 1. Anti-Lookahead Bias

**Strict barrier rules** ensure no future information is used:

- Entry signals evaluated at bar `t`, executed at bar `t+1` open
- Exit rules evaluated at bar `t-1`, executed at bar `t` open
- All indicators use only historical data up to current bar
- Mathematical proofs track all feature access

### 2. Feature Type System

Prevents **invalid operations**:
- `CrossUp(RSI14, EMA20)` → **REJECTED** (oscillator ≠ price level)
- `CrossUp(EMA10, EMA20)` → **ACCEPTED** (same price family)
- `CrossUp(VolSMA20, VolZ20)` → **REJECTED** (raw ≠ z-score)

### 3. Multi-Fidelity Evaluation

**Progressive filtering** saves computation:
- Stage 1 rejects ~80% of junk strategies (fast)
- Stage 2 rejects ~15% more (medium speed)
- Stage 3 applies full scoring (slow)
- Stage 4 validates OOS robustness (very slow)

### 4. Walk-Forward Validation

**Realistic performance estimation**:
```
Train[0-365]  → Test[365-455]  → OOS metrics
Train[30-395] → Test[395-485]  → OOS metrics
Train[60-425] → Test[425-515]  → OOS metrics
...
```

### 5. DSR-Lite Scoring

**Deflation penalty** prevents data snooping:
```
finalScore = baseScore - k × log(1 + testedCount / 10000)
```

As more strategies are tested, the bar for acceptance rises.

### 6. Diversity Tracking

**Coarse fingerprinting** prevents duplicate elites:
- Groups strategies by rule structure (thresholds ignored)
- Maximum 5 elites per fingerprint family
- Encourages exploration of different rule patterns

---

## Technical Indicators

### Moving Averages
- **EMA**: Exponential Moving Average (periods: 10, 20, 50, 100, 200)
- **HMA**: Hull Moving Average (periods: 9, 20, 50, 100, 200)
- **SMA**: Simple Moving Average (for volume)
- **WMA**: Weighted Moving Average (for HMA calculation)
- **Kijun-sen**: Ichimoku Base Line (26)

### Oscillators
- **RSI**: Relative Strength Index (periods: 7, 14, 21)
- **MFI**: Money Flow Index (14)
- **Stochastic**: %K and %D (periods: 5, 14)
- **Williams %R**: Williams Percent R (periods: 7, 14)

### Momentum
- **ROC**: Rate of Change (periods: 5, 10, 20)
- **MACD**: Moving Average Convergence Divergence
- **Force Index**: (periods: 2, 13)
- **Momentum**: (periods: 60, 240)
- **SLOPE**: Linear regression slope (20)

### Volatility
- **ATR**: Average True Range (periods: 7, 14)
- **Bollinger Bands**: (periods: 20, 50, std: 2.0)
- **Keltner Channels**: (period: 20)
- **Donchian Channels**: (periods: 20, 55)

### Volume
- **OBV**: On-Balance Volume
- **VolSMA/VolEMA**: Volume averages (periods: 20, 50)
- **VolZ**: Volume z-score (periods: 20, 50)
- **BuyRatio**: Taker buy volume / total volume
- **Imbalance**: (Buy - Sell) / Total
- **VolPerTrade**: Volume / Trade count

### Market Structure
- **HH/LL**: Highest High / Lowest Low (lookbacks: 20, 50, 100, 200)
- **SwingHigh/Low**: Rolling max/min (lookback: 20)
- **BOS**: Break of Structure
- **FVG**: Fair Value Gap (Up/Down)
- **Sweep**: Liquidity sweep detection (lookbacks: 20, 50, 100, 200)

### Candle Anatomy
- **BodyPct**: |Close - Open| / Range
- **WickUpPct**: Upper wick / Range
- **WickDownPct**: Lower wick / Range
- **ClosePos**: (Close - Low) / Range
- **Body**: |Close - Open|
- **HighLowDiff**: High - Low
- **RangeWidth**: Body / Range

### Trend/ADX
- **ADX**: Average Directional Index (14)
- **PlusDI/MinusDI**: Directional Indicators (14)

---

## Validation Pipeline

### Entry Rate Gates

**Purpose**: Prevent strategies that never trade

```go
// Adaptive thresholds based on window size
minEdges = multiplier × windowBars / baseWindowBars
maxEdges = 120 × windowBars / baseWindowBars

// Annualized rate calculation
edgesPerYear = (edges / windowBars) × barsPerYear
```

### Regime Filter Validation

**Sanity bands**:
```
1% ≤ regimePassRate ≤ 99%
```

Rejects:
- **All-blocking** regimes (< 1% pass)
- **Always-true** regimes (> 99% pass)

### Overlap Validation

**Ensures entry and regime filters can both be true**:
```
entry_true × regime_ok ≥ 20  (count)
OR
entry_true × regime_ok ≥ 0.05% (rate)
```

### OOS Monthly Constraints

**Per-month limits**:
- **Maximum DD**: 10-15% (configurable)
- **Minimum return**: -20% to +2% (floor)
- **Sparse month rejection**: > 80% zero-trade months
- **Active month requirement**: ≥ 50% months with trades

### Final Fitness Calculation

```go
fitness = 0.5 × oosGeoMonthly
        + 0.2 × (oosGeoMonthly - trainGeoMonthly)  // Overfit penalty
        + 0.1 × oosMedianMonthly
        + 0.1 × oosWinRate
        - 0.1 × complexity
```

---

## Output Files

### checkpoint.json
Search progress state for resume capability.

### winners.jsonl
Elite strategies that passed all validation (one JSON per line).

### best_every_10000.txt
Best strategy score at each 10,000 evaluation milestone.

### *.csv (Golden Mode)
- **trades.csv**: All trades with entry/exit details
- **states.csv**: Per-bar state tracking

---

## Dependencies

- **Go 1.24+**: Core language
- **Bubble Tea**: TUI framework
- **Gorilla WebSocket**: Real-time dashboard
- **golang.org/x/term**: Terminal detection

---

## Performance Characteristics

### Optimization Techniques

1. **Bytecode compilation**: 10-100x faster than AST evaluation
2. **Fast leaf path**: 1.2-1.5x speedup for simple rules
3. **Pooled stacks**: Reduced GC pressure
4. **Adaptive screening**: Early rejection of junk strategies
5. **Parallel workers**: Multi-core utilization

### Typical Performance

- **5-minute data**: ~1000-5000 strategies/second/core
- **1-hour data**: ~5000-20000 strategies/second/core
- **Walk-forward**: 10-50x slower than simple validation

### Memory Usage

- **Base**: ~200MB for 5-minute BTC dataset
- **Per worker**: ~50MB for feature arrays
- **Checkpoint**: ~1-5MB depending on elite count

---

## Troubleshooting

### Low Strategy Acceptance Rate

**Symptoms**: Very few or no elites found

**Solutions**:
1. Lower `-screen-relax` to 2 or 3
2. Reduce `-min-val-trades` threshold
3. Increase `-max-val-dd` limit
4. Check if timeframe has enough volatility

### Zero OOS Trades

**Symptoms**: Strategies pass train but have 0 OOS trades

**Causes**:
- Entry/regime filters don't overlap in OOS
- Edges per year too low (< 30)

**Solutions**:
1. Check entry rate in combined window
2. Verify regime filter pass rate (1-99%)
3. Reduce `-wf-min-edges-per-year`

### High Drawdown Strategies

**Symptoms**: Accepted elites have 60%+ DD

**Solutions**:
1. Reduce SL ATRMult cap (default: 10.0)
2. Lower `-max-val-dd`
3. Enable stricter screening (level 0)

---

## Version History

- **V1.61**: Current - Enhanced feature validation
- **V1.6**: Walk-forward improvements
- **V1.51**: Monthly OOS constraints
- **V1.5**: Multi-fidelity pipeline
- **V1.4**: Core backtesting engine

---

## License

[Your License Here]

---

## Contact

[Your Contact Information]
