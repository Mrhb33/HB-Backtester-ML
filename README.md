# HB Backtest Strategy Search Engine

A high-performance genetic algorithm-based backtesting system for automated trading strategy discovery using technical analysis indicators.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Feature System](#feature-system)
- [Strategy Representation](#strategy-representation)
- [Backtesting Engine](#backtesting-engine)
- [Evolution System](#evolution-system)
- [Multi-Fidelity Testing](#multi-fidelity-testing)
- [Walk-Forward Validation](#walk-forward-validation)
- [Scoring System](#scoring-system)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Development](#development)

## Overview

The HB Backtest Strategy Search Engine is a sophisticated tool for discovering profitable trading strategies through evolutionary computation. It uses a genetic algorithm to evolve rule-based trading strategies, rigorously validating them through walk-forward testing to avoid overfitting.

**Key capabilities:**
- Generate and evolve millions of trading strategies
- Test with realistic transaction costs and slippage
- Walk-forward validation to prevent overfitting
- Support for 60+ technical indicators
- Multi-fidelity evaluation pipeline for efficiency
- Comprehensive performance metrics and reporting

## Features

- **Genetic Algorithm**: Tournament selection, mutation, crossover, and diversity preservation
- **60+ Technical Indicators**: EMA, HMA, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic, Williams %R, SuperTrend, Keltner Channels, Donchian Channels, and more
- **Realistic Backtesting**: Transaction fees, slippage, stop-loss, take-profit, trailing stops
- **Walk-Forward Validation**: Time-series cross-validation to detect overfitting
- **Multi-Fidelity Pipeline**: Quick screening → surrogate model → full validation
- **Operating Modes**: Search, test, golden (detailed analysis), trace (per-bar states), validate, diagnose, verify

## Installation

### Prerequisites

- Go 1.24.0 or later
- Binance-style OHLCV CSV data (see [Data Format](#data-format))

### Building

```bash
go build -o hb_bactest_checker.exe
```

## Usage

### Basic Search

```bash
hb_bactest_checker.exe -mode search -data btc_5min_data.csv
```

### Walk-Forward Search (Recommended)

```bash
hb_bactest_checker.exe -mode search -data btc_5min_data.csv -wf
```

### Test Saved Strategies

```bash
hb_bactest_checker.exe -mode test -data btc_5min_data.csv
```

### Golden Mode (Detailed Analysis)

```bash
hb_bactest_checker.exe -mode golden -golden_seed 12345
```

### Trace Mode (Per-Bar Debug)

```bash
hb_bactest_checker.exe -mode trace -trace_seed 12345
```

### Command-Line Flags

```
-mode              Operating mode: search|test|golden|trace|validate|diagnose|verify (default "search")
-data              CSV data file path (default "btc_5min_data.csv")
-timeframe         Data timeframe: 5m, 15m, 60m, 1h, 4h (default "5m")
-wf                Enable walk-forward validation (default false)
-verbose           Enable verbose debug logs
-log_trades        Enable detailed per-trade logging
-recovery          Enable recovery mode (relaxed screening)
-fee_bps           Transaction fee in basis points (default 20)
-slip_bps          Slippage in basis points (default 5)
```

See [Configuration](#configuration) for all available flags.

## Architecture

### System Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  CSV Data File  │───▶│  Data Loader     │───▶│  Series Struct  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Strategy Gen   │───▶│  Feature Engine   │───▶│  Features (60+) │
│  (Genetic Algo)  │    └──────────────────┘    └─────────────────┘
└─────────────────┘                                    │
         │                                              ▼
         │                                    ┌─────────────────┐
         │                                    │  Strategy Tree  │
         │                                    │  (Entry/Exit)    │
         │                                    └─────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Fidelity Pipeline                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │  Screen  │───▶│  Surrogate   │───▶│  Full Backtest          │ │
│  │ (Quick)  │    │  (Filter)    │    │  (Core Engine)          │ │
│  └──────────┘    └──────────────┘    └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Walk-Forward Validation                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Fold 1: Train ──▶ Test ──▶ Metrics                       │  │
│  │  Fold 2: Train ──▶ Test ──▶ Metrics                       │  │
│  │  ...                                                     │  │
│  │  Fold N: Train ──▶ Test ──▶ Metrics                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                           │                                         │
│                           ▼                                         │
│              ┌──────────────────────────┐                          │
│              │  OOS Stats & Monthly      │                          │
│              │  Returns Validation      │                          │
│              └──────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Hall of Fame    │◀───│   Elite Filter   │───▶│  Winners.jsonl  │
│  (Top K)         │    │  (Constraints)    │    │  (Strategies)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

```
CSV Data → Series → Features → Strategy Generation → Multi-Fidelity Testing
                                                                  ↓
                                              Validation → Hall of Fame → Results
```

## Core Components

### Main Entry Point (`main.go`)

**Location**: `main.go:252-500`

The main function orchestrates the entire search process:

1. **Command-line parsing**: Handles all configuration flags
2. **Mode dispatch**: Routes to search, test, golden, trace, validate, diagnose, or verify mode
3. **Data loading**: Loads OHLCV CSV data
4. **Feature computation**: Calculates all 60+ technical indicators
5. **Generator loop**: Creates strategies via genetic algorithm
6. **Worker pool**: Parallel evaluation of candidates
7. **Elite management**: Maintains hall of fame of best strategies

**Operating Modes**:
- `search`: Main genetic algorithm search mode
- `test`: Backtest saved strategies from `winners.jsonl`
- `golden`: Detailed single-strategy analysis with trade export
- `trace`: Per-bar state debugging with proofs
- `validate`: Run validation suite
- `diagnose`: Diagnostic mode for troubleshooting
- `verify`: Bytecode vs AST verification mode

### Data Loading (`data.go`)

**Location**: `data.go:35-342`

**CSV Data Format** (Binance OHLCV):
```
OpenTime,Open,High,Low,Close,Volume,CloseTime,QuoteVolume,Trades,TakerBuyBase,TakerBuyQuote
```

Also supports simple 6-column format:
```
Timestamp,Open,High,Low,Close,Volume
```

**Series Structure** (`data.go:19-33`):
```go
type Series struct {
    T int
    OpenTimeMs, CloseTimeMs []int64
    Open, High, Low, Close []float32
    Volume, QuoteVolume, TakerBuyBase, TakerBuyQuote []float32
    Trades []int32
    CSVRowIndex []int  // Maps Series[i] to original CSV row
}
```

**Key Functions**:
- `LoadBinanceKlinesCSV()`: Loads and validates CSV data (`data.go:35`)
- Automatic timestamp ordering and deduplication
- Interval detection and close time calculation

### Feature Engineering (`features.go`)

**Location**: `features.go:345-1173`

The system computes **60+ features** across multiple categories:

#### Moving Averages
- **EMA**: Exponential Moving Average (10, 20, 50, 100, 200)
- **HMA**: Hull Moving Average (9, 20, 50, 100, 200)
- **Kijun-sen**: Ichimoku Base Line (26)

#### Oscillators
- **RSI**: Relative Strength Index (7, 14, 21)
- **Stochastic**: Stochastic K/D (14, 5)
- **Williams %R**: (14, 7)
- **MFI**: Money Flow Index (14)
- **ADX**: Average Directional Index (14)
- **PlusDI/MinusDI**: Directional Indicators (14)

#### Volatility
- **ATR**: Average True Range (7, 14)
- **Bollinger Bands**: Upper/Lower/Width (20, 50)
- **Keltner Channels**: Upper/Middle/Lower (20)
- **Donchian Channels**: Upper/Lower (20, 55)

#### Momentum
- **MACD**: MACD/Signal/Histogram (12, 26, 9)
- **ROC**: Rate of Change (10, 20)
- **Momentum**: (60, 240)
- **Force Index**: (2, 13)
- **SuperTrend**: (10)

#### Volume
- **OBV**: On-Balance Volume
- **VolSMA/VolEMA**: Volume moving averages (20, 50)
- **VolZ**: Volume Z-score (20, 50)
- **BuyRatio**: Taker buy ratio
- **Imbalance**: Buy/sell imbalance
- **VolPerTrade**: Volume per trade

#### Market Structure
- **SwingHigh/Low**: Rolling swing levels (20 lookback)
- **HH/LL**: Higher Highs/Lower Lows (20, 50, 100, 200)
- **SweepUp/Down**: Liquidity sweep detection (20, 50, 100, 200)
- **BOS**: Break of Structure
- **FVG**: Fair Value Gaps (Up/Down)

#### Candle Anatomy
- **Body**: Absolute candle body size
- **BodyPct**: Body as % of range
- **WickUpPct/WickDownPct**: Wick percentages
- **ClosePos**: Close position in range
- **HighLowDiff**: High - Low difference
- **RangeWidth**: Body/HighLowDiff ratio

#### Event Flags
- **Squeeze**: Bollinger-Keltner squeeze
- **SqueezeBreakUp/Down**: Squeeze breakout
- **StochBullCross/BearCross**: Stochastic crossovers

**Feature Type System** (`features.go:18-35`):

```go
type FeatureType uint8

const (
    FeatTypeUnknown FeatureType = iota
    FeatTypePriceLevel   // EMA*, BB_Upper/Lower*, SwingHigh/Low
    FeatTypePriceRange   // BB_Width*, Body, HighLowDiff
    FeatTypeOscillator   // RSI*, ADX, PlusDI, MFI
    FeatTypeZScore       // VolZ*
    FeatTypeNormalized   // Imbalance, BuyRatio, RangeWidth
    FeatTypeEventFlag    // BOS/FVG/Active style
    FeatTypeVolume       // OBV, VolSMA/EMA
    FeatTypeVolumeDerived // VolZ*, BuyRatio, Imbalance
    FeatTypeATR          // ATR*
    FeatTypeMomentum     // ROC*, MACD*, Hist
)
```

### Strategy Representation (`strategy.go`)

**Location**: `strategy.go:564-584`

**Strategy Structure**:
```go
type Strategy struct {
    Seed             int64
    FeeBps           float32
    SlippageBps      float32
    RiskPct          float32
    Direction        int         // 1=Long, -1=Short
    EntryRule        RuleTree
    EntryCompiled    CompiledRule
    ExitRule         RuleTree
    ExitCompiled     CompiledRule
    StopLoss         StopModel   // "fixed", "atr", "swing"
    TakeProfit       TPModel     // "fixed", "atr"
    Trail            TrailModel
    RegimeFilter     RuleTree
    RegimeCompiled   CompiledRule
    VolatilityFilter VolFilterModel
    MaxHoldBars      int
    MaxConsecLosses  int
    CooldownBars     int
    FeatureMapHash   string
}
```

#### Rule Tree Representation

**Operations** (`strategy.go:408-415`):
- `OpAnd`: Logical AND
- `OpOr`: Logical OR
- `OpNot`: Logical NOT
- `OpLeaf`: Leaf node

**Leaf Types** (`strategy.go:417-433`):
- `LeafGT`: Feature > Threshold
- `LeafLT`: Feature < Threshold
- `LeafCrossUp`: A crosses above B (A[t-1] ≤ B[t-1] && A[t] > B[t])
- `LeafCrossDown`: A crosses below B (A[t-1] ≥ B[t-1] && A[t] < B[t])
- `LeafBreakUp`: A breaks above B (A[t-1] ≤ B[t-1] && A[t] > B[t])
- `LeafBreakDown`: A breaks below B (A[t-1] ≥ B[t-1] && A[t] < B[t])
- `LeafRising`: A increasing over lookback
- `LeafFalling`: A decreasing over lookback
- `LeafBetween`: X ≤ A ≤ Y
- `LeafAbsGT`: |A| > Threshold
- `LeafAbsLT`: |A| < Threshold
- `LeafSlopeGT`: Slope of A > Threshold
- `LeafSlopeLT`: Slope of A < Threshold

**Stop-Loss Models**:
- `fixed`: Fixed percentage
- `atr`: ATR multiple (4.0-10.0)
- `swing`: Swing low/high

**Take-Profit Models**:
- `fixed`: Fixed percentage
- `atr`: ATR multiple (8.0-20.0)

### Backtesting Engine (`backtest.go`)

**Location**: `backtest.go:949-2174`

The core backtest engine simulates trading with realistic execution:

**Position States** (`backtest.go:162-168`):
```go
type PositionState int
const (
    Flat PositionState = iota
    Long
    Short
)
```

**Execution Flow**:

1. **Signal Detection** (bar t):
   - Evaluate regime filter
   - Evaluate entry rule
   - Check volatility filter
   - Check Active filter (volume ≥ 10% average)

2. **Entry Scheduling** (bar t):
   - Schedule entry for bar t+1
   - Store mathematical proofs (LeafProof)

3. **Entry Execution** (bar t+1):
   - Execute at open with slippage
   - Set stop-loss and take-profit

4. **Position Management** (each bar while in position):
   - Check gap-open TP/SL (price jumps past levels)
   - Check intrabar TP/SL
   - Check exit rule / max hold
   - Update trailing stop

**Exit Reasons**:
- `tp_hit`: Take profit hit intrabar
- `sl_hit`: Stop loss hit intrabar
- `trail_hit`: Trailing stop hit
- `tp_gap_open`: Take profit on gap open
- `sl_gap_open`: Stop loss on gap open
- `exit_rule`: Exit rule triggered
- `max_hold`: Maximum hold bars reached
- `end_of_data`: Force close at backtest end

**Performance Metrics** (`backtest.go:135-160`):
- **Return**: Total return (risk-adjusted and raw)
- **MaxDD**: Maximum drawdown (mark-to-market)
- **WinRate**: Winning trades percentage
- **Expectancy**: Average PnL per trade
- **ProfitFactor**: Gross wins / Gross losses (capped at 999.0)
- **SmoothVol**: EMA volatility of equity changes
- **DownsideVol**: Downside volatility for Sortino ratio

**Slippage Model** (`backtest.go:1072-1083`):
```go
slip := st.SlippageBps / 10000
if volZ < -2.0 {
    slip *= 4.0  // Extreme volatility
} else if volZ < -1.0 {
    slip *= 2.0  // High volatility
}
```

### Evolution System (`evolution.go`)

**Location**: `evolution.go:14-1177`

**Hall of Fame** (`evolution.go:25-30`):
```go
type HallOfFame struct {
    mu       sync.RWMutex
    K        int              // Capacity
    Elites   []Elite
    snapshot atomic.Value     // Lock-free reads
}
```

**Tournament Selection** (`evolution.go:154-188`):
- Sample K=4 random elites
- Return best performer
- Lock-free via atomic snapshot

**Mutation Operators** (`evolution.go:753-873`):

1. **Leaf mutations** (40%):
   - Change threshold (scale-aware)
   - Change feature A/B
   - Change leaf kind
   - Tweak lookback

2. **Subtree replace** (20%):
   - Replace random subtree with new random tree

3. **Op flip** (10%):
   - Flip AND ↔ OR

4. **NOT insert/remove** (10%):
   - Add/remove NOT nodes

5. **Swap branches** (10%):
   - Swap left/right children

6. **Prune** (10%):
   - Replace node with one child

**Crossover** (`evolution.go:921-981`):
- Mix entry/exit/regime rules from parents
- Mix risk models (SL/TP/Trail)

**Big Mutation** (`evolution.go:983-1115`):
- Radical mutation for escaping local maxima
- Mutates 4-7 parts simultaneously
- Larger step sizes

**Diversity Preservation** (`evolution.go:76-138`):
- `Fingerprint()`: Exact strategy signature
- `SkeletonFingerprint()`: Structure-only (ignores thresholds)
- `CoarseFingerprint()`: Bucketed thresholds
- Max 5 elites per fingerprint family

### Multi-Fidelity Testing (`multi_fidelity.go`)

**Location**: `multi_fidelity.go:283-1306`

**Three-Stage Pipeline**:

```
Stage 0: Entry Rate Precheck
  └─ Count entry edges (rising edges of all conditions)
  └─ Adaptive limits by window size
  └─ Soft scoring (penalty factor instead of hard reject)

Stage 1: Screen (Fast Filter)
  └─ 6-month window
  └─ Basic DD/trades gates
  └─ Max 95% DD (cheap filter)

Stage 2: Train (Full Window)
  └─ Full training window
  └─ Tighter DD gates (65-75%)
  └─ Minimum trade counts (15-80)

Stage 3: Validation
  └─ Full strict scoring
  └─ DSR-lite deflation penalty
  └─ Smoothness metrics
```

**Entry Rate Checking** (`multi_fidelity.go:170-274`):
```go
func checkEntryRate(full Series, fullF Features, st Strategy, w Window) checkEntryRateResult {
    // Count RISING EDGES (not bars true)
    // Scale limits by window size
    // Return soft penalty factor instead of hard reject
}
```

**Relax Levels** (`multi_fidelity.go:335-366`):
- **0 (Strict)**: minScreenTrades=30, minTrainTrades=80, maxTrainDD=0.50
- **1 (Normal)**: minScreenTrades=5, minTrainTrades=20, maxTrainDD=0.75
- **2 (Relaxed)**: minScreenTrades=15, minTrainTrades=40, maxTrainDD=0.60
- **3 (Very Relaxed)**: minScreenTrades=5, minTrainTrades=15, maxTrainDD=0.65

### Walk-Forward Validation (`walkforward.go`)

**Location**: `walkforward.go:14-854`

**Fold Structure** (`walkforward.go:54-60`):
```go
type Fold struct {
    TrainStart int  // Training window start (inclusive)
    TrainEnd   int  // Training window end (exclusive)
    TestStart  int  // Test window start (inclusive)
    TestEnd    int  // Test window end (exclusive)
    Warmup     int  // Warmup bars before train start
}
```

**Fold Generation** (`walkforward.go:168-250`):
- Timestamp-based boundaries (avoids bar-count issues)
- Half-open intervals: `[start, end)` (no overlap)
- Binary search for precise indices
- Validates minimum fold count

**OOS Constraints** (`walkforward.go:14-51`):
- `MinMonths`: Minimum out-of-sample months (default 3)
- `MinTotalTradesOOS`: Minimum total OOS trades (default 30)
- `MinTradesPerMonth`: Minimum trades per month (default 2)
- `MaxDrawdown`: Maximum allowed drawdown (default 0.70)
- `MinMonthReturn`: Minimum monthly return (default -0.35)
- `MinGeoMonthlyReturn`: Minimum geometric average monthly return (default 0.005)
- `MinMedianMonthly`: Minimum median monthly return (default 0.003)
- `MinActiveMonthsRatio`: Minimum ratio of active months (default 0.25)
- `MaxSparseMonthsRatio`: Maximum ratio of sparse months (default 0.80)

**Monthly Return Calculation**:
- Assigns trades to months by entry timestamp
- Handles MTM (mark-to-market) for open positions
- Computes geometric average of monthly returns

**Fitness Function** (`walkforward.go:803-853`):
```go
fitness = geoAvgMonthly
          - overfitPenalty (train/OOS gap)
          - volatilityPenalty (tiered: >15%, >10%)
          - drawdownPenalty
          - complexityPenalty
```

## Feature System

### Feature Type Taxonomy

The system uses a typed feature system to prevent invalid operations:

| Type | Description | Examples |
|------|-------------|----------|
| `PriceLevel` | Price-denominated indicators | EMA*, BB_Upper/Lower, SwingHigh/Low |
| `PriceRange` | Price range/size | Body, HighLowDiff, BB_Width |
| `Oscillator` | Bounded oscillators (0-100) | RSI*, ADX, MFI, Stoch |
| `ZScore` | Z-score normalized | VolZ* |
| `Normalized` | Normalized ratios | Imbalance, BuyRatio, RangeWidth |
| `EventFlag` | Binary/discrete events | BOS, FVG, Active |
| `Volume` | Volume-based | OBV, VolSMA |
| `VolumeDerived` | Volume-derived metrics | BuyRatio, Imbalance |
| `ATR` | Volatility | ATR* |
| `Momentum` | Rate of change | ROC*, MACD* |

### Cross Operation Compatibility

Cross operations (CrossUp, CrossDown, BreakUp, BreakDown) are only allowed between compatible feature types:

```
Price family: PriceLevel, PriceRange, ATR
Oscillator family: Oscillator
Volume family: Volume, VolumeDerived
Momentum family: Momentum
```

This prevents nonsense operations like `CrossUp(BB_Upper50, MACD_Hist)`.

### Threshold Clamping

Feature-specific threshold bounds prevent impossible conditions:

```go
RSI7:          [1, 99]
VolZ20:        [-3.5, 3.5]
Imbalance:      [-1.0, 1.0]
BuyRatio:       [0.0, 1.0]
BodyPct:        [0.0, 1.0]
StochK_14:      [1, 99]
WilliamsR_14:   [-99, -1]
```

## Scoring System

### Base Score Components (`backtest.go:277-401`)

```go
baseScore = logReturn * 6.0      // Log return, weighted
           + calmar * 2.0         // Return/DD ratio, capped at 10
           + expectancy * 200.0   // Expectancy, capped at ±5
           + tradesReward         // Log(trades) * 0.01
           - ddPenalty * 6.0      // Drawdown penalty
           - tradePenalty         // Penalty for <20 trades
           - retPenalty           // Penalty for negative returns
           - holdPenalty           // Penalty for tiny holds
```

**Caps**:
- `calmar`: capped at 10 (prevents DD gaming)
- `expectancy`: capped at ±5 (prevents gaming)
- `logReturn`: log(1+ret) for outlier reduction

### DSR-Lite Deflation Penalty

```go
deflated = baseScore - 0.5 * log(1 + testedCount / 10000)
```

Forces higher significance as testing progresses.

### Smoothness Metrics (`backtest.go:417-466`)

```go
sortino = ret / downsideVol  // Downside volatility only
smoothPenalty = smoothVol / |ret| * 0.125

finalScore = baseScore + 0.125 * sortino - smoothPenalty
```

**Reduced weights (Problem B fix)**: Smoothness and Sortino weights reduced 4x (from 0.5 to 0.125) during discovery to prevent "monthly consistency" from killing profitable strategies.

## Configuration

### Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-mode` | Operating mode | `search` |
| `-data` | CSV data file | `btc_5min_data.csv` |
| `-timeframe` | Data timeframe (5m, 15m, 1h, 4h) | `5m` |
| `-wf` | Enable walk-forward | `false` |
| `-wf_train_days` | WF training window | `360` |
| `-wf_test_days` | WF test window | `90` |
| `-wf_step_days` | WF step size | `60` |
| `-wf_max_dd` | Maximum OOS drawdown | `0.70` |
| `-wf_min_edges_per_year` | Entry rate threshold | `30.0` |
| `-wf_fold_min_edges_per_year` | Per-fold entry rate | `15.0` |
| `-fee_bps` | Transaction fee (bps) | `20` |
| `-slip_bps` | Slippage (bps) | `5` |
| `-recovery` | Enable recovery mode | `false` |
| `-verbose` | Verbose logging | `false` |
| `-log_trades` | Detailed trade logging | `false` |
| `-checkpoint` | Checkpoint file | `checkpoint.json` |
| `-resume` | Resume from checkpoint | `""` |

### Configuration Files

**`meta.json`**: Runtime parameters
```json
{
  "radical_p": 0.10,
  "sur_explore_p": 0.10,
  "max_val_dd": 0.45,
  "min_val_return": 0.01,
  "min_val_pf": 1.01,
  "screen_relax_level": 1
}
```

**`surrogate.json`**: Surrogate model coefficients
```json
{
  "intercept": 0.0,
  "weights": [...]
}
```

**`checkpoint.json`**: Saved state for resumption
```json
{
  "hof": [...],
  "meta": {...},
  "seen": [...]
}
```

**`winners.jsonl`**: Validated strategies (JSONL format)
```json
{"seed": 12345, "entry_rule": "...", "exit_rule": "...", ...}
```

**`best_every_10000.txt`**: Progress tracking

## Output Files

### Trade Exports

**Trades CSV** (`backtest.go:669-735`):
```csv
TradeID,SignalIndex,EntryIndex,ExitIndex,SignalTime,EntryTime,ExitTime,
Direction,EntryPrice,ExitPrice,PnL,HoldBars,ExitReason,StopPrice,TPPrice,
TrailActive,ProofSummary
```

**States CSV** (`backtest.go:577-644`):
```csv
CSVRow,BarIndex,Time,States,RegimeOK,EntryOK,ProofSummary
```

### Performance Reports

**Elite Log** (`main.go:142-164`):
```json
{
  "seed": 12345,
  "entry_rule": "AND(OR(GT(EMA20,X),...),...)",
  "exit_rule": "...",
  "train_score": 1.23,
  "val_score": 0.98,
  "train_return": 0.45,
  "val_return": 0.23,
  "train_max_dd": 0.12,
  "val_max_dd": 0.18,
  "train_win_rate": 0.65,
  "val_win_rate": 0.58,
  "train_trades": 150,
  "val_trades": 80
}
```

## Development

### Thread Safety

- **Atomic operations**: Global counters use `atomic.Int64` / `atomic.Bool`
- **Lock-free HOF reads**: Hall of Fame uses `atomic.Value` for snapshot
- **Mutex protection**: MetaState uses `sync.RWMutex`

### Important Concepts

**Lookahead Bias Prevention**:
- Entry signals at bar t execute at bar t+1 open
- Exit signals at bar t-1 execute at bar t open
- Features use t-1 for execution decisions
- Mathematical proofs stored for verification

**Warmup Periods**:
- 200 bars for EMA stabilization
- Additional warmup from max indicator lookback
- Dynamic warmup computation per strategy

**Slippage Modeling**:
- Base slippage from `-slip_bps` flag
- Volatility-adjusted (×2 for VolZ < -1, ×4 for VolZ < -2)
- Directional (worse fill on exit)

**Fee Calculation**:
- Entry + exit fees: `2 * fee_bps / 10000`
- Applied to PnL: `raw_pnl - fees`

### Dependencies

- **Go**: 1.24.0
- **Bubble Tea**: TUI framework (`github.com/charmbracelet/bubbletea`)
- **golang.org/x/term**: Terminal handling

### Troubleshooting

**Zero OOS Trades**:
- Check `-wf_min_edges_per_year` threshold
- Verify entry rate with `-verbose`
- Use trace mode to diagnose strategy behavior

**High Rejection Rate**:
- Lower `screen_relax_level` via meta.json
- Enable recovery mode: `-recovery`
- Check gate thresholds in startup logs

**Low Trade Generation**:
- Reduce edge constraints: `setEdgeMinMultiplier(1)`
- Disable entry rate gate temporarily (debug mode only)

**Verification Mode**:
- Run `-mode verify` to check bytecode vs AST consistency
- Reports discrepancies in rule evaluation

---

**Version**: 1.6
**License**: See LICENSE file
**Issues**: Report bugs via GitHub issues
