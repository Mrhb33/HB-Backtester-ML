package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"sync/atomic"
	"time"

	"hb_bactest_checker/logx"
)

const timeLayout = "2006-01-02 15:04:05-07:00"

type Series struct {
	T int

	OpenTimeMs, CloseTimeMs []int64

	Open, High, Low, Close []float32

	Volume, QuoteVolume, TakerBuyBase, TakerBuyQuote []float32
	Trades                                           []int32

	// CSVRowIndex maps Series[i] to original CSV row number (1-based)
	// This is critical because rows with invalid prices are skipped during loading,
	// causing the internal Series index to NOT match the original CSV row number.
	CSVRowIndex []int
}

func LoadBinanceKlinesCSV(path string) (Series, error) {
	f, err := os.Open(path)
	if err != nil {
		return Series{}, err
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReaderSize(f, 1<<20))
	r.ReuseRecord = true

	_, err = r.Read()
	if err != nil {
		return Series{}, err
	}

	var s Series
	csvRowIndex := 0 // Track actual CSV row number (1-based, after header)

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Series{}, err
		}

		csvRowIndex++ // ALWAYS increment - even for skipped rows

		// Validate record length - support both 6-column (simple) and 11-column (Binance) formats
		if len(rec) != 6 && len(rec) != 11 {
			fmt.Printf("%s  Warning: skipping row %d with unsupported column count (got %d, expected 6 or 11)\n", logx.Channel("VAL "), csvRowIndex, len(rec))
			continue
		}

		isSimpleFormat := len(rec) == 6

		// Parse open time - column 0 for both formats (named "timestamp" in simple format)
		openT, err := time.Parse(timeLayout, rec[0])
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid open time: %s\n", logx.Channel("VAL "), csvRowIndex, rec[0])
			continue
		}

		var closeT time.Time
		if isSimpleFormat {
			// For 6-column format, use temporary close time
			// Will be recalculated correctly after loading based on detected interval
			closeT = openT.Add(60 * time.Minute) // Placeholder
		} else {
			// For 11-column format, close_time is in column 6
			closeT, err = time.Parse(timeLayout, rec[6])
			if err != nil {
				fmt.Printf("%s  Warning: skipping row %d with invalid close time: %s\n", logx.Channel("VAL "), csvRowIndex, rec[6])
				continue
			}
		}

		// Parse OHLCV - columns 1-5 for both formats
		open, err := strconv.ParseFloat(rec[1], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid open price: %s\n", logx.Channel("VAL "), csvRowIndex, rec[1])
			continue
		}

		high, err := strconv.ParseFloat(rec[2], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid high price: %s\n", logx.Channel("VAL "), csvRowIndex, rec[2])
			continue
		}

		low, err := strconv.ParseFloat(rec[3], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid low price: %s\n", logx.Channel("VAL "), csvRowIndex, rec[3])
			continue
		}

		closep, err := strconv.ParseFloat(rec[4], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid close price: %s\n", logx.Channel("VAL "), csvRowIndex, rec[4])
			continue
		}

		vol, err := strconv.ParseFloat(rec[5], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid volume: %s\n", logx.Channel("VAL "), csvRowIndex, rec[5])
			continue
		}

		// For 11-column format, parse additional fields
		var qvol, tbb, tbq float64
		var tr int64
		if !isSimpleFormat {
			qvol, err = strconv.ParseFloat(rec[7], 32)
			if err != nil {
				fmt.Printf("%s  Warning: skipping row %d with invalid quote volume: %s\n", logx.Channel("VAL "), csvRowIndex, rec[7])
				continue
			}

			tr, err = strconv.ParseInt(rec[8], 10, 32)
			if err != nil {
				fmt.Printf("%s  Warning: skipping row %d with invalid trade count: %s\n", logx.Channel("VAL "), csvRowIndex, rec[8])
				continue
			}

			tbb, err = strconv.ParseFloat(rec[9], 32)
			if err != nil {
				fmt.Printf("%s  Warning: skipping row %d with invalid taker buy base: %s\n", logx.Channel("VAL "), csvRowIndex, rec[9])
				continue
			}

			tbq, err = strconv.ParseFloat(rec[10], 32)
			if err != nil {
				fmt.Printf("%s  Warning: skipping row %d with invalid taker buy quote: %s\n", logx.Channel("VAL "), csvRowIndex, rec[10])
				continue
			}
		}
		// For 6-column format, these fields default to 0

		// Validate price data is reasonable
		if open <= 0 || high <= 0 || low <= 0 || closep <= 0 {
			fmt.Printf("%s  Warning: skipping row %d with non-positive prices (open=%f, high=%f, low=%f, close=%f)\n", logx.Channel("VAL "), csvRowIndex, open, high, low, closep)
			continue
		}
		if high < low || high < open || high < closep || low > open || low > closep {
			fmt.Printf("%s  Warning: skipping row %d with invalid price relationships (open=%f, high=%f, low=%f, close=%f)\n", logx.Channel("VAL "), csvRowIndex, open, high, low, closep)
			continue
		}

		s.OpenTimeMs = append(s.OpenTimeMs, openT.UnixMilli())
		s.CloseTimeMs = append(s.CloseTimeMs, closeT.UnixMilli())

		s.Open = append(s.Open, float32(open))
		s.High = append(s.High, float32(high))
		s.Low = append(s.Low, float32(low))
		s.Close = append(s.Close, float32(closep))

		s.Volume = append(s.Volume, float32(vol))
		s.QuoteVolume = append(s.QuoteVolume, float32(qvol))

		s.Trades = append(s.Trades, int32(tr))
		s.TakerBuyBase = append(s.TakerBuyBase, float32(tbb))
		s.TakerBuyQuote = append(s.TakerBuyQuote, float32(tbq))

		// CRITICAL: Store the ORIGINAL CSV row number for index mapping
		s.CSVRowIndex = append(s.CSVRowIndex, csvRowIndex)
	}

	// === FIX CLOSE TIMES FOR 6-COLUMN FORMAT ===
	// Detect the median interval from open times and recalculate close times
	if len(s.OpenTimeMs) > 1 {
		// Collect intervals between consecutive open times
		intervals := make([]int64, 0, len(s.OpenTimeMs)-1)
		for i := 1; i < len(s.OpenTimeMs); i++ {
			delta := s.OpenTimeMs[i] - s.OpenTimeMs[i-1]
			if delta > 0 {
				intervals = append(intervals, delta)
			}
		}

		if len(intervals) > 0 {
			// Use median interval to avoid outliers
			sort.Slice(intervals, func(i, j int) bool { return intervals[i] < intervals[j] })
			medianInterval := intervals[len(intervals)/2]

			// Recalculate all CloseTimeMs as open_time + median_interval
			for i := 0; i < len(s.CloseTimeMs); i++ {
				s.CloseTimeMs[i] = s.OpenTimeMs[i] + medianInterval
			}

			// Log detected interval
			intervalMinutes := medianInterval / (60 * 1000)
			fmt.Printf("[DATA] Detected %d-minute bar interval from %d samples\n", intervalMinutes, len(intervals))

			// FIX PROBLEM B: Update globalTimeframeMinutes from detected interval
			// This ensures trades/year calculations use the correct timeframe
			atomic.StoreInt32(&globalTimeframeMinutes, int32(intervalMinutes))
			fmt.Printf("[DATA] Updated global timeframe to %dm (was using command-line flag)\n", intervalMinutes)
		}
	}
	// ==========================================

	s.T = len(s.Close)

	// === TIMESTAMP ORDERING FIX ===
	// Step 1: Reverse if fully descending (newest first common case)
	if len(s.CloseTimeMs) > 1 && s.CloseTimeMs[0] > s.CloseTimeMs[len(s.CloseTimeMs)-1] {
		reverseInt64(s.OpenTimeMs)
		reverseInt64(s.CloseTimeMs)
		reverseFloat32(s.Open)
		reverseFloat32(s.High)
		reverseFloat32(s.Low)
		reverseFloat32(s.Close)
		reverseFloat32(s.Volume)
		reverseFloat32(s.QuoteVolume)
		reverseFloat32(s.TakerBuyBase)
		reverseFloat32(s.TakerBuyQuote)
		reverseInt32(s.Trades)
		reverseInt(s.CSVRowIndex)
		fmt.Printf("[DATA] Reversed %d bars - timestamps were descending\n", len(s.CloseTimeMs))
	}

	// Step 2: Remove duplicates (keep LAST occurrence for same CloseTimeMs)
	s = deduplicateByCloseTime(s)

	// Step 3: Final validation - fail with detailed error if still invalid
	// Uses validateTimestampOrderingDetailed() defined in walkforward.go
	valid, badIdx, prevTs, curTs := validateTimestampOrderingDetailed(s)
	if !valid {
		csvRow := "unknown"
		if badIdx >= 0 && badIdx < len(s.CSVRowIndex) {
			csvRow = fmt.Sprintf("%d", s.CSVRowIndex[badIdx])
		}
		return Series{}, fmt.Errorf("timestamp ordering failed at index %d (CSV row %s): %d < %d - data has corrupted/out-of-order blocks",
			badIdx, csvRow, curTs, prevTs)
	}

	// Step 2A: Verify timestamps are in milliseconds (not seconds)
	const minMillisecondTs = 1000000000000 // Sep 2001 in milliseconds
	if len(s.CloseTimeMs) > 0 && s.CloseTimeMs[0] < minMillisecondTs {
		firstDate := time.UnixMilli(s.CloseTimeMs[0]).UTC().Format("2006-01-02")
		lastDate := time.UnixMilli(s.CloseTimeMs[len(s.CloseTimeMs)-1]).UTC().Format("2006-01-02")
		panic(fmt.Sprintf("CloseTimeMs looks like seconds, not milliseconds. First TS=%d (%s), Last TS=%d (%s). Convert to ms in CSV loader.",
			s.CloseTimeMs[0], firstDate, s.CloseTimeMs[len(s.CloseTimeMs)-1], lastDate))
	}
	// ============================

	return s, nil
}

func reverseInt64(slice []int64) {
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
}

func reverseFloat32(slice []float32) {
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
}

func reverseInt32(slice []int32) {
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
}

func reverseInt(slice []int) {
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
}

// deduplicateByCloseTime removes duplicate rows with same CloseTimeMs (keeps LAST)
func deduplicateByCloseTime(s Series) Series {
	if len(s.CloseTimeMs) <= 1 {
		return s
	}

	// First pass: find last occurrence index for each timestamp
	lastSeen := make(map[int64]int)
	for i := 0; i < len(s.CloseTimeMs); i++ {
		lastSeen[s.CloseTimeMs[i]] = i
	}

	// Second pass: keep row only if it's the last occurrence
	filtered := Series{
		OpenTimeMs:    make([]int64, 0, len(lastSeen)),
		CloseTimeMs:   make([]int64, 0, len(lastSeen)),
		Open:          make([]float32, 0, len(lastSeen)),
		High:          make([]float32, 0, len(lastSeen)),
		Low:           make([]float32, 0, len(lastSeen)),
		Close:         make([]float32, 0, len(lastSeen)),
		Volume:        make([]float32, 0, len(lastSeen)),
		QuoteVolume:   make([]float32, 0, len(lastSeen)),
		TakerBuyBase:  make([]float32, 0, len(lastSeen)),
		TakerBuyQuote: make([]float32, 0, len(lastSeen)),
		Trades:        make([]int32, 0, len(lastSeen)),
		CSVRowIndex:   make([]int, 0, len(lastSeen)),
	}

	for i := 0; i < len(s.CloseTimeMs); i++ {
		ts := s.CloseTimeMs[i]
		if lastSeen[ts] == i {
			filtered.OpenTimeMs = append(filtered.OpenTimeMs, s.OpenTimeMs[i])
			filtered.CloseTimeMs = append(filtered.CloseTimeMs, s.CloseTimeMs[i])
			filtered.Open = append(filtered.Open, s.Open[i])
			filtered.High = append(filtered.High, s.High[i])
			filtered.Low = append(filtered.Low, s.Low[i])
			filtered.Close = append(filtered.Close, s.Close[i])
			filtered.Volume = append(filtered.Volume, s.Volume[i])
			filtered.QuoteVolume = append(filtered.QuoteVolume, s.QuoteVolume[i])
			filtered.TakerBuyBase = append(filtered.TakerBuyBase, s.TakerBuyBase[i])
			filtered.TakerBuyQuote = append(filtered.TakerBuyQuote, s.TakerBuyQuote[i])
			filtered.Trades = append(filtered.Trades, s.Trades[i])
			filtered.CSVRowIndex = append(filtered.CSVRowIndex, s.CSVRowIndex[i])
		}
	}

	filtered.T = len(filtered.CloseTimeMs)
	removed := len(s.CloseTimeMs) - len(filtered.CloseTimeMs)
	if removed > 0 {
		fmt.Printf("[DATA] Removed %d duplicate timestamp rows (kept last occurrence)\n", removed)
	}
	return filtered
}
