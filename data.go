package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
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
	var nextCloseTime int64 // For 6-column format, derive close_time from next row's timestamp
	csvRowIndex := 0        // Track actual CSV row number (1-based, after header)

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
			// For 6-column format, close_time is derived from next row's open_time
			// Store current open_time as the next row's potential close_time
			if nextCloseTime > 0 {
				closeT = time.UnixMilli(nextCloseTime)
			} else {
				// First row - approximate close_time as open_time + 5 minutes
				// This will be corrected if we detect the actual interval later
				closeT = openT.Add(5 * time.Minute)
			}
			nextCloseTime = openT.UnixMilli()
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

	// Fix last row's close_time for 6-column format
	if len(s.CloseTimeMs) > 0 && s.CloseTimeMs[len(s.CloseTimeMs)-1] == s.OpenTimeMs[len(s.OpenTimeMs)-1] {
		// Use the same interval as the previous row
		if len(s.CloseTimeMs) >= 2 {
			interval := s.CloseTimeMs[len(s.CloseTimeMs)-2] - s.OpenTimeMs[len(s.OpenTimeMs)-2]
			s.CloseTimeMs[len(s.CloseTimeMs)-1] = s.OpenTimeMs[len(s.OpenTimeMs)-1] + interval
		}
	}

	s.T = len(s.Close)
	return s, nil
}
