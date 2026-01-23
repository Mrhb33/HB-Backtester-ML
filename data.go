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

		// Validate record length
		if len(rec) < 11 {
			fmt.Printf("%s  Warning: skipping row %d with insufficient columns (got %d, expected 11)\n", logx.Channel("VAL "), csvRowIndex, len(rec))
			continue
		}

		// Parse with error checking - skip invalid rows
		openT, err := time.Parse(timeLayout, rec[0])
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid open time: %s\n", logx.Channel("VAL "), csvRowIndex, rec[0])
			continue
		}

		closeT, err := time.Parse(timeLayout, rec[6])
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid close time: %s\n", logx.Channel("VAL "), csvRowIndex, rec[6])
			continue
		}

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

		qvol, err := strconv.ParseFloat(rec[7], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid quote volume: %s\n", logx.Channel("VAL "), csvRowIndex, rec[7])
			continue
		}

		tr, err := strconv.ParseInt(rec[8], 10, 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid trade count: %s\n", logx.Channel("VAL "), csvRowIndex, rec[8])
			continue
		}

		tbb, err := strconv.ParseFloat(rec[9], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid taker buy base: %s\n", logx.Channel("VAL "), csvRowIndex, rec[9])
			continue
		}

		tbq, err := strconv.ParseFloat(rec[10], 32)
		if err != nil {
			fmt.Printf("%s  Warning: skipping row %d with invalid taker buy quote: %s\n", logx.Channel("VAL "), csvRowIndex, rec[10])
			continue
		}

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

	s.T = len(s.Close)
	return s, nil
}
