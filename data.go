package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

const timeLayout = "2006-01-02 15:04:05-07:00"

type Series struct {
	T int

	OpenTimeMs, CloseTimeMs []int64

	Open, High, Low, Close []float32

	Volume, QuoteVolume, TakerBuyBase, TakerBuyQuote []float32
	Trades                                           []int32
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
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Series{}, err
		}

		// Validate record length
		if len(rec) < 11 {
			fmt.Printf("Warning: skipping row with insufficient columns (got %d, expected 11)\n", len(rec))
			continue
		}

		// Parse with error checking - skip invalid rows
		openT, err := time.Parse(timeLayout, rec[0])
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid open time: %s\n", rec[0])
			continue
		}

		closeT, err := time.Parse(timeLayout, rec[6])
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid close time: %s\n", rec[6])
			continue
		}

		open, err := strconv.ParseFloat(rec[1], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid open price: %s\n", rec[1])
			continue
		}

		high, err := strconv.ParseFloat(rec[2], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid high price: %s\n", rec[2])
			continue
		}

		low, err := strconv.ParseFloat(rec[3], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid low price: %s\n", rec[3])
			continue
		}

		closep, err := strconv.ParseFloat(rec[4], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid close price: %s\n", rec[4])
			continue
		}

		vol, err := strconv.ParseFloat(rec[5], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid volume: %s\n", rec[5])
			continue
		}

		qvol, err := strconv.ParseFloat(rec[7], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid quote volume: %s\n", rec[7])
			continue
		}

		tr, err := strconv.ParseInt(rec[8], 10, 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid trade count: %s\n", rec[8])
			continue
		}

		tbb, err := strconv.ParseFloat(rec[9], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid taker buy base: %s\n", rec[9])
			continue
		}

		tbq, err := strconv.ParseFloat(rec[10], 32)
		if err != nil {
			fmt.Printf("Warning: skipping row with invalid taker buy quote: %s\n", rec[10])
			continue
		}

		// Validate price data is reasonable
		if open <= 0 || high <= 0 || low <= 0 || closep <= 0 {
			fmt.Printf("Warning: skipping row with non-positive prices (open=%f, high=%f, low=%f, close=%f)\n", open, high, low, closep)
			continue
		}
		if high < low || high < open || high < closep || low > open || low > closep {
			fmt.Printf("Warning: skipping row with invalid price relationships (open=%f, high=%f, low=%f, close=%f)\n", open, high, low, closep)
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
	}

	s.T = len(s.Close)
	return s, nil
}
