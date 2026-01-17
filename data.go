package main

import (
	"bufio"
	"encoding/csv"
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

		openT, _ := time.Parse(timeLayout, rec[0])
		closeT, _ := time.Parse(timeLayout, rec[6])

		open, _ := strconv.ParseFloat(rec[1], 32)
		high, _ := strconv.ParseFloat(rec[2], 32)
		low, _ := strconv.ParseFloat(rec[3], 32)
		closep, _ := strconv.ParseFloat(rec[4], 32)

		vol, _ := strconv.ParseFloat(rec[5], 32)
		qvol, _ := strconv.ParseFloat(rec[7], 32)

		tr, _ := strconv.ParseInt(rec[8], 10, 32)

		tbb, _ := strconv.ParseFloat(rec[9], 32)
		tbq, _ := strconv.ParseFloat(rec[10], 32)

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
