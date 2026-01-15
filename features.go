package main

import (
	"fmt"
	"math"
)

type FeatureStats struct {
	Min  float32
	Max  float32
	Mean float32
	Std  float32
}

type Features struct {
	F     [][]float32
	Names []string
	Index map[string]int
	Stats []FeatureStats // Per-feature statistics for scale-aware mutations
}

func computeAllFeatures(s Series) Features {
	n := s.T
	f := Features{}

	f.Names = make([]string, 0, 60)
	f.F = make([][]float32, 0, 60)
	f.Index = make(map[string]int, 60)

	addFeature := func(name string, arr []float32) {
		f.Index[name] = len(f.F)
		f.F = append(f.F, arr)
		f.Names = append(f.Names, name)
	}

	emaPeriods := []int{10, 20, 50, 100, 200}
	for _, p := range emaPeriods {
		arr := make([]float32, n)
		computeEMA(s.Close, arr, p)
		addFeature(fmt.Sprintf("EMA%d", p), arr)
	}

	rsiPeriods := []int{7, 14, 21}
	for _, p := range rsiPeriods {
		arr := make([]float32, n)
		computeRSI(s.Close, arr, p)
		addFeature(fmt.Sprintf("RSI%d", p), arr)
	}

	atrPeriods := []int{7, 14}
	for _, p := range atrPeriods {
		arr := make([]float32, n)
		computeATR(s.High, s.Low, s.Close, arr, p)
		addFeature(fmt.Sprintf("ATR%d", p), arr)
	}

	for _, p := range []int{20, 50} {
		upper := make([]float32, n)
		lower := make([]float32, n)
		width := make([]float32, n)
		computeBollinger(s.Close, upper, lower, width, p, 2.0)
		addFeature(fmt.Sprintf("BB_Upper%d", p), upper)
		addFeature(fmt.Sprintf("BB_Lower%d", p), lower)
		addFeature(fmt.Sprintf("BB_Width%d", p), width)
	}

	macd := make([]float32, n)
	signal := make([]float32, n)
	hist := make([]float32, n)
	computeMACD(s.Close, macd, signal, hist, 12, 26, 9)
	addFeature("MACD", macd)
	addFeature("MACD_Signal", signal)
	addFeature("MACD_Hist", hist)

	obv := make([]float32, n)
	computeOBV(s.Close, s.Volume, obv)
	addFeature("OBV", obv)

	adxPeriods := []int{14}
	for _, p := range adxPeriods {
		adx, plusDI, minusDI := make([]float32, n), make([]float32, n), make([]float32, n)
		computeADX(s.High, s.Low, s.Close, adx, plusDI, minusDI, p)
		addFeature("ADX", adx)
		addFeature("PlusDI", plusDI)
		addFeature("MinusDI", minusDI)
	}

	mfi := make([]float32, n)
	computeMFI(s.High, s.Low, s.Close, s.Volume, mfi, 14)
	addFeature("MFI14", mfi)

	for _, p := range []int{10, 20} {
		roc := make([]float32, n)
		computeROC(s.Close, roc, p)
		addFeature(fmt.Sprintf("ROC%d", p), roc)
	}

	for _, p := range []int{20, 50} {
		volSMA := make([]float32, n)
		volEMA := make([]float32, n)
		volZ := make([]float32, n)
		computeSMA(s.Volume, volSMA, p)
		computeEMA(s.Volume, volEMA, p)
		computeZScore(s.Volume, volSMA, volZ, p)
		addFeature(fmt.Sprintf("VolSMA%d", p), volSMA)
		addFeature(fmt.Sprintf("VolEMA%d", p), volEMA)
		addFeature(fmt.Sprintf("VolZ%d", p), volZ)
	}

	buyRatio := make([]float32, n)
	imbalance := make([]float32, n)
	volPerTrade := make([]float32, n)
	active := make([]float32, n)

	for i := 0; i < n; i++ {
		if s.Volume[i] > 0 {
			buyRatio[i] = s.TakerBuyBase[i] / s.Volume[i]
			sellBase := s.Volume[i] - s.TakerBuyBase[i]
			imbalance[i] = (s.TakerBuyBase[i] - sellBase) / s.Volume[i]
		}
		if s.Trades[i] > 0 {
			volPerTrade[i] = s.Volume[i] / float32(s.Trades[i])
			active[i] = 1
		}
	}

	addFeature("BuyRatio", buyRatio)
	addFeature("Imbalance", imbalance)
	addFeature("VolPerTrade", volPerTrade)
	addFeature("Active", active)

	highLowDiff := make([]float32, n)
	body := make([]float32, n)
	rangeWidth := make([]float32, n)
	for i := 0; i < n; i++ {
		highLowDiff[i] = s.High[i] - s.Low[i]
		body[i] = float32(math.Abs(float64(s.Close[i] - s.Open[i])))
		if highLowDiff[i] > 0 {
			rangeWidth[i] = body[i] / highLowDiff[i]
		}
	}
	addFeature("HighLowDiff", highLowDiff)
	addFeature("Body", body)
	addFeature("RangeWidth", rangeWidth)

	swingHigh := make([]float32, n)
	swingLow := make([]float32, n)
	sweep := make([]float32, n)
	bos := make([]float32, n)
	displacement := make([]float32, n)
	fvgUp := make([]float32, n)
	fvgDown := make([]float32, n)

	leftLook := 3
	rightLook := 3
	for i := leftLook; i < n-rightLook; i++ {
		isSwingHigh := true
		isSwingLow := true
		for j := i - leftLook; j <= i+rightLook; j++ {
			if j != i {
				if s.High[j] >= s.High[i] {
					isSwingHigh = false
				}
				if s.Low[j] <= s.Low[i] {
					isSwingLow = false
				}
			}
		}
		if isSwingHigh {
			swingHigh[i] = s.High[i]
		}
		if isSwingLow {
			swingLow[i] = s.Low[i]
		}
	}

	for i := 4; i < n; i++ {
		if swingHigh[i-4] > 0 {
			swingHighPrice := swingHigh[i-4]
			if s.Low[i] < swingHighPrice {
				if s.Close[i] > swingHighPrice {
					sweep[i] = 1
				}
			}
		}
	}

	lastSwingHigh := float32(-1)
	for i := 0; i < n; i++ {
		if swingHigh[i] > 0 {
			lastSwingHigh = swingHigh[i]
		}
		if lastSwingHigh > 0 && s.Close[i] > lastSwingHigh {
			bos[i] = 1
		}
	}

	for i := 5; i < n; i++ {
		recentRange := s.High[i-5 : i+1]
		minR := recentRange[0]
		maxR := recentRange[0]
		for _, v := range recentRange {
			if v < minR {
				minR = v
			}
			if v > maxR {
				maxR = v
			}
		}
		if maxR-minR > 0 {
			displacement[i] = (s.Close[i] - minR) / (maxR - minR)
		}
	}

	for i := 2; i < n; i++ {
		if s.Low[i-1] > s.High[i] {
			fvgUp[i] = s.Low[i-1] - s.High[i]
		}
		if s.High[i-1] < s.Low[i] {
			fvgDown[i] = s.Low[i] - s.High[i-1]
		}
	}

	addFeature("SwingHigh", swingHigh)
	addFeature("SwingLow", swingLow)
	addFeature("Sweep", sweep)
	addFeature("BOS", bos)
	addFeature("Displacement", displacement)
	addFeature("FVGUp", fvgUp)
	addFeature("FVGDown", fvgDown)

	// Compute statistics for scale-aware mutations (will be recomputed on train window in main)
	f.Stats = make([]FeatureStats, len(f.F))

	return f
}

// computeStatsOnWindow computes feature statistics only on a specific window (train)
func computeStatsOnWindow(f *Features, startIdx, endIdx int) {
	for i := range f.F {
		arr := f.F[i]
		if len(arr) == 0 || endIdx <= startIdx {
			continue
		}

		// Clamp indices to array bounds
		start := startIdx
		end := endIdx
		if start < 0 {
			start = 0
		}
		if end > len(arr) {
			end = len(arr)
		}
		if start >= end {
			continue
		}

		// Compute min, max, mean on window only
		min := arr[start]
		max := arr[start]
		sum := float32(0)
		count := int32(0)
		for j := start; j < end; j++ {
			v := arr[j]
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
			count++
		}
		mean := sum / float32(count)

		// Compute standard deviation on window only
		variance := float32(0)
		for j := start; j < end; j++ {
			diff := arr[j] - mean
			variance += diff * diff
		}
		variance /= float32(count)
		std := float32(math.Sqrt(float64(variance)))

		f.Stats[i] = FeatureStats{
			Min:  min,
			Max:  max,
			Mean: mean,
			Std:  std,
		}
	}
}

func computeSMA(src, dst []float32, period int) {
	if len(src) < period {
		return
	}
	sum := float32(0.0)
	for i := 0; i < len(src); i++ {
		sum += src[i]
		if i >= period {
			sum -= src[i-period]
		}
		if i >= period-1 {
			dst[i] = sum / float32(period)
		}
	}
}

func computeEMA(src, dst []float32, period int) {
	if len(src) < 1 {
		return
	}
	multiplier := float32(2.0 / float32(period+1))
	dst[0] = src[0]
	for i := 1; i < len(src); i++ {
		dst[i] = (src[i]-dst[i-1])*multiplier + dst[i-1]
	}
}

func computeRSI(src, dst []float32, period int) {
	if len(src) < period+1 {
		return
	}
	gain := make([]float32, len(src))
	loss := make([]float32, len(src))

	for i := 1; i < len(src); i++ {
		change := src[i] - src[i-1]
		if change > 0 {
			gain[i] = change
			loss[i] = 0
		} else {
			gain[i] = 0
			loss[i] = -change
		}
	}

	avgGain := float32(0)
	avgLoss := float32(0)

	for i := 1; i <= period; i++ {
		avgGain += gain[i]
		avgLoss += loss[i]
	}
	avgGain /= float32(period)
	avgLoss /= float32(period)

	if avgLoss == 0 {
		dst[period] = 100
	} else {
		dst[period] = 100 - (100 / (1 + avgGain/avgLoss))
	}

	for i := period + 1; i < len(src); i++ {
		avgGain = (avgGain*float32(period-1) + gain[i]) / float32(period)
		avgLoss = (avgLoss*float32(period-1) + loss[i]) / float32(period)
		if avgLoss == 0 {
			dst[i] = 100
		} else {
			dst[i] = 100 - (100 / (1 + avgGain/avgLoss))
		}
	}
}

func computeATR(high, low, close, dst []float32, period int) {
	if len(high) < period+1 {
		return
	}
	tr := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		hl := high[i] - low[i]
		hc := float32(math.Abs(float64(high[i] - close[i-1])))
		lc := float32(math.Abs(float64(low[i] - close[i-1])))
		tr[i] = maxFloat32(hl, maxFloat32(hc, lc))
	}

	sum := float32(0)
	for i := 1; i <= period; i++ {
		sum += tr[i]
	}
	dst[period] = sum / float32(period)

	for i := period + 1; i < len(high); i++ {
		dst[i] = (dst[i-1]*float32(period-1) + tr[i]) / float32(period)
	}
}

func computeBollinger(src, upper, lower, width []float32, period int, stdDev float64) {
	sma := make([]float32, len(src))
	computeSMA(src, sma, period)

	for i := period - 1; i < len(src); i++ {
		sumSq := float32(0)
		for j := i - period + 1; j <= i; j++ {
			diff := src[j] - sma[i]
			sumSq += diff * diff
		}
		std := float32(math.Sqrt(float64(sumSq / float32(period))))
		upper[i] = sma[i] + float32(stdDev)*std
		lower[i] = sma[i] - float32(stdDev)*std
		if sma[i] > 0 {
			width[i] = (upper[i] - lower[i]) / sma[i]
		}
	}
}

func computeMACD(src, macd, signal, hist []float32, fastPeriod, slowPeriod, signalPeriod int) {
	fastEMA := make([]float32, len(src))
	slowEMA := make([]float32, len(src))
	computeEMA(src, fastEMA, fastPeriod)
	computeEMA(src, slowEMA, slowPeriod)

	for i := 0; i < len(src); i++ {
		macd[i] = fastEMA[i] - slowEMA[i]
	}
	computeEMA(macd, signal, signalPeriod)

	for i := 0; i < len(src); i++ {
		hist[i] = macd[i] - signal[i]
	}
}

func computeOBV(close, volume, dst []float32) {
	if len(close) < 1 {
		return
	}
	dst[0] = volume[0]
	for i := 1; i < len(close); i++ {
		if close[i] > close[i-1] {
			dst[i] = dst[i-1] + volume[i]
		} else if close[i] < close[i-1] {
			dst[i] = dst[i-1] - volume[i]
		} else {
			dst[i] = dst[i-1]
		}
	}
}

func computeADX(high, low, close, adx, plusDI, minusDI []float32, period int) {
	if len(high) < period*2+1 {
		return
	}

	tr := make([]float32, len(high))
	plusDM := make([]float32, len(high))
	minusDM := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		hl := high[i] - low[i]
		hc := float32(math.Abs(float64(high[i] - close[i-1])))
		lc := float32(math.Abs(float64(low[i] - close[i-1])))
		tr[i] = maxFloat32(hl, maxFloat32(hc, lc))

		upMove := high[i] - high[i-1]
		downMove := low[i-1] - low[i]
		if upMove > downMove && upMove > 0 {
			plusDM[i] = upMove
		} else {
			plusDM[i] = 0
		}
		if downMove > upMove && downMove > 0 {
			minusDM[i] = downMove
		} else {
			minusDM[i] = 0
		}
	}

	smoothedTR := make([]float32, len(high))
	smoothedPlusDM := make([]float32, len(high))
	smoothedMinusDM := make([]float32, len(high))

	for i := 1; i <= period; i++ {
		smoothedTR[period] += tr[i]
		smoothedPlusDM[period] += plusDM[i]
		smoothedMinusDM[period] += minusDM[i]
	}

	for i := period + 1; i < len(high); i++ {
		smoothedTR[i] = smoothedTR[i-1] - smoothedTR[i-1]/float32(period) + tr[i]
		smoothedPlusDM[i] = smoothedPlusDM[i-1] - smoothedPlusDM[i-1]/float32(period) + plusDM[i]
		smoothedMinusDM[i] = smoothedMinusDM[i-1] - smoothedMinusDM[i-1]/float32(period) + minusDM[i]
	}

	for i := period; i < len(high); i++ {
		if smoothedTR[i] > 0 {
			plusDI[i] = 100 * smoothedPlusDM[i] / smoothedTR[i]
			minusDI[i] = 100 * smoothedMinusDM[i] / smoothedTR[i]
		}
	}

	dx := make([]float32, len(high))
	for i := period; i < len(high); i++ {
		sum := plusDI[i] + minusDI[i]
		if sum > 0 {
			dx[i] = 100 * float32(math.Abs(float64(plusDI[i]-minusDI[i]))) / sum
		}
	}

	computeEMA(dx, adx, period)
}

func computeMFI(high, low, close, volume, dst []float32, period int) {
	if len(high) < period+1 {
		return
	}

	typicalPrice := make([]float32, len(high))
	rawMF := make([]float32, len(high))

	for i := 0; i < len(high); i++ {
		typicalPrice[i] = (high[i] + low[i] + close[i]) / 3
		rawMF[i] = typicalPrice[i] * volume[i]
	}

	posMF := make([]float32, len(high))
	negMF := make([]float32, len(high))

	for i := 1; i < len(high); i++ {
		if typicalPrice[i] > typicalPrice[i-1] {
			posMF[i] = rawMF[i]
		} else {
			negMF[i] = rawMF[i]
		}
	}

	sumPos := float32(0)
	sumNeg := float32(0)

	for i := 1; i <= period; i++ {
		sumPos += posMF[i]
		sumNeg += negMF[i]
	}

	if sumNeg == 0 {
		dst[period] = 100
	} else {
		dst[period] = 100 - (100 / (1 + sumPos/sumNeg))
	}

	for i := period + 1; i < len(high); i++ {
		sumPos = sumPos - posMF[i-period] + posMF[i]
		sumNeg = sumNeg - negMF[i-period] + negMF[i]

		if sumNeg == 0 {
			dst[i] = 100
		} else {
			dst[i] = 100 - (100 / (1 + sumPos/sumNeg))
		}
	}
}

func computeROC(src, dst []float32, period int) {
	if len(src) < period {
		return
	}
	for i := period; i < len(src); i++ {
		if src[i-period] != 0 {
			dst[i] = 100 * (src[i] - src[i-period]) / src[i-period]
		}
	}
}

func computeZScore(src, mean, dst []float32, period int) {
	if len(src) < period {
		return
	}
	for i := period - 1; i < len(src); i++ {
		sum := float32(0)
		sumSq := float32(0)
		for j := i - period + 1; j <= i; j++ {
			sum += src[j]
			sumSq += src[j] * src[j]
		}
		meanVal := sum / float32(period)
		variance := (sumSq / float32(period)) - (meanVal * meanVal)
		if variance > 0 {
			std := float32(math.Sqrt(float64(variance)))
			if std > 0 {
				dst[i] = (src[i] - meanVal) / std
			}
		}
	}
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
