package main

import (
	"math/rand"
	"sync"
)

type CellKey struct {
	FreqBin int // trade frequency bin
	WRBin   int // winrate bin (proxy for style)
	SkewBin int // expectancy/skew bin (personality)
}

type Archive struct {
	cells map[CellKey]Elite
	keys  []CellKey // for O(1) random sample
	dirty bool
	mu    sync.RWMutex // mutex for thread-safe access
}

func NewArchive() *Archive {
	return &Archive{cells: make(map[CellKey]Elite)}
}

// BehaviorDescriptor represents a 3D behavior descriptor
type BehaviorDescriptor struct {
	FreqBin int // Trade frequency bin
	WRBin   int // Win rate bin
	SkewBin int // Skew/expectancy bin
}

// computeBehaviorDescriptor computes the 3D behavior descriptor for a result
func computeBehaviorDescriptor(r Result) BehaviorDescriptor {
	// Bin 1: Trade frequency (6 bins)
	freqBin := freqBin(r.Trades)

	// Bin 2: Win rate (6 bins)
	wrBin := wrBin(r.WinRate)

	// Bin 3: Skew (using expectancy)
	skewBin := 0
	if r.Expectancy < -0.002 {
		skewBin = 0
	} else if r.Expectancy < -0.001 {
		skewBin = 1
	} else if r.Expectancy < 0.0 {
		skewBin = 2
	} else if r.Expectancy < 0.001 {
		skewBin = 3
	} else if r.Expectancy < 0.002 {
		skewBin = 4
	} else {
		skewBin = 5
	}

	return BehaviorDescriptor{
		FreqBin: freqBin,
		WRBin:   wrBin,
		SkewBin: skewBin,
	}
}

func freqBin(trades int) int {
	// 0..5 bins
	switch {
	case trades < 50:
		return 0
	case trades < 150:
		return 1
	case trades < 300:
		return 2
	case trades < 600:
		return 3
	case trades < 1200:
		return 4
	default:
		return 5
	}
}

func wrBin(wr float32) int {
	// 0..5 bins
	switch {
	case wr < 0.35:
		return 0
	case wr < 0.45:
		return 1
	case wr < 0.52:
		return 2
	case wr < 0.58:
		return 3
	case wr < 0.65:
		return 4
	default:
		return 5
	}
}

func skewBin(expectancy float32) int {
	// 0..5 bins based on expectancy (per-trade edge)
	switch {
	case expectancy < -0.002:
		return 0
	case expectancy < -0.001:
		return 1
	case expectancy < 0.0:
		return 2
	case expectancy < 0.001:
		return 3
	case expectancy < 0.002:
		return 4
	default:
		return 5
	}
}

func (a *Archive) Add(val Result, e Elite) {
	a.mu.Lock()
	defer a.mu.Unlock()

	k := CellKey{FreqBin: freqBin(val.Trades), WRBin: wrBin(val.WinRate), SkewBin: skewBin(val.Expectancy)}
	cur, ok := a.cells[k]
	if !ok || e.ValScore > cur.ValScore {
		a.cells[k] = e
		a.dirty = true
	}
}

func (a *Archive) rebuildKeys() {
	// Internal function - must be called while holding lock (handled by caller)
	a.keys = a.keys[:0]
	for k := range a.cells {
		a.keys = append(a.keys, k)
	}
	a.dirty = false
}

func (a *Archive) Len() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.cells)
}

func (a *Archive) Sample(rng *rand.Rand) (Elite, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.cells) == 0 {
		return Elite{}, false
	}
	if a.dirty || len(a.keys) != len(a.cells) {
		// rebuildKeys is now internal, called while holding lock
		a.keys = a.keys[:0]
		for k := range a.cells {
			a.keys = append(a.keys, k)
		}
		a.dirty = false
	}
	// Uniform sampling across bins for diversity
	k := a.keys[rng.Intn(len(a.keys))]
	return a.cells[k], true
}

// SampleUniform picks a random elite from HOF with probability proportional to bin diversity
// This reduces bias towards high-scoring bins and encourages exploration of all behavioral spaces
func SampleUniformFromHOF(h *HallOfFame, a *Archive, rng *rand.Rand) (Elite, bool) {
	// 50%: sample from Archive (increased from 30% for diversity)
	// 50%: sample from HOF (normal behavior)
	if a.Len() > 0 && rng.Float32() < 0.50 {
		return a.Sample(rng)
	}
	return h.Sample(rng)
}

func (a *Archive) Size() int {
	return a.Len()
}
