package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

type Surrogate struct {
	mu       sync.Mutex
	w        []float64
	lr       float64
	l2       float64
	exploreP float64 // epsilon exploration
	rng      *rand.Rand
}

func NewSurrogate(dim int) *Surrogate {
	return &Surrogate{
		w:        make([]float64, dim),
		lr:       0.02,
		l2:       1e-4,
		exploreP: 0.10,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// NewSurrogateWithSeed creates a surrogate with a specific seed for reproducible runs
func NewSurrogateWithSeed(dim int, seed int64) *Surrogate {
	return &Surrogate{
		w:        make([]float64, dim),
		lr:       0.02,
		l2:       1e-4,
		exploreP: 0.10,
		rng:      rand.New(rand.NewSource(seed)),
	}
}

func (m *Surrogate) Score(x []float64) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	var s float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		s += m.w[i] * x[i]
	}
	return s
}

// scoreNoLock computes score without locking (must be called with lock already held)
func (m *Surrogate) scoreNoLock(x []float64) float64 {
	var s float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		s += m.w[i] * x[i]
	}
	return s
}

func (m *Surrogate) Accept(x []float64, threshold float64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	// epsilon-greedy exploration so we don't get stuck
	if m.rng.Float64() < m.exploreP {
		return true
	}
	return m.scoreNoLock(x) >= threshold
}

// Train toward target y (use valScore or passed flag)
func (m *Surrogate) Update(x []float64, y float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// prediction
	var pred float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		pred += m.w[i] * x[i]
	}

	// Huber-like clamp to avoid huge gradients
	err := pred - y
	if err > 5 {
		err = 5
	}
	if err < -5 {
		err = -5
	}

	for i := 0; i < len(m.w) && i < len(x); i++ {
		grad := err*x[i] + m.l2*m.w[i]
		m.w[i] -= m.lr * grad
	}
}

// Save surrogate weights to file
func (m *Surrogate) Save(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, err := json.Marshal(m.w)
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// Load surrogate weights from file
func (m *Surrogate) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var w []float64
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Resize weight vector if needed
	if len(w) != len(m.w) {
		m.w = make([]float64, len(w))
	}
	copy(m.w, w)

	return nil
}

// -------- Feature extraction (fixed-size vector) --------

// Keep it simple and stable.
// Dimension = 64 is fine to start.
const SurDim = 64

func ExtractSurFeatures(s Strategy) []float64 {
	x := make([]float64, SurDim)

	// risk params (normalize gently)
	x[0] = float64(s.RiskPct)
	x[1] = float64(s.StopLoss.ATRMult)
	if s.StopLoss.Kind == "fixed" {
		x[1] = float64(s.StopLoss.Value) / 100.0
	}
	x[2] = float64(s.TakeProfit.ATRMult)
	if s.TakeProfit.Kind == "fixed" {
		x[2] = float64(s.TakeProfit.Value) / 100.0
	}
	x[3] = boolToFloat(s.Trail.Active)

	// rule tree stats (entry rule)
	nodes, leaves, ands, ors, nots := countTree(s.EntryRule.Root)
	x[10] = float64(nodes) / 32.0
	x[11] = float64(leaves) / 32.0
	x[12] = float64(ands) / 16.0
	x[13] = float64(ors) / 16.0
	x[14] = float64(nots) / 16.0

	// rule tree stats (exit rule)
	nodes2, leaves2, ands2, ors2, nots2 := countTree(s.ExitRule.Root)
	x[15] = float64(nodes2) / 32.0
	x[16] = float64(leaves2) / 32.0
	x[17] = float64(ands2) / 16.0
	x[18] = float64(ors2) / 16.0
	x[19] = float64(nots2) / 16.0

	// leaf kind histogram (first 20 kinds)
	kindCounts := make([]int, 32)
	collectLeafKinds(s.EntryRule.Root, kindCounts)
	for k := 0; k < 20; k++ {
		x[20+k] = float64(kindCounts[k]) / 10.0
	}

	// exit rule leaf kind histogram
	kindCounts2 := make([]int, 32)
	collectLeafKinds(s.ExitRule.Root, kindCounts2)
	for k := 0; k < 10; k++ {
		x[40+k] = float64(kindCounts2[k]) / 10.0
	}

	// bias term
	x[63] = 1.0

	// clamp to avoid numeric weirdness
	for i := range x {
		if math.IsNaN(x[i]) || math.IsInf(x[i], 0) {
			x[i] = 0
		}
		if x[i] > 10 {
			x[i] = 10
		}
		if x[i] < -10 {
			x[i] = -10
		}
	}
	return x
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func countTree(n *RuleNode) (nodes, leaves, ands, ors, nots int) {
	if n == nil {
		return
	}
	nodes = 1
	if n.Op == OpLeaf {
		leaves = 1
		return
	}
	if n.Op == OpAnd {
		ands = 1
	}
	if n.Op == OpOr {
		ors = 1
	}
	if n.Op == OpNot {
		nots = 1
	}
	a1, b1, c1, d1, e1 := countTree(n.L)
	a2, b2, c2, d2, e2 := countTree(n.R)
	return nodes + a1 + a2, leaves + b1 + b2, ands + c1 + c2, ors + d1 + d2, nots + e1 + e2
}

func collectLeafKinds(n *RuleNode, kc []int) {
	if n == nil {
		return
	}
	if n.Op == OpLeaf {
		k := int(n.Leaf.Kind)
		if k >= 0 && k < len(kc) {
			kc[k]++
		}
		return
	}
	collectLeafKinds(n.L, kc)
	collectLeafKinds(n.R, kc)
}
