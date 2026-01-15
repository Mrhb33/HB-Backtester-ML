package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

// SurrogateV2 is the enhanced surrogate with behavior descriptors
type SurrogateV2 struct {
	mu       sync.Mutex
	w        []float64
	lr       float64
	l2       float64
	exploreP float64 // epsilon exploration
	rng      *rand.Rand

	// Rank-based acceptance parameters
	sampleWindow     int // Number of recent validation results to track
	recentScores     []float64
	acceptPercentile float64 // Accept if predicted score in top percentile
}

// NewSurrogateV2 creates enhanced surrogate with behavior descriptors
func NewSurrogateV2(dim int) *SurrogateV2 {
	return &SurrogateV2{
		w:                make([]float64, dim),
		lr:               0.02,
		l2:               1e-4,
		exploreP:         0.10,
		rng:              rand.New(rand.NewSource(time.Now().UnixNano())),
		sampleWindow:     500,
		recentScores:     make([]float64, 0, 500),
		acceptPercentile: 0.70, // Accept top 30% by default
	}
}

// NewSurrogateV2WithSeed creates surrogate with specific seed
func NewSurrogateV2WithSeed(dim int, seed int64) *SurrogateV2 {
	return &SurrogateV2{
		w:                make([]float64, dim),
		lr:               0.02,
		l2:               1e-4,
		exploreP:         0.10,
		rng:              rand.New(rand.NewSource(seed)),
		sampleWindow:     500,
		recentScores:     make([]float64, 0, 500),
		acceptPercentile: 0.70,
	}
}

// ExtractSurFeaturesV2 extracts features with behavior descriptors
func ExtractSurFeaturesV2(s Strategy, result Result) []float64 {
	// Base features from V1
	base := ExtractSurFeatures(s)

	// Append behavior descriptors (3 dimensions)
	desc := computeBehaviorDescriptor(result)

	// Expand base array to accommodate behavior descriptors
	if len(base) < SurDim+3 {
		newBase := make([]float64, SurDim+3)
		copy(newBase, base)
		base = newBase
	}

	// Normalize behavior descriptors to [0, 1]
	base[SurDim] = float64(desc.FreqBin) / 5.0
	base[SurDim+1] = float64(desc.WRBin) / 5.0
	base[SurDim+2] = float64(desc.SkewBin) / 5.0

	return base[:SurDim+3]
}

// AddScore adds a validation score to recent history
func (m *SurrogateV2) AddScore(score float32) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.recentScores = append(m.recentScores, float64(score))

	// Keep only last sampleWindow scores
	if len(m.recentScores) > m.sampleWindow {
		m.recentScores = m.recentScores[len(m.recentScores)-m.sampleWindow:]
	}
}

// GetAcceptThreshold computes rank-based acceptance threshold
func (m *SurrogateV2) GetAcceptThreshold() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.recentScores) < 10 {
		// Not enough data - use permissive threshold
		return -1.0
	}

	// Copy and sort scores
	sorted := make([]float64, len(m.recentScores))
	copy(sorted, m.recentScores)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })

	// Find percentile
	percentileIdx := int(float64(len(sorted)) * (1.0 - m.acceptPercentile))
	if percentileIdx >= len(sorted) {
		percentileIdx = len(sorted) - 1
	}
	if percentileIdx < 0 {
		percentileIdx = 0
	}

	return sorted[percentileIdx]
}

// AcceptRankBased checks if strategy passes rank-based filter
func (m *SurrogateV2) AcceptRankBased(x []float64) bool {
	// Get threshold before locking to avoid deadlock
	m.mu.Lock()
	threshold := m.getAcceptThresholdNoLock()
	m.mu.Unlock()

	// Epsilon-greedy exploration
	if m.rng.Float64() < m.exploreP {
		return true
	}

	// Check prediction
	m.mu.Lock()
	defer m.mu.Unlock()
	pred := m.scoreNoLock(x)
	return pred >= threshold
}

// getAcceptThresholdNoLock computes rank-based acceptance threshold without locking
func (m *SurrogateV2) getAcceptThresholdNoLock() float64 {
	if len(m.recentScores) < 10 {
		// Not enough data - use permissive threshold
		return -1.0
	}

	// Copy and sort scores
	sorted := make([]float64, len(m.recentScores))
	copy(sorted, m.recentScores)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })

	// Find percentile
	percentileIdx := int(float64(len(sorted)) * (1.0 - m.acceptPercentile))
	if percentileIdx >= len(sorted) {
		percentileIdx = len(sorted) - 1
	}
	if percentileIdx < 0 {
		percentileIdx = 0
	}

	return sorted[percentileIdx]
}

// Score computes predicted score (same as V1)
func (m *SurrogateV2) Score(x []float64) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	var s float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		s += m.w[i] * x[i]
	}
	return s
}

// scoreNoLock computes score without locking
func (m *SurrogateV2) scoreNoLock(x []float64) float64 {
	var s float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		s += m.w[i] * x[i]
	}
	return s
}

// Update trains the surrogate with new validation result
func (m *SurrogateV2) Update(x []float64, y float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// prediction
	var pred float64
	for i := 0; i < len(m.w) && i < len(x); i++ {
		pred += m.w[i] * x[i]
	}

	// Huber-like clamp
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

// Save saves weights and recent scores
func (m *SurrogateV2) Save(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	data := struct {
		W                []float64
		RecentScores     []float64
		AcceptPercentile float64
	}{
		W:                m.w,
		RecentScores:     m.recentScores,
		AcceptPercentile: m.acceptPercentile,
	}

	encoded, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(path, encoded, 0644)
}

// Load loads weights and recent scores
func (m *SurrogateV2) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var loaded struct {
		W                []float64
		RecentScores     []float64
		AcceptPercentile float64
	}

	if err := json.Unmarshal(data, &loaded); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Resize weight vector if needed
	if len(loaded.W) != len(m.w) {
		m.w = make([]float64, len(loaded.W))
	}
	copy(m.w, loaded.W)

	// Restore recent scores
	m.recentScores = loaded.RecentScores

	// Restore acceptance percentile (optional, if saved)
	if loaded.AcceptPercentile > 0 && loaded.AcceptPercentile <= 1.0 {
		m.acceptPercentile = loaded.AcceptPercentile
	}

	return nil
}
