package main

import (
	"sync"
)

// TradePool provides reusable trade arrays to reduce allocations
type TradePool struct {
	mu      sync.Mutex
	pool    [][]Position
	maxSize int
}

// NewTradePool creates a trade pool
func NewTradePool(maxSize int) *TradePool {
	return &TradePool{
		pool:    make([][]Position, 0, maxSize),
		maxSize: maxSize,
	}
}

// Get retrieves a trade array from pool or creates new one
func (p *TradePool) Get() []Position {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.pool) > 0 {
		// Pop from pool
		lastIdx := len(p.pool) - 1
		trades := p.pool[lastIdx]
		p.pool = p.pool[:lastIdx]
		return trades[:0] // Reset length but keep capacity
	}

	// No pool available - create new
	return make([]Position, 0, 100)
}

// Put returns a trade array to pool
func (p *TradePool) Put(trades []Position) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Don't exceed pool size
	if len(p.pool) >= p.maxSize {
		return
	}

	// Reset and add to pool
	p.pool = append(p.pool, trades[:0])
}

// BufferPool provides reusable float32 buffers
type BufferPool struct {
	mu      sync.Mutex
	pools   map[int][][]float32 // Key: buffer size, Value: slice of buffers
	maxEach int
}

// NewBufferPool creates a buffer pool
func NewBufferPool(maxEach int) *BufferPool {
	return &BufferPool{
		pools:   make(map[int][][]float32),
		maxEach: maxEach,
	}
}

// Get retrieves a buffer of given size
func (p *BufferPool) Get(size int) []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()

	pool, exists := p.pools[size]
	if !exists {
		pool = make([][]float32, 0, p.maxEach)
		p.pools[size] = pool
	}

	if len(pool) > 0 {
		lastIdx := len(pool) - 1
		buf := pool[lastIdx]
		p.pools[size] = pool[:lastIdx]
		return buf[:0] // Reset length
	}

	return make([]float32, 0, size)
}

// Put returns a buffer to pool
func (p *BufferPool) Put(buf []float32) {
	if buf == nil {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	size := cap(buf)
	if size == 0 {
		return
	}

	pool, exists := p.pools[size]
	if !exists {
		pool = make([][]float32, 0, p.maxEach)
		p.pools[size] = pool
	}

	if len(pool) >= p.maxEach {
		return
	}

	p.pools[size] = append(pool, buf[:0])
}

// Global pool instances (can be shared across workers)
var (
	tradePool  = NewTradePool(100) // 100 trade arrays in pool
	bufferPool = NewBufferPool(50) // 50 buffers of each size
)

// GetTradeBuffer gets a reusable trade buffer
func GetTradeBuffer() []Position {
	return tradePool.Get()
}

// PutTradeBuffer returns a trade buffer to pool
func PutTradeBuffer(buf []Position) {
	tradePool.Put(buf)
}

// GetFloatBuffer gets a reusable float32 buffer
func GetFloatBuffer(size int) []float32 {
	return bufferPool.Get(size)
}

// PutFloatBuffer returns a float32 buffer to pool
func PutFloatBuffer(buf []float32) {
	bufferPool.Put(buf)
}
