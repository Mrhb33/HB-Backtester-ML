package main

import (
	"sort"
	"sync"
)

const NumShards = 128 // Power of 2, tuned for worker count (40% of CPU)

// ShardedSeenMap reduces contention by sharding the seen map across 128 mutexes
// Each fingerprint is hashed to a specific shard, allowing concurrent access
type ShardedSeenMap struct {
	shards [NumShards]struct {
		mu    sync.Mutex
		items map[string]struct{}
	}
}

// NewShardedSeenMap creates a new sharded seen map with pre-allocated shards
func NewShardedSeenMap() *ShardedSeenMap {
	ssm := &ShardedSeenMap{}
	for i := 0; i < NumShards; i++ {
		// Pre-size to reduce rehashing
		ssm.shards[i].items = make(map[string]struct{}, 256)
	}
	return ssm
}

// fnv1aHash implements FNV-1a hash - very fast, good distribution
func fnv1aHash(s string) uint32 {
	hash := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		hash ^= uint32(s[i])
		hash *= 16777619
	}
	return hash
}

// CheckAndSet checks if a fingerprint exists in the seen map
// Returns false if already seen (duplicate), true if new (added)
func (ssm *ShardedSeenMap) CheckAndSet(fp string) bool {
	hash := fnv1aHash(fp)
	shardIdx := hash & (NumShards - 1) // Fast modulo for power of 2

	shard := &ssm.shards[shardIdx]
	shard.mu.Lock()
	defer shard.mu.Unlock()

	if _, ok := shard.items[fp]; ok {
		return false // Already seen
	}
	shard.items[fp] = struct{}{}
	return true // New fingerprint
}

// Snapshot returns a sorted snapshot of all fingerprints in the map
// Used for checkpointing - must be reproducible
func (ssm *ShardedSeenMap) Snapshot() []string {
	result := make([]string, 0, 4096)
	for i := 0; i < NumShards; i++ {
		shard := &ssm.shards[i]
		shard.mu.Lock()
		for fp := range shard.items {
			result = append(result, fp)
		}
		shard.mu.Unlock()
	}
	// Sort for reproducible checkpoints
	sort.Strings(result)
	return result
}

// Restore adds fingerprints from a snapshot (used when loading checkpoint)
func (ssm *ShardedSeenMap) Restore(fingerprints []string) {
	for _, fp := range fingerprints {
		hash := fnv1aHash(fp)
		shardIdx := hash & (NumShards - 1)
		shard := &ssm.shards[shardIdx]
		shard.mu.Lock()
		shard.items[fp] = struct{}{}
		shard.mu.Unlock()
	}
}
