package main

// Complexity holds complexity metrics for a strategy
type Complexity struct {
	NodeCount       int
	LeafCount       int
	UniqueFeatures  map[int]bool // Feature indices used
	ParamCount      int          // Numeric thresholds
	MaxDepth        int
	OperatorCount   map[string]int // AND, OR, NOT, etc.
}

// ComputeCombinedComplexity computes combined complexity for entry + exit rules
func ComputeCombinedComplexity(entryRoot, exitRoot *RuleNode) Complexity {
	entryCx := ComputeComplexity(entryRoot)
	exitCx := ComputeComplexity(exitRoot)

	// Combine complexities
	combined := Complexity{
		NodeCount:      entryCx.NodeCount + exitCx.NodeCount,
		LeafCount:      entryCx.LeafCount + exitCx.LeafCount,
		UniqueFeatures: make(map[int]bool),
		ParamCount:     entryCx.ParamCount + exitCx.ParamCount,
		MaxDepth:       max(entryCx.MaxDepth, exitCx.MaxDepth),
		OperatorCount:  make(map[string]int),
	}

	// Merge unique features from both rules
	for f := range entryCx.UniqueFeatures {
		combined.UniqueFeatures[f] = true
	}
	for f := range exitCx.UniqueFeatures {
		combined.UniqueFeatures[f] = true
	}

	// Merge operator counts
	for op, count := range entryCx.OperatorCount {
		combined.OperatorCount[op] += count
	}
	for op, count := range exitCx.OperatorCount {
		combined.OperatorCount[op] += count
	}

	return combined
}

// ComputeComplexity computes complexity for a single rule tree
func ComputeComplexity(node *RuleNode) Complexity {
	cx := Complexity{
		UniqueFeatures: make(map[int]bool),
		OperatorCount:  make(map[string]int),
	}

	if node == nil {
		return cx
	}

	// Walk AST recursively - pass pointer to avoid copying
	walkTree(node, &cx, 0)

	return cx
}

// walkTree recursively walks the AST and computes complexity
func walkTree(node *RuleNode, cx *Complexity, depth int) {
	if node == nil {
		return
	}

	// Update max depth
	if depth > cx.MaxDepth {
		cx.MaxDepth = depth
	}

	cx.NodeCount++

	if node.Op == OpLeaf {
		cx.LeafCount++
		// Track unique features
		cx.UniqueFeatures[node.Leaf.A] = true
		// For Cross/Between leaves, also track B
		if node.Leaf.Kind == LeafCrossUp || node.Leaf.Kind == LeafCrossDown ||
			node.Leaf.Kind == LeafBetween || node.Leaf.Kind == LeafSlopeGT ||
			node.Leaf.Kind == LeafSlopeLT {
			cx.UniqueFeatures[node.Leaf.B] = true
		}
		// Count numeric parameters (thresholds)
		if node.Leaf.Kind != LeafCrossUp && node.Leaf.Kind != LeafCrossDown &&
			node.Leaf.Kind != LeafRising && node.Leaf.Kind != LeafFalling {
			cx.ParamCount++
		}
		// For Between leaves, count both thresholds
		if node.Leaf.Kind == LeafBetween {
			cx.ParamCount++
		}
		return
	}

	// Count operators
	switch node.Op {
	case OpAnd:
		cx.OperatorCount["AND"]++
	case OpOr:
		cx.OperatorCount["OR"]++
	case OpNot:
		cx.OperatorCount["NOT"]++
	}

	// Recurse
	walkTree(node.L, cx, depth+1)
	walkTree(node.R, cx, depth+1)
}

// pruneRedundantNodes removes redundant subtrees to simplify the tree

// UniqueFeatureCount returns the number of unique features used
func (c Complexity) UniqueFeatureCount() int {
	return len(c.UniqueFeatures)
}

// pruneRedundantNodes removes redundant subtrees to simplify the tree
// Returns the simplified tree
func pruneRedundantNodes(node *RuleNode) *RuleNode {
	if node == nil {
		return nil
	}

	// First recurse to simplify children
	node.L = pruneRedundantNodes(node.L)
	node.R = pruneRedundantNodes(node.R)

	// After simplifying children, check for redundant patterns at this node
	// Handle cases where children became nil after simplification

	// (A AND A) -> A
	// (A OR A) -> A
	if node.L != nil && node.R != nil && ruleTreeEqual(node.L, node.R) {
		if node.Op == OpAnd || node.Op == OpOr {
			return node.L
		}
	}

	// (A AND True) -> A, (A OR False) -> A
	if node.Op == OpAnd && isTrueNode(node.R) {
		return node.L
	}
	if node.Op == OpAnd && isTrueNode(node.L) {
		return node.R
	}
	if node.Op == OpOr && isFalseNode(node.R) {
		return node.L
	}
	if node.Op == OpOr && isFalseNode(node.L) {
		return node.R
	}

	// NOT(NOT(A)) -> A
	if node.Op == OpNot && node.L != nil && node.L.Op == OpNot {
		return node.L.L
	}

	// If one child is nil, return the other (simplify single-child nodes)
	if node.L == nil {
		return node.R
	}
	if node.R == nil {
		return node.L
	}

	return node
}

// ruleTreeEqual checks if two rule trees are structurally equal
func ruleTreeEqual(a, b *RuleNode) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Op != b.Op {
		return false
	}
	if a.Op == OpLeaf {
		return a.Leaf.A == b.Leaf.A &&
			a.Leaf.B == b.Leaf.B &&
			a.Leaf.Kind == b.Leaf.Kind &&
			a.Leaf.Lookback == b.Leaf.Lookback
	}
	return ruleTreeEqual(a.L, b.L) && ruleTreeEqual(a.R, b.R)
}

// isTrueNode checks if a node is always true (currently no such pattern in our trees)
func isTrueNode(node *RuleNode) bool {
	return false // No "always true" leaf nodes in our system
}

// isFalseNode checks if a node is always false (currently no such pattern in our trees)
func isFalseNode(node *RuleNode) bool {
	return false // No "always false" leaf nodes in our system
}
