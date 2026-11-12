//go:build arm64 && !purego

package tensor

// reluPreservingNaNPrefix processes complete vectors within the current poll
// interval. The caller retains the scalar tail and all context checks.
func reluPreservingNaNPrefix(input, destination *Tensor, offset, end int) (hasNaN bool, next int) {
	next = offset + ((end - offset) &^ 3)
	if next == offset {
		return false, next
	}

	return reluPreservingNaNBlocks(input.floats[offset:next], destination.floats[offset:next]), next
}

// reluPreservingNaNBlocks requires equal, disjoint, nonempty slices whose length
// is a multiple of four. Integer operations preserve every NaN and subnormal bit.
//
//go:noescape
func reluPreservingNaNBlocks(source, destination []float32) (hasNaN bool)
