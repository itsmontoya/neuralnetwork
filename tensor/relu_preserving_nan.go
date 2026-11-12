package tensor

import (
	"context"
	"fmt"
	"math"
)

// ReLUPreservingNaNInto rectifies float32 input into a caller-owned destination,
// preserving every original NaN bit, including sign, payload and signaling bits.
// Positive values (including subnormals and +Inf) are unchanged; other non-NaN
// values become +0. On success hasNaN reports whether input contains any NaN.
// ReLUInto retains its separate NaN-to-zero contract.
//
// Destination must have input's exact shape and disjoint storage, including for
// scalars and empties. Input then destination validation precedes context checks;
// validation errors and a nil context leave destination unchanged. A nil context
// is an error. No references are retained and no payload scratch is required.
// Shared read-only inputs with distinct destinations support concurrent calls.
//
// After validation, ctx.Err is checked before traversal, before each chunk of
// at most 1024 elements (including the first), and after traversal, even if empty.
// Context errors are returned directly, with hasNaN=false. Cancellation can leave
// a rectified prefix and an unchanged suffix, or a fully written destination if
// detected by the final check. A failed pre-traversal check leaves it unchanged.
func ReLUPreservingNaNInto(ctx context.Context, input, destination *Tensor) (hasNaN bool, err error) {
	if err = validateTensorType("input", input, Float32); err != nil {
		return false, err
	}
	if err = validateDestination(destination, Float32, input.shape, input); err != nil {
		return false, err
	}
	if ctx == nil {
		err = fmt.Errorf("tensor: context is nil")
		return false, err
	}
	if err = ctx.Err(); err != nil {
		return false, err
	}

	for offset := 0; offset < len(input.floats); {
		if err = ctx.Err(); err != nil {
			return false, err
		}
		end := min(offset+1024, len(input.floats))
		chunkNaN, next := reluPreservingNaNPrefix(input, destination, offset, end)
		hasNaN = chunkNaN || hasNaN
		for index, value := range input.floats[next:end] {
			bits := math.Float32bits(value)
			if bits&0x7fffffff > 0x7f800000 {
				hasNaN = true
			} else if bits&0x80000000 != 0 {
				value = 0
			}
			destination.floats[next+index] = value
		}
		offset = end
	}
	if err = ctx.Err(); err != nil {
		return false, err
	}
	return hasNaN, nil
}
