package tensor

import (
	"slices"
	"testing"
)

func Test_BroadcastShapeInto_Ownership(t *testing.T) {
	var (
		left   = []int64{2, 1, 3}
		right  = []int64{4, 1}
		buffer = []int64{99, 99, 99, 99}
		shape  []int64
		owned  []int64
		err    error
	)

	if shape, err = broadcastShapeInto(left, right, buffer[:0]); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(shape, []int64{2, 4, 3}) || &shape[0] != &buffer[0] || buffer[3] != 99 {
		t.Fatalf("shape=%v buffer=%v", shape, buffer)
	}

	if owned, err = BroadcastShape(left, right); err != nil {
		t.Fatal(err)
	}

	left[0], right[0], buffer[0] = 7, 8, 9
	if !slices.Equal(owned, []int64{2, 4, 3}) {
		t.Fatalf("public shape shares temporary storage: %v", owned)
	}

	if shape, err = broadcastShapeInto([]int64{2, 3, 4}, nil, buffer[:1:1]); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(shape, []int64{2, 3, 4}) || &shape[0] == &buffer[0] {
		t.Fatalf("insufficient capacity did not allocate: %v", shape)
	}
}

func Test_BroadcastStridesInto_ResetAndOverflow(t *testing.T) {
	var (
		buffer  = []int64{99, 99, 99, 99}
		strides []int64
		err     error
	)

	if strides, err = broadcastStridesInto([]int64{4, 1}, []int64{2, 4, 3}, buffer[:0]); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(strides, []int64{0, 1, 0}) || &strides[0] != &buffer[0] || buffer[3] != 99 {
		t.Fatalf("strides=%v buffer=%v", strides, buffer)
	}

	if strides, err = broadcastStridesInto([]int64{1, 3}, []int64{2, 4, 3}, buffer); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(strides, []int64{0, 0, 1}) {
		t.Fatalf("stale strides survived reuse: %v", strides)
	}

	// Empty products still require every suffix stride to fit the host index.
	maximum := int64(int(^uint(0) >> 1))
	empty := []int64{0, maximum, 2}
	_, originalErr := contiguousStrides(empty)
	_, err = broadcastStridesInto(empty, empty, buffer)
	if originalErr == nil || err == nil || originalErr.Error() != err.Error() {
		t.Fatalf("suffix overflow changed: got %v, want %v", err, originalErr)
	}
}

func Test_BroadcastScalarMetadata_RemainsOwnedEmpty(t *testing.T) {
	var (
		shape []int64
		err   error
	)

	if shape, err = BroadcastShape(nil, nil); err != nil || shape == nil || len(shape) != 0 {
		t.Fatalf("scalar shape=%v, err=%v", shape, err)
	}

	if shape, err = broadcastDimensions(nil, nil); err != nil || shape == nil || len(shape) != 0 {
		t.Fatalf("scalar dimensions=%v, err=%v", shape, err)
	}

	if shape, err = broadcastStrides(nil, nil); err != nil || shape == nil || len(shape) != 0 {
		t.Fatalf("scalar strides=%v, err=%v", shape, err)
	}
}
