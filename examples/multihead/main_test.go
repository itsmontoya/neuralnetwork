package main

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SegmentArgmax(t *testing.T) {
	var (
		values *matrix.Matrix
		index  int
		err    error
	)

	if values, err = matrix.FromSlice(1, 5, []float32{0.1, 0.9, 0.7, 0.2, 0.1}); err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}
	if index, err = segmentArgmax(values, 0, 2); err != nil {
		t.Fatalf("first segmentArgmax returned error: %v", err)
	}
	if index != 1 {
		t.Fatalf("first segment index = %d, want 1", index)
	}
	if index, err = segmentArgmax(values, 2, 3); err != nil {
		t.Fatalf("second segmentArgmax returned error: %v", err)
	}
	if index != 0 {
		t.Fatalf("second segment index = %d, want 0", index)
	}
}
