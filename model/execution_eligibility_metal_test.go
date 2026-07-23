//go:build darwin && cgo && metal && !purego

package model

import (
	"math"
	"testing"
)

func Test_DenseMatMulOperations(t *testing.T) {
	type testcase struct {
		name       string
		rows       int
		inputSize  int
		outputSize int
		want       uint64
	}

	tests := []testcase{
		{name: "regular", rows: 64, inputSize: 128, outputSize: 128, want: 1 << 20},
		{name: "zero", rows: 0, inputSize: 128, outputSize: 128, want: 0},
		{
			name:       "overflow",
			rows:       math.MaxInt,
			inputSize:  math.MaxInt,
			outputSize: math.MaxInt,
			want:       math.MaxUint64,
		},
	}

	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var got uint64

			got = denseMatMulOperations(test.rows, test.inputSize, test.outputSize)
			if got != test.want {
				t.Fatalf("denseMatMulOperations = %d, want %d", got, test.want)
			}
		})
	}
}
