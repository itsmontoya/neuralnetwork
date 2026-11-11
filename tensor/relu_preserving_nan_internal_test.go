package tensor

import (
	"context"
	"math"
	"slices"
	"testing"
)

func Test_ReLUPreservingNaNInto_InvalidStorage(t *testing.T) {
	var tests = []struct {
		name  string
		value Tensor
	}{
		{"missing identity", Tensor{dataType: Float32, floats: []float32{1}}},
		{"invalid type", Tensor{identity: new(byte), dataType: Invalid}},
		{"missing values", Tensor{identity: new(byte), dataType: Float32}},
		{"extra integer storage", Tensor{identity: new(byte), dataType: Float32, floats: []float32{1}, integers: []int64{1}}},
		{"negative dimension", Tensor{identity: new(byte), dataType: Float32, shape: []int64{-1}, floats: []float32{1}}},
		{"wrong count", Tensor{identity: new(byte), dataType: Float32, shape: []int64{2}, floats: []float32{1}}},
	}
	for _, tt := range tests {
		for _, argument := range []string{"input", "destination"} {
			t.Run(tt.name+"/"+argument, func(t *testing.T) {
				var (
					input, output *Tensor
					before        []float32
					err           error
				)

				if input, err = FromFloat32(nil, []float32{2}); err != nil {
					t.Fatal(err)
				}
				if output, err = FromFloat32(nil, []float32{91}); err != nil {
					t.Fatal(err)
				}
				if argument == "input" {
					input = &tt.value
				} else {
					output = &tt.value
				}
				before = slices.Clone(output.floats)
				hasNaN, runErr := ReLUPreservingNaNInto(context.Background(), input, output)
				if runErr == nil || hasNaN || !slices.Equal(before, output.floats) {
					t.Fatalf("hasNaN=%t err=%v destination=%v before=%v", hasNaN, runErr, output.floats, before)
				}
			})
		}
	}
}

func Test_ReLUPreservingNaNInto_ScalarAliases(t *testing.T) {
	var (
		input *Tensor
		err   error
	)

	if input, err = FromFloat32(nil, []float32{math.Float32frombits(0xff801234)}); err != nil {
		t.Fatal(err)
	}
	shallow := *input
	for _, output := range []*Tensor{input, &shallow} {
		hasNaN, runErr := ReLUPreservingNaNInto(context.Background(), input, output)
		if runErr == nil || hasNaN || math.Float32bits(input.floats[0]) != 0xff801234 {
			t.Fatalf("hasNaN=%t err=%v bits=%08x", hasNaN, runErr, math.Float32bits(input.floats[0]))
		}
	}
}
