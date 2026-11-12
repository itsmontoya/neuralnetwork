package tensor

import (
	"context"
	"fmt"
	"math"
	"math/rand"
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

func Test_ReLUPreservingNaNInto_AlignedStorageBits(t *testing.T) {
	var (
		sizes   = []int{0, 1, 2, 3, 4, 5, 7, 15, 16, 17, 1023, 1024, 1025, 3136, 6272, 8193}
		special = []uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
			0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
			0x7f800001, 0xff800001, 0x7fbfffff, 0xffbfffff, 0x7fc12345, 0xffc54321,
			0x7fffffff, 0xffffffff}
	)

	for _, size := range sizes {
		for inputOffset := 0; inputOffset < 4; inputOffset++ {
			for outputOffset := 0; outputOffset < 4; outputOffset++ {
				t.Run(fmt.Sprintf("%d/%d/%d", size, inputOffset, outputOffset), func(t *testing.T) {
					var (
						input, output *Tensor
						backing       = make([]float32, size+outputOffset+4)
						source        = make([]float32, size+inputOffset+4)
						random        = rand.New(rand.NewSource(int64(size)))
						wantNaN       bool
						err           error
					)

					if input, err = New(Float32, []int64{int64(size)}); err != nil {
						t.Fatal(err)
					}

					if output, err = New(Float32, input.shape); err != nil {
						t.Fatal(err)
					}

					for index := range source {
						bits := random.Uint32()
						if index < len(special) {
							bits = special[index]
						}

						source[index] = math.Float32frombits(bits)
					}
					for index := range backing {
						backing[index] = 91
					}
					before := slices.Clone(source)
					input.floats = source[inputOffset : inputOffset+size]
					output.floats = backing[outputOffset : outputOffset+size]
					hasNaN, runErr := ReLUPreservingNaNInto(context.Background(), input, output)
					if runErr != nil {
						t.Fatal(runErr)
					}

					for index, value := range input.floats {
						var (
							want  = math.Float32bits(value)
							isNaN = math.IsNaN(float64(value))
						)

						wantNaN = wantNaN || isNaN
						if !isNaN && !(value > 0) {
							want = 0
						}

						if got := math.Float32bits(output.floats[index]); got != want {
							t.Fatalf("index=%d got=%08x want=%08x", index, got, want)
						}
					}
					if hasNaN != wantNaN {
						t.Fatalf("hasNaN=%t want=%t", hasNaN, wantNaN)
					}

					for index, value := range source {
						if math.Float32bits(value) != math.Float32bits(before[index]) {
							t.Fatalf("input changed at %d", index)
						}
					}
					for index, value := range backing {
						if (index < outputOffset || index >= outputOffset+size) && value != 91 {
							t.Fatalf("destination guard changed at %d", index)
						}
					}
				})
			}
		}
	}
}
