package tensor_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_ReLUPreservingNaNInto_ExceptionalBits(t *testing.T) {
	var tests = []struct {
		input uint32
		want  uint32
	}{
		{0, 0}, {0x80000000, 0},
		{1, 1}, {0x80000001, 0}, {0x007fffff, 0x007fffff}, {0x807fffff, 0},
		{0x00800000, 0x00800000}, {0x80800000, 0},
		{0x3fa00000, 0x3fa00000}, {0xbfa00000, 0},
		{0x7f7fffff, 0x7f7fffff}, {0xff7fffff, 0},
		{0x7f800000, 0x7f800000}, {0xff800000, 0},
		{0x7fc12345, 0x7fc12345}, {0xffc12345, 0xffc12345},
		{0x7f800001, 0x7f800001}, {0xff800001, 0xff800001},
		{0x7fbfffff, 0x7fbfffff}, {0xffbfffff, 0xffbfffff},
		{0x7fffffff, 0x7fffffff}, {0xffffffff, 0xffffffff},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%08x", tt.input), func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				values        []float32
				hasNaN        bool
				err           error
			)

			if input, err = tensor.FromFloat32(nil, valuesFromBits([]uint32{tt.input})); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromFloat32(nil, []float32{91}); err != nil {
				t.Fatal(err)
			}
			hasNaN, err = tensor.ReLUPreservingNaNInto(context.Background(), input, output)
			if err != nil || hasNaN != math.IsNaN(float64(math.Float32frombits(tt.input))) {
				t.Fatalf("hasNaN=%t err=%v", hasNaN, err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, values, []uint32{tt.want})
			if values, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, values, []uint32{tt.input})
		})
	}
}

func Test_ReLUPreservingNaNInto_LegacyCorrection(t *testing.T) {
	var (
		sizes    = []int{0, 1, 2, 3, 4, 5, 7, 14, 15, 16, 17, 31, 32, 33, 42, 1023, 1024, 1025, 3136, 6272, 8193}
		patterns = []string{"finite", "mixed", "random bits", "all NaN", "first NaN", "middle NaN", "last NaN"}
		nanBits  = []uint32{0x7fc12345, 0xffc54321, 0x7f800001, 0xff800002, 0x7fffffff, 0xffffffff}
	)

	for _, size := range sizes {
		for _, pattern := range patterns {
			t.Run(fmt.Sprintf("%d/%s", size, pattern), func(t *testing.T) {
				var (
					bits          = make([]uint32, size)
					shape         = []int64{int64(size)}
					random        = rand.New(rand.NewSource(int64(size)))
					input, output *tensor.Tensor
					want, got     []float32
					hasNaN        bool
					wantNaN       bool
					err           error
				)

				for index := range bits {
					bits[index] = math.Float32bits(float32(index%17) - 8)
					switch pattern {
					case "mixed":
						if index%3 == 0 {
							bits[index] = nanBits[index%len(nanBits)]
						}
					case "random bits":
						bits[index] = random.Uint32()
					case "all NaN":
						bits[index] = nanBits[index%len(nanBits)]
					}
					if pattern == "first NaN" && index == 0 || pattern == "middle NaN" && index == size/2 || pattern == "last NaN" && index == size-1 {
						bits[index] = 0xff801234
					}
					if math.IsNaN(float64(math.Float32frombits(bits[index]))) {
						wantNaN = true
					}
				}
				if size == 1 {
					shape = nil
				} else if size == 42 {
					shape = []int64{2, 3, 7}
				} else if size == 0 {
					shape = []int64{2, 0, 3}
				}
				if input, err = tensor.FromFloat32(shape, valuesFromBits(bits)); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				want = legacyCorrectedReLU(t, input)
				hasNaN, err = tensor.ReLUPreservingNaNInto(context.Background(), input, output)
				if err != nil || hasNaN != wantNaN {
					t.Fatalf("hasNaN=%t want=%t err=%v", hasNaN, wantNaN, err)
				}
				if !slices.Equal(output.Shape(), shape) {
					t.Fatalf("shape=%v want=%v", output.Shape(), shape)
				}
				if got, err = output.Float32Values(); err != nil {
					t.Fatal(err)
				}
				for index, value := range want {
					if math.Float32bits(got[index]) != math.Float32bits(value) {
						t.Fatalf("index=%d got=%08x want=%08x", index, math.Float32bits(got[index]), math.Float32bits(value))
					}
				}
				if got, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				checkFloat32Bits(t, got, bits)
			})
		}
	}
}

// legacyCorrectedReLU uses the existing backend followed by the runtime's
// copying NaN repair, independently of the new primitive's bit classification.
func legacyCorrectedReLU(t *testing.T, input *tensor.Tensor) (values []float32) {
	t.Helper()
	var (
		output    *tensor.Tensor
		source    []float32
		corrected []float32
		err       error
	)

	if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
		t.Fatal(err)
	}
	if err = tensor.ReLUInto(input, output); err != nil {
		t.Fatal(err)
	}
	if source, err = input.Float32Values(); err != nil {
		t.Fatal(err)
	}
	for index, value := range source {
		if !math.IsNaN(float64(value)) {
			continue
		}
		if corrected == nil {
			if corrected, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
		}
		corrected[index] = value
	}
	if corrected != nil {
		if output, err = tensor.FromFloat32(input.Shape(), corrected); err != nil {
			t.Fatal(err)
		}
	}
	if values, err = output.Float32Values(); err != nil {
		t.Fatal(err)
	}
	return values
}

func Test_ReLUPreservingNaNInto_Cancellation(t *testing.T) {
	for _, size := range []int{0, 1, 3, 4, 5, 7, 1023, 1024, 1025, 2049, 8193} {
		checks := 2 + (size+1023)/1024
		for boundary := 1; boundary <= checks+1; boundary++ {
			t.Run(fmt.Sprintf("%d/check%d", size, boundary), func(t *testing.T) {
				var (
					source        = make([]float32, size)
					before        = make([]float32, size)
					input, output *tensor.Tensor
					got           []float32
					hasNaN        bool
					err           error
				)

				for index := range source {
					source[index] = math.Float32frombits(0xff801234)
					before[index] = 91
				}
				if input, err = tensor.FromFloat32([]int64{int64(size)}, source); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.FromFloat32(input.Shape(), before); err != nil {
					t.Fatal(err)
				}
				ctx := newReLUCancelContext(boundary)
				defer ctx.cancel()
				hasNaN, err = tensor.ReLUPreservingNaNInto(ctx, input, output)
				if boundary <= checks {
					if !errors.Is(err, context.Canceled) || hasNaN || ctx.checks != boundary {
						t.Fatalf("checks=%d hasNaN=%t err=%v", ctx.checks, hasNaN, err)
					}
				} else if err != nil || hasNaN != (size > 0) || ctx.checks != checks {
					t.Fatalf("checks=%d hasNaN=%t err=%v", ctx.checks, hasNaN, err)
				}
				written := size
				if boundary <= checks {
					written = min(max(0, boundary-2)*1024, size)
				}
				if got, err = output.Float32Values(); err != nil {
					t.Fatal(err)
				}
				for index, value := range got {
					want := uint32(0xff801234)
					if index >= written {
						want = math.Float32bits(91)
					}
					if math.Float32bits(value) != want {
						t.Fatalf("index=%d got=%08x want=%08x", index, math.Float32bits(value), want)
					}
				}
			})
		}
	}
}

func Test_ReLUPreservingNaNInto_ErrorPrecedence(t *testing.T) {
	var (
		input, output *tensor.Tensor
		got           []float32
		err           error
	)

	if input, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32(nil, []float32{91}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	for _, tt := range []struct {
		name        string
		ctx         context.Context
		input       *tensor.Tensor
		destination *tensor.Tensor
		want        string
	}{
		{"input before context", ctx, nil, output, "input"},
		{"input before destination", ctx, nil, nil, "input"},
		{"destination before context", ctx, input, nil, "destination"},
		{"nil context", nil, input, output, "context is nil"},
		{"deadline", ctx, input, output, "context deadline exceeded"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			hasNaN, runErr := tensor.ReLUPreservingNaNInto(tt.ctx, tt.input, tt.destination)
			if runErr == nil || hasNaN || !strings.Contains(runErr.Error(), tt.want) {
				t.Fatalf("hasNaN=%t err=%v", hasNaN, runErr)
			}
			if tt.name == "deadline" && runErr != context.DeadlineExceeded {
				t.Fatalf("context error was wrapped: %v", runErr)
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, got, []uint32{math.Float32bits(91)})
		})
	}
}

func Test_ReLUPreservingNaNInto_SingleNaNLanes(t *testing.T) {
	for _, size := range []int{4, 5, 7, 8, 9, 1024, 1025, 2049} {
		for _, position := range []int{0, 1, 2, 3, size - 1} {
			for _, bits := range []uint32{0x7f800001, 0xff800001, 0x7fc12345, 0xffc54321} {
				t.Run(fmt.Sprintf("%d/%d/%08x", size, position, bits), func(t *testing.T) {
					var (
						values        = make([]float32, size)
						input, output *tensor.Tensor
						got           []float32
						err           error
					)

					for index := range values {
						values[index] = 1
					}
					values[position] = math.Float32frombits(bits)
					if input, err = tensor.FromFloat32([]int64{int64(size)}, values); err != nil {
						t.Fatal(err)
					}

					if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
						t.Fatal(err)
					}

					hasNaN, runErr := tensor.ReLUPreservingNaNInto(context.Background(), input, output)
					if runErr != nil || !hasNaN {
						t.Fatalf("hasNaN=%t err=%v", hasNaN, runErr)
					}

					if got, err = output.Float32Values(); err != nil {
						t.Fatal(err)
					}

					for index, value := range got {
						if math.Float32bits(value) != math.Float32bits(values[index]) {
							t.Fatalf("index=%d got=%08x want=%08x", index, math.Float32bits(value), math.Float32bits(values[index]))
						}
					}
				})
			}
		}
	}
}

func preservingNaNReLU(input, output *tensor.Tensor) (err error) {
	_, err = tensor.ReLUPreservingNaNInto(context.Background(), input, output)
	return err
}
