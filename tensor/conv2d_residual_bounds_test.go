//go:build arm64 && !purego && (darwin || linux)

package tensor

import (
	"math"
	"syscall"
	"testing"
	"unsafe"
)

func Test_Conv2D_ResidualSumsBounds(t *testing.T) {
	for _, atEnd := range []bool{false, true} {
		var (
			input   = conv2DGuardedFloats(t, 1, atEnd)
			weights = conv2DGuardedFloats(t, 4, atEnd)
			output  = conv2DGuardedFloats(t, 4, atEnd)
			want    = [4]float32{6, 10, 14, 22}
		)

		input[0] = 2
		copy(weights, []float32{3, 5, 7, 11})
		conv2DResidualSums(input, weights, (*[4]float32)(output), 1, 1, 1, 1, 1, 1, 1)
		for lane, expected := range want {
			if output[lane] != expected {
				t.Fatalf("atEnd=%v output[%d]=%v, want %v", atEnd, lane, output[lane], expected)
			}
		}
		conv2DResidualBias((*[4]float32)(output), weights)
		for lane, expected := range []float32{9, 15, 21, 33} {
			if output[lane] != expected {
				t.Fatalf("atEnd=%v biased output[%d]=%v, want %v", atEnd, lane, output[lane], expected)
			}
		}
	}
}

func Test_Conv2D_ResidualPayloadBounds(t *testing.T) {
	var tests = []struct {
		name                   string
		channels, size, kernel int64
		pad                    int64
	}{
		{name: "smallest payload", channels: 1, size: 1, kernel: 1},
		{name: "multiple channels and rows", channels: 2, size: 3, kernel: 3},
		{name: "clipped and entirely padded windows", channels: 2, size: 1, kernel: 3, pad: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, atEnd := range []bool{false, true} {
				var (
					input, weights, output *Tensor
					shape                  []int64
					config                 = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{tt.pad, tt.pad, tt.pad, tt.pad}, Groups: 1}
					err                    error
				)

				if input, err = New(Float32, []int64{1, tt.channels, tt.size, tt.size}); err != nil {
					t.Fatal(err)
				}
				if weights, err = New(Float32, []int64{4, tt.channels, tt.kernel, tt.kernel}); err != nil {
					t.Fatal(err)
				}
				if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
					t.Fatal(err)
				}
				if output, err = New(Float32, shape); err != nil {
					t.Fatal(err)
				}
				input.floats = conv2DGuardedFloats(t, len(input.floats), atEnd)
				weights.floats = conv2DGuardedFloats(t, len(weights.floats), atEnd)
				output.floats = conv2DGuardedFloats(t, len(output.floats), atEnd)
				for index := range input.floats {
					input.floats[index] = float32(index%7-3) * 0.25
				}
				for index := range weights.floats {
					weights.floats[index] = float32(index%11-5) * 0.125
				}
				if err = Conv2DInto(input, weights, nil, output, config); err != nil {
					t.Fatal(err)
				}
				for row := 0; row < int(shape[2]); row++ {
					for col := 0; col < int(shape[3]); col++ {
						var (
							first, second, third, fourth = conv2DFourChannels(input, weights, 0, 0, row, col, config)
							want                         = [4]float32{first, second, third, fourth}
							plane                        = int(shape[2] * shape[3])
						)

						for lane, expected := range want {
							var offset = lane*plane + row*int(shape[3]) + col

							if math.Float32bits(output.floats[offset]) != math.Float32bits(expected) {
								t.Fatalf("atEnd=%v output[%d]=%v, want %v", atEnd, offset, output.floats[offset], expected)
							}
						}
					}
				}
			}
		})
	}
}

// conv2DGuardedFloats places an exact-capacity payload against an inaccessible
// page on either side, so assembly reads and writes cannot rely on spare space.
func conv2DGuardedFloats(t *testing.T, count int, atEnd bool) (values []float32) {
	t.Helper()
	var (
		pageSize = syscall.Getpagesize()
		pages    []byte
		offset   = pageSize
		err      error
	)

	if count <= 0 || count*4 > pageSize {
		t.Fatalf("guarded payload size %d does not fit one page", count)
	}
	if pages, err = syscall.Mmap(-1, 0, 3*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE); err != nil {
		t.Fatalf("map guarded payload: %v", err)
	}
	t.Cleanup(func() {
		if unmapErr := syscall.Munmap(pages); unmapErr != nil {
			t.Errorf("unmap guarded payload: %v", unmapErr)
		}
	})
	if err = syscall.Mprotect(pages[:pageSize], syscall.PROT_NONE); err != nil {
		t.Fatalf("protect leading page: %v", err)
	}
	if err = syscall.Mprotect(pages[2*pageSize:], syscall.PROT_NONE); err != nil {
		t.Fatalf("protect trailing page: %v", err)
	}
	if atEnd {
		offset = 2*pageSize - count*4
	}
	values = unsafe.Slice((*float32)(unsafe.Pointer(&pages[offset])), count)
	return values
}
