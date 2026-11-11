package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func Test_MaxPool2DSmallWindows_Bits(t *testing.T) {
	var (
		bits   = []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f801234, 0xff805678, 0x3f800000, 0xbf800000, 1, 0x80000001}
		random = rand.New(rand.NewSource(9))
		err    error
	)

	for _, kernel := range []int64{2, 3} {
		for _, strideH := range []int64{1, 2, 3, 4} {
			for _, strideW := range []int64{1, 2, 3, 4} {
				for height := int64(0); height < 10; height++ {
					for width := int64(0); width < 12; width++ {
						var (
							shape         = []int64{2, 3, height, width}
							config        = Pool2DConfig{Kernel: [2]int64{kernel, kernel}, Strides: [2]int64{strideH, strideW}, Dilations: [2]int64{1, 1}}
							outputShape   []int64
							input, output *Tensor
							values        = make([]float32, 6*height*width)
						)

						for index := range values {
							valueBits := random.Uint32()
							if index%2 == 0 {
								valueBits = bits[(index/2)%len(bits)]
							}

							values[index] = math.Float32frombits(valueBits)
						}
						if input, err = FromFloat32(shape, values); err != nil {
							t.Fatal(err)
						}

						if outputShape, err = Pool2DShape(shape, config); err != nil {
							t.Fatal(err)
						}

						if output, err = New(Float32, outputShape); err != nil {
							t.Fatal(err)
						}

						if err = MaxPool2DInto(input, output, config); err != nil {
							t.Fatal(err)
						}

						checkMaxPool2DBits(t, input, output, config)
					}
				}
			}
		}
	}
}

func Test_MaxPool2DSmallWindows_OrderedValues(t *testing.T) {
	type testcase struct {
		name string
		bits []uint32
		want uint32
	}
	var tests = []testcase{
		{"negative zero first", []uint32{0x80000000, 0, 0, 0}, 0x80000000},
		{"positive zero first", []uint32{0, 0x80000000, 0x80000000, 0x80000000}, 0},
		{"first NaN survives finite values", []uint32{0x7fc01234, 0x7f800000, 0, 0xff800000}, 0x7fc01234},
		{"last NaN wins", []uint32{0x7fc01234, 0, 0xffc05678, 0x7f800000}, 0xffc05678},
		{"signaling NaN bits", []uint32{0x7fc01234, 0x7f801234, 0, 0xff800000}, 0x7f801234},
		{"negative infinity", []uint32{0xff800000, 0xff800000, 0xff800000, 0xff800000}, 0xff800000},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, kernel := range []int64{2, 3} {
				var (
					config        = Pool2DConfig{Kernel: [2]int64{kernel, kernel}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
					values        = make([]float32, kernel*kernel)
					input, output *Tensor
					err           error
				)

				for index := range values {
					valueBits := test.bits[len(test.bits)-1]
					if index < len(test.bits) {
						valueBits = test.bits[index]
					}

					values[index] = math.Float32frombits(valueBits)
				}
				if input, err = FromFloat32([]int64{1, 1, kernel, kernel}, values); err != nil {
					t.Fatal(err)
				}

				if output, err = New(Float32, []int64{1, 1, 1, 1}); err != nil {
					t.Fatal(err)
				}

				if err = MaxPool2DInto(input, output, config); err != nil {
					t.Fatal(err)
				}

				if got := math.Float32bits(output.floats[0]); got != test.want {
					t.Fatalf("kernel=%d got=%08x want=%08x", kernel, got, test.want)
				}
			}
		})
	}
}

func Test_MaxPool2DSmallWindows_BoundariesAndFallback(t *testing.T) {
	type testcase struct {
		name                 string
		shape                []int64
		kernel, strides      [2]int64
		dilation             int64
		pads                 [4]int64
		ceilMode, includePad bool
		handled              bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "single 2", shape: []int64{1, 1, 2, 2}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}, handled: true},
			{name: "single 3", shape: []int64{1, 1, 3, 3}, kernel: [2]int64{3, 3}, strides: [2]int64{1, 1}, handled: true},
			{name: "maximum strides", shape: []int64{2, 3, 3, 4}, kernel: [2]int64{3, 3}, strides: [2]int64{limit, limit}, handled: true},
			{name: "uneven planes", shape: []int64{3, 2, 7, 11}, kernel: [2]int64{2, 2}, strides: [2]int64{3, 4}, handled: true},
			{name: "include pad ignored", shape: []int64{1, 2, 5, 7}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 3}, includePad: true, handled: true},
			{name: "empty batch", shape: []int64{0, 3, 5, 7}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}},
			{name: "empty channels", shape: []int64{2, 0, 5, 7}, kernel: [2]int64{3, 3}, strides: [2]int64{1, 1}},
			{name: "short height", shape: []int64{2, 3, 1, 7}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}},
			{name: "short width", shape: []int64{2, 3, 7, 2}, kernel: [2]int64{3, 3}, strides: [2]int64{1, 1}},
			{name: "top pad", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 3}, pads: [4]int64{1, 0, 0, 0}},
			{name: "left pad", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 3}, pads: [4]int64{0, 1, 0, 0}},
			{name: "bottom pad", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 3}, pads: [4]int64{0, 0, 1, 0}},
			{name: "right pad", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 3}, pads: [4]int64{0, 0, 0, 1}},
			{name: "empty padded windows", shape: []int64{1, 2, 0, 0}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}, pads: [4]int64{2, 2, 2, 2}},
			{name: "dilated", shape: []int64{2, 3, 7, 9}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 3}, dilation: 2},
			{name: "ceil partial", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 3}, ceilMode: true},
			{name: "ceil complete", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 2}, ceilMode: true},
			{name: "rectangular", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{2, 3}, strides: [2]int64{2, 3}},
			{name: "pointwise", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{1, 1}, strides: [2]int64{2, 3}},
			{name: "larger kernel", shape: []int64{2, 3, 5, 7}, kernel: [2]int64{4, 4}, strides: [2]int64{2, 3}},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, output *Tensor
				shape         []int64
				before        []float32
				storage       []float32
				config        = Pool2DConfig{
					Kernel: tt.kernel, Strides: tt.strides, Dilations: [2]int64{1, 1},
					Pads: tt.pads, CeilMode: tt.ceilMode, CountIncludePad: tt.includePad,
				}
				err error
			)
			if tt.dilation != 0 {
				config.Dilations = [2]int64{tt.dilation, tt.dilation}
			}
			if input, err = New(Float32, tt.shape); err != nil {
				t.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = float32(index%23 - 17)
				if index%7 == 0 {
					input.floats[index] = math.Float32frombits(0xff800123)
				}
			}
			before = append([]float32(nil), input.floats...)
			if shape, err = Pool2DShape(tt.shape, config); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			// Guard both ends while keeping a validated payload of the exact length.
			storage = make([]float32, len(output.floats)+2)
			for index := range storage {
				storage[index] = 91
			}
			output.floats = storage[1 : len(storage)-1]
			if len(output.floats) != 0 {
				if handled := maxPool2DSmallWindowsInto(input, output, config); handled != tt.handled {
					t.Fatalf("handled=%v, want %v", handled, tt.handled)
				}
				if !tt.handled {
					for _, value := range output.floats {
						if value != 91 {
							t.Fatal("declined fast path changed destination")
						}
					}
				}
			}
			if err = MaxPool2DInto(input, output, config); err != nil {
				t.Fatal(err)
			}
			checkMaxPool2DBits(t, input, output, config)
			if storage[0] != 91 || storage[len(storage)-1] != 91 {
				t.Fatal("pooling changed destination guards")
			}
			for index, value := range before {
				if math.Float32bits(input.floats[index]) != math.Float32bits(value) {
					t.Fatalf("pooling changed input at %d", index)
				}
			}
		})
	}
}

func Test_MaxPool2DSmallWindows_ErrorsPreserveDestination(t *testing.T) {
	for _, kernel := range []int64{2, 3} {
		for _, name := range []string{"alias", "wrong shape", "short payload", "zero stride"} {
			t.Run(fmt.Sprintf("%s/%dx%d", name, kernel, kernel), func(t *testing.T) {
				var (
					input, output *Tensor
					config        = Pool2DConfig{Kernel: [2]int64{kernel, kernel}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
					shape         = []int64{1, 1, 2, 2}
					before        []float32
					err           error
				)
				if input, err = New(Float32, []int64{1, 1, kernel + 1, kernel + 1}); err != nil {
					t.Fatal(err)
				}
				if name == "wrong shape" {
					shape = []int64{1, 1, 1, 1}
				}
				if output, err = New(Float32, shape); err != nil {
					t.Fatal(err)
				}
				for index := range output.floats {
					output.floats[index] = 91
				}
				switch name {
				case "alias":
					output.identity = input.identity
				case "short payload":
					output.floats = output.floats[:len(output.floats)-1]
				case "zero stride":
					config.Strides[0] = 0
				}
				before = append([]float32(nil), output.floats...)
				if err = MaxPool2DInto(input, output, config); err == nil {
					t.Fatal("invalid call accepted")
				}
				for index, value := range before {
					if output.floats[index] != value {
						t.Fatalf("error changed destination at %d", index)
					}
				}
			})
		}
	}
}

func Test_MaxPool2DSmallWindows_NaNPositions(t *testing.T) {
	for _, kernel := range []int64{2, 3} {
		for position := int64(0); position < kernel*kernel; position++ {
			for _, bits := range []uint32{0x7fc01234, 0xffc05678, 0x7f801234, 0xff805678} {
				var (
					values        = make([]float32, kernel*kernel)
					config        = Pool2DConfig{Kernel: [2]int64{kernel, kernel}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
					input, output *Tensor
					err           error
				)
				values[0] = math.Float32frombits(0x7fc0abcd)
				values[position] = math.Float32frombits(bits)
				if input, err = FromFloat32([]int64{1, 1, kernel, kernel}, values); err != nil {
					t.Fatal(err)
				}
				if output, err = New(Float32, []int64{1, 1, 1, 1}); err != nil {
					t.Fatal(err)
				}
				if err = MaxPool2DInto(input, output, config); err != nil {
					t.Fatal(err)
				}
				if got := math.Float32bits(output.floats[0]); got != bits {
					t.Fatalf("kernel=%d position=%d got=%08x want=%08x", kernel, position, got, bits)
				}
				checkMaxPool2DBits(t, input, output, config)
			}
		}
	}
}

func checkMaxPool2DBits(t *testing.T, input, output *Tensor, config Pool2DConfig) {
	t.Helper()
	var offset int
	for batch := 0; batch < int(output.shape[0]); batch++ {
		for channel := 0; channel < int(output.shape[1]); channel++ {
			for row := 0; row < int(output.shape[2]); row++ {
				for col := 0; col < int(output.shape[3]); col++ {
					var (
						want = math.Float32bits(maxPool2DValue(input, batch, channel, row, col, config))
						got  = math.Float32bits(output.floats[offset])
					)
					if got != want {
						t.Fatalf("shape=%v config=%+v offset=%d got=%08x want=%08x", input.shape, config, offset, got, want)
					}
					offset++
				}
			}
		}
	}
}
