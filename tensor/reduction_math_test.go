package tensor_test

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_ReductionsMath_Reference(t *testing.T) {
	type fixture struct {
		Name             string   `json:"name"`
		Operation        string   `json:"operation"`
		Opset            int      `json:"opset"`
		InputShape       []int64  `json:"input_shape"`
		InputBits        []uint32 `json:"input_bits"`
		Axes             []int64  `json:"axes"`
		KeepDimensions   bool     `json:"keep_dimensions"`
		AxesInputPresent bool     `json:"axes_input_present"`
		RightShape       []int64  `json:"right_shape"`
		RightBits        []uint32 `json:"right_bits"`
		OutputShape      []int64  `json:"output_shape"`
		OutputBits       []uint32 `json:"output_bits"`
		ReferenceResult  string   `json:"reference_result"`
		ReferenceError   string   `json:"reference_error"`
		ReferenceShape   []int64  `json:"reference_shape"`
		ReferenceBits    []uint32 `json:"reference_bits"`
	}
	var (
		contents []byte
		fixtures []fixture
		err      error
	)

	if contents, err = os.ReadFile("testdata/reductions_math.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 99 {
		t.Fatalf("fixture count=%d, want 99", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				input, right, output *tensor.Tensor
				shape                []int64
				values               []float32
				fixtureErr           error
				reduction            bool
			)

			if input, fixtureErr = tensor.FromFloat32(tt.InputShape, valuesFromBits(tt.InputBits)); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if tt.Operation == "Pow" {
				if right, fixtureErr = tensor.FromFloat32(tt.RightShape, valuesFromBits(tt.RightBits)); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
			}
			if output, fixtureErr = tensor.New(tensor.Float32, tt.OutputShape); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			switch tt.Operation {
			case "ReduceSum", "ReduceMean", "ReduceMax":
				reduction = true
				shape, fixtureErr = tensor.ReductionShape(tt.InputShape, tt.Axes, tt.KeepDimensions)
			case "Pow":
				shape, fixtureErr = tensor.BroadcastShape(tt.InputShape, tt.RightShape)
			default:
				shape = input.Shape()
			}
			if fixtureErr != nil || !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("planned shape=%v err=%v, want %v", shape, fixtureErr, tt.OutputShape)
			}
			switch tt.Operation {
			case "ReduceSum":
				fixtureErr = tensor.ReduceSumInto(input, output, tt.Axes, tt.KeepDimensions)
			case "ReduceMean":
				fixtureErr = tensor.ReduceMeanInto(input, output, tt.Axes, tt.KeepDimensions)
			case "ReduceMax":
				fixtureErr = tensor.ReduceMaxInto(input, output, tt.Axes, tt.KeepDimensions)
			case "Exp":
				fixtureErr = tensor.ExpInto(input, output)
			case "Log":
				fixtureErr = tensor.LogInto(input, output)
			case "Sqrt":
				fixtureErr = tensor.SqrtInto(input, output)
			case "Pow":
				fixtureErr = tensor.PowInto(input, right, output)
			default:
				t.Fatalf("unknown operation %q", tt.Operation)
			}
			if fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if values, fixtureErr = output.Float32Values(); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			checkProductValues(t, values, valuesFromBits(tt.OutputBits))
			if !reduction {
				for index, pattern := range tt.OutputBits {
					if math.Float32frombits(pattern) == 0 && math.Float32bits(values[index]) != pattern {
						t.Fatalf("zero sign at %d: got %08x, want %08x", index, math.Float32bits(values[index]), pattern)
					}
				}
			}
			if values, fixtureErr = input.Float32Values(); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			checkFloat32Bits(t, values, tt.InputBits)
			if right != nil {
				if values, fixtureErr = right.Float32Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				checkFloat32Bits(t, values, tt.RightBits)
			}
			if tt.ReferenceResult != "agree" || tt.ReferenceError != "" || !slices.Equal(tt.ReferenceShape, tt.OutputShape) {
				t.Fatalf("reference result=%q shape=%v error=%q", tt.ReferenceResult, tt.ReferenceShape, tt.ReferenceError)
			}
			checkProductValues(t, valuesFromBits(tt.ReferenceBits), valuesFromBits(tt.OutputBits))
			if reduction {
				if tt.Opset != 18 || tt.AxesInputPresent != (len(tt.Axes) != 0) {
					t.Fatalf("reduction adapter opset=%d axes input=%v", tt.Opset, tt.AxesInputPresent)
				}
			} else if tt.Opset != 13 {
				t.Fatalf("math adapter opset=%d, want 13", tt.Opset)
			}
		})
	}
}

func Test_ReductionShape(t *testing.T) {
	type testcase struct {
		name    string
		input   []int64
		axes    []int64
		keep    bool
		want    []int64
		wantErr bool
	}
	var (
		maxHost = int64(int(^uint(0) >> 1))
		tests   = []testcase{
			{name: "reordered axes", input: []int64{2, 3, 4}, axes: []int64{2, 0}, want: []int64{3}},
			{name: "negative kept axes", input: []int64{2, 3, 4}, axes: []int64{-1, -3}, keep: true, want: []int64{1, 3, 1}},
			{name: "all scalar", input: []int64{2, 3}, axes: []int64{1, 0}, want: []int64{}},
			{name: "all kept", input: []int64{2, 3}, axes: []int64{0, 1}, keep: true, want: []int64{1, 1}},
			{name: "identity", input: []int64{2, 3}, want: []int64{2, 3}},
			{name: "scalar identity", want: []int64{}},
			{name: "empty reduction", input: []int64{2, 0, 3}, axes: []int64{1}, want: []int64{2, 3}},
			{name: "empty output", input: []int64{0, 2, 3}, axes: []int64{2}, want: []int64{0, 2}},
			{name: "empty unused size", input: []int64{0, maxHost, maxHost}, axes: []int64{1, 2}, want: []int64{0}},
			{name: "negative shape", input: []int64{0, -1}, wantErr: true},
			{name: "input byte overflow", input: []int64{maxHost/4 + 1}, wantErr: true},
			{name: "output overflow after zero removed", input: []int64{0, maxHost, maxHost}, axes: []int64{0}, wantErr: true},
			{name: "scalar axis", axes: []int64{0}, wantErr: true},
			{name: "too high", input: []int64{2, 3}, axes: []int64{2}, wantErr: true},
			{name: "too low", input: []int64{2, 3}, axes: []int64{-3}, wantErr: true},
			{name: "minimum axis", input: []int64{2, 3}, axes: []int64{math.MinInt64}, wantErr: true},
			{name: "duplicate", input: []int64{2, 3}, axes: []int64{1, 1}, wantErr: true},
			{name: "normalized duplicate", input: []int64{2, 3}, axes: []int64{0, -2}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				originalInput = slices.Clone(tt.input)
				originalAxes  = slices.Clone(tt.axes)
				shape         []int64
				err           error
			)
			shape, err = tensor.ReductionShape(tt.input, tt.axes, tt.keep)
			if (err != nil) != tt.wantErr {
				t.Fatalf("shape=%v err=%v, wantErr=%v", shape, err, tt.wantErr)
			}
			if !slices.Equal(tt.input, originalInput) || !slices.Equal(tt.axes, originalAxes) {
				t.Fatal("shape planner changed arguments")
			}
			if tt.wantErr {
				if shape != nil {
					t.Fatalf("error returned shape=%v", shape)
				}
				return
			}
			if !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v, want %v", shape, tt.want)
			}
			if len(shape) > 0 {
				shape[0] = 99
				if !slices.Equal(tt.input, originalInput) || !slices.Equal(tt.axes, originalAxes) {
					t.Fatal("shape result aliases arguments")
				}
			}
		})
	}
}

func Test_Reductions_LiteralsAndEmptyIdentities(t *testing.T) {
	type testcase struct {
		name  string
		run   func(*tensor.Tensor, *tensor.Tensor, []int64, bool) error
		want  []float32
		all   float32
		empty float32
	}
	var tests = []testcase{
		{name: "sum", run: tensor.ReduceSumInto, want: []float32{30, 48}, all: 78, empty: 0},
		{name: "mean", run: tensor.ReduceMeanInto, want: []float32{5, 8}, all: 6.5, empty: float32(math.NaN())},
		{name: "max", run: tensor.ReduceMaxInto, want: []float32{9, 12}, all: 12, empty: float32(math.Inf(-1))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, empty, output, identity *tensor.Tensor
				shape                          []int64
				values                         []float32
				identityBits                   = []uint32{0x80000000, 0x7fc01234, 0x7f800000}
				err                            error
			)
			if input, err = tensor.FromFloat32([]int64{2, 2, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}); err != nil {
				t.Fatal(err)
			}
			for _, keep := range []bool{false, true} {
				shape = []int64{2}
				if keep {
					shape = []int64{1, 2, 1}
				}
				if output, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				if err = tt.run(input, output, []int64{2, 0}, keep); err != nil {
					t.Fatal(err)
				}
				if values, err = output.Float32Values(); err != nil {
					t.Fatal(err)
				}
				checkProductValues(t, values, tt.want)
			}
			if output, err = tensor.New(tensor.Float32, nil); err != nil {
				t.Fatal(err)
			}
			if err = tt.run(input, output, []int64{0, 1, 2}, false); err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, values, []float32{tt.all})
			if empty, err = tensor.New(tensor.Float32, []int64{2, 0, 3}); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, []int64{2, 3}); err != nil {
				t.Fatal(err)
			}
			if err = tt.run(empty, output, []int64{1}, false); err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			for _, value := range values {
				checkProductValues(t, []float32{value}, []float32{tt.empty})
				if tt.name == "sum" && math.Float32bits(value) != 0 {
					t.Fatalf("empty sum bits=%08x, want positive zero", math.Float32bits(value))
				}
			}
			if identity, err = tensor.FromFloat32([]int64{3}, valuesFromBits(identityBits)); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, identity.Shape()); err != nil {
				t.Fatal(err)
			}
			if err = tt.run(identity, output, nil, true); err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, values, identityBits)
		})
	}
}

func Test_Math_LiteralDomains(t *testing.T) {
	type testcase struct {
		name  string
		run   func(*tensor.Tensor, *tensor.Tensor) error
		input []float32
		want  []float32
	}
	var (
		negativeZero = math.Float32frombits(0x80000000)
		infinity     = float32(math.Inf(1))
		nan          = float32(math.NaN())
		tests        = []testcase{
			{name: "exp", run: tensor.ExpInto, input: []float32{-infinity, negativeZero, 0, 1, infinity, nan}, want: []float32{0, 1, 1, float32(math.E), infinity, nan}},
			{name: "log", run: tensor.LogInto, input: []float32{-4, negativeZero, 0, 1, infinity, -infinity, nan}, want: []float32{nan, -infinity, -infinity, 0, infinity, nan, nan}},
			{name: "sqrt", run: tensor.SqrtInto, input: []float32{-4, negativeZero, 0, 4, infinity, -infinity, nan}, want: []float32{nan, negativeZero, 0, 2, infinity, nan, nan}},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				values        []float32
				err           error
			)
			if input, err = tensor.FromFloat32([]int64{int64(len(tt.input))}, tt.input); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
				t.Fatal(err)
			}
			if err = tt.run(input, output); err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, values, tt.want)
			for index, expected := range tt.want {
				if expected == 0 && math.Float32bits(values[index]) != math.Float32bits(expected) {
					t.Fatalf("zero sign at %d: got %08x, want %08x", index, math.Float32bits(values[index]), math.Float32bits(expected))
				}
			}
		})
	}
	var (
		left, right, output *tensor.Tensor
		values              []float32
		err                 error
	)
	if left, err = tensor.FromFloat32([]int64{5}, []float32{negativeZero, -2, 0, 1, nan}); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{5}, []float32{-3, 3, -1, nan, 0}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, []int64{5}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.PowInto(left, right, output); err != nil {
		t.Fatal(err)
	}
	if values, err = output.Float32Values(); err != nil {
		t.Fatal(err)
	}
	checkProductValues(t, values, []float32{-infinity, -8, infinity, 1, 1})
}
