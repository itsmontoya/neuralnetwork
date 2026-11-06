package tensor_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Pooling_Reference(t *testing.T) {
	type fixture struct {
		Name                string   `json:"name"`
		Operation           string   `json:"operation"`
		InputShape          []int64  `json:"input_shape"`
		InputBits           []uint32 `json:"input_bits"`
		Kernel              [2]int64 `json:"kernel"`
		Strides             [2]int64 `json:"strides"`
		Dilations           [2]int64 `json:"dilations"`
		Pads                [4]int64 `json:"pads"`
		CeilMode            bool     `json:"ceil_mode"`
		CountIncludePad     bool     `json:"count_include_pad"`
		OutputShape         []int64  `json:"output_shape"`
		OutputBits          []uint32 `json:"output_bits"`
		ExpectedDivisors    []int64  `json:"expected_divisors"`
		Reference           string   `json:"reference"`
		ReferenceError      string   `json:"reference_error"`
		ReferenceDifference string   `json:"reference_difference"`
		ORTResult           string   `json:"ort_result"`
		ORTError            string   `json:"ort_error"`
		ORTDifference       string   `json:"ort_difference"`
	}
	var (
		contents                                []byte
		fixtures                                []fixture
		differences                             int
		ortAgreement, ortDifferences, ortErrors int
		err                                     error
	)

	if contents, err = os.ReadFile("testdata/pooling.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 65 {
		t.Fatalf("fixture count=%d, want 65", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				config        tensor.Pool2DConfig
				shape         []int64
				values        []float32
				fixtureErr    error
			)
			config.Kernel, config.Strides, config.Dilations = tt.Kernel, tt.Strides, tt.Dilations
			config.Pads, config.CeilMode, config.CountIncludePad = tt.Pads, tt.CeilMode, tt.CountIncludePad
			if input, fixtureErr = tensor.FromFloat32(tt.InputShape, valuesFromBits(tt.InputBits)); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if tt.Operation == "GlobalAveragePool" {
				shape, fixtureErr = tensor.GlobalAveragePoolShape(input.Shape())
			} else {
				shape, fixtureErr = tensor.Pool2DShape(input.Shape(), config)
			}
			if fixtureErr != nil || !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("shape=%v error=%v, want %v", shape, fixtureErr, tt.OutputShape)
			}
			if output, fixtureErr = tensor.New(tensor.Float32, shape); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			switch tt.Operation {
			case "MaxPool":
				fixtureErr = tensor.MaxPool2DInto(input, output, config)
			case "AveragePool":
				fixtureErr = tensor.AveragePool2DInto(input, output, config)
			case "GlobalAveragePool":
				fixtureErr = tensor.GlobalAveragePoolInto(input, output)
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
			if tt.Name == "odd ceil partial beyond declared pads avg include" {
				checkProductValues(t, values, []float32{3, 2.25, 3.75, 2.25})
			}
			if tt.Name == "odd ceil partial beyond declared pads avg exclude" {
				checkProductValues(t, values, []float32{3, 4.5, 7.5, 9})
			}
			if values, fixtureErr = input.Float32Values(); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			checkFloat32Bits(t, values, tt.InputBits)
			if tt.Reference == "numpy" {
				if tt.ReferenceError == "" && tt.ReferenceDifference == "" {
					t.Fatal("independent reference must record evaluator error or difference")
				}
				differences++
			} else if tt.Reference != "onnx" || tt.ReferenceError != "" || tt.ReferenceDifference != "" {
				t.Fatalf("unexpected reference=%q error=%q difference=%q", tt.Reference, tt.ReferenceError, tt.ReferenceDifference)
			}
			if tt.Name == "odd ceil partial beyond declared pads avg include" && !slices.Equal(tt.ExpectedDivisors, []int64{4, 4, 4, 4}) {
				t.Fatalf("trailing ceil divisors=%v, want full kernel areas [4 4 4 4]", tt.ExpectedDivisors)
			}
			switch tt.ORTResult {
			case "agree":
				if tt.ORTError != "" || tt.ORTDifference != "" {
					t.Fatal("ORT agreement has contradictory metadata")
				}
				ortAgreement++
			case "different":
				if tt.ORTDifference == "" || tt.ORTError != "" {
					t.Fatal("ORT difference must record the observed disagreement")
				}
				ortDifferences++
			case "error":
				if tt.ORTError == "" {
					t.Fatal("ORT rejection must retain its error")
				}
				ortErrors++
			default:
				t.Fatalf("unexpected ORT result %q", tt.ORTResult)
			}
		})
	}
	if differences != 17 {
		t.Fatalf("documented evaluator differences=%d, want 17", differences)
	}
	if ortAgreement != 40 || ortDifferences != 4 || ortErrors != 21 {
		t.Fatalf("ORT metadata: agreement=%d differences=%d errors=%d, want 40/4/21", ortAgreement, ortDifferences, ortErrors)
	}
}

func Test_Pool2DShape(t *testing.T) {
	type testcase struct {
		name    string
		input   []int64
		config  tensor.Pool2DConfig
		want    []int64
		invalid bool
	}
	var (
		limit              = int64(int(^uint(0) >> 1))
		valid              = tensor.Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		ceilNegative       = tensor.Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true}
		ceilTrim           = tensor.Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{3, 3}, Dilations: [2]int64{1, 1}, Pads: [4]int64{0, 0, 2, 2}, CeilMode: true}
		hugeTrim           = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{0, 0, limit - 1, limit - 1}, CeilMode: true}
		outputBytes        = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{limit / 4, 0, 0, 0}}
		outputElements     = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{limit / 2, limit / 2, 0, 0}}
		coordinateOverflow = tensor.Pool2DConfig{Kernel: [2]int64{limit - 1, 1}, Strides: [2]int64{limit - 1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{2, 0, 0, 0}, CeilMode: true}
		tests              = []testcase{
			{name: "rectangular", input: []int64{2, 3, 4, 5}, config: valid, want: []int64{2, 3, 3, 4}},
			{name: "negative ceil numerator", input: []int64{1, 1, 2, 2}, config: ceilNegative, want: []int64{1, 1, 1, 1}},
			{name: "remove trailing ceil starts", input: []int64{1, 1, 3, 3}, config: ceilTrim, want: []int64{1, 1, 1, 1}},
			{name: "huge trailing pads capped without iteration", input: []int64{1, 1, 1, 1}, config: hugeTrim, want: []int64{1, 1, 1, 1}},
			{name: "oversized floor", input: []int64{1, 1, 0, 1}, config: valid, want: []int64{1, 1, 0, 0}},
			{name: "empty batch", input: []int64{0, 2, 4, 5}, config: valid, want: []int64{0, 2, 3, 4}},
			{name: "empty channels", input: []int64{2, 0, 4, 5}, config: valid, want: []int64{2, 0, 3, 4}},
			{name: "scalar", config: valid, invalid: true},
			{name: "rank three", input: []int64{1, 2, 3}, config: valid, invalid: true},
			{name: "negative dimension", input: []int64{0, 1, -1, 2}, config: valid, invalid: true},
			{name: "zero config", input: []int64{1, 1, 2, 2}, invalid: true},
			{name: "input bytes overflow", input: []int64{1, 1, limit/4 + 1, 1}, config: valid, invalid: true},
			{name: "output bytes overflow", input: []int64{1, 1, 1, 1}, config: outputBytes, invalid: true},
			{name: "output element overflow", input: []int64{1, 1, 0, 0}, config: outputElements, invalid: true},
			{name: "ceil coordinate overflow despite empty batch", input: []int64{0, 1, limit - 2, 1}, config: coordinateOverflow, invalid: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var shape, err = tensor.Pool2DShape(tt.input, tt.config)
			if (err != nil) != tt.invalid || !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v error=%v, want %v invalid=%v", shape, err, tt.want, tt.invalid)
			}
			if !tt.invalid {
				shape[0] = 99
				if tt.input[0] == 99 {
					t.Fatal("planner returned aliased input shape")
				}
			}
		})
	}
}

func Test_Pooling_GeometryErrorsPreserveDestination(t *testing.T) {
	type testcase struct {
		name   string
		config tensor.Pool2DConfig
	}
	var (
		limit     = int64(int(^uint(0) >> 1))
		valid     = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		tests     = []testcase{}
		functions = []struct {
			name string
			call func(*tensor.Tensor, *tensor.Tensor, tensor.Pool2DConfig) error
		}{{"max", tensor.MaxPool2DInto}, {"average", tensor.AveragePool2DInto}}
	)
	for field := 0; field < 4; field++ {
		var negativePad = valid
		negativePad.Pads[field] = -1
		tests = append(tests, testcase{name: fmt.Sprintf("negative pad %d", field), config: negativePad})
	}
	for axis := 0; axis < 2; axis++ {
		var zeroKernel, zeroStride, zeroDilation, overflowDilation, overflowPad = valid, valid, valid, valid, valid
		zeroKernel.Kernel[axis] = 0
		zeroStride.Strides[axis] = 0
		zeroDilation.Dilations[axis] = 0
		overflowDilation.Kernel[axis], overflowDilation.Dilations[axis] = 2, limit
		overflowPad.Pads[axis] = limit
		tests = append(tests,
			testcase{name: fmt.Sprintf("zero kernel %d", axis), config: zeroKernel},
			testcase{name: fmt.Sprintf("zero stride %d", axis), config: zeroStride},
			testcase{name: fmt.Sprintf("zero dilation %d", axis), config: zeroDilation},
			testcase{name: fmt.Sprintf("effective kernel overflow %d", axis), config: overflowDilation},
			testcase{name: fmt.Sprintf("padding overflow %d", axis), config: overflowPad})
	}
	for _, function := range functions {
		for _, tt := range tests {
			t.Run(function.name+"/"+tt.name, func(t *testing.T) {
				var (
					input, output *tensor.Tensor
					values        []float32
					err           error
				)
				if input, err = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{3}); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{91}); err != nil {
					t.Fatal(err)
				}
				if err = function.call(input, output, tt.config); err == nil {
					t.Fatal("invalid geometry accepted")
				}
				if values, err = output.Float32Values(); err != nil || !slices.Equal(values, []float32{91}) {
					t.Fatalf("error changed destination: values=%v error=%v", values, err)
				}
			})
		}
	}
}

func Test_GlobalAveragePoolShape(t *testing.T) {
	type testcase struct {
		name        string
		input, want []int64
		invalid     bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "spatial", input: []int64{2, 3, 4, 5}, want: []int64{2, 3, 1, 1}},
			{name: "empty spatial", input: []int64{2, 3, 0, 0}, want: []int64{2, 3, 1, 1}},
			{name: "empty batch", input: []int64{0, 3, 4, 5}, want: []int64{0, 3, 1, 1}},
			{name: "scalar", invalid: true},
			{name: "rank two", input: []int64{2, 3}, invalid: true},
			{name: "negative", input: []int64{0, 1, -1, 2}, invalid: true},
			{name: "output bytes overflow", input: []int64{1, limit/4 + 1, 0, 0}, invalid: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var shape, err = tensor.GlobalAveragePoolShape(tt.input)
			if (err != nil) != tt.invalid || !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v error=%v, want %v invalid=%v", shape, err, tt.want, tt.invalid)
			}
			if !tt.invalid {
				shape[0] = 99
				if tt.input[0] == 99 {
					t.Fatal("planner returned aliased input shape")
				}
			}
		})
	}
}

func Test_Pooling_ArgumentErrors(t *testing.T) {
	var (
		config    = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		functions = []struct {
			name  string
			shape []int64
			call  func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{"max", []int64{1, 1, 2, 2}, func(input, output *tensor.Tensor) (err error) { return tensor.MaxPool2DInto(input, output, config) }},
			{"average", []int64{1, 1, 2, 2}, func(input, output *tensor.Tensor) (err error) { return tensor.AveragePool2DInto(input, output, config) }},
			{"global", []int64{1, 1, 1, 1}, tensor.GlobalAveragePoolInto},
		}
		cases = []string{"nil input", "zero input", "integer input", "scalar input", "nil destination", "zero destination", "integer destination", "wrong destination shape"}
	)
	for _, function := range functions {
		for _, name := range cases {
			t.Run(function.name+"/"+name, func(t *testing.T) {
				var (
					input, output     *tensor.Tensor
					zero              tensor.Tensor
					sentinels, values []float32
					integers          []int64
					err               error
				)
				if input, err = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4}); err != nil {
					t.Fatal(err)
				}
				sentinels = []float32{91}
				if function.name != "global" {
					sentinels = []float32{91, 91, 91, 91}
				}
				if output, err = tensor.FromFloat32(function.shape, sentinels); err != nil {
					t.Fatal(err)
				}
				switch name {
				case "nil input":
					input = nil
				case "zero input":
					input = &zero
				case "integer input":
					input, err = tensor.FromInt64([]int64{1, 1, 2, 2}, []int64{1, 2, 3, 4})
				case "scalar input":
					input, err = tensor.FromFloat32(nil, []float32{1})
				case "nil destination":
					output = nil
				case "zero destination":
					output = &zero
				case "integer destination":
					integers = make([]int64, len(sentinels))
					for index := range integers {
						integers[index] = 91
					}
					output, err = tensor.FromInt64(function.shape, integers)
				case "wrong destination shape":
					output, err = tensor.FromFloat32([]int64{int64(len(sentinels))}, sentinels)
				}
				if err != nil {
					t.Fatal(err)
				}
				if err = function.call(input, output); err == nil {
					t.Fatal("invalid argument accepted")
				}
				if output.Validate() != nil {
					return
				}
				if output.DataType() == tensor.Int64 {
					var got []int64
					if got, err = output.Int64Values(); err != nil || !slices.Equal(got, integers) {
						t.Fatalf("error changed integer destination: values=%v error=%v", got, err)
					}
					return
				}
				if values, err = output.Float32Values(); err != nil || !slices.Equal(values, sentinels) {
					t.Fatalf("error changed destination: values=%v error=%v", values, err)
				}
			})
		}
	}
}

func Test_Pooling_Aliases(t *testing.T) {
	var (
		config    = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		functions = []struct {
			name string
			call func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{"max", func(input, output *tensor.Tensor) (err error) { return tensor.MaxPool2DInto(input, output, config) }},
			{"average", func(input, output *tensor.Tensor) (err error) { return tensor.AveragePool2DInto(input, output, config) }},
			{"global", tensor.GlobalAveragePoolInto},
		}
	)
	for _, function := range functions {
		for _, empty := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/empty=%v", function.name, empty), func(t *testing.T) {
				var (
					shape   = []int64{1, 1, 1, 1}
					want    = []float32{7}
					input   *tensor.Tensor
					shallow tensor.Tensor
					values  []float32
					err     error
				)
				if empty {
					shape[0], want = 0, nil
				}
				if input, err = tensor.FromFloat32(shape, want); err != nil {
					t.Fatal(err)
				}
				shallow = *input
				for _, alias := range []*tensor.Tensor{input, &shallow} {
					if err = function.call(input, alias); err == nil {
						t.Fatal("destination alias accepted")
					}
				}
				if values, err = input.Float32Values(); err != nil || !slices.Equal(values, want) {
					t.Fatalf("alias failure changed input: values=%v error=%v", values, err)
				}
			})
		}
	}
}

func Test_Pooling_EmptyAndDivisorLimits(t *testing.T) {
	var (
		limit         = int64(int(^uint(0) >> 1))
		input, output *tensor.Tensor
		values        []float32
		err           error
		config        = tensor.Pool2DConfig{Kernel: [2]int64{limit/2 + 1, 3}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{limit / 2, 2, 1, 1}}
	)
	if input, err = tensor.New(tensor.Float32, []int64{1, 1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{91}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.AveragePool2DInto(input, output, config); err == nil {
		t.Fatal("unrepresentable average kernel-area divisor accepted")
	}
	if values, err = output.Float32Values(); err != nil || !slices.Equal(values, []float32{91}) {
		t.Fatalf("divisor failure changed output: values=%v error=%v", values, err)
	}
	if err = tensor.MaxPool2DInto(input, output, config); err != nil {
		t.Fatal(err)
	}
	if values, err = output.Float32Values(); err != nil || len(values) != 1 || !math.IsInf(float64(values[0]), -1) {
		t.Fatalf("all-padding max=%v error=%v, want -Inf", values, err)
	}
	config.Kernel, config.Pads = [2]int64{limit, limit}, [4]int64{}
	if input, err = tensor.New(tensor.Float32, []int64{0, limit, limit, limit}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, []int64{0, limit, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.MaxPool2DInto(input, output, config); err != nil {
		t.Fatal(err)
	}
	if err = tensor.AveragePool2DInto(input, output, config); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GlobalAveragePoolInto(input, output); err != nil {
		t.Fatal(err)
	}
	config = tensor.Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{2, 2}}
	if input, err = tensor.New(tensor.Float32, []int64{0, 1, 3, 3}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, []int64{0, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.AveragePool2DInto(input, output, config); err == nil {
		t.Fatal("nonunit average dilation accepted for empty output")
	}
}

func Test_Pooling_HugeCeilPadding(t *testing.T) {
	var (
		limit         = int64(int(^uint(0) >> 1))
		config        = tensor.Pool2DConfig{Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{0, 0, limit - 1, limit - 1}, CeilMode: true, CountIncludePad: true}
		input, output *tensor.Tensor
		values        []float32
		err           error
	)
	if input, err = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{-3}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, []int64{1, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	for _, call := range []func(*tensor.Tensor, *tensor.Tensor, tensor.Pool2DConfig) error{tensor.MaxPool2DInto, tensor.AveragePool2DInto} {
		if err = call(input, output, config); err != nil {
			t.Fatal(err)
		}
		if values, err = output.Float32Values(); err != nil || !slices.Equal(values, []float32{-3}) {
			t.Fatalf("huge ceil padding values=%v error=%v, want [-3]", values, err)
		}
	}
}

func Test_Pooling_ExistingLayers(t *testing.T) {
	var (
		values      = make([]float32, 80)
		input       *tensor.Tensor
		inputMatrix *matrix.Matrix
		spatial     layer.SpatialShape
		err         error
	)
	for index := range values {
		values[index] = float32(index%13-9) / 7
	}
	if input, err = tensor.FromFloat32([]int64{2, 2, 4, 5}, values); err != nil {
		t.Fatal(err)
	}
	if inputMatrix, err = matrix.FromSlice(2, 40, values); err != nil {
		t.Fatal(err)
	}
	if spatial, err = layer.NewSpatialShape(2, 4, 5); err != nil {
		t.Fatal(err)
	}
	for _, operation := range []string{"max", "average", "global"} {
		t.Run(operation, func(t *testing.T) {
			var (
				function      layer.Layer
				maxConfig     layer.MaxPool2DConfig
				averageConfig layer.AveragePool2DConfig
				globalConfig  layer.AdaptiveAveragePool2DConfig
				config        = tensor.Pool2DConfig{Kernel: [2]int64{2, 3}, Strides: [2]int64{1, 2}, Dilations: [2]int64{1, 1}}
				output        *tensor.Tensor
				outputMatrix  *matrix.Matrix
				shape         []int64
				got, want     []float32
				caseErr       error
			)
			switch operation {
			case "max":
				if maxConfig, caseErr = layer.NewMaxPool2DConfig(spatial, 2, 3, 1, 2); caseErr != nil {
					t.Fatal(caseErr)
				}
				function, caseErr = layer.NewMaxPool2D(maxConfig)
			case "average":
				if averageConfig, caseErr = layer.NewAveragePool2DConfig(spatial, 2, 3, 1, 2); caseErr != nil {
					t.Fatal(caseErr)
				}
				function, caseErr = layer.NewAveragePool2D(averageConfig)
			case "global":
				if globalConfig, caseErr = layer.NewAdaptiveAveragePool2DConfig(spatial, 1, 1); caseErr != nil {
					t.Fatal(caseErr)
				}
				function, caseErr = layer.NewAdaptiveAveragePool2D(globalConfig)
			}
			if caseErr != nil {
				t.Fatal(caseErr)
			}
			if outputMatrix, caseErr = function.Forward(inputMatrix); caseErr != nil {
				t.Fatal(caseErr)
			}
			if want, caseErr = outputMatrix.Values(); caseErr != nil {
				t.Fatal(caseErr)
			}
			if operation == "global" {
				shape, caseErr = tensor.GlobalAveragePoolShape(input.Shape())
			} else {
				shape, caseErr = tensor.Pool2DShape(input.Shape(), config)
			}
			if caseErr != nil {
				t.Fatal(caseErr)
			}
			if output, caseErr = tensor.New(tensor.Float32, shape); caseErr != nil {
				t.Fatal(caseErr)
			}
			switch operation {
			case "max":
				caseErr = tensor.MaxPool2DInto(input, output, config)
			case "average":
				caseErr = tensor.AveragePool2DInto(input, output, config)
			case "global":
				caseErr = tensor.GlobalAveragePoolInto(input, output)
			}
			if caseErr != nil {
				t.Fatal(caseErr)
			}
			if got, caseErr = output.Float32Values(); caseErr != nil {
				t.Fatal(caseErr)
			}
			checkProductValues(t, got, want)
		})
	}
}

func Test_Pooling_OutputLifetimeAndConcurrency(t *testing.T) {
	var (
		values    = make([]float32, 36)
		config    = tensor.Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
		input     *tensor.Tensor
		err       error
		functions = []struct {
			name  string
			shape []int64
			call  func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{"max", []int64{2, 2, 2, 2}, func(source, destination *tensor.Tensor) (callErr error) {
				return tensor.MaxPool2DInto(source, destination, config)
			}},
			{"average", []int64{2, 2, 2, 2}, func(source, destination *tensor.Tensor) (callErr error) {
				return tensor.AveragePool2DInto(source, destination, config)
			}},
			{"global", []int64{2, 2, 1, 1}, tensor.GlobalAveragePoolInto},
		}
	)
	for index := range values {
		values[index] = float32(index-9) / 7
	}
	if input, err = tensor.FromFloat32([]int64{2, 2, 3, 3}, values); err != nil {
		t.Fatal(err)
	}
	for _, function := range functions {
		t.Run(function.name, func(t *testing.T) {
			var (
				first, second *tensor.Tensor
				want, got     []float32
				wait          sync.WaitGroup
				failures      = make(chan error, 4)
				caseErr       error
			)
			if first, caseErr = tensor.New(tensor.Float32, function.shape); caseErr != nil {
				t.Fatal(caseErr)
			}
			if second, caseErr = tensor.New(tensor.Float32, function.shape); caseErr != nil {
				t.Fatal(caseErr)
			}
			if caseErr = function.call(input, first); caseErr != nil {
				t.Fatal(caseErr)
			}
			if want, caseErr = first.Float32Values(); caseErr != nil {
				t.Fatal(caseErr)
			}
			if caseErr = function.call(input, second); caseErr != nil {
				t.Fatal(caseErr)
			}
			if got, caseErr = first.Float32Values(); caseErr != nil || !slices.Equal(got, want) {
				t.Fatalf("later call changed first output: values=%v error=%v", got, caseErr)
			}
			for worker := 0; worker < 4; worker++ {
				wait.Add(1)
				go func() {
					defer wait.Done()
					var (
						output    *tensor.Tensor
						result    []float32
						workerErr error
					)
					if output, workerErr = tensor.New(tensor.Float32, function.shape); workerErr != nil {
						failures <- workerErr
						return
					}
					for iteration := 0; iteration < 8; iteration++ {
						if workerErr = function.call(input, output); workerErr != nil {
							failures <- workerErr
							return
						}
						if result, workerErr = output.Float32Values(); workerErr != nil || !slices.Equal(result, want) {
							failures <- fmt.Errorf("concurrent result=%v error=%v, want %v", result, workerErr, want)
							return
						}
					}
				}()
			}
			wait.Wait()
			close(failures)
			for caseErr = range failures {
				t.Error(caseErr)
			}
			if got, caseErr = input.Float32Values(); caseErr != nil || !slices.Equal(got, values) {
				t.Fatalf("concurrent calls changed input: values=%v error=%v", got, caseErr)
			}
		})
	}
}
