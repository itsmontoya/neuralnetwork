package tensor_test

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Conv2D_Reference(t *testing.T) {
	type fixture struct {
		Name           string   `json:"name"`
		InputShape     []int64  `json:"input_shape"`
		InputBits      []uint32 `json:"input_bits"`
		WeightsShape   []int64  `json:"weights_shape"`
		WeightsBits    []uint32 `json:"weights_bits"`
		HasBias        bool     `json:"has_bias"`
		BiasBits       []uint32 `json:"bias_bits"`
		Strides        [2]int64 `json:"strides"`
		Dilations      [2]int64 `json:"dilations"`
		Pads           [4]int64 `json:"pads"`
		Groups         int64    `json:"groups"`
		OutputShape    []int64  `json:"output_shape"`
		OutputBits     []uint32 `json:"output_bits"`
		Reference      string   `json:"reference"`
		ReferenceError string   `json:"reference_error"`
	}
	var (
		contents    []byte
		fixtures    []fixture
		differences int
		err         error
	)

	if contents, err = os.ReadFile("testdata/convolution.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 46 {
		t.Fatalf("fixture count=%d, want 46", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				input, weights, bias, output *tensor.Tensor
				shape                        []int64
				values                       []float32
				fixtureErr                   error
				config                       tensor.Conv2DConfig
			)

			config.Strides, config.Dilations = tt.Strides, tt.Dilations
			config.Pads, config.Groups = tt.Pads, tt.Groups
			if input, fixtureErr = tensor.FromFloat32(tt.InputShape, valuesFromBits(tt.InputBits)); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if weights, fixtureErr = tensor.FromFloat32(tt.WeightsShape, valuesFromBits(tt.WeightsBits)); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if tt.HasBias {
				if bias, fixtureErr = tensor.FromFloat32([]int64{tt.WeightsShape[0]}, valuesFromBits(tt.BiasBits)); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
			}
			if shape, fixtureErr = tensor.Conv2DShape(input.Shape(), weights.Shape(), config); fixtureErr != nil || !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("shape=%v error=%v, want %v", shape, fixtureErr, tt.OutputShape)
			}
			if output, fixtureErr = tensor.New(tensor.Float32, shape); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if fixtureErr = tensor.Conv2DInto(input, weights, bias, output, config); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if values, fixtureErr = output.Float32Values(); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			checkProductValues(t, values, valuesFromBits(tt.OutputBits))
			for index, source := range []*tensor.Tensor{input, weights, bias} {
				if source == nil {
					continue
				}
				if values, fixtureErr = source.Float32Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				checkFloat32Bits(t, values, [][]uint32{tt.InputBits, tt.WeightsBits, tt.BiasBits}[index])
			}
			if tt.Reference == "numpy" {
				if tt.ReferenceError == "" {
					t.Fatal("NumPy-only case must record the pinned evaluator rejection")
				}
				differences++
			} else if tt.Reference != "onnx" || tt.ReferenceError != "" {
				t.Fatalf("unexpected reference=%q error=%q", tt.Reference, tt.ReferenceError)
			}
		})
	}
	if differences != 6 {
		t.Fatalf("documented evaluator rejections=%d, want 6", differences)
	}
}

func Test_Conv2DShape(t *testing.T) {
	type testcase struct {
		name    string
		input   []int64
		weights []int64
		config  tensor.Conv2DConfig
		want    []int64
		invalid bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		valid = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		tests = []testcase{
			{name: "rectangular", input: []int64{1, 2, 4, 5}, weights: []int64{3, 2, 2, 3}, config: valid, want: []int64{1, 3, 3, 3}},
			{name: "empty oversized kernel", input: []int64{1, 1, 1, 1}, weights: []int64{1, 1, 4, 3}, config: valid, want: []int64{1, 1, 0, 0}},
			{name: "huge unused empty batch", input: []int64{0, 1, limit, 1}, weights: []int64{1, 1, 1, 1}, config: valid, want: []int64{0, 1, limit, 1}},
			{name: "scalar input", weights: []int64{1, 1, 1, 1}, config: valid, invalid: true},
			{name: "rank three weights", input: []int64{1, 1, 2, 2}, weights: []int64{1, 1, 1}, config: valid, invalid: true},
			{name: "negative input", input: []int64{1, 1, -1, 2}, weights: []int64{1, 1, 1, 1}, config: valid, invalid: true},
			{name: "zero channels", input: []int64{1, 0, 2, 2}, weights: []int64{1, 0, 1, 1}, config: valid, invalid: true},
			{name: "zero filters", input: []int64{1, 1, 2, 2}, weights: []int64{0, 1, 1, 1}, config: valid, invalid: true},
			{name: "zero kernel", input: []int64{1, 1, 2, 2}, weights: []int64{1, 1, 0, 1}, config: valid, invalid: true},
			{name: "weight channels mismatch", input: []int64{1, 2, 2, 2}, weights: []int64{1, 1, 1, 1}, config: valid, invalid: true},
			{name: "input payload overflow", input: []int64{1, 1, limit, 1}, weights: []int64{1, 1, 1, 1}, config: valid, invalid: true},
			{name: "weight payload overflow", input: []int64{0, 1, 1, 1}, weights: []int64{limit, 1, 1, 1}, config: valid, invalid: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape        []int64
				inputBefore  = slices.Clone(tt.input)
				weightBefore = slices.Clone(tt.weights)
				err          error
			)
			shape, err = tensor.Conv2DShape(tt.input, tt.weights, tt.config)
			if (err != nil) != tt.invalid || !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v error=%v, want %v invalid=%v", shape, err, tt.want, tt.invalid)
			}
			if !slices.Equal(tt.input, inputBefore) || !slices.Equal(tt.weights, weightBefore) {
				t.Fatal("shape planner mutated an argument")
			}
			if !tt.invalid && len(shape) > 0 {
				shape[0] = 99
				if !slices.Equal(tt.input, inputBefore) || !slices.Equal(tt.weights, weightBefore) {
					t.Fatal("shape planner returned borrowed metadata")
				}
			}
		})
	}
}

func Test_Conv2D_GeometryErrors(t *testing.T) {
	type testcase struct {
		name    string
		input   []int64
		weights []int64
		config  tensor.Conv2DConfig
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		valid = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		tests = []testcase{
			{name: "zero config"}, {name: "zero groups"}, {name: "negative groups"},
			{name: "input not divisible"}, {name: "filters not divisible"},
			{name: "zero height stride"}, {name: "negative width stride"},
			{name: "zero width dilation"}, {name: "negative height dilation"},
			{name: "negative top pad"}, {name: "negative left pad"},
			{name: "negative bottom pad"}, {name: "negative right pad"},
			{name: "effective kernel overflow"}, {name: "padded extent overflow"},
			{name: "output element overflow"}, {name: "output byte overflow"},
		}
	)
	for index := range tests {
		tests[index].input = []int64{1, 2, 2, 2}
		tests[index].weights = []int64{2, 2, 1, 1}
		tests[index].config = valid
	}
	tests[0].config = tensor.Conv2DConfig{}
	tests[1].config.Groups, tests[2].config.Groups = 0, -1
	tests[3].config.Groups, tests[3].input[1] = 2, 3
	tests[4].config.Groups, tests[4].weights = 2, []int64{3, 1, 1, 1}
	tests[5].config.Strides[0], tests[6].config.Strides[1] = 0, -1
	tests[7].config.Dilations[1], tests[8].config.Dilations[0] = 0, -1
	for index := 0; index < 4; index++ {
		tests[9+index].config.Pads[index] = -1
	}
	tests[13].weights[2], tests[13].config.Dilations[0] = 2, limit
	tests[14].input, tests[14].config.Pads[0] = []int64{0, 2, limit, 1}, 1
	tests[15].input, tests[15].config.Pads[0] = []int64{limit, 2, 0, 1}, 1
	tests[16].input, tests[16].config.Pads[0] = []int64{limit / 8, 2, 0, 1}, 2
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var shape, err = tensor.Conv2DShape(tt.input, tt.weights, tt.config)
			if err == nil || shape != nil {
				t.Fatalf("shape=%v error=%v, want nil shape and error", shape, err)
			}
		})
	}
}

func Test_Conv2D_ArgumentErrors(t *testing.T) {
	type testcase struct {
		name string
	}
	var tests = []testcase{
		{name: "nil input"}, {name: "zero input"}, {name: "integer input"},
		{name: "nil weights"}, {name: "zero weights"}, {name: "integer weights"},
		{name: "zero bias"}, {name: "integer bias"}, {name: "scalar bias"},
		{name: "wrong bias length"}, {name: "wrong bias rank"},
		{name: "nil destination"}, {name: "zero destination"},
		{name: "integer destination"}, {name: "wrong destination shape"},
		{name: "invalid config"}, {name: "invalid bias despite empty batch"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *tensor.Tensor
				floats                       []float32
				integers                     []int64
				zero                         tensor.Tensor
				config                       = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
				err                          error
			)
			input, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
			weights, _ = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{2})
			bias, _ = tensor.FromFloat32([]int64{1}, []float32{1})
			output, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{71, 72, 73, 74})
			switch tt.name {
			case "nil input":
				input = nil
			case "zero input":
				input = &zero
			case "integer input":
				input, _ = tensor.New(tensor.Int64, []int64{1, 1, 2, 2})
			case "nil weights":
				weights = nil
			case "zero weights":
				weights = &zero
			case "integer weights":
				weights, _ = tensor.New(tensor.Int64, []int64{1, 1, 1, 1})
			case "zero bias":
				bias = &zero
			case "integer bias":
				bias, _ = tensor.New(tensor.Int64, []int64{1})
			case "scalar bias":
				bias, _ = tensor.FromFloat32(nil, []float32{1})
			case "wrong bias length":
				bias, _ = tensor.FromFloat32([]int64{2}, []float32{1, 2})
			case "wrong bias rank":
				bias, _ = tensor.FromFloat32([]int64{1, 1}, []float32{1})
			case "nil destination":
				output = nil
			case "zero destination":
				output = &zero
			case "integer destination":
				output, _ = tensor.FromInt64([]int64{1, 1, 2, 2}, []int64{71, 72, 73, 74})
			case "wrong destination shape":
				output, _ = tensor.FromFloat32([]int64{4}, []float32{71, 72, 73, 74})
			case "invalid config":
				config.Strides[0] = 0
			case "invalid bias despite empty batch":
				input, _ = tensor.New(tensor.Float32, []int64{0, 1, 2, 2})
				output, _ = tensor.New(tensor.Float32, []int64{0, 1, 2, 2})
				bias, _ = tensor.FromFloat32(nil, []float32{1})
			}
			if err = tensor.Conv2DInto(input, weights, bias, output, config); err == nil {
				t.Fatal("invalid arguments accepted")
			}
			if output.Validate() != nil || output.Len() == 0 {
				return
			}
			if output.DataType() == tensor.Int64 {
				integers, err = output.Int64Values()
				if err != nil || !slices.Equal(integers, []int64{71, 72, 73, 74}) {
					t.Fatalf("destination changed: %v error=%v", integers, err)
				}
				return
			}
			floats, err = output.Float32Values()
			if err != nil || !slices.Equal(floats, []float32{71, 72, 73, 74}) {
				t.Fatalf("destination changed: %v error=%v", floats, err)
			}
		})
	}
}

func Test_Conv2D_AliasesAndEmptyValues(t *testing.T) {
	var config = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
	for _, empty := range []bool{false, true} {
		for _, shallow := range []bool{false, true} {
			t.Run(fmt.Sprintf("input empty=%v shallow=%v", empty, shallow), func(t *testing.T) {
				var (
					input, weights, output *tensor.Tensor
					copied                 tensor.Tensor
					values                 []float32
					err                    error
				)
				if empty {
					input, _ = tensor.New(tensor.Float32, []int64{0, 1, 2, 2})
				} else {
					input, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
				}
				weights, _ = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{2})
				output = input
				if shallow {
					copied = *input
					output = &copied
				}
				if err = tensor.Conv2DInto(input, weights, nil, output, config); err == nil {
					t.Fatal("aliased input accepted")
				}
				values, _ = input.Float32Values()
				if !empty && !slices.Equal(values, []float32{1, 2, 3, 4}) {
					t.Fatalf("input mutated: %v", values)
				}
			})
		}
	}
	for _, shallow := range []bool{false, true} {
		t.Run(fmt.Sprintf("weights shallow=%v", shallow), func(t *testing.T) {
			var (
				input, weights, output *tensor.Tensor
				copied                 tensor.Tensor
				values                 []float32
				err                    error
			)
			input, _ = tensor.New(tensor.Float32, []int64{1, 1, 3, 3})
			weights, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
			output = weights
			if shallow {
				copied = *weights
				output = &copied
			}
			if err = tensor.Conv2DInto(input, weights, nil, output, config); err == nil {
				t.Fatal("aliased weights accepted")
			}
			values, _ = weights.Float32Values()
			if !slices.Equal(values, []float32{1, 2, 3, 4}) {
				t.Fatalf("weights mutated: %v", values)
			}
		})
	}
	var (
		input, weights, output *tensor.Tensor
		shape                  []int64
		values                 []float32
		limit                  = int64(int(^uint(0) >> 1))
		err                    error
	)
	input, _ = tensor.New(tensor.Float32, []int64{0, 1, limit, 1})
	weights, _ = tensor.FromFloat32([]int64{1, 1, 1, 1}, []float32{2})
	shape, _ = tensor.Conv2DShape(input.Shape(), weights.Shape(), config)
	output, _ = tensor.New(tensor.Float32, shape)
	if err = tensor.Conv2DInto(input, weights, nil, output, config); err != nil {
		t.Fatal(err)
	}
	input, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	output, _ = tensor.New(tensor.Float32, []int64{1, 1, 1, 1})
	if err = tensor.Conv2DInto(input, input, nil, output, config); err != nil {
		t.Fatal(err)
	}
	values, _ = output.Float32Values()
	if !slices.Equal(values, []float32{30}) {
		t.Fatalf("shared read-only input/weights result=%v, want [30]", values)
	}
}

func Test_Conv2D_ExistingLayer(t *testing.T) {
	type testcase struct {
		name                      string
		channels, height, width   int
		filters, kernelH, kernelW int
		strideH, strideW          int
		padH, padW, batches       int
	}
	var tests = []testcase{
		{name: "valid", channels: 2, height: 4, width: 5, filters: 3, kernelH: 2, kernelW: 3, strideH: 1, strideW: 1, batches: 2},
		{name: "padded unequal strides", channels: 2, height: 4, width: 5, filters: 3, kernelH: 3, kernelW: 2, strideH: 2, strideW: 1, padH: 1, padW: 2, batches: 2},
		{name: "pointwise", channels: 3, height: 2, width: 3, filters: 2, kernelH: 1, kernelW: 1, strideH: 1, strideW: 1, batches: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				spatial               layer.SpatialShape
				layerConfig           layer.Conv2DConfig
				conv                  *layer.Conv2D
				inputMatrix, expected *matrix.Matrix
				input, weights, bias  *tensor.Tensor
				output                *tensor.Tensor
				shape                 []int64
				inputValues           = make([]float32, tt.batches*tt.channels*tt.height*tt.width)
				weightValues          = make([]float32, tt.filters*tt.channels*tt.kernelH*tt.kernelW)
				packed                = make([]float32, len(weightValues))
				biasValues            = make([]float32, tt.filters)
				values, want          []float32
				fanIn                 = tt.channels * tt.kernelH * tt.kernelW
				config                tensor.Conv2DConfig
				err                   error
			)
			for index := range inputValues {
				inputValues[index] = float32(index%11-5) / 4
			}
			for index := range weightValues {
				weightValues[index] = float32(index%7-3) / 4
			}
			for channel := 0; channel < tt.filters; channel++ {
				biasValues[channel] = float32(channel-1) / 4
				for index := 0; index < fanIn; index++ {
					packed[index*tt.filters+channel] = weightValues[channel*fanIn+index]
				}
			}
			if spatial, err = layer.NewSpatialShape(tt.channels, tt.height, tt.width); err != nil {
				t.Fatal(err)
			}
			if layerConfig, err = layer.NewConv2DConfig(spatial, tt.filters, tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW); err != nil {
				t.Fatal(err)
			}
			if conv, err = layer.NewConv2D(layerConfig, func(rows, cols int) (out *matrix.Matrix, initErr error) {
				out, initErr = matrix.FromSlice(rows, cols, packed)
				return out, initErr
			}); err != nil {
				t.Fatal(err)
			}
			if err = conv.Biases().Values().CopyValuesFrom(biasValues); err != nil {
				t.Fatal(err)
			}
			if inputMatrix, err = matrix.FromSlice(tt.batches, tt.channels*tt.height*tt.width, inputValues); err != nil {
				t.Fatal(err)
			}
			if expected, err = conv.Forward(inputMatrix); err != nil {
				t.Fatal(err)
			}
			input, _ = tensor.FromFloat32([]int64{int64(tt.batches), int64(tt.channels), int64(tt.height), int64(tt.width)}, inputValues)
			weights, _ = tensor.FromFloat32([]int64{int64(tt.filters), int64(tt.channels), int64(tt.kernelH), int64(tt.kernelW)}, weightValues)
			bias, _ = tensor.FromFloat32([]int64{int64(tt.filters)}, biasValues)
			config.Strides, config.Dilations = [2]int64{int64(tt.strideH), int64(tt.strideW)}, [2]int64{1, 1}
			config.Pads, config.Groups = [4]int64{int64(tt.padH), int64(tt.padW), int64(tt.padH), int64(tt.padW)}, 1
			shape, _ = tensor.Conv2DShape(input.Shape(), weights.Shape(), config)
			output, _ = tensor.New(tensor.Float32, shape)
			if err = tensor.Conv2DInto(input, weights, bias, output, config); err != nil {
				t.Fatal(err)
			}
			values, _ = output.Float32Values()
			want, _ = expected.Values()
			checkProductValues(t, values, want)
		})
	}
}

func Test_Conv2D_DynamicWeightsAndConcurrency(t *testing.T) {
	var (
		input, weights, nextWeights, first, second *tensor.Tensor
		values                                     []float32
		config                                     = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		workers                                    sync.WaitGroup
		results                                    = make(chan error, 4)
		err                                        error
	)
	input, _ = tensor.FromFloat32([]int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	weights, _ = tensor.FromFloat32([]int64{5, 1, 1, 1}, []float32{2, 2, 2, 2, 2})
	nextWeights, _ = tensor.FromFloat32([]int64{5, 1, 1, 1}, []float32{3, 3, 3, 3, 3})
	first, _ = tensor.New(tensor.Float32, []int64{1, 5, 2, 2})
	second, _ = tensor.New(tensor.Float32, []int64{1, 5, 2, 2})
	if err = tensor.Conv2DInto(input, weights, nil, first, config); err != nil {
		t.Fatal(err)
	}
	if err = tensor.Conv2DInto(input, nextWeights, nil, second, config); err != nil {
		t.Fatal(err)
	}
	values, _ = first.Float32Values()
	if !slices.Equal(values, slices.Repeat([]float32{2, 4, 6, 8}, 5)) {
		t.Fatalf("earlier output changed: %v", values)
	}
	values, _ = second.Float32Values()
	if !slices.Equal(values, slices.Repeat([]float32{3, 6, 9, 12}, 5)) {
		t.Fatalf("dynamic weights result=%v", values)
	}
	for worker := 0; worker < 4; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			var (
				output    *tensor.Tensor
				got       []float32
				workerErr error
			)
			if output, workerErr = tensor.New(tensor.Float32, first.Shape()); workerErr != nil {
				results <- workerErr
				return
			}
			for run := 0; run < 10; run++ {
				if workerErr = tensor.Conv2DInto(input, weights, nil, output, config); workerErr != nil {
					results <- workerErr
					return
				}
				got, _ = output.Float32Values()
				if !slices.Equal(got, slices.Repeat([]float32{2, 4, 6, 8}, 5)) {
					results <- fmt.Errorf("concurrent result=%v", got)
					return
				}
			}
		}()
	}
	workers.Wait()
	close(results)
	for err = range results {
		t.Error(err)
	}
}
