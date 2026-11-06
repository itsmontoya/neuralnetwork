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

func Test_BatchNormalization_Reference(t *testing.T) {
	type fixture struct {
		Name            string   `json:"name"`
		InputShape      []int64  `json:"input_shape"`
		InputBits       []uint32 `json:"input_bits"`
		ScaleBits       []uint32 `json:"scale_bits"`
		BiasBits        []uint32 `json:"bias_bits"`
		MeanBits        []uint32 `json:"mean_bits"`
		VarianceBits    []uint32 `json:"variance_bits"`
		EpsilonBits     uint32   `json:"epsilon_bits"`
		OutputBits      []uint32 `json:"output_bits"`
		ReferenceResult string   `json:"reference_result"`
		ReferenceError  string   `json:"reference_error"`
		ReferenceShape  []int64  `json:"reference_shape"`
		ReferenceBits   []uint32 `json:"reference_bits"`
		ORTResult       string   `json:"ort_result"`
		ORTError        string   `json:"ort_error"`
		ORTShape        []int64  `json:"ort_shape"`
		ORTBits         []uint32 `json:"ort_bits"`
	}
	var (
		contents                           []byte
		fixtures                           []fixture
		referenceAgreements, ortAgreements int
		err                                error
	)

	if contents, err = os.ReadFile("testdata/normalization.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 31 {
		t.Fatalf("fixture count=%d, want 31", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				arguments  = make([]*tensor.Tensor, 5)
				inputBits  = [][]uint32{tt.InputBits, tt.ScaleBits, tt.BiasBits, tt.MeanBits, tt.VarianceBits}
				output     *tensor.Tensor
				shape      []int64
				values     []float32
				fixtureErr error
			)

			for index := range arguments {
				shape = []int64{tt.InputShape[1]}
				if index == 0 {
					shape = tt.InputShape
				}
				if arguments[index], fixtureErr = tensor.FromFloat32(shape, valuesFromBits(inputBits[index])); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
			}
			if output, fixtureErr = tensor.New(tensor.Float32, tt.InputShape); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if fixtureErr = tensor.BatchNormalizationInto(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], output, math.Float32frombits(tt.EpsilonBits)); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if !slices.Equal(output.Shape(), tt.InputShape) {
				t.Fatalf("shape=%v, want %v", output.Shape(), tt.InputShape)
			}
			if values, fixtureErr = output.Float32Values(); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			checkProductValues(t, values, valuesFromBits(tt.OutputBits))
			for index, input := range arguments {
				if values, fixtureErr = input.Float32Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				checkFloat32Bits(t, values, inputBits[index])
			}
			if tt.ReferenceError != "" || !slices.Equal(tt.ReferenceShape, tt.InputShape) {
				t.Fatalf("unexpected evaluator shape/error: %v %q", tt.ReferenceShape, tt.ReferenceError)
			}
			if tt.ReferenceResult == "agree" {
				checkProductValues(t, valuesFromBits(tt.ReferenceBits), valuesFromBits(tt.OutputBits))
				referenceAgreements++
			} else if tt.ReferenceResult != "different" || len(tt.ReferenceBits) != len(tt.OutputBits) {
				t.Fatalf("unexpected evaluator result=%q", tt.ReferenceResult)
			}
			if tt.ORTError != "" || !slices.Equal(tt.ORTShape, tt.InputShape) {
				t.Fatalf("unexpected ORT shape/error: %v %q", tt.ORTShape, tt.ORTError)
			}
			if tt.ORTResult == "agree" {
				checkProductValues(t, valuesFromBits(tt.ORTBits), valuesFromBits(tt.OutputBits))
				ortAgreements++
			} else if tt.ORTResult != "different" || tt.Name != "zero variance and zero epsilon" {
				t.Fatalf("unexpected ORT result=%q for %s", tt.ORTResult, tt.Name)
			}
		})
	}
	if referenceAgreements != 8 || ortAgreements != 30 {
		t.Fatalf("reference agreements: evaluator=%d ORT=%d, want 8 and 30", referenceAgreements, ortAgreements)
	}
}

func Test_BatchNormalization_LiteralAndDomainValues(t *testing.T) {
	type testcase struct {
		name     string
		input    []float32
		scale    float32
		bias     float32
		mean     float32
		variance float32
		want     []float32
	}
	var tests = []testcase{
		{name: "learned statistics", input: []float32{3, 5}, scale: 2, bias: 0.5, mean: 1, variance: 4, want: []float32{2.5, 4.5}},
		{name: "zero divisor", input: []float32{-1, 0, 1}, scale: 1, want: []float32{float32(math.Inf(-1)), float32(math.NaN()), float32(math.Inf(1))}},
		{name: "negative variance", input: []float32{1, 2}, scale: 1, variance: -1, want: []float32{float32(math.NaN()), float32(math.NaN())}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				parameters    = make([]*tensor.Tensor, 4)
				values        []float32
				err           error
			)
			if input, err = tensor.FromFloat32([]int64{int64(len(tt.input)), 1}, tt.input); err != nil {
				t.Fatal(err)
			}
			for index, value := range []float32{tt.scale, tt.bias, tt.mean, tt.variance} {
				if parameters[index], err = tensor.FromFloat32([]int64{1}, []float32{value}); err != nil {
					t.Fatal(err)
				}
			}
			if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
				t.Fatal(err)
			}
			if err = tensor.BatchNormalizationInto(input, parameters[0], parameters[1], parameters[2], parameters[3], output, 0); err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, values, tt.want)
		})
	}
}

func Test_BatchNormalization_ArgumentErrors(t *testing.T) {
	var names = []string{"input", "scale", "bias", "mean", "variance", "destination"}
	for argument, name := range names {
		var changes = []string{"nil", "zero", "type", "shape", "scalar"}
		if argument > 0 && argument < 5 {
			changes = append(changes, "short")
		}
		for _, change := range changes {
			t.Run(name+"/"+change, func(t *testing.T) {
				var (
					arguments   = make([]*tensor.Tensor, 6)
					shapes      = [][]int64{{2, 2}, {2}, {2}, {2}, {2}, {2, 2}}
					inputs      = [][]float32{{3, 8, 5, 11}, {2, 3}, {0.5, -1}, {1, 2}, {4, 9}, {19, 23, 29, 31}}
					zero        tensor.Tensor
					before      []float32
					beforeInt   []int64
					beforeShape []int64
					after       []float32
					afterInt    []int64
					err         error
				)
				for index := range arguments {
					if arguments[index], err = tensor.FromFloat32(shapes[index], inputs[index]); err != nil {
						t.Fatal(err)
					}
				}
				switch change {
				case "nil":
					arguments[argument] = nil
				case "zero":
					arguments[argument] = &zero
				case "type":
					arguments[argument], err = tensor.New(tensor.Int64, shapes[argument])
				case "shape":
					shapes[argument] = []int64{1, 2}
					if argument == 0 || argument == 5 {
						shapes[argument] = []int64{4}
					}
					arguments[argument], err = tensor.FromFloat32(shapes[argument], inputs[argument])
				case "scalar":
					arguments[argument], err = tensor.FromFloat32(nil, []float32{19})
				case "short":
					arguments[argument], err = tensor.FromFloat32([]int64{1}, []float32{19})
				}
				if err != nil {
					t.Fatal(err)
				}
				beforeShape = arguments[5].Shape()
				if arguments[5].DataType() == tensor.Float32 {
					if before, err = arguments[5].Float32Values(); err != nil {
						t.Fatal(err)
					}
				} else if arguments[5].DataType() == tensor.Int64 {
					if beforeInt, err = arguments[5].Int64Values(); err != nil {
						t.Fatal(err)
					}
				}
				if err = tensor.BatchNormalizationInto(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], 0.125); err == nil {
					t.Fatal("invalid argument succeeded")
				}
				if !slices.Equal(arguments[5].Shape(), beforeShape) {
					t.Fatal("destination shape changed on error")
				}
				if arguments[5].DataType() == tensor.Float32 {
					if after, err = arguments[5].Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(before, after) {
						t.Fatalf("destination changed: before=%v after=%v", before, after)
					}
				} else if arguments[5].DataType() == tensor.Int64 {
					if afterInt, err = arguments[5].Int64Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(beforeInt, afterInt) {
						t.Fatal("integer destination changed on error")
					}
				}
			})
		}
	}
}

func Test_BatchNormalization_EpsilonErrors(t *testing.T) {
	var epsilons = []float32{-1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	for _, epsilon := range epsilons {
		t.Run(fmt.Sprint(epsilon), func(t *testing.T) {
			var (
				input, output, parameter *tensor.Tensor
				values                   []float32
				err                      error
			)
			if input, err = tensor.FromFloat32([]int64{1, 1}, []float32{3}); err != nil {
				t.Fatal(err)
			}
			if parameter, err = tensor.FromFloat32([]int64{1}, []float32{1}); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromFloat32(input.Shape(), []float32{19}); err != nil {
				t.Fatal(err)
			}
			if err = tensor.BatchNormalizationInto(input, parameter, parameter, parameter, parameter, output, epsilon); err == nil {
				t.Fatal("invalid epsilon succeeded")
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, []float32{19}) {
				t.Fatalf("destination changed: %v", values)
			}
		})
	}
}

func Test_BatchNormalization_EmptyValidation(t *testing.T) {
	var (
		limit  = int64(int(^uint(0) >> 1))
		shapes = [][]int64{{0, 1, limit, limit}, {1, 1, 0, limit, limit}, {1, 0, limit, limit}}
	)
	for _, shape := range shapes {
		for _, change := range []string{"valid", "nil scale", "nil bias", "nil mean", "nil variance", "wrong variance", "bad epsilon", "alias"} {
			t.Run(fmt.Sprint(shape)+"/"+change, func(t *testing.T) {
				var (
					input, output *tensor.Tensor
					parameters            = make([]*tensor.Tensor, 4)
					epsilon       float32 = 0.125
					err           error
				)
				if input, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				for index := range parameters {
					if parameters[index], err = tensor.New(tensor.Float32, []int64{shape[1]}); err != nil {
						t.Fatal(err)
					}
				}
				switch change {
				case "nil scale":
					parameters[0] = nil
				case "nil bias":
					parameters[1] = nil
				case "nil mean":
					parameters[2] = nil
				case "nil variance":
					parameters[3] = nil
				case "wrong variance":
					parameters[3], err = tensor.New(tensor.Float32, []int64{1, shape[1]})
				case "bad epsilon":
					epsilon = float32(math.NaN())
				case "alias":
					output = input
				}
				if err != nil {
					t.Fatal(err)
				}
				err = tensor.BatchNormalizationInto(input, parameters[0], parameters[1], parameters[2], parameters[3], output, epsilon)
				if (err != nil) != (change != "valid") {
					t.Fatalf("error=%v, change=%s", err, change)
				}
			})
		}
	}
}

func Test_BatchNormalization_InputAliases(t *testing.T) {
	for _, shape := range [][]int64{{1, 2}, {0, 2}, {1, 2, 0, 3}} {
		for _, shallow := range []bool{false, true} {
			t.Run(fmt.Sprint(shape, shallow), func(t *testing.T) {
				var (
					input, output, parameter *tensor.Tensor
					copied                   tensor.Tensor
					before, after            []float32
					err                      error
				)
				if input, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				if parameter, err = tensor.FromFloat32([]int64{2}, []float32{1, 2}); err != nil {
					t.Fatal(err)
				}
				output = input
				if shallow {
					copied = *input
					output = &copied
				}
				if before, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				if err = tensor.BatchNormalizationInto(input, parameter, parameter, parameter, parameter, output, 0.125); err == nil {
					t.Fatal("destination alias succeeded")
				}
				if after, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(before, after) {
					t.Fatal("input alias changed on error")
				}
			})
		}
	}
}

func Test_BatchNormalization_ExistingLayers(t *testing.T) {
	for _, spatial := range []bool{false, true} {
		t.Run(fmt.Sprint("spatial=", spatial), func(t *testing.T) {
			var (
				featureLayer              *layer.BatchNormalization
				spatialLayer              *layer.BatchNormalization2D
				forward                   layer.Layer
				spatialShape              layer.SpatialShape
				shape                     = []int64{3, 2}
				inputValues               = []float32{3, 8, 5, 11, -1, 2}
				parameters                = [][]float32{{2, 3}, {0.5, -1}, {1, 2}, {4, 9}}
				parameterTensors          = make([]*tensor.Tensor, 4)
				state                     []*matrix.Matrix
				input, output             *tensor.Tensor
				matrixInput, matrixOutput *matrix.Matrix
				got, want, after          []float32
				err                       error
			)

			if spatial {
				shape = []int64{2, 2, 2, 2}
				inputValues = []float32{3, 5, -1, 2, 8, 11, 2, -4, 9, 1, 4, 7, -2, 3, 12, 6}
				if spatialShape, err = layer.NewSpatialShape(2, 2, 2); err != nil {
					t.Fatal(err)
				}
				if spatialLayer, err = layer.NewBatchNormalization2D(layer.BatchNormalization2DConfig{InputShape: spatialShape, Epsilon: 0.25, Momentum: 0.8}); err != nil {
					t.Fatal(err)
				}
				spatialLayer.SetTraining(false)
				forward = spatialLayer
				state = []*matrix.Matrix{spatialLayer.Gamma().Values(), spatialLayer.Beta().Values(), spatialLayer.RunningMean(), spatialLayer.RunningVariance(), spatialLayer.Gamma().Gradient(), spatialLayer.Beta().Gradient()}
			} else {
				if featureLayer, err = layer.NewBatchNormalizationWithConfig(2, 0.8, 0.25); err != nil {
					t.Fatal(err)
				}
				featureLayer.SetTraining(false)
				forward = featureLayer
				state = []*matrix.Matrix{featureLayer.Gamma().Values(), featureLayer.Beta().Values(), featureLayer.RunningMean(), featureLayer.RunningVariance(), featureLayer.Gamma().Gradient(), featureLayer.Beta().Gradient()}
			}
			for index := range parameters {
				if parameterTensors[index], err = tensor.FromFloat32([]int64{2}, parameters[index]); err != nil {
					t.Fatal(err)
				}
				if err = state[index].CopyValuesFrom(parameters[index]); err != nil {
					t.Fatal(err)
				}
			}
			if input, err = tensor.FromFloat32(shape, inputValues); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, shape); err != nil {
				t.Fatal(err)
			}
			if err = tensor.BatchNormalizationInto(input, parameterTensors[0], parameterTensors[1], parameterTensors[2], parameterTensors[3], output, 0.25); err != nil {
				t.Fatal(err)
			}
			if matrixInput, err = matrix.FromSlice(int(shape[0]), len(inputValues)/int(shape[0]), inputValues); err != nil {
				t.Fatal(err)
			}
			if matrixOutput, err = forward.Forward(matrixInput); err != nil {
				t.Fatal(err)
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if want, err = matrixOutput.Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, got, want)
			for index, value := range state {
				if after, err = value.Values(); err != nil {
					t.Fatal(err)
				}
				want = []float32{0, 0}
				if index < len(parameters) {
					want = parameters[index]
				}
				if !slices.Equal(after, want) {
					t.Fatalf("layer state %d changed: got %v want %v", index, after, want)
				}
			}
			if after, err = matrixInput.Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(after, inputValues) {
				t.Fatal("matrix input changed")
			}
			if after, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(after, inputValues) {
				t.Fatal("tensor input changed")
			}
			if spatial && spatialLayer.Training() || !spatial && featureLayer.Training() {
				t.Fatal("evaluation mode changed")
			}
		})
	}
}

func Test_BatchNormalization_OutputLifetimeAndConcurrency(t *testing.T) {
	var (
		input, parameter, first, later *tensor.Tensor
		want, values                   []float32
		group                          sync.WaitGroup
		failures                       = make(chan error, 8)
		err                            error
	)
	if input, err = tensor.FromFloat32([]int64{2, 2}, []float32{3, 8, 5, 11}); err != nil {
		t.Fatal(err)
	}
	if parameter, err = tensor.FromFloat32([]int64{2}, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if first, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
		t.Fatal(err)
	}
	if err = tensor.BatchNormalizationInto(input, parameter, parameter, parameter, parameter, first, 0.125); err != nil {
		t.Fatal(err)
	}
	if want, err = first.Float32Values(); err != nil {
		t.Fatal(err)
	}
	for worker := 0; worker < 8; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			var (
				output    *tensor.Tensor
				actual    []float32
				workerErr error
			)
			if output, workerErr = tensor.New(tensor.Float32, input.Shape()); workerErr != nil {
				failures <- workerErr
				return
			}
			for repeat := 0; repeat < 16; repeat++ {
				if workerErr = tensor.BatchNormalizationInto(input, parameter, parameter, parameter, parameter, output, 0.125); workerErr != nil {
					failures <- workerErr
					return
				}
				if actual, workerErr = output.Float32Values(); workerErr != nil {
					failures <- workerErr
					return
				}
				if !slices.Equal(actual, want) {
					failures <- fmt.Errorf("concurrent output=%v, want %v", actual, want)
					return
				}
			}
		}()
	}
	group.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
	if later, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
		t.Fatal(err)
	}
	if err = tensor.BatchNormalizationInto(input, parameter, parameter, parameter, parameter, later, 2.5); err != nil {
		t.Fatal(err)
	}
	if values, err = first.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, want) {
		t.Fatal("later calls changed previously returned output")
	}
	if values, err = parameter.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, []float32{1, 2}) {
		t.Fatal("shared parameters changed")
	}
	if values, err = input.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, []float32{3, 8, 5, 11}) {
		t.Fatal("shared input changed")
	}
}
