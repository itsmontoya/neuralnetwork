package tensor_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Activations_Reference(t *testing.T) {
	type fixture struct {
		Name           string   `json:"name"`
		Operation      string   `json:"operation"`
		Axis           int64    `json:"axis"`
		InputShape     []int64  `json:"input_shape"`
		InputBits      []uint32 `json:"input_bits"`
		OutputShape    []int64  `json:"output_shape"`
		OutputBits     []uint32 `json:"output_bits"`
		ReferenceError string   `json:"reference_error"`
	}
	var (
		contents                         []byte
		fixtures                         []fixture
		reluDifferences, emptyAxisErrors int
		err                              error
	)

	if contents, err = os.ReadFile("testdata/activations.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 38 {
		t.Fatalf("fixture count=%d, want 38", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				got, want     []float32
			)

			if input, err = tensor.FromFloat32(tt.InputShape, valuesFromBits(tt.InputBits)); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, tt.OutputShape); err != nil {
				t.Fatal(err)
			}
			switch tt.Operation {
			case "Relu":
				err = tensor.ReLUInto(input, output)
			case "Sigmoid":
				err = tensor.SigmoidInto(input, output)
			case "Tanh":
				err = tensor.TanhInto(input, output)
			case "Softmax":
				err = tensor.SoftmaxInto(input, output, tt.Axis)
			default:
				t.Fatalf("unknown fixture operation %q", tt.Operation)
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(output.Shape(), tt.InputShape) || !slices.Equal(output.Shape(), tt.OutputShape) {
				t.Fatalf("shape=%v, want %v", output.Shape(), tt.OutputShape)
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if tt.ReferenceError != "" {
				if tt.Operation != "Softmax" || tt.InputShape[tt.Axis] != 0 || len(got) != 0 {
					t.Fatalf("unexpected reference rejection: %s", tt.ReferenceError)
				}
				emptyAxisErrors++
			}
			want = valuesFromBits(tt.OutputBits)
			if tt.Operation == "Relu" {
				for index, inputBits := range tt.InputBits {
					if !math.IsNaN(float64(math.Float32frombits(inputBits))) {
						continue
					}
					if !math.IsNaN(float64(want[index])) || math.Float32bits(got[index]) != 0 {
						t.Fatal("ReLU NaN must preserve library +0 behavior and record reference NaN")
					}
					want[index] = 0
					reluDifferences++
				}
			}
			checkActivationValues(t, got, want)
			if got, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, got, tt.InputBits)
		})
	}
	if reluDifferences != 1 || emptyAxisErrors != 2 {
		t.Fatalf("documented reference differences: ReLU=%d empty-axis=%d, want 1 and 2", reluDifferences, emptyAxisErrors)
	}
}

func Test_ReLUInto_Bits(t *testing.T) {
	var tests = []struct {
		name  string
		input uint32
		want  uint32
	}{
		{name: "positive", input: 0x3fa00000, want: 0x3fa00000},
		{name: "negative", input: 0xbfa00000, want: 0},
		{name: "positive zero", input: 0, want: 0},
		{name: "negative zero", input: 0x80000000, want: 0},
		{name: "positive subnormal", input: 0x00000001, want: 0x00000001},
		{name: "negative subnormal", input: 0x80000001, want: 0},
		{name: "largest finite", input: 0x7f7fffff, want: 0x7f7fffff},
		{name: "positive infinity", input: 0x7f800000, want: 0x7f800000},
		{name: "negative infinity", input: 0xff800000, want: 0},
		{name: "positive quiet NaN", input: 0x7fc12345, want: 0},
		{name: "negative quiet NaN", input: 0xffc12345, want: 0},
		{name: "positive signaling NaN", input: 0x7f800001, want: 0},
		{name: "negative signaling NaN", input: 0xff800001, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				values        []float32
				err           error
			)

			if input, err = tensor.FromFloat32(nil, []float32{math.Float32frombits(tt.input)}); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromFloat32(nil, []float32{19}); err != nil {
				t.Fatal(err)
			}
			if err = tensor.ReLUInto(input, output); err != nil {
				t.Fatal(err)
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

func Test_Activations_ArgumentErrors(t *testing.T) {
	var (
		operations = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "ReLU", run: tensor.ReLUInto},
			{name: "ReLUPreservingNaN", run: preservingNaNReLU},
			{name: "Sigmoid", run: tensor.SigmoidInto},
			{name: "Tanh", run: tensor.TanhInto},
			{name: "Softmax", run: func(input, output *tensor.Tensor) (err error) {
				err = tensor.SoftmaxInto(input, output, 0)
				return err
			}},
		}
		tests = []string{"nil input", "zero input", "integer input", "nil destination", "zero destination", "integer destination", "wrong shape"}
	)
	for _, operation := range operations {
		for _, name := range tests {
			t.Run(operation.name+"/"+name, func(t *testing.T) {
				var (
					input, output       *tensor.Tensor
					zero                tensor.Tensor
					before, after       []float32
					beforeInt, afterInt []int64
					err                 error
				)
				if input, err = tensor.FromFloat32([]int64{2}, []float32{1, 2}); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.FromFloat32([]int64{2}, []float32{19, 23}); err != nil {
					t.Fatal(err)
				}
				switch name {
				case "nil input":
					input = nil
				case "zero input":
					input = &zero
				case "integer input":
					input, err = tensor.FromInt64([]int64{2}, []int64{1, 2})
				case "nil destination":
					output = nil
				case "zero destination":
					output = &zero
				case "integer destination":
					output, err = tensor.FromInt64([]int64{2}, []int64{19, 23})
				case "wrong shape":
					output, err = tensor.FromFloat32([]int64{1, 2}, []float32{19, 23})
				}
				if err != nil {
					t.Fatal(err)
				}
				if output.DataType() == tensor.Float32 {
					if before, err = output.Float32Values(); err != nil {
						t.Fatal(err)
					}
				} else if output.DataType() == tensor.Int64 {
					if beforeInt, err = output.Int64Values(); err != nil {
						t.Fatal(err)
					}
				}
				if err = operation.run(input, output); err == nil {
					t.Fatal("invalid arguments succeeded")
				}
				if output.DataType() == tensor.Float32 {
					if after, err = output.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(before, after) {
						t.Fatalf("destination changed on error: got %v, before %v", after, before)
					}
				} else if output.DataType() == tensor.Int64 {
					if afterInt, err = output.Int64Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(beforeInt, afterInt) {
						t.Fatalf("integer destination changed on error: got %v, before %v", afterInt, beforeInt)
					}
				}
			})
		}
	}
}

func Test_Softmax_InvalidAxes(t *testing.T) {
	var axes = []int64{2, -3, math.MinInt64, math.MaxInt64}
	for _, axis := range axes {
		t.Run(fmt.Sprint(axis), func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				got           []float32
				err           error
			)
			if input, err = tensor.New(tensor.Float32, []int64{2, 3}); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromFloat32([]int64{2, 3}, []float32{7, 8, 9, 10, 11, 12}); err != nil {
				t.Fatal(err)
			}
			if err = tensor.SoftmaxInto(input, output, axis); err == nil {
				t.Fatal("invalid axis succeeded")
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, []float32{7, 8, 9, 10, 11, 12}) {
				t.Fatalf("destination changed on invalid axis: %v", got)
			}
		})
	}
	var (
		input, output *tensor.Tensor
		got           []float32
		err           error
	)
	if input, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32(nil, []float32{7}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.SoftmaxInto(input, output, 0); err == nil {
		t.Fatal("scalar Softmax succeeded")
	}
	if got, err = output.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []float32{7}) {
		t.Fatalf("scalar destination changed on error: %v", got)
	}
	if input, err = tensor.New(tensor.Float32, []int64{2, 0, 3}); err != nil {
		t.Fatal(err)
	}
	if output, err = input.Clone(); err != nil {
		t.Fatal(err)
	}
	if err = tensor.SoftmaxInto(input, output, 3); err == nil {
		t.Fatal("empty Softmax bypassed invalid axis")
	}
}

func Test_Activations_AliasesAndEmptySizes(t *testing.T) {
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		shapes       = [][]int64{{2}, {0}, {0, maxHostIndex, maxHostIndex}}
		operations   = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "ReLU", run: tensor.ReLUInto},
			{name: "ReLUPreservingNaN", run: preservingNaNReLU},
			{name: "Sigmoid", run: tensor.SigmoidInto},
			{name: "Tanh", run: tensor.TanhInto},
			{name: "Softmax", run: func(input, output *tensor.Tensor) (err error) {
				err = tensor.SoftmaxInto(input, output, 0)
				return err
			}},
		}
	)
	for _, operation := range operations {
		for _, shape := range shapes {
			t.Run(operation.name+fmt.Sprint(shape), func(t *testing.T) {
				var (
					input, output *tensor.Tensor
					shallow       tensor.Tensor
					before, after []float32
					err           error
				)
				if input, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				if before, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				if err = operation.run(input, input); err == nil {
					t.Fatal("pointer destination alias succeeded")
				}
				shallow = *input
				if err = operation.run(input, &shallow); err == nil {
					t.Fatal("shallow-copy destination alias succeeded")
				}
				if after, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(before, after) {
					t.Fatalf("aliased input changed: %v", after)
				}
				if output, err = input.Clone(); err != nil {
					t.Fatal(err)
				}
				if err = operation.run(input, output); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func Test_Activations_ExistingImplementations(t *testing.T) {
	var (
		operations = []struct {
			name     string
			run      func(*tensor.Tensor, *tensor.Tensor) error
			function activation.DestinationActivation
		}{
			{name: "ReLU", run: tensor.ReLUInto, function: activation.ReLU{}},
			{name: "Sigmoid", run: tensor.SigmoidInto, function: activation.Sigmoid{}},
			{name: "Tanh", run: tensor.TanhInto, function: activation.Tanh{}},
			{name: "Softmax", function: activation.Softmax{}, run: func(input, output *tensor.Tensor) (err error) {
				err = tensor.SoftmaxInto(input, output, 1)
				return err
			}},
		}
		inputValues = []float32{-4, math.Float32frombits(0x80000000), 0, 1, 3, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	)
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				input, output             *tensor.Tensor
				inputMatrix, outputMatrix *matrix.Matrix
				got, want                 []float32
				err                       error
			)
			if input, err = tensor.FromFloat32([]int64{2, 4}, inputValues); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
				t.Fatal(err)
			}
			if inputMatrix, err = matrix.FromSlice(2, 4, inputValues); err != nil {
				t.Fatal(err)
			}
			if outputMatrix, err = matrix.New(2, 4); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(input, output); err != nil {
				t.Fatal(err)
			}
			if err = operation.function.ForwardInto(inputMatrix, outputMatrix); err != nil {
				t.Fatal(err)
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if want, err = outputMatrix.Values(); err != nil {
				t.Fatal(err)
			}
			checkActivationValues(t, got, want)
			for index := range got {
				if !math.IsNaN(float64(want[index])) && math.Float32bits(got[index]) != math.Float32bits(want[index]) {
					t.Fatalf("existing implementation differs at %d: got %g want %g", index, got[index], want[index])
				}
			}
		})
	}
}

func Test_Activations_OutputLifetimeAndConcurrency(t *testing.T) {
	var operations = []struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor) error
	}{
		{name: "ReLU", run: tensor.ReLUInto},
		{name: "ReLUPreservingNaN", run: preservingNaNReLU},
		{name: "Sigmoid", run: tensor.SigmoidInto},
		{name: "Tanh", run: tensor.TanhInto},
		{name: "Softmax", run: func(input, output *tensor.Tensor) (err error) {
			err = tensor.SoftmaxInto(input, output, 1)
			return err
		}},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				input, first, second *tensor.Tensor
				before, after        []float32
				workers              sync.WaitGroup
				results              = make(chan error, 8)
				err                  error
			)
			if input, err = tensor.FromFloat32([]int64{2, 3}, []float32{1, 2, 3, -2, 0, 2}); err != nil {
				t.Fatal(err)
			}
			if first, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
				t.Fatal(err)
			}
			if second, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(input, first); err != nil {
				t.Fatal(err)
			}
			if before, err = first.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(first, second); err != nil {
				t.Fatal(err)
			}
			if after, err = first.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before, after) {
				t.Fatalf("later call changed earlier output: got %v, want %v", after, before)
			}
			for worker := 0; worker < 8; worker++ {
				workers.Add(1)
				go func() {
					var (
						output    *tensor.Tensor
						values    []float32
						workerErr error
					)
					defer workers.Done()
					if output, workerErr = tensor.New(tensor.Float32, input.Shape()); workerErr != nil {
						results <- workerErr
						return
					}
					for repeat := 0; repeat < 8; repeat++ {
						if workerErr = operation.run(input, output); workerErr != nil {
							results <- workerErr
							return
						}
					}
					if values, workerErr = output.Float32Values(); workerErr != nil {
						results <- workerErr
						return
					}
					if !slices.Equal(values, before) {
						results <- fmt.Errorf("concurrent output=%v, want %v", values, before)
						return
					}
					results <- nil
				}()
			}
			workers.Wait()
			close(results)
			for err = range results {
				if err != nil {
					t.Fatal(err)
				}
			}
			if after, err = first.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before, after) {
				t.Fatal("concurrent calls overwrote earlier output")
			}
			if after, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(after, []float32{1, 2, 3, -2, 0, 2}) {
				t.Fatal("concurrent calls changed shared input")
			}
		})
	}
}

func checkActivationValues(t *testing.T, got, want []float32) {
	var (
		actual, expected float64
		index            int
	)

	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("value count=%d, want %d", len(got), len(want))
	}
	for index = range got {
		actual, expected = float64(got[index]), float64(want[index])
		if math.IsNaN(expected) {
			if !math.IsNaN(actual) {
				t.Fatalf("value %d=%g, want NaN", index, actual)
			}
			continue
		}
		if math.IsInf(expected, 0) {
			if actual != expected {
				t.Fatalf("value %d=%g, want %g", index, actual, expected)
			}
			continue
		}
		if expected == 0 && math.Float32bits(got[index]) != math.Float32bits(want[index]) {
			t.Fatalf("zero bits at %d: got %08x want %08x", index, math.Float32bits(got[index]), math.Float32bits(want[index]))
		}
		if math.IsNaN(actual) || math.IsInf(actual, 0) || math.Abs(actual-expected) > 1e-6+1e-6*math.Abs(expected) {
			t.Fatalf("value %d=%g, want %g within atol=rtol=1e-6", index, actual, expected)
		}
	}
}
