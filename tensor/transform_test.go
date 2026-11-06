package tensor_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Transforms_Reference(t *testing.T) {
	type payload struct {
		Bits     []uint32 `json:"bits"`
		Integers []int64  `json:"integers"`
	}
	type fixture struct {
		Name           string  `json:"name"`
		Operation      string  `json:"operation"`
		DataType       string  `json:"data_type"`
		InputShape     []int64 `json:"input_shape"`
		Input          payload `json:"input"`
		Axis           int64   `json:"axis"`
		Axes           []int64 `json:"axes"`
		Target         []int64 `json:"target"`
		OutputShape    []int64 `json:"output_shape"`
		Output         payload `json:"output"`
		Reference      string  `json:"reference"`
		ReferenceError string  `json:"reference_error"`
	}
	var (
		contents        []byte
		fixtures        []fixture
		referenceErrors int
		err             error
	)

	if contents, err = os.ReadFile("testdata/transforms.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 54 {
		t.Fatalf("fixture count=%d, want 54", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				shape         []int64
				floats        []float32
				integers      []int64
				dataType      tensor.DataType
			)

			if tt.DataType == "float32" {
				dataType = tensor.Float32
				input, err = tensor.FromFloat32(tt.InputShape, valuesFromBits(tt.Input.Bits))
			} else if tt.DataType == "int64" {
				dataType = tensor.Int64
				input, err = tensor.FromInt64(tt.InputShape, tt.Input.Integers)
			} else {
				t.Fatalf("unknown fixture data type %q", tt.DataType)
			}
			if err != nil {
				t.Fatal(err)
			}
			switch tt.Operation {
			case "Reshape":
				shape, err = tensor.ReshapeShape(tt.InputShape, tt.Target)
			case "Flatten":
				shape, err = tensor.FlattenShape(tt.InputShape, tt.Axis)
			case "Transpose":
				shape, err = tensor.TransposeShape(tt.InputShape, tt.Axes)
			case "Squeeze":
				shape, err = tensor.SqueezeShape(tt.InputShape, tt.Axes)
			case "Unsqueeze":
				shape, err = tensor.UnsqueezeShape(tt.InputShape, tt.Axes)
			default:
				t.Fatalf("unknown fixture operation %q", tt.Operation)
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("planned shape=%v, want %v", shape, tt.OutputShape)
			}
			if output, err = tensor.New(dataType, shape); err != nil {
				t.Fatal(err)
			}
			switch tt.Operation {
			case "Reshape":
				err = tensor.ReshapeInto(input, output)
			case "Flatten":
				err = tensor.FlattenInto(input, output, tt.Axis)
			case "Transpose":
				err = tensor.TransposeInto(input, output, tt.Axes)
			case "Squeeze":
				err = tensor.SqueezeInto(input, output, tt.Axes)
			case "Unsqueeze":
				err = tensor.UnsqueezeInto(input, output, tt.Axes)
			}
			if err != nil {
				t.Fatal(err)
			}
			if output.DataType() != dataType || !slices.Equal(output.Shape(), tt.OutputShape) {
				t.Fatalf("output metadata: type=%d shape=%v", output.DataType(), output.Shape())
			}
			if dataType == tensor.Float32 {
				if floats, err = output.Float32Values(); err != nil {
					t.Fatal(err)
				}
				checkTransformBits(t, floats, tt.Output.Bits)
				if floats, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				checkTransformBits(t, floats, tt.Input.Bits)
			} else {
				if integers, err = output.Int64Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(integers, tt.Output.Integers) {
					t.Fatalf("integer output=%v, want %v", integers, tt.Output.Integers)
				}
				if integers, err = input.Int64Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(integers, tt.Input.Integers) {
					t.Fatalf("integer input changed: %v", integers)
				}
			}
			if !slices.Equal(input.Shape(), tt.InputShape) || tt.Reference == "" {
				t.Fatal("input shape changed or reference provenance missing")
			}
			if tt.ReferenceError != "" {
				if tt.Operation != "Flatten" || output.Len() != 0 {
					t.Fatalf("unexpected reference rejection: %s", tt.ReferenceError)
				}
				referenceErrors++
			}
		})
	}
	if referenceErrors != 2 {
		t.Fatalf("documented empty-prefix reference errors=%d, want 2", referenceErrors)
	}
}

func Test_TransformShapes(t *testing.T) {
	type testcase struct {
		name      string
		operation string
		input     []int64
		argument  []int64
		axis      int64
		want      []int64
		wantErr   bool
	}
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		tests        = []testcase{
			{name: "reshape scalar", operation: "reshape"},
			{name: "reshape mismatch", operation: "reshape", input: []int64{2, 3}, argument: []int64{5}, wantErr: true},
			{name: "reshape negative target", operation: "reshape", input: []int64{2}, argument: []int64{-1}, wantErr: true},
			{name: "reshape negative input after zero", operation: "reshape", input: []int64{0, -1}, argument: []int64{0}, wantErr: true},
			{name: "reshape large unused dimensions", operation: "reshape", input: []int64{0, maxHostIndex, maxHostIndex}, argument: []int64{maxHostIndex, 0}, want: []int64{maxHostIndex, 0}},
			{name: "reshape target element overflow", operation: "reshape", input: []int64{0}, argument: []int64{maxHostIndex, 2}, wantErr: true},
			{name: "reshape target byte overflow", operation: "reshape", input: []int64{0}, argument: []int64{maxHostIndex/8 + 1}, wantErr: true},
			{name: "reshape input byte overflow", operation: "reshape", input: []int64{maxHostIndex/8 + 1}, wantErr: true},
			{name: "flatten negative first endpoint", operation: "flatten", input: []int64{2, 3}, axis: -2, want: []int64{1, 6}},
			{name: "flatten scalar", operation: "flatten", want: []int64{1, 1}},
			{name: "flatten invalid scalar axis", operation: "flatten", axis: -1, wantErr: true},
			{name: "flatten axis too high", operation: "flatten", input: []int64{2, 3}, axis: 3, wantErr: true},
			{name: "flatten axis too low", operation: "flatten", input: []int64{2, 3}, axis: -3, wantErr: true},
			{name: "flatten minimum axis", operation: "flatten", input: []int64{2}, axis: math.MinInt64, wantErr: true},
			{name: "flatten prefix overflow despite empty suffix", operation: "flatten", input: []int64{maxHostIndex, maxHostIndex, 0}, axis: 2, wantErr: true},
			{name: "flatten suffix overflow despite empty prefix", operation: "flatten", input: []int64{0, maxHostIndex, maxHostIndex}, axis: 1, wantErr: true},
			{name: "transpose scalar", operation: "transpose"},
			{name: "transpose missing axes", operation: "transpose", input: []int64{2, 3}, wantErr: true},
			{name: "transpose extra axes", operation: "transpose", input: []int64{2}, argument: []int64{0, 1}, wantErr: true},
			{name: "transpose duplicate normalized axes", operation: "transpose", input: []int64{2, 3}, argument: []int64{0, -2}, wantErr: true},
			{name: "transpose out of range", operation: "transpose", input: []int64{2, 3}, argument: []int64{0, 2}, wantErr: true},
			{name: "transpose minimum axis", operation: "transpose", input: []int64{2}, argument: []int64{math.MinInt64}, wantErr: true},
			{name: "transpose negative dimension", operation: "transpose", input: []int64{0, -1}, argument: []int64{1, 0}, wantErr: true},
			{name: "squeeze selected unit", operation: "squeeze", input: []int64{1, 2, 1}, argument: []int64{-1}, want: []int64{1, 2}},
			{name: "squeeze empty axes identity", operation: "squeeze", input: []int64{1, 2, 1}, want: []int64{1, 2, 1}},
			{name: "squeeze duplicate normalized axes", operation: "squeeze", input: []int64{1, 2, 1}, argument: []int64{0, -3}, wantErr: true},
			{name: "squeeze nonunit", operation: "squeeze", input: []int64{1, 2, 1}, argument: []int64{1}, wantErr: true},
			{name: "squeeze zero axis", operation: "squeeze", input: []int64{0}, argument: []int64{0}, wantErr: true},
			{name: "squeeze scalar axis", operation: "squeeze", argument: []int64{0}, wantErr: true},
			{name: "squeeze out of range", operation: "squeeze", input: []int64{1}, argument: []int64{-2}, wantErr: true},
			{name: "squeeze invalid input", operation: "squeeze", input: []int64{-1}, wantErr: true},
			{name: "unsqueeze negative first position", operation: "unsqueeze", input: []int64{2}, argument: []int64{-2}, want: []int64{1, 2}},
			{name: "unsqueeze scalar empty axes", operation: "unsqueeze"},
			{name: "unsqueeze duplicate normalized axes", operation: "unsqueeze", input: []int64{2}, argument: []int64{2, -1}, wantErr: true},
			{name: "unsqueeze high axis", operation: "unsqueeze", input: []int64{2}, argument: []int64{2}, wantErr: true},
			{name: "unsqueeze low axis", operation: "unsqueeze", input: []int64{2}, argument: []int64{-3}, wantErr: true},
			{name: "unsqueeze maximum axis", operation: "unsqueeze", argument: []int64{math.MaxInt64}, wantErr: true},
			{name: "unsqueeze invalid input", operation: "unsqueeze", input: []int64{0, -1}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				inputCopy    = slices.Clone(tt.input)
				argumentCopy = slices.Clone(tt.argument)
				shape        []int64
				err          error
			)
			switch tt.operation {
			case "reshape":
				shape, err = tensor.ReshapeShape(tt.input, tt.argument)
			case "flatten":
				shape, err = tensor.FlattenShape(tt.input, tt.axis)
			case "transpose":
				shape, err = tensor.TransposeShape(tt.input, tt.argument)
			case "squeeze":
				shape, err = tensor.SqueezeShape(tt.input, tt.argument)
			case "unsqueeze":
				shape, err = tensor.UnsqueezeShape(tt.input, tt.argument)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("shape=%v error=%v, wantErr=%v", shape, err, tt.wantErr)
			}
			if !slices.Equal(tt.input, inputCopy) || !slices.Equal(tt.argument, argumentCopy) {
				t.Fatal("planner changed arguments")
			}
			if tt.wantErr {
				if shape != nil {
					t.Fatal("invalid planner returned a shape")
				}
				return
			}
			if !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v, want %v", shape, tt.want)
			}
			if len(shape) > 0 {
				shape[0] = 99
			}
			if !slices.Equal(tt.input, inputCopy) || !slices.Equal(tt.argument, argumentCopy) {
				t.Fatal("planner changed arguments or returned borrowed shape metadata")
			}
		})
	}
}

func Test_Transforms_ArgumentErrors(t *testing.T) {
	var (
		operations = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "reshape", run: tensor.ReshapeInto},
			{name: "flatten", run: func(input, output *tensor.Tensor) (err error) {
				return tensor.FlattenInto(input, output, 1)
			}},
			{name: "transpose", run: func(input, output *tensor.Tensor) (err error) {
				return tensor.TransposeInto(input, output, []int64{0, 1})
			}},
			{name: "squeeze", run: func(input, output *tensor.Tensor) (err error) {
				return tensor.SqueezeInto(input, output, nil)
			}},
			{name: "unsqueeze", run: func(input, output *tensor.Tensor) (err error) {
				return tensor.UnsqueezeInto(input, output, nil)
			}},
		}
		tests     = []string{"nil input", "zero input", "nil destination", "zero destination", "wrong type", "wrong shape"}
		dataTypes = []tensor.DataType{tensor.Float32, tensor.Int64}
	)
	for _, operation := range operations {
		for _, dataType := range dataTypes {
			for _, name := range tests {
				t.Run(operation.name+fmt.Sprint(dataType)+"/"+name, func(t *testing.T) {
					var (
						input, output       *tensor.Tensor
						zero                tensor.Tensor
						before, after       []float32
						beforeInt, afterInt []int64
						err                 error
					)
					if input, err = tensor.New(dataType, []int64{2, 2}); err != nil {
						t.Fatal(err)
					}
					if dataType == tensor.Float32 {
						output, err = tensor.FromFloat32([]int64{2, 2}, []float32{7, 8, 9, 10})
					} else {
						output, err = tensor.FromInt64([]int64{2, 2}, []int64{7, 8, 9, 10})
					}
					if err != nil {
						t.Fatal(err)
					}
					switch name {
					case "nil input":
						input = nil
					case "zero input":
						input = &zero
					case "nil destination":
						output = nil
					case "zero destination":
						output = &zero
					case "wrong type":
						if dataType == tensor.Float32 {
							output, err = tensor.FromInt64([]int64{2, 2}, []int64{7, 8, 9, 10})
						} else {
							output, err = tensor.FromFloat32([]int64{2, 2}, []float32{7, 8, 9, 10})
						}
					case "wrong shape":
						if dataType == tensor.Float32 {
							output, err = tensor.FromFloat32([]int64{1, 4}, []float32{7, 8, 9, 10})
						} else {
							output, err = tensor.FromInt64([]int64{1, 4}, []int64{7, 8, 9, 10})
						}
					}
					if err != nil {
						t.Fatal(err)
					}
					// Reshape accepts any equal-count concrete destination shape.
					if operation.name == "reshape" && name == "wrong shape" {
						if dataType == tensor.Float32 {
							output, err = tensor.FromFloat32([]int64{3}, []float32{7, 8, 9})
						} else {
							output, err = tensor.FromInt64([]int64{3}, []int64{7, 8, 9})
						}
						if err != nil {
							t.Fatal(err)
						}
					}
					if output.DataType() == tensor.Float32 {
						before, err = output.Float32Values()
					} else if output.DataType() == tensor.Int64 {
						beforeInt, err = output.Int64Values()
					}
					if err != nil {
						t.Fatal(err)
					}
					if err = operation.run(input, output); err == nil {
						t.Fatal("invalid arguments succeeded")
					}
					if output.DataType() == tensor.Float32 {
						if after, err = output.Float32Values(); err != nil {
							t.Fatal(err)
						}
						if !slices.Equal(before, after) {
							t.Fatalf("destination changed on error: %v", after)
						}
					} else if output.DataType() == tensor.Int64 {
						if afterInt, err = output.Int64Values(); err != nil {
							t.Fatal(err)
						}
						if !slices.Equal(beforeInt, afterInt) {
							t.Fatalf("integer destination changed on error: %v", afterInt)
						}
					}
				})
			}
		}
	}
}

func Test_Transforms_Aliases(t *testing.T) {
	var operations = []struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor) error
	}{
		{name: "reshape", run: tensor.ReshapeInto},
		{name: "flatten", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.FlattenInto(input, output, 1)
		}},
		{name: "transpose", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.TransposeInto(input, output, []int64{0, 1})
		}},
		{name: "squeeze", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.SqueezeInto(input, output, nil)
		}},
		{name: "unsqueeze", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.UnsqueezeInto(input, output, nil)
		}},
	}
	for _, operation := range operations {
		for _, dataType := range []tensor.DataType{tensor.Float32, tensor.Int64} {
			for _, shape := range [][]int64{{2, 2}, {0, 2}} {
				t.Run(operation.name+fmt.Sprint(dataType, shape), func(t *testing.T) {
					var (
						input, output *tensor.Tensor
						shallow       tensor.Tensor
						err           error
					)
					if input, err = tensor.New(dataType, shape); err != nil {
						t.Fatal(err)
					}
					if err = operation.run(input, input); err == nil {
						t.Fatal("pointer destination alias succeeded")
					}
					shallow = *input
					if err = operation.run(input, &shallow); err == nil {
						t.Fatal("shallow-copy destination alias succeeded")
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
}

func Test_Transforms_InvalidParametersPreserveDestination(t *testing.T) {
	var operations = []struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor) error
	}{
		{name: "flatten axis", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.FlattenInto(input, output, 3)
		}},
		{name: "transpose duplicate axes", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.TransposeInto(input, output, []int64{0, -2})
		}},
		{name: "transpose missing axes", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.TransposeInto(input, output, nil)
		}},
		{name: "squeeze nonunit", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.SqueezeInto(input, output, []int64{0})
		}},
		{name: "squeeze duplicate axes", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.SqueezeInto(input, output, []int64{0, -2})
		}},
		{name: "unsqueeze duplicate axes", run: func(input, output *tensor.Tensor) (err error) {
			return tensor.UnsqueezeInto(input, output, []int64{0, -4})
		}},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				input, output *tensor.Tensor
				values        []int64
				err           error
			)
			if input, err = tensor.FromInt64([]int64{2, 2}, []int64{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromInt64([]int64{2, 2}, []int64{7, 8, 9, 10}); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(input, output); err == nil {
				t.Fatal("invalid parameters succeeded")
			}
			if values, err = output.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, []int64{7, 8, 9, 10}) {
				t.Fatalf("destination changed on error: %v", values)
			}
		})
	}
}

func Test_Transforms_OutputLifetimeAndConcurrency(t *testing.T) {
	var operations = []struct {
		name  string
		shape []int64
		want  []int64
		run   func(*tensor.Tensor, *tensor.Tensor) error
	}{
		{name: "reshape", shape: []int64{4}, want: []int64{1, 2, 3, 4}, run: tensor.ReshapeInto},
		{name: "flatten", shape: []int64{1, 4}, want: []int64{1, 2, 3, 4}, run: func(input, output *tensor.Tensor) (err error) {
			return tensor.FlattenInto(input, output, 1)
		}},
		{name: "transpose", shape: []int64{1, 2, 2}, want: []int64{1, 3, 2, 4}, run: func(input, output *tensor.Tensor) (err error) {
			return tensor.TransposeInto(input, output, []int64{0, 2, 1})
		}},
		{name: "squeeze", shape: []int64{2, 2}, want: []int64{1, 2, 3, 4}, run: func(input, output *tensor.Tensor) (err error) {
			return tensor.SqueezeInto(input, output, []int64{0})
		}},
		{name: "unsqueeze", shape: []int64{1, 1, 2, 2}, want: []int64{1, 2, 3, 4}, run: func(input, output *tensor.Tensor) (err error) {
			return tensor.UnsqueezeInto(input, output, []int64{0})
		}},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				input, first, second *tensor.Tensor
				values               []int64
				workers              sync.WaitGroup
				results              = make(chan error, 4)
				err                  error
			)
			if input, err = tensor.FromInt64([]int64{1, 2, 2}, []int64{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			if first, err = tensor.New(tensor.Int64, operation.shape); err != nil {
				t.Fatal(err)
			}
			if second, err = tensor.New(tensor.Int64, []int64{4}); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(input, first); err != nil {
				t.Fatal(err)
			}
			if err = tensor.ReshapeInto(first, second); err != nil {
				t.Fatal(err)
			}
			for worker := 0; worker < 4; worker++ {
				workers.Add(1)
				go func() {
					var (
						output    *tensor.Tensor
						got       []int64
						workerErr error
					)
					defer workers.Done()
					if output, workerErr = tensor.New(tensor.Int64, operation.shape); workerErr != nil {
						results <- workerErr
						return
					}
					for repeat := 0; repeat < 4; repeat++ {
						if workerErr = operation.run(input, output); workerErr != nil {
							results <- workerErr
							return
						}
					}
					if got, workerErr = output.Int64Values(); workerErr != nil {
						results <- workerErr
						return
					}
					if !slices.Equal(got, operation.want) {
						results <- fmt.Errorf("concurrent values=%v, want %v", got, operation.want)
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
			for _, output := range []*tensor.Tensor{first, second} {
				if values, err = output.Int64Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(values, operation.want) {
					t.Fatalf("later call changed earlier values: %v", values)
				}
			}
			if values, err = input.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, []int64{1, 2, 3, 4}) {
				t.Fatalf("shared input changed: %v", values)
			}
		})
	}
}

func Test_Transforms_EmptyUnusedSizes(t *testing.T) {
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		operations   = []struct {
			name   string
			input  []int64
			output []int64
			run    func(*tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "reshape", input: []int64{0, maxHostIndex, maxHostIndex}, output: []int64{maxHostIndex, 0, maxHostIndex}, run: tensor.ReshapeInto},
			{name: "flatten", input: []int64{0, maxHostIndex, maxHostIndex}, output: []int64{1, 0}, run: func(input, output *tensor.Tensor) (err error) {
				return tensor.FlattenInto(input, output, 0)
			}},
			{name: "transpose", input: []int64{0, maxHostIndex, maxHostIndex}, output: []int64{maxHostIndex, maxHostIndex, 0}, run: func(input, output *tensor.Tensor) (err error) {
				return tensor.TransposeInto(input, output, []int64{2, 1, 0})
			}},
			{name: "squeeze", input: []int64{1, 0, maxHostIndex, maxHostIndex}, output: []int64{0, maxHostIndex, maxHostIndex}, run: func(input, output *tensor.Tensor) (err error) {
				return tensor.SqueezeInto(input, output, []int64{0})
			}},
			{name: "unsqueeze", input: []int64{0, maxHostIndex, maxHostIndex}, output: []int64{1, 0, maxHostIndex, maxHostIndex}, run: func(input, output *tensor.Tensor) (err error) {
				return tensor.UnsqueezeInto(input, output, []int64{0})
			}},
		}
	)
	for _, operation := range operations {
		for _, dataType := range []tensor.DataType{tensor.Float32, tensor.Int64} {
			t.Run(operation.name+fmt.Sprint(dataType), func(t *testing.T) {
				var (
					input, output *tensor.Tensor
					err           error
				)
				if input, err = tensor.New(dataType, operation.input); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.New(dataType, operation.output); err != nil {
					t.Fatal(err)
				}
				if err = operation.run(input, output); err != nil {
					t.Fatal(err)
				}
				if output.Len() != 0 || !slices.Equal(output.Shape(), operation.output) {
					t.Fatalf("empty output metadata changed: %v", output.Shape())
				}
			})
		}
	}
}

func checkTransformBits(t *testing.T, values []float32, bits []uint32) {
	var index int

	t.Helper()
	if len(values) != len(bits) {
		t.Fatalf("value count=%d, want %d", len(values), len(bits))
	}
	for index = range bits {
		if math.Float32bits(values[index]) != bits[index] {
			t.Fatalf("value %d: got bits %08x, want %08x", index, math.Float32bits(values[index]), bits[index])
		}
	}
}
