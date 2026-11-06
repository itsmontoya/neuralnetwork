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

func Test_Indexing_Reference(t *testing.T) {
	type payload struct {
		Shape    []int64  `json:"shape"`
		Bits     []uint32 `json:"bits"`
		Integers []int64  `json:"integers"`
	}
	type fixture struct {
		Name                string    `json:"name"`
		Operation           string    `json:"operation"`
		DataType            string    `json:"data_type"`
		Inputs              []payload `json:"inputs"`
		Axis                int64     `json:"axis"`
		IndicesShape        []int64   `json:"indices_shape"`
		Indices             []int64   `json:"indices"`
		OutputShape         []int64   `json:"output_shape"`
		Output              payload   `json:"output"`
		Reference           string    `json:"reference"`
		ReferenceShape      []int64   `json:"reference_shape"`
		ReferenceDifference string    `json:"reference_difference"`
	}
	var (
		contents    []byte
		fixtures    []fixture
		differences int
		err         error
	)

	if contents, err = os.ReadFile("testdata/indexing.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 48 {
		t.Fatalf("fixture count=%d, want 48", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				inputs          = make([]*tensor.Tensor, len(tt.Inputs))
				shapes          = make([][]int64, len(tt.Inputs))
				indices, output *tensor.Tensor
				shape           []int64
				floats          []float32
				integers        []int64
				dataType        tensor.DataType
				fixtureErr      error
			)
			if tt.DataType == "float32" {
				dataType = tensor.Float32
			} else if tt.DataType == "int64" {
				dataType = tensor.Int64
			} else {
				t.Fatalf("unknown fixture type %q", tt.DataType)
			}
			for index, input := range tt.Inputs {
				shapes[index] = input.Shape
				if dataType == tensor.Float32 {
					inputs[index], fixtureErr = tensor.FromFloat32(input.Shape, valuesFromBits(input.Bits))
				} else {
					inputs[index], fixtureErr = tensor.FromInt64(input.Shape, input.Integers)
				}
				if fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
			}
			if tt.Operation == "Concat" {
				shape, fixtureErr = tensor.ConcatShape(shapes, tt.Axis)
			} else if tt.Operation == "Gather" {
				if indices, fixtureErr = tensor.FromInt64(tt.IndicesShape, tt.Indices); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				shape, fixtureErr = tensor.GatherShape(shapes[0], tt.IndicesShape, tt.Axis)
			} else {
				t.Fatalf("unknown operation %q", tt.Operation)
			}
			if fixtureErr != nil || !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("planned shape=%v error=%v, want %v", shape, fixtureErr, tt.OutputShape)
			}
			if output, fixtureErr = tensor.New(dataType, shape); fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if tt.Operation == "Concat" {
				fixtureErr = tensor.ConcatInto(inputs, output, tt.Axis)
			} else {
				fixtureErr = tensor.GatherInto(inputs[0], indices, output, tt.Axis)
			}
			if fixtureErr != nil {
				t.Fatal(fixtureErr)
			}
			if output.DataType() != dataType || !slices.Equal(output.Shape(), tt.OutputShape) {
				t.Fatalf("output metadata type=%d shape=%v", output.DataType(), output.Shape())
			}
			if dataType == tensor.Float32 {
				if floats, fixtureErr = output.Float32Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				checkTransformBits(t, floats, tt.Output.Bits)
			} else {
				if integers, fixtureErr = output.Int64Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				if !slices.Equal(integers, tt.Output.Integers) {
					t.Fatalf("output=%v, want %v", integers, tt.Output.Integers)
				}
			}
			for index, input := range inputs {
				if !slices.Equal(input.Shape(), tt.Inputs[index].Shape) {
					t.Fatal("input shape changed")
				}
				if dataType == tensor.Float32 {
					if floats, fixtureErr = input.Float32Values(); fixtureErr != nil {
						t.Fatal(fixtureErr)
					}
					checkTransformBits(t, floats, tt.Inputs[index].Bits)
				} else {
					if integers, fixtureErr = input.Int64Values(); fixtureErr != nil {
						t.Fatal(fixtureErr)
					}
					if !slices.Equal(integers, tt.Inputs[index].Integers) {
						t.Fatal("input values changed")
					}
				}
			}
			if indices != nil {
				if integers, fixtureErr = indices.Int64Values(); fixtureErr != nil {
					t.Fatal(fixtureErr)
				}
				if !slices.Equal(integers, tt.Indices) || !slices.Equal(indices.Shape(), tt.IndicesShape) {
					t.Fatal("indices changed")
				}
			}
			if tt.Reference == "" {
				t.Fatal("missing reference provenance")
			}
			if tt.ReferenceDifference != "" {
				if tt.Operation != "Gather" || len(tt.Indices) != 0 || !slices.Equal(tt.ReferenceShape, []int64{0}) {
					t.Fatal("unexpected reference difference")
				}
				differences++
			} else if !slices.Equal(tt.ReferenceShape, tt.OutputShape) {
				t.Fatal("undocumented reference shape difference")
			}
		})
	}
	if differences != 6 {
		t.Fatalf("recorded empty-index reference differences=%d, want 6", differences)
	}
}

func Test_IndexingShapes(t *testing.T) {
	type testcase struct {
		name    string
		concat  bool
		inputs  [][]int64
		indices []int64
		axis    int64
		want    []int64
		wantErr bool
	}
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		tests        = []testcase{
			{name: "concat no inputs", concat: true, wantErr: true},
			{name: "concat scalar", concat: true, inputs: [][]int64{nil}, wantErr: true},
			{name: "concat negative axis", concat: true, inputs: [][]int64{{2, 1, 3}, {2, 2, 3}}, axis: -2, want: []int64{2, 3, 3}},
			{name: "concat rank mismatch", concat: true, inputs: [][]int64{{2, 3}, {3}}, wantErr: true},
			{name: "concat nonaxis mismatch", concat: true, inputs: [][]int64{{2, 3}, {1, 4}}, wantErr: true},
			{name: "concat negative dimension after zero", concat: true, inputs: [][]int64{{0, -1}}, wantErr: true},
			{name: "concat high axis", concat: true, inputs: [][]int64{{2}}, axis: 1, wantErr: true},
			{name: "concat minimum axis", concat: true, inputs: [][]int64{{2}}, axis: math.MinInt64, wantErr: true},
			{name: "concat axis sum overflow despite empty", concat: true, inputs: [][]int64{{0, maxHostIndex}, {0, 1}}, axis: 1, wantErr: true},
			{name: "concat output bytes overflow", concat: true, inputs: [][]int64{{maxHostIndex/16 + 1}, {maxHostIndex/16 + 1}}, wantErr: true},
			{name: "concat input bytes overflow", concat: true, inputs: [][]int64{{maxHostIndex/8 + 1}}, wantErr: true},
			{name: "concat unused empty products", concat: true, inputs: [][]int64{{0, maxHostIndex, maxHostIndex}, {0, maxHostIndex, maxHostIndex}}, want: []int64{0, maxHostIndex, maxHostIndex}},
			{name: "gather scalar data", inputs: [][]int64{nil}, wantErr: true},
			{name: "gather scalar index", inputs: [][]int64{{2, 3}}, axis: 1, want: []int64{2}},
			{name: "gather multidimensional index", inputs: [][]int64{{2, 3, 4}}, indices: []int64{2, 1}, axis: -2, want: []int64{2, 2, 1, 4}},
			{name: "gather empty index shape", inputs: [][]int64{{2, 3, 4}}, indices: []int64{2, 0, 1}, axis: 1, want: []int64{2, 2, 0, 1, 4}},
			{name: "gather high axis", inputs: [][]int64{{2}}, indices: []int64{0}, axis: 1, wantErr: true},
			{name: "gather minimum axis", inputs: [][]int64{{2}}, axis: math.MinInt64, wantErr: true},
			{name: "gather invalid data dimension", inputs: [][]int64{{0, -1}}, indices: []int64{0}, wantErr: true},
			{name: "gather invalid index dimension", inputs: [][]int64{{2}}, indices: []int64{0, -1}, wantErr: true},
			{name: "gather index bytes overflow", inputs: [][]int64{{2}}, indices: []int64{maxHostIndex/8 + 1}, wantErr: true},
			{name: "gather output bytes overflow", inputs: [][]int64{{2, 2}}, indices: []int64{maxHostIndex/16 + 1}, wantErr: true},
			{name: "gather unused empty products", inputs: [][]int64{{0, maxHostIndex, maxHostIndex}}, indices: []int64{0}, axis: 1, want: []int64{0, 0, maxHostIndex}},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				inputsCopy = make([][]int64, len(tt.inputs))
				indexCopy  = slices.Clone(tt.indices)
				shape      []int64
				err        error
			)
			for index := range tt.inputs {
				inputsCopy[index] = slices.Clone(tt.inputs[index])
			}
			if tt.concat {
				shape, err = tensor.ConcatShape(tt.inputs, tt.axis)
			} else {
				shape, err = tensor.GatherShape(tt.inputs[0], tt.indices, tt.axis)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("shape=%v error=%v, wantErr=%v", shape, err, tt.wantErr)
			}
			if !tt.wantErr && !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v, want %v", shape, tt.want)
			}
			if len(shape) > 0 {
				shape[0] = 99
			}
			for index := range tt.inputs {
				if !slices.Equal(tt.inputs[index], inputsCopy[index]) {
					t.Fatal("planner changed or retained input shape")
				}
			}
			if !slices.Equal(tt.indices, indexCopy) {
				t.Fatal("planner changed or retained indices shape")
			}
		})
	}
}

func Test_Indexing_ArgumentErrors(t *testing.T) {
	var (
		operations = []string{"Concat", "Gather"}
		tests      = []string{"nil data", "zero data", "nil second", "zero second", "second type", "scalar data", "nil destination", "zero destination", "destination type", "destination shape", "high axis", "minimum axis", "invalid second"}
	)
	for _, operation := range operations {
		for _, name := range tests {
			t.Run(operation+"/"+name, func(t *testing.T) {
				var (
					data, second, destination *tensor.Tensor
					zero                      tensor.Tensor
					shape                     = []int64{2, 2}
					axis                      int64
					before, after             []int64
					beforeFloat, afterFloat   []float32
					err                       error
				)
				if data, err = tensor.FromInt64([]int64{2, 2}, []int64{1, 2, 3, 4}); err != nil {
					t.Fatal(err)
				}
				if operation == "Concat" {
					second, err = data.Clone()
					shape = []int64{4, 2}
				} else {
					second, err = tensor.FromInt64([]int64{2}, []int64{1, 0})
				}
				if err != nil {
					t.Fatal(err)
				}
				before = make([]int64, int(shape[0]*shape[1]))
				for index := range before {
					before[index] = int64(index + 7)
				}
				if destination, err = tensor.FromInt64(shape, before); err != nil {
					t.Fatal(err)
				}
				switch name {
				case "nil data":
					data = nil
				case "zero data":
					data = &zero
				case "nil second":
					second = nil
				case "zero second":
					second = &zero
				case "second type":
					second, err = tensor.New(tensor.Float32, second.Shape())
				case "scalar data":
					data, err = tensor.FromInt64(nil, []int64{1})
				case "nil destination":
					destination = nil
				case "zero destination":
					destination = &zero
				case "destination type":
					beforeFloat = make([]float32, len(before))
					for index := range beforeFloat {
						beforeFloat[index] = float32(index + 7)
					}
					destination, err = tensor.FromFloat32(shape, beforeFloat)
				case "destination shape":
					destination, err = tensor.FromInt64([]int64{int64(len(before))}, before)
				case "high axis":
					axis = 2
				case "minimum axis":
					axis = math.MinInt64
				case "invalid second":
					if operation == "Concat" {
						second, err = tensor.FromInt64([]int64{2, 1}, []int64{5, 6})
					} else {
						second, err = tensor.FromInt64([]int64{2}, []int64{0, 2})
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				if operation == "Concat" {
					err = tensor.ConcatInto([]*tensor.Tensor{data, second}, destination, axis)
				} else {
					err = tensor.GatherInto(data, second, destination, axis)
				}
				if err == nil {
					t.Fatal("invalid arguments succeeded")
				}
				if destination.DataType() == tensor.Int64 {
					if after, err = destination.Int64Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(after, before) {
						t.Fatalf("destination changed on error: %v", after)
					}
				} else if destination.DataType() == tensor.Float32 {
					if afterFloat, err = destination.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(afterFloat, beforeFloat) {
						t.Fatalf("float destination changed on error: %v", afterFloat)
					}
				}
			})
		}
	}
}

func Test_Gather_IndexValidation(t *testing.T) {
	type testcase struct {
		name    string
		shape   []int64
		indices []int64
		output  []int64
		wantErr bool
	}
	var tests = []testcase{
		{name: "late high index", shape: []int64{2, 3}, indices: []int64{0, 3}, output: []int64{2, 2}, wantErr: true},
		{name: "late low index", shape: []int64{2, 3}, indices: []int64{-1, -4}, output: []int64{2, 2}, wantErr: true},
		{name: "minimum integer", shape: []int64{2, 3}, indices: []int64{math.MinInt64}, output: []int64{2, 1}, wantErr: true},
		{name: "maximum integer", shape: []int64{2, 3}, indices: []int64{math.MaxInt64}, output: []int64{2, 1}, wantErr: true},
		{name: "zero data axis", shape: []int64{2, 0}, indices: []int64{0}, output: []int64{2, 1}, wantErr: true},
		{name: "invalid despite zero prefix", shape: []int64{0, 3, 2}, indices: []int64{0, 3}, output: []int64{0, 2, 2}, wantErr: true},
		{name: "invalid despite zero suffix", shape: []int64{2, 3, 0}, indices: []int64{-4}, output: []int64{2, 1, 0}, wantErr: true},
		{name: "negative boundary", shape: []int64{2, 3}, indices: []int64{-3, -1}, output: []int64{2, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				data, indices, output *tensor.Tensor
				inputValues           []int64
				before, after         []int64
				size                  int64 = 1
				err                   error
			)
			for _, dimension := range tt.shape {
				size *= dimension
			}
			inputValues = make([]int64, int(size))
			for index := range inputValues {
				inputValues[index] = int64(index + 1)
			}
			if data, err = tensor.FromInt64(tt.shape, inputValues); err != nil {
				t.Fatal(err)
			}
			if indices, err = tensor.FromInt64([]int64{int64(len(tt.indices))}, tt.indices); err != nil {
				t.Fatal(err)
			}
			size = 1
			for _, dimension := range tt.output {
				size *= dimension
			}
			before = make([]int64, int(size))
			for index := range before {
				before[index] = int64(index + 99)
			}
			if output, err = tensor.FromInt64(tt.output, before); err != nil {
				t.Fatal(err)
			}
			err = tensor.GatherInto(data, indices, output, 1)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error=%v, wantErr=%v", err, tt.wantErr)
			}
			if after, err = output.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if tt.wantErr && !slices.Equal(after, before) {
				t.Fatalf("invalid index changed destination: %v", after)
			}
			if !tt.wantErr && !slices.Equal(after, []int64{1, 3, 4, 6}) {
				t.Fatalf("negative boundary values=%v", after)
			}
		})
	}
}

func Test_Indexing_AliasesAndRepeatedInputs(t *testing.T) {
	var (
		shapes = [][]int64{{2}, {0}}
		types  = []tensor.DataType{tensor.Float32, tensor.Int64}
	)
	for _, shape := range shapes {
		for _, dataType := range types {
			t.Run(fmt.Sprint(shape, dataType), func(t *testing.T) {
				var (
					data, indices, output, empty *tensor.Tensor
					shallow                      tensor.Tensor
					inputValues, integers        []int64
					floatInput, floats           []float32
					indexValues                  []int64
					err                          error
				)
				if shape[0] != 0 {
					inputValues = []int64{3, 4}
					floatInput = []float32{3, 4}
				}
				if dataType == tensor.Float32 {
					data, err = tensor.FromFloat32(shape, floatInput)
				} else {
					data, err = tensor.FromInt64(shape, inputValues)
				}
				if err != nil {
					t.Fatal(err)
				}
				if empty, err = tensor.New(dataType, []int64{0}); err != nil {
					t.Fatal(err)
				}
				if shape[0] != 0 {
					indexValues = []int64{1, 0}
				}
				if indices, err = tensor.FromInt64(shape, indexValues); err != nil {
					t.Fatal(err)
				}
				shallow = *data
				for _, alias := range []*tensor.Tensor{data, &shallow} {
					if err = tensor.ConcatInto([]*tensor.Tensor{empty, data}, alias, 0); err == nil {
						t.Fatal("concat destination alias succeeded")
					}
					if err = tensor.GatherInto(data, indices, alias, 0); err == nil {
						t.Fatal("gather data destination alias succeeded")
					}
				}
				if dataType == tensor.Int64 {
					shallow = *indices
					for _, alias := range []*tensor.Tensor{indices, &shallow} {
						if err = tensor.GatherInto(data, indices, alias, 0); err == nil {
							t.Fatal("gather indices destination alias succeeded")
						}
					}
				}
				if output, err = tensor.New(dataType, []int64{2 * shape[0]}); err != nil {
					t.Fatal(err)
				}
				if err = tensor.ConcatInto([]*tensor.Tensor{data, data}, output, 0); err != nil {
					t.Fatalf("repeated read-only concat inputs failed: %v", err)
				}
				if dataType == tensor.Float32 {
					if floats, err = data.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(floats, floatInput) {
						t.Fatal("alias error changed float data")
					}
				} else {
					if integers, err = data.Int64Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(integers, inputValues) {
						t.Fatal("alias error changed integer data")
					}
				}
				if integers, err = indices.Int64Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(integers, indexValues) {
					t.Fatal("alias error changed indices")
				}
			})
		}
	}
	var (
		shared, output *tensor.Tensor
		values         []int64
		err            error
	)
	if shared, err = tensor.FromInt64([]int64{2}, []int64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Int64, []int64{2}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GatherInto(shared, shared, output, 0); err != nil {
		t.Fatal(err)
	}
	if values, err = output.Int64Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, []int64{0, 1}) {
		t.Fatalf("shared read-only data/indices result=%v", values)
	}
}

func Test_Indexing_EmptyUnusedSizes(t *testing.T) {
	var (
		maxHostIndex          = int64(int(^uint(0) >> 1))
		data, indices, output *tensor.Tensor
		err                   error
	)
	if data, err = tensor.New(tensor.Int64, []int64{0, maxHostIndex, maxHostIndex}); err != nil {
		t.Fatal(err)
	}
	if output, err = data.Clone(); err != nil {
		t.Fatal(err)
	}
	if err = tensor.ConcatInto([]*tensor.Tensor{data, data}, output, 0); err != nil {
		t.Fatal(err)
	}
	if indices, err = tensor.FromInt64(nil, []int64{maxHostIndex - 1}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Int64, []int64{0, maxHostIndex}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GatherInto(data, indices, output, 1); err != nil {
		t.Fatal(err)
	}
	if indices, err = tensor.FromInt64(nil, []int64{maxHostIndex}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.GatherInto(data, indices, output, 1); err == nil {
		t.Fatal("invalid index ignored because output is empty")
	}
	if err = tensor.ConcatInto(nil, output, 0); err == nil {
		t.Fatal("empty concat input list succeeded")
	}
}

func Test_Indexing_OutputLifetimeAndConcurrency(t *testing.T) {
	var operations = []struct {
		name  string
		other []int64
		shape []int64
		want  []int64
		run   func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error
	}{
		{name: "Concat", other: []int64{5, 6}, shape: []int64{3, 2}, want: []int64{1, 2, 3, 4, 5, 6}, run: func(data, other, output *tensor.Tensor) (err error) {
			return tensor.ConcatInto([]*tensor.Tensor{data, other}, output, 0)
		}},
		{name: "Gather", other: []int64{1, 0}, shape: []int64{2, 2}, want: []int64{3, 4, 1, 2}, run: func(data, other, output *tensor.Tensor) (err error) {
			return tensor.GatherInto(data, other, output, 0)
		}},
	}
	for _, operation := range operations {
		t.Run(operation.name, func(t *testing.T) {
			var (
				data, other, first, second, changed *tensor.Tensor
				otherShape                          = []int64{2}
				values                              []int64
				workers                             sync.WaitGroup
				results                             = make(chan error, 4)
				err                                 error
			)
			if data, err = tensor.FromInt64([]int64{2, 2}, []int64{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			if changed, err = tensor.FromInt64([]int64{2, 2}, []int64{11, 12, 13, 14}); err != nil {
				t.Fatal(err)
			}
			if operation.name == "Concat" {
				otherShape = []int64{1, 2}
			}
			if other, err = tensor.FromInt64(otherShape, operation.other); err != nil {
				t.Fatal(err)
			}
			if first, err = tensor.New(tensor.Int64, operation.shape); err != nil {
				t.Fatal(err)
			}
			if second, err = tensor.New(tensor.Int64, operation.shape); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(data, other, first); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(changed, other, second); err != nil {
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
						if workerErr = operation.run(data, other, output); workerErr != nil {
							results <- workerErr
							return
						}
					}
					if got, workerErr = output.Int64Values(); workerErr != nil {
						results <- workerErr
						return
					}
					if !slices.Equal(got, operation.want) {
						results <- fmt.Errorf("concurrent output=%v, want %v", got, operation.want)
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
			if values, err = first.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, operation.want) {
				t.Fatalf("later call changed earlier output: %v", values)
			}
			if values, err = data.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, []int64{1, 2, 3, 4}) {
				t.Fatal("shared data changed")
			}
			if values, err = other.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, operation.other) {
				t.Fatal("shared other input changed")
			}
		})
	}
}
