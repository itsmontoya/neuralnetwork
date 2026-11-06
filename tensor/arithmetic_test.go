package tensor_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Arithmetic_Reference(t *testing.T) {
	type fixture struct {
		Name        string   `json:"name"`
		Operation   string   `json:"operation"`
		LeftShape   []int64  `json:"left_shape"`
		LeftBits    []uint32 `json:"left_bits"`
		RightShape  []int64  `json:"right_shape"`
		RightBits   []uint32 `json:"right_bits"`
		OutputShape []int64  `json:"output_shape"`
		OutputBits  []uint32 `json:"output_bits"`
	}
	var (
		contents   []byte
		fixtures   []fixture
		err        error
		operations = map[string]func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error{
			"Add": tensor.AddInto, "Sub": tensor.SubInto,
			"Mul": tensor.MulInto, "Div": tensor.DivInto,
		}
	)

	if contents, err = os.ReadFile("testdata/arithmetic.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 40 {
		t.Fatalf("fixture count=%d, want 40", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				left, right, destination *tensor.Tensor
				got                      []float32
				operation                = operations[tt.Operation]
			)

			if operation == nil {
				t.Fatalf("unknown fixture operation %q", tt.Operation)
			}
			if left, err = tensor.FromFloat32(tt.LeftShape, valuesFromBits(tt.LeftBits)); err != nil {
				t.Fatal(err)
			}
			if right, err = tensor.FromFloat32(tt.RightShape, valuesFromBits(tt.RightBits)); err != nil {
				t.Fatal(err)
			}
			if destination, err = tensor.New(tensor.Float32, tt.OutputShape); err != nil {
				t.Fatal(err)
			}
			if err = operation(left, right, destination); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(destination.Shape(), tt.OutputShape) {
				t.Fatalf("shape=%v, want %v", destination.Shape(), tt.OutputShape)
			}
			if got, err = destination.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, got, tt.OutputBits)
			if got, err = left.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, got, tt.LeftBits)
			if got, err = right.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkFloat32Bits(t, got, tt.RightBits)
		})
	}
}

func Test_Arithmetic_ArgumentErrors(t *testing.T) {
	type testcase struct {
		name             string
		leftShape        []int64
		rightShape       []int64
		destinationShape []int64
		leftType         tensor.DataType
		rightType        tensor.DataType
		destinationType  tensor.DataType
		invalid          string
	}
	var (
		tests = []testcase{
			{name: "left nil", invalid: "left nil"},
			{name: "right nil", invalid: "right nil"},
			{name: "destination nil", invalid: "destination nil"},
			{name: "left zero", invalid: "left zero"},
			{name: "right zero", invalid: "right zero"},
			{name: "destination zero", invalid: "destination zero"},
			{name: "left integer", leftType: tensor.Int64},
			{name: "right integer", rightType: tensor.Int64},
			{name: "destination integer", destinationType: tensor.Int64},
			{name: "incompatible shapes", leftShape: []int64{2}, rightShape: []int64{3}, destinationShape: []int64{3}},
			{name: "wrong destination shape", destinationShape: []int64{2}},
			{name: "wrong destination rank", leftShape: []int64{2}, rightShape: []int64{2}, destinationShape: []int64{1, 2}},
			{name: "empty integer input", leftShape: []int64{0}, rightShape: []int64{1}, destinationShape: []int64{0}, leftType: tensor.Int64},
			{name: "empty integer destination", leftShape: []int64{0}, rightShape: []int64{1}, destinationShape: []int64{0}, destinationType: tensor.Int64},
			{name: "zero dimension incompatible", leftShape: []int64{0}, rightShape: []int64{2}, destinationShape: []int64{0}},
		}
		operations = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "Add", run: tensor.AddInto}, {name: "Sub", run: tensor.SubInto},
			{name: "Mul", run: tensor.MulInto}, {name: "Div", run: tensor.DivInto},
		}
	)
	for _, operation := range operations {
		for _, tt := range tests {
			t.Run(operation.name+"/"+tt.name, func(t *testing.T) {
				var (
					left, right, destination *tensor.Tensor
					zero                     tensor.Tensor
					dataTypes                = []tensor.DataType{tt.leftType, tt.rightType, tt.destinationType}
					shapes                   = [][]int64{tt.leftShape, tt.rightShape, tt.destinationShape}
					values                   [3]*tensor.Tensor
					before, after            []float32
					beforeInt, afterInt      []int64
					err                      error
				)

				for index, dataType := range dataTypes {
					if dataType == tensor.Invalid {
						dataType = tensor.Float32
					}
					if values[index], err = tensor.New(dataType, shapes[index]); err != nil {
						t.Fatal(err)
					}
				}
				left, right, destination = values[0], values[1], values[2]
				switch tt.invalid {
				case "left nil":
					left = nil
				case "right nil":
					right = nil
				case "destination nil":
					destination = nil
				case "left zero":
					left = &zero
				case "right zero":
					right = &zero
				case "destination zero":
					destination = &zero
				}
				if destination != nil && destination.DataType() == tensor.Float32 {
					before = make([]float32, int(destination.Len()))
					for index := range before {
						before[index] = 99
					}
					destination, err = tensor.FromFloat32(destination.Shape(), before)
				} else if destination != nil && destination.DataType() == tensor.Int64 {
					beforeInt = make([]int64, int(destination.Len()))
					for index := range beforeInt {
						beforeInt[index] = 99
					}
					destination, err = tensor.FromInt64(destination.Shape(), beforeInt)
				}
				if err != nil {
					t.Fatal(err)
				}
				if err = operation.run(left, right, destination); err == nil {
					t.Fatal("invalid arguments succeeded")
				}
				if destination != nil && destination.DataType() == tensor.Float32 {
					if after, err = destination.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(before, after) {
						t.Fatalf("destination changed on error: got %v, before %v", after, before)
					}
				} else if destination != nil && destination.DataType() == tensor.Int64 {
					if afterInt, err = destination.Int64Values(); err != nil {
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

func Test_Arithmetic_Aliases(t *testing.T) {
	var (
		operations = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "Add", run: tensor.AddInto}, {name: "Sub", run: tensor.SubInto},
			{name: "Mul", run: tensor.MulInto}, {name: "Div", run: tensor.DivInto},
		}
		tests = []struct {
			name   string
			shape  []int64
			values []float32
		}{
			{name: "nonempty", shape: []int64{2}, values: []float32{2, 4}},
			{name: "empty", shape: []int64{0}},
		}
	)
	for _, operation := range operations {
		for _, tt := range tests {
			t.Run(operation.name+"/"+tt.name, func(t *testing.T) {
				var (
					left, right, destination *tensor.Tensor
					shallow                  tensor.Tensor
					shape                    = tt.shape
					values                   = tt.values
					got                      []float32
					err                      error
				)
				if left, err = tensor.FromFloat32(shape, values); err != nil {
					t.Fatal(err)
				}
				if right, err = left.Clone(); err != nil {
					t.Fatal(err)
				}
				for _, input := range []*tensor.Tensor{left, right} {
					if err = operation.run(left, right, input); err == nil {
						t.Fatal("pointer destination alias succeeded")
					}
					shallow = *input
					if err = operation.run(left, right, &shallow); err == nil {
						t.Fatal("shallow-copy destination alias succeeded")
					}
					if got, err = input.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(got, values) {
						t.Fatalf("aliased input changed: got %v, want %v", got, values)
					}
				}
				if destination, err = tensor.New(tensor.Float32, shape); err != nil {
					t.Fatal(err)
				}
				if err = operation.run(left, left, destination); err != nil {
					t.Fatalf("read-only repeated inputs rejected: %v", err)
				}
			})
		}
	}
}

func Test_Arithmetic_EmptyUnusedStrides(t *testing.T) {
	var (
		maxHostIndex             = int64(int(^uint(0) >> 1))
		left, right, destination *tensor.Tensor
		err                      error
		operations               = []func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error{
			tensor.AddInto, tensor.SubInto, tensor.MulInto, tensor.DivInto,
		}
	)
	if left, err = tensor.New(tensor.Float32, []int64{0, maxHostIndex, maxHostIndex}); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if destination, err = left.Clone(); err != nil {
		t.Fatal(err)
	}
	for _, operation := range operations {
		if err = operation(left, right, destination); err != nil {
			t.Fatal(err)
		}
	}
}

func Test_Arithmetic_ConcurrentAndOutputLifetime(t *testing.T) {
	var (
		left, right, first, second *tensor.Tensor
		workers                    sync.WaitGroup
		results                    = make(chan error, 8)
		got                        []float32
		err                        error
	)
	if left, err = tensor.FromFloat32([]int64{2, 2}, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{2}, []float32{10, 20}); err != nil {
		t.Fatal(err)
	}
	if first, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
		t.Fatal(err)
	}
	if second, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.AddInto(left, right, first); err != nil {
		t.Fatal(err)
	}
	if err = tensor.MulInto(left, right, second); err != nil {
		t.Fatal(err)
	}
	if got, err = first.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []float32{11, 22, 13, 24}) {
		t.Fatalf("later call overwrote first output: %v", got)
	}
	for range 8 {
		workers.Add(1)
		go func() {
			var (
				output    *tensor.Tensor
				values    []float32
				workerErr error
			)
			defer workers.Done()
			if output, workerErr = tensor.New(tensor.Float32, []int64{2, 2}); workerErr == nil {
				workerErr = tensor.DivInto(left, right, output)
			}
			if workerErr == nil {
				values, workerErr = output.Float32Values()
			}
			if workerErr == nil && !slices.Equal(values, []float32{0.1, 0.1, 0.3, 0.2}) {
				workerErr = fmt.Errorf("concurrent quotient=%v", values)
			}
			results <- workerErr
		}()
	}
	workers.Wait()
	close(results)
	for result := range results {
		if result != nil {
			t.Fatal(result)
		}
	}
}

func Test_DivInto_PreservesMatrixDivisionContract(t *testing.T) {
	var (
		left, right, destination *matrix.Matrix
		err                      error
	)
	if left, err = matrix.FromSlice(1, 2, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if right, err = matrix.FromSlice(1, 2, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if destination, err = matrix.New(1, 2); err != nil {
		t.Fatal(err)
	}
	if err = left.DivideElementsInto(right, destination); err == nil {
		t.Fatal("matrix division accepted a zero denominator")
	}
}

func valuesFromBits(bits []uint32) (values []float32) {
	var index int

	values = make([]float32, len(bits))
	for index = range bits {
		values[index] = math.Float32frombits(bits[index])
	}
	return values
}

func checkFloat32Bits(t *testing.T, values []float32, bits []uint32) {
	var (
		index int
		want  float32
	)

	t.Helper()
	if len(values) != len(bits) {
		t.Fatalf("value count=%d, want %d", len(values), len(bits))
	}
	for index = range bits {
		want = math.Float32frombits(bits[index])
		if math.IsNaN(float64(want)) && math.IsNaN(float64(values[index])) {
			continue
		}
		if math.Float32bits(values[index]) != bits[index] {
			t.Fatalf("value %d: got %g (%08x), want %g (%08x)", index, values[index], math.Float32bits(values[index]), want, bits[index])
		}
	}
}
