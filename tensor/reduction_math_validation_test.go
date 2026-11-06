package tensor_test

import (
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Reductions_ArgumentErrors(t *testing.T) {
	type operation struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor, []int64, bool) error
	}
	type testcase struct {
		name        string
		input       *tensor.Tensor
		destination *tensor.Tensor
		axes        []int64
	}
	var (
		input, output, wrongShape, integerInput, integerOutput, empty, emptyOutput *tensor.Tensor
		zero, shallow, emptyShallow                                                tensor.Tensor
		tests                                                                      []testcase
		operations                                                                 = []operation{
			{name: "sum", run: tensor.ReduceSumInto},
			{name: "mean", run: tensor.ReduceMeanInto},
			{name: "max", run: tensor.ReduceMaxInto},
		}
		err error
	)
	if input, err = tensor.FromFloat32([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32([]int64{2}, []float32{77, 88}); err != nil {
		t.Fatal(err)
	}
	if wrongShape, err = tensor.FromFloat32([]int64{1, 2}, []float32{77, 88}); err != nil {
		t.Fatal(err)
	}
	if integerInput, err = tensor.FromInt64([]int64{2, 3}, []int64{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if integerOutput, err = tensor.FromInt64([]int64{2}, []int64{77, 88}); err != nil {
		t.Fatal(err)
	}
	if empty, err = tensor.New(tensor.Float32, []int64{0, 3}); err != nil {
		t.Fatal(err)
	}
	if emptyOutput, err = tensor.New(tensor.Float32, []int64{0}); err != nil {
		t.Fatal(err)
	}
	shallow, emptyShallow = *input, *empty
	tests = []testcase{
		{name: "nil input", destination: output, axes: []int64{1}},
		{name: "zero input", input: &zero, destination: output, axes: []int64{1}},
		{name: "integer input", input: integerInput, destination: output, axes: []int64{1}},
		{name: "nil destination", input: input, axes: []int64{1}},
		{name: "zero destination", input: input, destination: &zero, axes: []int64{1}},
		{name: "integer destination", input: input, destination: integerOutput, axes: []int64{1}},
		{name: "wrong destination shape", input: input, destination: wrongShape, axes: []int64{1}},
		{name: "out of range axis", input: input, destination: output, axes: []int64{2}},
		{name: "normalized duplicate", input: input, destination: output, axes: []int64{0, -2}},
		{name: "empty invalid axis", input: empty, destination: emptyOutput, axes: []int64{2}},
		{name: "pointer alias identity", input: input, destination: input},
		{name: "shallow alias identity", input: input, destination: &shallow},
		{name: "empty pointer alias", input: empty, destination: empty},
		{name: "empty shallow alias", input: empty, destination: &emptyShallow},
	}
	for _, op := range operations {
		for _, tt := range tests {
			t.Run(op.name+"/"+tt.name, func(t *testing.T) {
				var (
					shape    = tt.destination.Shape()
					dataType = tt.destination.DataType()
					before   []float32
					values   []float32
					integers []int64
					caseErr  error
				)
				if dataType == tensor.Float32 {
					if before, caseErr = tt.destination.Float32Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
				}
				if caseErr = op.run(tt.input, tt.destination, tt.axes, false); caseErr == nil {
					t.Fatal("invalid arguments succeeded")
				}
				if dataType != tt.destination.DataType() || !slices.Equal(shape, tt.destination.Shape()) {
					t.Fatal("error changed destination metadata")
				}
				if dataType == tensor.Float32 {
					if values, caseErr = tt.destination.Float32Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
					if !slices.Equal(values, before) {
						t.Fatalf("error changed destination: got %v, want %v", values, before)
					}
				} else if dataType == tensor.Int64 {
					if integers, caseErr = tt.destination.Int64Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
					if !slices.Equal(integers, []int64{77, 88}) {
						t.Fatalf("error changed integer destination: %v", integers)
					}
				}
			})
		}
	}
}

func Test_Math_ArgumentErrors(t *testing.T) {
	type operation struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor) error
	}
	type testcase struct {
		name        string
		input       *tensor.Tensor
		destination *tensor.Tensor
	}
	var (
		input, output, wrongShape, integer, empty *tensor.Tensor
		zero, shallow, emptyShallow               tensor.Tensor
		tests                                     []testcase
		operations                                = []operation{
			{name: "exp", run: tensor.ExpInto},
			{name: "log", run: tensor.LogInto},
			{name: "sqrt", run: tensor.SqrtInto},
		}
		err error
	)
	if input, err = tensor.FromFloat32([]int64{2}, []float32{1, 4}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32([]int64{2}, []float32{77, 88}); err != nil {
		t.Fatal(err)
	}
	if wrongShape, err = tensor.FromFloat32([]int64{1, 2}, []float32{77, 88}); err != nil {
		t.Fatal(err)
	}
	if integer, err = tensor.FromInt64([]int64{2}, []int64{77, 88}); err != nil {
		t.Fatal(err)
	}
	if empty, err = tensor.New(tensor.Float32, []int64{0, 3}); err != nil {
		t.Fatal(err)
	}
	shallow, emptyShallow = *input, *empty
	tests = []testcase{
		{name: "nil input", destination: output},
		{name: "zero input", input: &zero, destination: output},
		{name: "integer input", input: integer, destination: output},
		{name: "nil destination", input: input},
		{name: "zero destination", input: input, destination: &zero},
		{name: "integer destination", input: input, destination: integer},
		{name: "wrong shape", input: input, destination: wrongShape},
		{name: "pointer alias", input: input, destination: input},
		{name: "shallow alias", input: input, destination: &shallow},
		{name: "empty pointer alias", input: empty, destination: empty},
		{name: "empty shallow alias", input: empty, destination: &emptyShallow},
	}
	for _, op := range operations {
		for _, tt := range tests {
			t.Run(op.name+"/"+tt.name, func(t *testing.T) {
				var (
					shape    = tt.destination.Shape()
					dataType = tt.destination.DataType()
					before   []float32
					values   []float32
					integers []int64
					caseErr  error
				)
				if dataType == tensor.Float32 {
					if before, caseErr = tt.destination.Float32Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
				}
				if caseErr = op.run(tt.input, tt.destination); caseErr == nil {
					t.Fatal("invalid arguments succeeded")
				}
				if dataType != tt.destination.DataType() || !slices.Equal(shape, tt.destination.Shape()) {
					t.Fatal("error changed destination metadata")
				}
				if dataType == tensor.Float32 {
					if values, caseErr = tt.destination.Float32Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
					if !slices.Equal(values, before) {
						t.Fatalf("error changed destination: got %v, want %v", values, before)
					}
				} else if dataType == tensor.Int64 {
					if integers, caseErr = tt.destination.Int64Values(); caseErr != nil {
						t.Fatal(caseErr)
					}
					if !slices.Equal(integers, []int64{77, 88}) {
						t.Fatalf("error changed integer destination: %v", integers)
					}
				}
			})
		}
	}
}

func Test_Pow_ArgumentErrors(t *testing.T) {
	type testcase struct {
		name                     string
		left, right, destination *tensor.Tensor
	}
	var (
		left, right, output, wrongShape, incompatible, integer, scalar, empty *tensor.Tensor
		zero, shallow, emptyShallow                                           tensor.Tensor
		tests                                                                 []testcase
		err                                                                   error
	)
	if left, err = tensor.FromFloat32([]int64{2, 1}, []float32{2, 3}); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.FromFloat32([]int64{1, 3}, []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.FromFloat32([]int64{2, 3}, []float32{71, 72, 73, 74, 75, 76}); err != nil {
		t.Fatal(err)
	}
	if wrongShape, err = tensor.FromFloat32([]int64{3, 2}, []float32{71, 72, 73, 74, 75, 76}); err != nil {
		t.Fatal(err)
	}
	if incompatible, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
		t.Fatal(err)
	}
	if integer, err = tensor.FromInt64([]int64{2, 3}, []int64{71, 72, 73, 74, 75, 76}); err != nil {
		t.Fatal(err)
	}
	if scalar, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if empty, err = tensor.New(tensor.Float32, []int64{0, 3}); err != nil {
		t.Fatal(err)
	}
	shallow, emptyShallow = *left, *empty
	tests = []testcase{
		{name: "nil left", right: right, destination: output},
		{name: "nil right", left: left, destination: output},
		{name: "zero left", left: &zero, right: right, destination: output},
		{name: "zero right", left: left, right: &zero, destination: output},
		{name: "integer left", left: integer, right: right, destination: output},
		{name: "integer right", left: left, right: integer, destination: output},
		{name: "nil destination", left: left, right: right},
		{name: "zero destination", left: left, right: right, destination: &zero},
		{name: "integer destination", left: left, right: right, destination: integer},
		{name: "wrong shape", left: left, right: right, destination: wrongShape},
		{name: "incompatible shapes", left: incompatible, right: right, destination: output},
		{name: "left alias", left: left, right: scalar, destination: left},
		{name: "right alias", left: scalar, right: right, destination: right},
		{name: "shallow alias", left: left, right: scalar, destination: &shallow},
		{name: "empty pointer alias", left: empty, right: scalar, destination: empty},
		{name: "empty shallow alias", left: empty, right: scalar, destination: &emptyShallow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape    = tt.destination.Shape()
				dataType = tt.destination.DataType()
				before   []float32
				values   []float32
				integers []int64
				caseErr  error
			)
			if dataType == tensor.Float32 {
				if before, caseErr = tt.destination.Float32Values(); caseErr != nil {
					t.Fatal(caseErr)
				}
			}
			if caseErr = tensor.PowInto(tt.left, tt.right, tt.destination); caseErr == nil {
				t.Fatal("invalid arguments succeeded")
			}
			if dataType != tt.destination.DataType() || !slices.Equal(shape, tt.destination.Shape()) {
				t.Fatal("error changed destination metadata")
			}
			if dataType == tensor.Float32 {
				if values, caseErr = tt.destination.Float32Values(); caseErr != nil {
					t.Fatal(caseErr)
				}
				if !slices.Equal(values, before) {
					t.Fatalf("error changed destination: got %v, want %v", values, before)
				}
			} else if dataType == tensor.Int64 {
				if integers, caseErr = tt.destination.Int64Values(); caseErr != nil {
					t.Fatal(caseErr)
				}
				if !slices.Equal(integers, []int64{71, 72, 73, 74, 75, 76}) {
					t.Fatalf("error changed integer destination: %v", integers)
				}
			}
		})
	}
}

func Test_ReductionsMath_HugeEmpty(t *testing.T) {
	type reduction struct {
		name string
		run  func(*tensor.Tensor, *tensor.Tensor, []int64, bool) error
	}
	var (
		maxHost               = int64(int(^uint(0) >> 1))
		input, output, scalar *tensor.Tensor
		shape                 []int64
		err                   error
		reductions            = []reduction{
			{name: "sum", run: tensor.ReduceSumInto},
			{name: "mean", run: tensor.ReduceMeanInto},
			{name: "max", run: tensor.ReduceMaxInto},
		}
	)
	if input, err = tensor.New(tensor.Float32, []int64{0, maxHost, maxHost}); err != nil {
		t.Fatal(err)
	}
	for _, operation := range reductions {
		for _, keep := range []bool{false, true} {
			shape = []int64{0}
			if keep {
				shape = []int64{0, 1, 1}
			}
			if output, err = tensor.New(tensor.Float32, shape); err != nil {
				t.Fatal(err)
			}
			if err = operation.run(input, output, []int64{1, 2}, keep); err != nil {
				t.Fatalf("%s empty output: %v", operation.name, err)
			}
		}
		if output, err = tensor.New(tensor.Float32, nil); err != nil {
			t.Fatal(err)
		}
		if err = operation.run(input, output, []int64{0, 1, 2}, false); err != nil {
			t.Fatalf("%s empty reduction: %v", operation.name, err)
		}
	}
	if output, err = tensor.New(tensor.Float32, input.Shape()); err != nil {
		t.Fatal(err)
	}
	for _, operation := range []func(*tensor.Tensor, *tensor.Tensor) error{tensor.ExpInto, tensor.LogInto, tensor.SqrtInto} {
		if err = operation(input, output); err != nil {
			t.Fatal(err)
		}
	}
	if scalar, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if err = tensor.PowInto(input, scalar, output); err != nil {
		t.Fatal(err)
	}
}

func Test_ReductionsMath_OutputLifetimeAndConcurrency(t *testing.T) {
	type operation struct {
		name      string
		shape     []int64
		run       func(*tensor.Tensor, *tensor.Tensor) error
		reference *tensor.Tensor
		want      []float32
	}
	var (
		input, exponent *tensor.Tensor
		operations      []operation
		group           sync.WaitGroup
		index           int
		err             error
	)
	if input, err = tensor.FromFloat32([]int64{2, 2}, []float32{1, 4, 9, 16}); err != nil {
		t.Fatal(err)
	}
	if exponent, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}
	operations = []operation{
		{name: "sum", shape: []int64{2}, run: func(source, destination *tensor.Tensor) (runErr error) {
			return tensor.ReduceSumInto(source, destination, []int64{1}, false)
		}},
		{name: "mean", shape: []int64{2}, run: func(source, destination *tensor.Tensor) (runErr error) {
			return tensor.ReduceMeanInto(source, destination, []int64{1}, false)
		}},
		{name: "max", shape: []int64{2}, run: func(source, destination *tensor.Tensor) (runErr error) {
			return tensor.ReduceMaxInto(source, destination, []int64{1}, false)
		}},
		{name: "exp", shape: input.Shape(), run: tensor.ExpInto},
		{name: "log", shape: input.Shape(), run: tensor.LogInto},
		{name: "sqrt", shape: input.Shape(), run: tensor.SqrtInto},
		{name: "pow", shape: input.Shape(), run: func(source, destination *tensor.Tensor) (runErr error) {
			return tensor.PowInto(source, exponent, destination)
		}},
	}
	for index = range operations {
		if operations[index].reference, err = tensor.New(tensor.Float32, operations[index].shape); err != nil {
			t.Fatal(err)
		}
		if err = operations[index].run(input, operations[index].reference); err != nil {
			t.Fatal(err)
		}
		if operations[index].want, err = operations[index].reference.Float32Values(); err != nil {
			t.Fatal(err)
		}
	}
	for worker := 0; worker < 4; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for _, op := range operations {
				var (
					output *tensor.Tensor
					values []float32
					runErr error
				)
				if output, runErr = tensor.New(tensor.Float32, op.shape); runErr != nil {
					t.Error(runErr)
					return
				}
				if runErr = op.run(input, output); runErr != nil {
					t.Errorf("%s: %v", op.name, runErr)
					return
				}
				if values, runErr = output.Float32Values(); runErr != nil {
					t.Error(runErr)
					return
				}
				checkProductValues(t, values, op.want)
			}
		}()
	}
	group.Wait()
	for _, op := range operations {
		var values []float32
		if values, err = op.reference.Float32Values(); err != nil {
			t.Fatal(err)
		}
		checkProductValues(t, values, op.want)
	}
	var (
		repeated *tensor.Tensor
		values   []float32
	)
	if repeated, err = tensor.New(tensor.Float32, nil); err != nil {
		t.Fatal(err)
	}
	if err = tensor.PowInto(exponent, exponent, repeated); err != nil {
		t.Fatal(err)
	}
	if values, err = repeated.Float32Values(); err != nil {
		t.Fatal(err)
	}
	checkProductValues(t, values, []float32{4})
	if values, err = input.Float32Values(); err != nil {
		t.Fatal(err)
	}
	checkProductValues(t, values, []float32{1, 4, 9, 16})
}
