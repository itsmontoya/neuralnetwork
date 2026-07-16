package matrix_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const epsilon = 1e-5

func Test_New(t *testing.T) {
	var (
		got *matrix.Matrix
		err error
	)

	got, err = matrix.New(2, 3)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if got.Rows() != 2 {
		t.Fatalf("Rows() = %d, want 2", got.Rows())
	}

	if got.Cols() != 3 {
		t.Fatalf("Cols() = %d, want 3", got.Cols())
	}

	requireMatrixValues(t, got, []float32{0, 0, 0, 0, 0, 0})
}

func Test_New_ValidatesShape(t *testing.T) {
	type testcase struct {
		name string
		rows int
		cols int
	}

	tests := []testcase{
		{
			name: "zero rows",
			rows: 0,
			cols: 2,
		},
		{
			name: "zero cols",
			rows: 2,
			cols: 0,
		},
		{
			name: "negative rows",
			rows: -1,
			cols: 2,
		},
		{
			name: "negative cols",
			rows: 2,
			cols: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got *matrix.Matrix
				err error
			)

			got, err = matrix.New(tt.rows, tt.cols)
			if err == nil {
				t.Fatalf("New(%d, %d) error = nil, want error", tt.rows, tt.cols)
			}

			if got != nil {
				t.Fatalf("New(%d, %d) returned matrix on error", tt.rows, tt.cols)
			}
		})
	}
}

func Test_FromSlice(t *testing.T) {
	var (
		values []float32
		got    *matrix.Matrix
		value  float32
		err    error
	)

	values = []float32{1, 2, 3, 4, 5, 6}
	got, err = matrix.FromSlice(2, 3, values)
	if err != nil {
		t.Fatalf("FromSlice returned error: %v", err)
	}

	value, err = got.At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 1, epsilon)

	value, err = got.At(0, 2)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 3, epsilon)

	value, err = got.At(1, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 4, epsilon)

	values[0] = 99
	value, err = got.At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 1, epsilon)
}

func Test_FromSlice_ValidatesLength(t *testing.T) {
	var (
		got *matrix.Matrix
		err error
	)

	got, err = matrix.FromSlice(2, 3, []float32{1, 2, 3})
	if err == nil {
		t.Fatal("FromSlice error = nil, want error")
	}

	if got != nil {
		t.Fatal("FromSlice returned matrix on error")
	}
}

func Test_NewRandom(t *testing.T) {
	var (
		random         *rand.Rand
		expectedRandom *rand.Rand
		got            *matrix.Matrix
		expected       []float32
		index          int
		err            error
	)

	random = rand.New(rand.NewSource(7))
	expectedRandom = rand.New(rand.NewSource(7))
	got, err = matrix.NewRandom(2, 3, random)
	if err != nil {
		t.Fatalf("NewRandom returned error: %v", err)
	}

	expected = make([]float32, 6)
	for index = range expected {
		expected[index] = float32(expectedRandom.Float64())
	}

	requireMatrixValues(t, got, expected)
}

func Test_NewRandom_ValidatesRandomSource(t *testing.T) {
	var (
		got *matrix.Matrix
		err error
	)

	got, err = matrix.NewRandom(2, 3, nil)
	if err == nil {
		t.Fatal("NewRandom error = nil, want error")
	}

	if got != nil {
		t.Fatal("NewRandom returned matrix on error")
	}
}

func Test_AtAndSet(t *testing.T) {
	var (
		got   *matrix.Matrix
		value float32
		err   error
	)

	got = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})

	value, err = got.At(1, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 3, epsilon)

	err = got.Set(1, 0, 7)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, got, []float32{1, 2, 7, 4})
}

func Test_AtAndSet_ValidateIndexes(t *testing.T) {
	var (
		got   *matrix.Matrix
		value float32
		err   error
	)

	got = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})

	value, err = got.At(2, 0)
	if err == nil {
		t.Fatalf("At returned value %g and nil error, want error", value)
	}

	err = got.Set(0, 2, 9)
	if err == nil {
		t.Fatal("Set error = nil, want error")
	}
}

func Test_Fill(t *testing.T) {
	var (
		got *matrix.Matrix
		err error
	)

	got = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	err = got.Fill(8)
	if err != nil {
		t.Fatalf("Fill returned error: %v", err)
	}

	requireMatrixValues(t, got, []float32{8, 8, 8, 8, 8, 8})
}

func Test_CloneAndCopyFrom(t *testing.T) {
	var (
		source *matrix.Matrix
		target *matrix.Matrix
		clone  *matrix.Matrix
		values []float32
		value  float32
		err    error
	)

	source = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})

	clone, err = source.Clone()
	if err != nil {
		t.Fatalf("Clone returned error: %v", err)
	}

	target = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	err = target.CopyFrom(source)
	if err != nil {
		t.Fatalf("CopyFrom returned error: %v", err)
	}

	err = source.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, clone, []float32{1, 2, 3, 4})
	requireMatrixValues(t, target, []float32{1, 2, 3, 4})

	values, err = target.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	values[0] = -1
	value, err = target.At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 1, epsilon)
}

func Test_ValuesInto(t *testing.T) {
	var (
		source *matrix.Matrix
		values []float32
		value  float32
		err    error
	)

	source = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	values = make([]float32, 4)

	err = source.ValuesInto(values)
	if err != nil {
		t.Fatalf("ValuesInto returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, values, []float32{1, 2, 3, 4}, epsilon)

	values[0] = -1
	value, err = source.At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, value, 1, epsilon)
}

func Test_ValuesInto_ValidatesLength(t *testing.T) {
	var (
		source *matrix.Matrix
		err    error
	)

	source = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	err = source.ValuesInto(make([]float32, 3))
	if err == nil {
		t.Fatal("ValuesInto length error = nil, want error")
	}
}

func Test_CopyValuesFrom(t *testing.T) {
	var (
		target *matrix.Matrix
		values []float32
		err    error
	)

	target = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	values = []float32{1, 2, 3, 4}

	err = target.CopyValuesFrom(values)
	if err != nil {
		t.Fatalf("CopyValuesFrom returned error: %v", err)
	}

	values[0] = -1
	requireMatrixValues(t, target, []float32{1, 2, 3, 4})
}

func Test_CopyValuesFrom_ValidatesLength(t *testing.T) {
	var (
		target *matrix.Matrix
		err    error
	)

	target = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	err = target.CopyValuesFrom([]float32{1, 2, 3})
	if err == nil {
		t.Fatal("CopyValuesFrom length error = nil, want error")
	}
}

func Test_SelectRows(t *testing.T) {
	var (
		source *matrix.Matrix
		got    *matrix.Matrix
		err    error
	)

	source = mustMatrix(t, 3, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
	})

	got, err = source.SelectRows([]int{2, 0, 2})
	if err != nil {
		t.Fatalf("SelectRows returned error: %v", err)
	}

	requireMatrixValues(t, got, []float32{
		5, 6,
		1, 2,
		5, 6,
	})

	err = source.Set(2, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	requireMatrixValues(t, got, []float32{
		5, 6,
		1, 2,
		5, 6,
	})
}

func Test_SelectRowsInto(t *testing.T) {
	var (
		source      *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	source = mustMatrix(t, 3, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
	})
	destination = mustMatrix(t, 3, 2, []float32{
		-1, -1,
		-1, -1,
		-1, -1,
	})

	if err = source.SelectRowsInto([]int{2, 0, 2}, destination); err != nil {
		t.Fatalf("SelectRowsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{
		5, 6,
		1, 2,
		5, 6,
	})

	if err = destination.Set(0, 0, 99); err != nil {
		t.Fatalf("destination Set returned error: %v", err)
	}

	requireMatrixValues(t, source, []float32{
		1, 2,
		3, 4,
		5, 6,
	})
}

func Test_SelectRowsInto_ValidatesDestinationAndIndexes(t *testing.T) {
	type testcase struct {
		name        string
		indexes     []int
		destination func(source *matrix.Matrix) *matrix.Matrix
	}

	var tests []testcase
	tests = []testcase{
		{
			name:    "empty indexes",
			indexes: []int{},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = mustMatrix(t, 1, 2, []float32{0, 0})
				return destination
			},
		},
		{
			name:    "negative index",
			indexes: []int{-1},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = mustMatrix(t, 1, 2, []float32{0, 0})
				return destination
			},
		},
		{
			name:    "index too large",
			indexes: []int{3},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = mustMatrix(t, 1, 2, []float32{0, 0})
				return destination
			},
		},
		{
			name:    "wrong rows",
			indexes: []int{0, 1},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = mustMatrix(t, 1, 2, []float32{0, 0})
				return destination
			},
		},
		{
			name:    "wrong columns",
			indexes: []int{0, 1},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = mustMatrix(t, 2, 1, []float32{0, 0})
				return destination
			},
		},
		{
			name:    "nil destination",
			indexes: []int{0},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				return nil
			},
		},
		{
			name:    "aliased destination",
			indexes: []int{2, 1, 0},
			destination: func(source *matrix.Matrix) (destination *matrix.Matrix) {
				destination = source
				return destination
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				source      *matrix.Matrix
				destination *matrix.Matrix
				err         error
			)

			source = mustMatrix(t, 3, 2, []float32{1, 2, 3, 4, 5, 6})
			destination = tt.destination(source)
			err = source.SelectRowsInto(tt.indexes, destination)
			if err == nil {
				t.Fatal("SelectRowsInto error = nil, want error")
			}
		})
	}
}

func Test_SelectRows_ValidatesIndexes(t *testing.T) {
	type testcase struct {
		name    string
		indexes []int
	}

	tests := []testcase{
		{
			name:    "empty",
			indexes: []int{},
		},
		{
			name:    "negative",
			indexes: []int{-1},
		},
		{
			name:    "too large",
			indexes: []int{2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				source *matrix.Matrix
				got    *matrix.Matrix
				err    error
			)

			source = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
			got, err = source.SelectRows(tt.indexes)
			if err == nil {
				t.Fatal("SelectRows error = nil, want error")
			}

			if got != nil {
				t.Fatal("SelectRows returned matrix on error")
			}
		})
	}
}

func Test_CopyFrom_ValidatesShape(t *testing.T) {
	var (
		target *matrix.Matrix
		source *matrix.Matrix
		err    error
	)

	target = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	source = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	err = target.CopyFrom(source)
	if err == nil {
		t.Fatal("CopyFrom error = nil, want error")
	}
}

func Test_Pairwise(t *testing.T) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		sum    float32
		visits int
		err    error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 2, 2, []float32{10, 20, 30, 40})

	err = left.Pairwise(right, func(row, col int, leftValue, rightValue float32) (err error) {
		sum += float32(row+col) + leftValue + rightValue
		visits++
		return nil
	})
	if err != nil {
		t.Fatalf("Pairwise returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, sum, 114, epsilon)
	if visits != 4 {
		t.Fatalf("Pairwise visits = %d, want 4", visits)
	}
}

func Test_Pairwise_ValidatesInputs(t *testing.T) {
	var (
		left  *matrix.Matrix
		right *matrix.Matrix
		err   error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})

	err = left.Pairwise(right, nil)
	if err == nil {
		t.Fatal("Pairwise nil function error = nil, want error")
	}

	err = left.Pairwise(right, func(row, col int, leftValue, rightValue float32) (err error) {
		return nil
	})
	if err == nil {
		t.Fatal("Pairwise shape error = nil, want error")
	}
}

func Test_PairwiseInto(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 2, 2, []float32{10, 20, 30, 40})
	destination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})

	err = left.PairwiseInto(right, destination, func(row, col int, leftValue, rightValue float32) (value float32, err error) {
		value = float32(row+col) + leftValue + rightValue
		return value, nil
	})
	if err != nil {
		t.Fatalf("PairwiseInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{11, 23, 34, 46})
}

func Test_PairwiseInto_ValidatesInputs(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	destination = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})

	err = left.PairwiseInto(left, left, nil)
	if err == nil {
		t.Fatal("PairwiseInto nil function error = nil, want error")
	}

	err = left.PairwiseInto(right, left, func(row, col int, leftValue, rightValue float32) (value float32, err error) {
		return 0, nil
	})
	if err == nil {
		t.Fatal("PairwiseInto input shape error = nil, want error")
	}

	err = left.PairwiseInto(left, destination, func(row, col int, leftValue, rightValue float32) (value float32, err error) {
		return 0, nil
	})
	if err == nil {
		t.Fatal("PairwiseInto destination shape error = nil, want error")
	}
}

func Test_ElementwiseOperations(t *testing.T) {
	var (
		left     *matrix.Matrix
		right    *matrix.Matrix
		result   *matrix.Matrix
		original []float32
		err      error
	)

	left = mustMatrix(t, 2, 2, []float32{2, 4, 6, 8})
	right = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	original = []float32{2, 4, 6, 8}

	result, err = left.Add(right)
	if err != nil {
		t.Fatalf("Add returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{3, 6, 9, 12})

	result, err = left.Subtract(right)
	if err != nil {
		t.Fatalf("Subtract returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{1, 2, 3, 4})

	result, err = left.MultiplyElements(right)
	if err != nil {
		t.Fatalf("MultiplyElements returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{2, 8, 18, 32})

	result, err = left.DivideElements(right)
	if err != nil {
		t.Fatalf("DivideElements returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{2, 2, 2, 2})
	requireMatrixValues(t, left, original)
	requireMatrixValues(t, right, []float32{1, 2, 3, 4})
}

func Test_ElementwiseDestinationOperations(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{2, 4, 6, 8})
	right = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	destination = mustMatrix(t, 2, 2, []float32{-1, -1, -1, -1})

	err = left.AddInto(right, destination)
	if err != nil {
		t.Fatalf("AddInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{3, 6, 9, 12})

	err = left.SubtractInto(right, destination)
	if err != nil {
		t.Fatalf("SubtractInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{1, 2, 3, 4})

	err = left.MultiplyElementsInto(right, destination)
	if err != nil {
		t.Fatalf("MultiplyElementsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{2, 8, 18, 32})

	err = left.DivideElementsInto(right, destination)
	if err != nil {
		t.Fatalf("DivideElementsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{2, 2, 2, 2})

	err = left.AddInPlace(right)
	if err != nil {
		t.Fatalf("AddInPlace returned error: %v", err)
	}

	requireMatrixValues(t, left, []float32{3, 6, 9, 12})

	err = left.AddScaledInPlace(right, -0.5)
	if err != nil {
		t.Fatalf("AddScaledInPlace returned error: %v", err)
	}

	requireMatrixValues(t, left, []float32{2.5, 5, 7.5, 10})

	err = left.AddMappedInPlace(right, func(value float32) (mapped float32) {
		mapped = value * 0.25
		return mapped
	})
	if err != nil {
		t.Fatalf("AddMappedInPlace returned error: %v", err)
	}

	requireMatrixValues(t, left, []float32{2.75, 5.5, 8.25, 11})

	err = left.MultiplyScalarInPlace(2)
	if err != nil {
		t.Fatalf("MultiplyScalarInPlace returned error: %v", err)
	}

	requireMatrixValues(t, left, []float32{5.5, 11, 16.5, 22})
}

func Test_AddMappedInPlace_ValidatesInputs(t *testing.T) {
	var (
		left       *matrix.Matrix
		right      *matrix.Matrix
		mismatched *matrix.Matrix
		err        error
	)

	left = mustMatrix(t, 1, 2, []float32{1, 2})
	right = mustMatrix(t, 1, 2, []float32{3, 4})
	mismatched = mustMatrix(t, 2, 1, []float32{3, 4})

	err = left.AddMappedInPlace(right, nil)
	if err == nil {
		t.Fatal("AddMappedInPlace nil function error = nil, want error")
	}

	err = left.AddMappedInPlace(mismatched, func(value float32) (mapped float32) {
		return value
	})
	if err == nil {
		t.Fatal("AddMappedInPlace shape error = nil, want error")
	}
}

func Test_AddMappedInPlace_AllowsAliasedInput(t *testing.T) {
	var (
		values *matrix.Matrix
		err    error
	)

	values = mustMatrix(t, 1, 3, []float32{-2, 0, 3})
	err = values.AddMappedInPlace(values, func(value float32) (mapped float32) {
		mapped = value * value
		return mapped
	})
	if err != nil {
		t.Fatalf("AddMappedInPlace returned error: %v", err)
	}

	requireMatrixValues(t, values, []float32{2, 0, 12})
}

func Test_AdamUpdateInPlace(t *testing.T) {
	var (
		values       *matrix.Matrix
		gradients    *matrix.Matrix
		firstMoment  *matrix.Matrix
		secondMoment *matrix.Matrix
		err          error
	)

	values = mustMatrix(t, 1, 2, []float32{1, -2})
	gradients = mustMatrix(t, 1, 2, []float32{2, -4})
	firstMoment = mustMatrix(t, 1, 2, []float32{0, 0})
	secondMoment = mustMatrix(t, 1, 2, []float32{0, 0})

	err = values.AdamUpdateInPlace(gradients, firstMoment, secondMoment, 0.1, 0.5, 0.25, 0.1, 0.5, 0.75)
	if err != nil {
		t.Fatalf("AdamUpdateInPlace returned error: %v", err)
	}

	requireMatrixValues(t, firstMoment, []float32{1, -2})
	requireMatrixValues(t, secondMoment, []float32{3, 12})
	requireMatrixValues(t, values, []float32{0.9047619047619048, -1.902439024390244})
}

func Test_AdamUpdateInPlace_ValidatesInputs(t *testing.T) {
	var (
		values       *matrix.Matrix
		gradients    *matrix.Matrix
		firstMoment  *matrix.Matrix
		secondMoment *matrix.Matrix
		mismatched   *matrix.Matrix
		err          error
	)

	values = mustMatrix(t, 1, 2, []float32{1, 2})
	gradients = mustMatrix(t, 1, 2, []float32{0.1, 0.2})
	firstMoment = mustMatrix(t, 1, 2, []float32{0, 0})
	secondMoment = mustMatrix(t, 1, 2, []float32{0, 0})
	mismatched = mustMatrix(t, 2, 1, []float32{0.1, 0.2})

	err = values.AdamUpdateInPlace(mismatched, firstMoment, secondMoment, 0.1, 0.5, 0.25, 0.1, 0.5, 0.75)
	if err == nil {
		t.Fatal("AdamUpdateInPlace gradient shape error = nil, want error")
	}

	err = values.AdamUpdateInPlace(gradients, mismatched, secondMoment, 0.1, 0.5, 0.25, 0.1, 0.5, 0.75)
	if err == nil {
		t.Fatal("AdamUpdateInPlace first moment shape error = nil, want error")
	}

	err = values.AdamUpdateInPlace(values, firstMoment, secondMoment, 0.1, 0.5, 0.25, 0.1, 0.5, 0.75)
	if err == nil {
		t.Fatal("AdamUpdateInPlace alias error = nil, want error")
	}

	err = values.AdamUpdateInPlace(gradients, firstMoment, secondMoment, 0.1, 0.5, 0.25, 0.1, 0, 0.75)
	if err == nil {
		t.Fatal("AdamUpdateInPlace first correction error = nil, want error")
	}

	err = values.AdamUpdateInPlace(gradients, firstMoment, secondMoment, 0.1, 0.5, 0.25, 0.1, 0.5, 0)
	if err == nil {
		t.Fatal("AdamUpdateInPlace second correction error = nil, want error")
	}
}

func Test_ElementwiseDestinationOperations_ValidateShape(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	destination = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})

	err = left.SubtractInto(right, destination)
	if err == nil {
		t.Fatal("SubtractInto destination shape error = nil, want error")
	}

	err = left.MultiplyElementsInto(right, destination)
	if err == nil {
		t.Fatal("MultiplyElementsInto destination shape error = nil, want error")
	}

	err = left.DivideElementsInto(right, destination)
	if err == nil {
		t.Fatal("DivideElementsInto destination shape error = nil, want error")
	}
}

func Test_ElementwiseOperations_ValidateShape(t *testing.T) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	result, err = left.Add(right)
	if err == nil {
		t.Fatalf("Add returned result %v and nil error, want error", result)
	}
}

func Test_DivideElements_ValidatesZeroDenominator(t *testing.T) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
	)

	left = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	right = mustMatrix(t, 1, 3, []float32{1, 0, 3})
	result, err = left.DivideElements(right)
	if err == nil {
		t.Fatalf("DivideElements returned result %v and nil error, want error", result)
	}
}

func Test_DivideElementsInto_ValidatesZeroDenominator(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	right = mustMatrix(t, 1, 3, []float32{1, 0, 3})
	destination = mustMatrix(t, 1, 3, []float32{0, 0, 0})

	err = left.DivideElementsInto(right, destination)
	if err == nil {
		t.Fatal("DivideElementsInto error = nil, want error")
	}
}

func Test_ScalarOperations(t *testing.T) {
	var (
		input    *matrix.Matrix
		result   *matrix.Matrix
		original []float32
		err      error
	)

	input = mustMatrix(t, 2, 2, []float32{2, 4, 6, 8})
	original = []float32{2, 4, 6, 8}

	result, err = input.AddScalar(3)
	if err != nil {
		t.Fatalf("AddScalar returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{5, 7, 9, 11})

	result, err = input.MultiplyScalar(0.5)
	if err != nil {
		t.Fatalf("MultiplyScalar returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{1, 2, 3, 4})

	result, err = input.DivideScalar(2)
	if err != nil {
		t.Fatalf("DivideScalar returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{1, 2, 3, 4})
	requireMatrixValues(t, input, original)
}

func Test_ScalarDestinationOperations(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 2, []float32{2, 4, 6, 8})
	destination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})

	err = input.AddScalarInto(3, destination)
	if err != nil {
		t.Fatalf("AddScalarInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{5, 7, 9, 11})

	err = input.MultiplyScalarInto(0.5, destination)
	if err != nil {
		t.Fatalf("MultiplyScalarInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{1, 2, 3, 4})

	err = input.DivideScalarInto(2, destination)
	if err != nil {
		t.Fatalf("DivideScalarInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{1, 2, 3, 4})
}

func Test_ScalarDestinationOperations_ValidateShape(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 2, []float32{2, 4, 6, 8})
	destination = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})

	err = input.AddScalarInto(3, destination)
	if err == nil {
		t.Fatal("AddScalarInto destination shape error = nil, want error")
	}

	err = input.MultiplyScalarInto(0.5, destination)
	if err == nil {
		t.Fatal("MultiplyScalarInto destination shape error = nil, want error")
	}

	err = input.DivideScalarInto(2, destination)
	if err == nil {
		t.Fatal("DivideScalarInto destination shape error = nil, want error")
	}
}

func Test_DivideScalar_ValidatesZeroDenominator(t *testing.T) {
	var (
		input  *matrix.Matrix
		result *matrix.Matrix
		err    error
	)

	input = mustMatrix(t, 1, 2, []float32{1, 2})
	result, err = input.DivideScalar(0)
	if err == nil {
		t.Fatalf("DivideScalar returned result %v and nil error, want error", result)
	}
}

func Test_DivideScalarInto_ValidatesZeroDenominator(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 1, 2, []float32{1, 2})
	destination = mustMatrix(t, 1, 2, []float32{0, 0})

	err = input.DivideScalarInto(0, destination)
	if err == nil {
		t.Fatal("DivideScalarInto error = nil, want error")
	}
}

func Test_MatMul(t *testing.T) {
	var (
		left         *matrix.Matrix
		right        *matrix.Matrix
		result       *matrix.Matrix
		originalLeft []float32
		err          error
	)

	left = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	right = mustMatrix(t, 3, 2, []float32{7, 8, 9, 10, 11, 12})
	originalLeft = []float32{1, 2, 3, 4, 5, 6}

	result, err = left.MatMul(right)
	if err != nil {
		t.Fatalf("MatMul returned error: %v", err)
	}

	if result.Rows() != 2 {
		t.Fatalf("result Rows() = %d, want 2", result.Rows())
	}

	if result.Cols() != 2 {
		t.Fatalf("result Cols() = %d, want 2", result.Cols())
	}

	requireMatrixValues(t, result, []float32{58, 64, 139, 154})
	requireMatrixValues(t, left, originalLeft)
	requireMatrixValues(t, right, []float32{7, 8, 9, 10, 11, 12})
}

func Test_MatMulInto(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	right = mustMatrix(t, 3, 2, []float32{7, 8, 9, 10, 11, 12})
	destination = mustMatrix(t, 2, 2, []float32{100, 100, 100, 100})

	err = left.MatMulInto(right, destination)
	if err != nil {
		t.Fatalf("MatMulInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{58, 64, 139, 154})
}

func Test_MatMulLeftTransposeInto(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	right = mustMatrix(t, 2, 2, []float32{7, 8, 9, 10})
	destination = mustMatrix(t, 3, 2, []float32{100, 100, 100, 100, 100, 100})

	err = left.MatMulLeftTransposeInto(right, destination)
	if err != nil {
		t.Fatalf("MatMulLeftTransposeInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{43, 48, 59, 66, 75, 84})
}

func Test_MatMulRightTransposeInto(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	right = mustMatrix(t, 2, 3, []float32{7, 8, 9, 10, 11, 12})
	destination = mustMatrix(t, 2, 2, []float32{100, 100, 100, 100})

	err = left.MatMulRightTransposeInto(right, destination)
	if err != nil {
		t.Fatalf("MatMulRightTransposeInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{50, 68, 122, 167})
}

func Test_MatMul_ValidatesShape(t *testing.T) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 3, 1, []float32{1, 2, 3})
	result, err = left.MatMul(right)
	if err == nil {
		t.Fatalf("MatMul returned result %v and nil error, want error", result)
	}
}

func Test_MatMulInto_ValidatesDestination(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 2, 2, []float32{5, 6, 7, 8})
	destination = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})

	err = left.MatMulInto(right, destination)
	if err == nil {
		t.Fatal("MatMulInto shape error = nil, want error")
	}

	err = left.MatMulInto(right, left)
	if err == nil {
		t.Fatal("MatMulInto alias error = nil, want error")
	}
}

func Test_MatMulTransposeInto_ValidatesDestination(t *testing.T) {
	var (
		left        *matrix.Matrix
		right       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	left = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	right = mustMatrix(t, 2, 2, []float32{5, 6, 7, 8})
	destination = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})

	err = left.MatMulLeftTransposeInto(right, destination)
	if err == nil {
		t.Fatal("MatMulLeftTransposeInto shape error = nil, want error")
	}

	err = left.MatMulLeftTransposeInto(right, left)
	if err == nil {
		t.Fatal("MatMulLeftTransposeInto alias error = nil, want error")
	}

	err = left.MatMulRightTransposeInto(right, destination)
	if err == nil {
		t.Fatal("MatMulRightTransposeInto shape error = nil, want error")
	}

	err = left.MatMulRightTransposeInto(right, right)
	if err == nil {
		t.Fatal("MatMulRightTransposeInto alias error = nil, want error")
	}
}

func Test_Transpose(t *testing.T) {
	var (
		input    *matrix.Matrix
		result   *matrix.Matrix
		original []float32
		err      error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	original = []float32{1, 2, 3, 4, 5, 6}

	result, err = input.Transpose()
	if err != nil {
		t.Fatalf("Transpose returned error: %v", err)
	}

	if result.Rows() != 3 {
		t.Fatalf("result Rows() = %d, want 3", result.Rows())
	}

	if result.Cols() != 2 {
		t.Fatalf("result Cols() = %d, want 2", result.Cols())
	}

	requireMatrixValues(t, result, []float32{1, 4, 2, 5, 3, 6})
	requireMatrixValues(t, input, original)
}

func Test_TransposeInto(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	destination = mustMatrix(t, 3, 2, []float32{0, 0, 0, 0, 0, 0})

	err = input.TransposeInto(destination)
	if err != nil {
		t.Fatalf("TransposeInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{1, 4, 2, 5, 3, 6})

	err = input.TransposeInto(input)
	if err == nil {
		t.Fatal("TransposeInto alias error = nil, want error")
	}
}

func Test_RowAndColumnSums(t *testing.T) {
	var (
		input      *matrix.Matrix
		rowSums    []float32
		columnSums []float32
		err        error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})

	rowSums, err = input.RowSums()
	if err != nil {
		t.Fatalf("RowSums returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, rowSums, []float32{6, 15}, epsilon)

	columnSums, err = input.ColumnSums()
	if err != nil {
		t.Fatalf("ColumnSums returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, columnSums, []float32{5, 7, 9}, epsilon)
}

func Test_RowSumsInto(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	destination = mustMatrix(t, 2, 1, []float32{100, 100})

	err = input.RowSumsInto(destination)
	if err != nil {
		t.Fatalf("RowSumsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{6, 15})

	err = input.RowSumsInto(input)
	if err == nil {
		t.Fatal("RowSumsInto alias error = nil, want error")
	}
}

func Test_ColumnSumsInto(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	destination = mustMatrix(t, 1, 3, []float32{100, 100, 100})

	err = input.ColumnSumsInto(destination)
	if err != nil {
		t.Fatalf("ColumnSumsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{5, 7, 9})

	input = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	err = input.ColumnSumsInto(input)
	if err == nil {
		t.Fatal("ColumnSumsInto alias error = nil, want error")
	}
}

func Test_ColumnSumsIntoWideShape(t *testing.T) {
	var (
		inputValues       []float32
		destinationValues []float32
		want              []float32
		input             *matrix.Matrix
		destination       *matrix.Matrix
		err               error
		col               int
	)

	inputValues = make([]float32, 34)
	destinationValues = make([]float32, 17)
	want = make([]float32, 17)
	for col = 0; col < 17; col++ {
		inputValues[col] = float32(col + 1)
		inputValues[17+col] = float32(-2 * (col + 1))
		destinationValues[col] = 100
		want[col] = inputValues[col] + inputValues[17+col]
	}

	input = mustMatrix(t, 2, 17, inputValues)
	destination = mustMatrix(t, 1, 17, destinationValues)

	err = input.ColumnSumsInto(destination)
	if err != nil {
		t.Fatalf("ColumnSumsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, want)
}

func Test_ReductionDestinationOperations_ValidateShape(t *testing.T) {
	var (
		input             *matrix.Matrix
		rowDestination    *matrix.Matrix
		columnDestination *matrix.Matrix
		err               error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	rowDestination = mustMatrix(t, 1, 2, []float32{0, 0})
	columnDestination = mustMatrix(t, 3, 1, []float32{0, 0, 0})

	err = input.RowSumsInto(rowDestination)
	if err == nil {
		t.Fatal("RowSumsInto destination shape error = nil, want error")
	}

	err = input.ColumnSumsInto(columnDestination)
	if err == nil {
		t.Fatal("ColumnSumsInto destination shape error = nil, want error")
	}

	err = input.AccumulateColumnSumsInto(columnDestination)
	if err == nil {
		t.Fatal("AccumulateColumnSumsInto destination shape error = nil, want error")
	}
}

func Test_AccumulateColumnSumsInto(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	destination = mustMatrix(t, 1, 3, []float32{10, 20, 30})

	err = input.AccumulateColumnSumsInto(destination)
	if err != nil {
		t.Fatalf("AccumulateColumnSumsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{15, 27, 39})

	err = input.AccumulateColumnSumsInto(input)
	if err == nil {
		t.Fatal("AccumulateColumnSumsInto alias error = nil, want error")
	}
}

func Test_AccumulateColumnSumsIntoWideShape(t *testing.T) {
	var (
		inputValues       []float32
		destinationValues []float32
		want              []float32
		input             *matrix.Matrix
		destination       *matrix.Matrix
		err               error
		col               int
	)

	inputValues = make([]float32, 34)
	destinationValues = make([]float32, 17)
	want = make([]float32, 17)
	for col = 0; col < 17; col++ {
		inputValues[col] = float32(col + 1)
		inputValues[17+col] = float32(-2 * (col + 1))
		destinationValues[col] = float32(100 + col)
		want[col] = destinationValues[col] + inputValues[col] + inputValues[17+col]
	}

	input = mustMatrix(t, 2, 17, inputValues)
	destination = mustMatrix(t, 1, 17, destinationValues)

	err = input.AccumulateColumnSumsInto(destination)
	if err != nil {
		t.Fatalf("AccumulateColumnSumsInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, want)
}

func Test_AddRowVectorInPlace(t *testing.T) {
	var (
		input     *matrix.Matrix
		rowVector *matrix.Matrix
		err       error
	)

	input = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	rowVector = mustMatrix(t, 1, 3, []float32{10, 20, 30})

	err = input.AddRowVectorInPlace(rowVector)
	if err != nil {
		t.Fatalf("AddRowVectorInPlace returned error: %v", err)
	}

	requireMatrixValues(t, input, []float32{11, 22, 33, 14, 25, 36})
}

func Test_Apply(t *testing.T) {
	var (
		input    *matrix.Matrix
		result   *matrix.Matrix
		original []float32
		err      error
	)

	input = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	original = []float32{1, 2, 3, 4}

	result, err = input.Apply(func(value float32) (result float32) {
		result = value * value
		return result
	})
	if err != nil {
		t.Fatalf("Apply returned error: %v", err)
	}

	requireMatrixValues(t, result, []float32{1, 4, 9, 16})
	requireMatrixValues(t, input, original)
}

func Test_ApplyInto(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	destination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})

	err = input.ApplyInto(func(value float32) (result float32) {
		result = value * value
		return result
	}, destination)
	if err != nil {
		t.Fatalf("ApplyInto returned error: %v", err)
	}

	requireMatrixValues(t, destination, []float32{1, 4, 9, 16})
}

func Test_Apply_ValidatesFunction(t *testing.T) {
	var (
		input  *matrix.Matrix
		result *matrix.Matrix
		err    error
	)

	input = mustMatrix(t, 1, 2, []float32{1, 2})
	result, err = input.Apply(nil)
	if err == nil {
		t.Fatalf("Apply returned result %v and nil error, want error", result)
	}
}

func Test_ApplyInto_ValidatesFunction(t *testing.T) {
	var (
		input       *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	input = mustMatrix(t, 1, 2, []float32{1, 2})
	destination = mustMatrix(t, 1, 2, []float32{0, 0})

	err = input.ApplyInto(nil, destination)
	if err == nil {
		t.Fatal("ApplyInto error = nil, want error")
	}
}

func Test_NilMatrixValidation(t *testing.T) {
	var (
		input  *matrix.Matrix
		values []float32
		err    error
	)

	values, err = input.Values()
	if err == nil {
		t.Fatalf("Values returned values %v and nil error, want error", values)
	}
}

func mustMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	tb.Helper()

	var err error
	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float32) {
	tb.Helper()

	var (
		values []float32
		err    error
	)

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}
