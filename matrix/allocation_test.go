package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationMatrixResult *matrix.Matrix
var allocationMatrixValues []float32
var allocationMatrixFloat float32

func Test_MatrixDestinationAllocations(t *testing.T) {
	var (
		left                      *matrix.Matrix
		right                     *matrix.Matrix
		nonZeroRight              *matrix.Matrix
		destination               *matrix.Matrix
		matMulLeft                *matrix.Matrix
		matMulRight               *matrix.Matrix
		matMulDestination         *matrix.Matrix
		leftTransposeRight        *matrix.Matrix
		leftTransposeDestination  *matrix.Matrix
		rightTransposeRight       *matrix.Matrix
		rightTransposeDestination *matrix.Matrix
		transposeDestination      *matrix.Matrix
		rowSumsDestination        *matrix.Matrix
		columnSumsDestination     *matrix.Matrix
		rowVector                 *matrix.Matrix
		values                    []float32
		adamValues                *matrix.Matrix
		adamGradient              *matrix.Matrix
		adamFirstMoment           *matrix.Matrix
		adamSecondMoment          *matrix.Matrix
		err                       error
	)

	left = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	right = mustMatrix(t, 2, 3, []float32{6, 5, 4, 3, 2, 1})
	nonZeroRight = mustMatrix(t, 2, 3, []float32{6, 5, 4, 3, 2, 1})
	destination = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})
	matMulLeft = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	matMulRight = mustMatrix(t, 3, 2, []float32{1, 2, 3, 4, 5, 6})
	matMulDestination = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	leftTransposeRight = mustMatrix(t, 2, 4, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	leftTransposeDestination = mustMatrix(t, 3, 4, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	rightTransposeRight = mustMatrix(t, 4, 3, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	rightTransposeDestination = mustMatrix(t, 2, 4, []float32{0, 0, 0, 0, 0, 0, 0, 0})
	transposeDestination = mustMatrix(t, 3, 2, []float32{0, 0, 0, 0, 0, 0})
	rowSumsDestination = mustMatrix(t, 2, 1, []float32{0, 0})
	columnSumsDestination = mustMatrix(t, 1, 3, []float32{0, 0, 0})
	rowVector = mustMatrix(t, 1, 3, []float32{0.5, 1.5, 2.5})
	values = []float32{9, 8, 7, 6, 5, 4}
	adamValues = mustMatrix(t, 1, 3, []float32{0.5, -0.25, 0.75})
	adamGradient = mustMatrix(t, 1, 3, []float32{0.1, -0.2, 0.3})
	adamFirstMoment = mustMatrix(t, 1, 3, []float32{0, 0, 0})
	adamSecondMoment = mustMatrix(t, 1, 3, []float32{0, 0, 0})

	tests := []struct {
		name string
		run  func()
	}{
		{
			name: "ValuesInto",
			run: func() {
				if err = left.ValuesInto(values); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "CopyValuesFrom",
			run: func() {
				if err = destination.CopyValuesFrom(values); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddInto",
			run: func() {
				if err = left.AddInto(right, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddInPlace",
			run: func() {
				if err = destination.AddInPlace(right); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddScaledInPlace",
			run: func() {
				if err = destination.AddScaledInPlace(right, 0.25); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddMappedInPlace",
			run: func() {
				if err = destination.AddMappedInPlace(right, allocationDouble); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "SubtractInto",
			run: func() {
				if err = left.SubtractInto(right, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MultiplyElementsInto",
			run: func() {
				if err = left.MultiplyElementsInto(right, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "DivideElementsInto",
			run: func() {
				if err = left.DivideElementsInto(nonZeroRight, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddScalarInto",
			run: func() {
				if err = left.AddScalarInto(1.5, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MultiplyScalarInto",
			run: func() {
				if err = left.MultiplyScalarInto(2, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MultiplyScalarInPlace",
			run: func() {
				if err = destination.MultiplyScalarInPlace(0.5); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "DivideScalarInto",
			run: func() {
				if err = left.DivideScalarInto(2, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MatMulInto",
			run: func() {
				if err = matMulLeft.MatMulInto(matMulRight, matMulDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MatMulLeftTransposeInto",
			run: func() {
				if err = matMulLeft.MatMulLeftTransposeInto(leftTransposeRight, leftTransposeDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "MatMulRightTransposeInto",
			run: func() {
				if err = matMulLeft.MatMulRightTransposeInto(rightTransposeRight, rightTransposeDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "TransposeInto",
			run: func() {
				if err = left.TransposeInto(transposeDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "RowSumsInto",
			run: func() {
				if err = left.RowSumsInto(rowSumsDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "ColumnSumsInto",
			run: func() {
				if err = left.ColumnSumsInto(columnSumsDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AccumulateColumnSumsInto",
			run: func() {
				if err = left.AccumulateColumnSumsInto(columnSumsDestination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AddRowVectorInPlace",
			run: func() {
				if err = destination.AddRowVectorInPlace(rowVector); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "ApplyInto",
			run: func() {
				if err = left.ApplyInto(allocationDouble, destination); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "Pairwise",
			run: func() {
				allocationMatrixFloat = 0
				if err = left.Pairwise(right, allocationSumPair); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "PairwiseInto",
			run: func() {
				if err = left.PairwiseInto(right, destination, allocationAddPair); err != nil {
					panic(err)
				}
			},
		},
		{
			name: "AdamUpdateInPlace",
			run: func() {
				err = adamValues.AdamUpdateInPlace(
					adamGradient,
					adamFirstMoment,
					adamSecondMoment,
					0.001,
					0.9,
					0.999,
					1e-8,
					0.1,
					0.001,
				)
				if err != nil {
					panic(err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireMaxAllocs(t, tt.name, 0, tt.run)
		})
	}

	allocationMatrixResult = destination
}

func Test_MatrixCopyAllocationCeilings(t *testing.T) {
	var (
		indexes     []int
		source      *matrix.Matrix
		destination *matrix.Matrix
		result      *matrix.Matrix
		values      []float32
		err         error
	)

	indexes = []int{1, 0}
	source = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	destination = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})

	requireMaxAllocs(t, "SelectRowsInto", 0, func() {
		if err = source.SelectRowsInto(indexes, destination); err != nil {
			panic(err)
		}
	})

	requireMaxAllocs(t, "Values", 1, func() {
		values, err = source.Values()
		if err != nil {
			panic(err)
		}
	})
	allocationMatrixValues = values

	requireMaxAllocs(t, "Clone", 2, func() {
		result, err = source.Clone()
		if err != nil {
			panic(err)
		}
	})
	allocationMatrixResult = result

	requireMaxAllocs(t, "SelectRows", 2, func() {
		result, err = source.SelectRows(indexes)
		if err != nil {
			panic(err)
		}
	})
	allocationMatrixResult = result
}

func allocationDouble(value float32) (out float32) {
	out = value * 2
	return out
}

func allocationSumPair(row, col int, left, right float32) (err error) {
	allocationMatrixFloat += left + right
	return nil
}

func allocationAddPair(row, col int, left, right float32) (value float32, err error) {
	value = left + right
	return value, nil
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
