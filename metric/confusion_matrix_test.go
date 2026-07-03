package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_NewBinaryConfusionMatrix(t *testing.T) {
	var (
		predictions     *matrix.Matrix
		targets         *matrix.Matrix
		confusionMatrix *metric.ConfusionMatrix
		counts          [][]int
		count           int
		value           float64
		err             error
	)

	predictions = mustMatrix(t, 5, 1, []float64{0.2, 0.8, 0.6, 0.4, 0.9})
	targets = mustMatrix(t, 5, 1, []float64{0, 1, 0, 1, 1})

	confusionMatrix, err = metric.NewBinaryConfusionMatrix(predictions, targets)
	if err != nil {
		t.Fatalf("NewBinaryConfusionMatrix returned error: %v", err)
	}

	if confusionMatrix.ClassCount() != 2 {
		t.Fatalf("ClassCount = %d, want 2", confusionMatrix.ClassCount())
	}

	if confusionMatrix.Total() != 5 {
		t.Fatalf("Total = %d, want 5", confusionMatrix.Total())
	}

	counts, err = confusionMatrix.Counts()
	if err != nil {
		t.Fatalf("Counts returned error: %v", err)
	}

	requireCounts(t, counts, [][]int{
		{1, 1},
		{1, 2},
	})

	counts[0][0] = 99
	count, err = confusionMatrix.At(0, 0)
	if err != nil {
		t.Fatalf("At returned error: %v", err)
	}

	if count != 1 {
		t.Fatalf("At after Counts mutation = %d, want 1", count)
	}

	value, err = confusionMatrix.Accuracy()
	if err != nil {
		t.Fatalf("Accuracy returned error: %v", err)
	}
	requireAlmostEqual(t, value, 3.0/5.0)

	value, err = confusionMatrix.Precision(1)
	if err != nil {
		t.Fatalf("Precision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)

	value, err = confusionMatrix.Recall(1)
	if err != nil {
		t.Fatalf("Recall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)

	value, err = confusionMatrix.F1(1)
	if err != nil {
		t.Fatalf("F1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)

	value, err = confusionMatrix.MacroPrecision()
	if err != nil {
		t.Fatalf("MacroPrecision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 7.0/12.0)

	value, err = confusionMatrix.MacroRecall()
	if err != nil {
		t.Fatalf("MacroRecall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 7.0/12.0)

	value, err = confusionMatrix.MacroF1()
	if err != nil {
		t.Fatalf("MacroF1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 7.0/12.0)
}

func Test_NewBinaryConfusionMatrixWithThreshold(t *testing.T) {
	var (
		predictions     *matrix.Matrix
		targets         *matrix.Matrix
		confusionMatrix *metric.ConfusionMatrix
		counts          [][]int
		err             error
	)

	predictions = mustMatrix(t, 3, 1, []float64{0.49, 0.5, 0.51})
	targets = mustMatrix(t, 3, 1, []float64{0, 0, 1})

	confusionMatrix, err = metric.NewBinaryConfusionMatrixWithThreshold(predictions, targets, 0.51)
	if err != nil {
		t.Fatalf("NewBinaryConfusionMatrixWithThreshold returned error: %v", err)
	}

	counts, err = confusionMatrix.Counts()
	if err != nil {
		t.Fatalf("Counts returned error: %v", err)
	}

	requireCounts(t, counts, [][]int{
		{2, 0},
		{0, 1},
	})
}

func Test_NewCategoricalConfusionMatrix(t *testing.T) {
	var (
		predictions     *matrix.Matrix
		targets         *matrix.Matrix
		confusionMatrix *metric.ConfusionMatrix
		counts          [][]int
		value           float64
		err             error
	)

	predictions = mustMatrix(t, 4, 3, []float64{
		0.1, 0.8, 0.1,
		0.6, 0.2, 0.2,
		0.1, 0.2, 0.7,
		0.1, 0.3, 0.6,
	})
	targets = mustMatrix(t, 4, 3, []float64{
		0, 1, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})

	confusionMatrix, err = metric.NewCategoricalConfusionMatrix(predictions, targets)
	if err != nil {
		t.Fatalf("NewCategoricalConfusionMatrix returned error: %v", err)
	}

	counts, err = confusionMatrix.Counts()
	if err != nil {
		t.Fatalf("Counts returned error: %v", err)
	}

	requireCounts(t, counts, [][]int{
		{1, 0, 0},
		{0, 1, 1},
		{0, 0, 1},
	})

	value, err = confusionMatrix.Accuracy()
	if err != nil {
		t.Fatalf("Accuracy returned error: %v", err)
	}
	requireAlmostEqual(t, value, 3.0/4.0)

	value, err = confusionMatrix.Precision(2)
	if err != nil {
		t.Fatalf("Precision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 0.5)

	value, err = confusionMatrix.Recall(1)
	if err != nil {
		t.Fatalf("Recall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 0.5)

	value, err = confusionMatrix.MacroPrecision()
	if err != nil {
		t.Fatalf("MacroPrecision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 5.0/6.0)

	value, err = confusionMatrix.MacroRecall()
	if err != nil {
		t.Fatalf("MacroRecall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 5.0/6.0)

	value, err = confusionMatrix.MacroF1()
	if err != nil {
		t.Fatalf("MacroF1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 7.0/9.0)
}

func Test_ConfusionMatrix_ValidatesClassIndex(t *testing.T) {
	var (
		predictions     *matrix.Matrix
		targets         *matrix.Matrix
		confusionMatrix *metric.ConfusionMatrix
		err             error
	)

	predictions = mustMatrix(t, 1, 1, []float64{0.5})
	targets = mustMatrix(t, 1, 1, []float64{1})

	confusionMatrix, err = metric.NewBinaryConfusionMatrix(predictions, targets)
	if err != nil {
		t.Fatalf("NewBinaryConfusionMatrix returned error: %v", err)
	}

	_, err = confusionMatrix.At(0, 2)
	if err == nil {
		t.Fatal("At error = nil, want error")
	}
}

func Test_ConfusionMatrix_ValidatesInputs(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 1, 2, []float64{0.4, 0.6})
	targets = mustMatrix(t, 1, 2, []float64{0, 1})

	_, err = metric.NewBinaryConfusionMatrix(predictions, targets)
	if err == nil {
		t.Fatal("NewBinaryConfusionMatrix error = nil, want error")
	}

	predictions = mustMatrix(t, 1, 3, []float64{0.1, 0.8, 0.1})
	targets = mustMatrix(t, 1, 3, []float64{0.5, 0.5, 0})

	_, err = metric.NewCategoricalConfusionMatrix(predictions, targets)
	if err == nil {
		t.Fatal("NewCategoricalConfusionMatrix error = nil, want error")
	}
}

func requireCounts(tb testing.TB, got, want [][]int) {
	var (
		row int
		col int
	)

	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("count row length = %d, want %d", len(got), len(want))
	}

	for row = range got {
		if len(got[row]) != len(want[row]) {
			tb.Fatalf("count column length at row %d = %d, want %d", row, len(got[row]), len(want[row]))
		}

		for col = range got[row] {
			if got[row][col] == want[row][col] {
				continue
			}

			tb.Fatalf("count at row %d column %d = %d, want %d", row, col, got[row][col], want[row][col])
		}
	}
}
