package data_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const epsilon = 1e-5

func Test_NewDataset_ValidatesSampleCount(t *testing.T) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		got     *data.Dataset
		err     error
	)

	inputs = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	targets = mustMatrix(t, 1, 1, []float32{1})

	got, err = data.NewDataset(inputs, targets)
	if err == nil {
		t.Fatal("NewDataset error = nil, want error")
	}

	if got != nil {
		t.Fatal("NewDataset returned dataset on error")
	}
}

func Test_Dataset_CopiesMatrices(t *testing.T) {
	var (
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		dataset         *data.Dataset
		datasetInputs   *matrix.Matrix
		datasetTargets  *matrix.Matrix
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		err             error
	)

	inputs = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	targets = mustMatrix(t, 2, 1, []float32{10, 20})

	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}

	err = inputs.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	err = targets.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	datasetInputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	datasetTargets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, datasetInputs, []float32{1, 2, 3, 4})
	requireMatrixValues(t, datasetTargets, []float32{10, 20})

	err = datasetInputs.Set(0, 0, 77)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	err = datasetTargets.Set(0, 0, 77)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	returnedInputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	returnedTargets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{1, 2, 3, 4})
	requireMatrixValues(t, returnedTargets, []float32{10, 20})
}

func Test_Dataset_InputsIntoAndTargetsIntoCopyMatrices(t *testing.T) {
	var (
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		dataset         *data.Dataset
		inputsCopy      *matrix.Matrix
		targetsCopy     *matrix.Matrix
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		err             error
	)

	inputs = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	targets = mustMatrix(t, 2, 1, []float32{10, 20})
	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}

	inputsCopy = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	targetsCopy = mustMatrix(t, 2, 1, []float32{0, 0})
	if err = dataset.InputsInto(inputsCopy); err != nil {
		t.Fatalf("InputsInto returned error: %v", err)
	}

	if err = dataset.TargetsInto(targetsCopy); err != nil {
		t.Fatalf("TargetsInto returned error: %v", err)
	}

	requireMatrixValues(t, inputsCopy, []float32{1, 2, 3, 4})
	requireMatrixValues(t, targetsCopy, []float32{10, 20})

	if err = inputsCopy.Set(0, 0, 99); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	if err = targetsCopy.Set(0, 0, 99); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	if returnedInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if returnedTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{1, 2, 3, 4})
	requireMatrixValues(t, returnedTargets, []float32{10, 20})
}

func Test_Dataset_InputsIntoAndTargetsIntoRejectWrongShape(t *testing.T) {
	var (
		dataset     *data.Dataset
		wrongInputs *matrix.Matrix
		err         error
	)

	dataset = mustDatasetWithSamples(t, 2)
	wrongInputs = mustMatrix(t, 1, 2, []float32{0, 0})

	err = dataset.InputsInto(wrongInputs)
	if err == nil {
		t.Fatal("InputsInto error = nil, want error")
	}

	err = dataset.TargetsInto(wrongInputs)
	if err == nil {
		t.Fatal("TargetsInto error = nil, want error")
	}
}

func Test_Dataset_BatchesReturnsExpectedCounts(t *testing.T) {
	type testcase struct {
		name        string
		samples     int
		batchSize   int
		wantSamples []int
	}

	tests := []testcase{
		{
			name:        "even batches",
			samples:     4,
			batchSize:   2,
			wantSamples: []int{2, 2},
		},
		{
			name:        "last partial batch",
			samples:     5,
			batchSize:   2,
			wantSamples: []int{2, 2, 1},
		},
		{
			name:        "batch larger than dataset",
			samples:     3,
			batchSize:   5,
			wantSamples: []int{3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dataset *data.Dataset
				batches []*data.Batch
				index   int
				err     error
			)

			dataset = mustDatasetWithSamples(t, tt.samples)
			batches, err = dataset.Batches(tt.batchSize, nil)
			if err != nil {
				t.Fatalf("Batches returned error: %v", err)
			}

			if len(batches) != len(tt.wantSamples) {
				t.Fatalf("Batches length = %d, want %d", len(batches), len(tt.wantSamples))
			}

			for index = range batches {
				if batches[index].SampleCount() == tt.wantSamples[index] {
					continue
				}

				t.Fatalf(
					"batch %d sample count = %d, want %d",
					index,
					batches[index].SampleCount(),
					tt.wantSamples[index],
				)
			}
		})
	}
}

func Test_Dataset_BatchesReturnsLastPartialBatchValues(t *testing.T) {
	var (
		dataset *data.Dataset
		batches []*data.Batch
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	dataset = mustDataset(t,
		5,
		2,
		[]float32{
			1, 10,
			2, 20,
			3, 30,
			4, 40,
			5, 50,
		},
		1,
		[]float32{101, 102, 103, 104, 105},
	)

	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	inputs, err = batches[2].Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = batches[2].Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	if batches[2].SampleCount() != 1 {
		t.Fatalf("last batch sample count = %d, want 1", batches[2].SampleCount())
	}

	requireMatrixValues(t, inputs, []float32{5, 50})
	requireMatrixValues(t, targets, []float32{105})
}

func Test_Dataset_BatchesCopiesReturnedMatrices(t *testing.T) {
	var (
		dataset         *data.Dataset
		batches         []*data.Batch
		batchInputs     *matrix.Matrix
		batchTargets    *matrix.Matrix
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		err             error
	)

	dataset = mustDataset(t,
		2,
		2,
		[]float32{
			1, 10,
			2, 20,
		},
		1,
		[]float32{101, 102},
	)

	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	batchInputs, err = batches[0].Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	batchTargets, err = batches[0].Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	err = batchInputs.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	err = batchTargets.Set(0, 0, 99)
	if err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	returnedInputs, err = batches[0].Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	returnedTargets, err = batches[0].Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{
		1, 10,
		2, 20,
	})
	requireMatrixValues(t, returnedTargets, []float32{101, 102})
}

func Test_Batch_InputsIntoAndTargetsIntoCopyMatrices(t *testing.T) {
	var (
		dataset         *data.Dataset
		batches         []*data.Batch
		batchInputs     *matrix.Matrix
		batchTargets    *matrix.Matrix
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		err             error
	)

	dataset = mustDataset(t,
		2,
		2,
		[]float32{
			1, 10,
			2, 20,
		},
		1,
		[]float32{101, 102},
	)

	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	batchInputs = mustMatrix(t, 2, 2, []float32{0, 0, 0, 0})
	batchTargets = mustMatrix(t, 2, 1, []float32{0, 0})
	if err = batches[0].InputsInto(batchInputs); err != nil {
		t.Fatalf("InputsInto returned error: %v", err)
	}

	if err = batches[0].TargetsInto(batchTargets); err != nil {
		t.Fatalf("TargetsInto returned error: %v", err)
	}

	requireMatrixValues(t, batchInputs, []float32{
		1, 10,
		2, 20,
	})
	requireMatrixValues(t, batchTargets, []float32{101, 102})

	if err = batchInputs.Set(0, 0, 99); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	if err = batchTargets.Set(0, 0, 99); err != nil {
		t.Fatalf("Set returned error: %v", err)
	}

	if returnedInputs, err = batches[0].Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if returnedTargets, err = batches[0].Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{
		1, 10,
		2, 20,
	})
	requireMatrixValues(t, returnedTargets, []float32{101, 102})
}

func Test_Batch_InputsIntoAndTargetsIntoRejectWrongShape(t *testing.T) {
	var (
		dataset     *data.Dataset
		batches     []*data.Batch
		wrongInputs *matrix.Matrix
		err         error
	)

	dataset = mustDatasetWithSamples(t, 2)
	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	wrongInputs = mustMatrix(t, 1, 2, []float32{0, 0})
	err = batches[0].InputsInto(wrongInputs)
	if err == nil {
		t.Fatal("InputsInto error = nil, want error")
	}

	err = batches[0].TargetsInto(wrongInputs)
	if err == nil {
		t.Fatal("TargetsInto error = nil, want error")
	}
}

func Test_Dataset_BatchesShufflesDeterministicallyAndKeepsRowsAligned(t *testing.T) {
	var (
		dataset      *data.Dataset
		first        []*data.Batch
		second       []*data.Batch
		firstInputs  []float32
		firstTargets []float32
		secondInputs []float32
		inputValue   float32
		targetValue  float32
		row          int
		err          error
	)

	dataset = mustDataset(t,
		6,
		2,
		[]float32{
			1, 10,
			2, 20,
			3, 30,
			4, 40,
			5, 50,
			6, 60,
		},
		1,
		[]float32{101, 102, 103, 104, 105, 106},
	)

	first, err = dataset.Batches(2, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	second, err = dataset.Batches(2, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	firstInputs = flattenBatchInputs(t, first)
	firstTargets = flattenBatchTargets(t, first)
	secondInputs = flattenBatchInputs(t, second)

	testutil.RequireSliceAlmostEqual(t, firstInputs, secondInputs, epsilon)

	for row = 0; row < len(firstTargets); row++ {
		inputValue = firstInputs[row*2]
		targetValue = firstTargets[row]

		testutil.RequireAlmostEqual(t, firstInputs[row*2+1], inputValue*10, epsilon)
		testutil.RequireAlmostEqual(t, targetValue, inputValue+100, epsilon)
	}
}

func Test_Dataset_BatchesRejectsInvalidBatchSize(t *testing.T) {
	var (
		dataset *data.Dataset
		batches []*data.Batch
		err     error
	)

	dataset = mustDatasetWithSamples(t, 2)
	batches, err = dataset.Batches(0, nil)
	if err == nil {
		t.Fatal("Batches error = nil, want error")
	}

	if batches != nil {
		t.Fatal("Batches returned batches on error")
	}
}

func Test_Dataset_SplitPreservesOrderWithoutShuffle(t *testing.T) {
	var (
		dataset      *data.Dataset
		train        *data.Dataset
		test         *data.Dataset
		trainInputs  *matrix.Matrix
		trainTargets *matrix.Matrix
		testInputs   *matrix.Matrix
		testTargets  *matrix.Matrix
		err          error
	)

	dataset = mustDataset(t,
		4,
		2,
		[]float32{
			1, 10,
			2, 20,
			3, 30,
			4, 40,
		},
		1,
		[]float32{101, 102, 103, 104},
	)

	train, test, err = dataset.Split(0.25, nil)
	if err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	if train.SampleCount() != 3 {
		t.Fatalf("train sample count = %d, want 3", train.SampleCount())
	}

	if test.SampleCount() != 1 {
		t.Fatalf("test sample count = %d, want 1", test.SampleCount())
	}

	trainInputs, err = train.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	trainTargets, err = train.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	testInputs, err = test.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	testTargets, err = test.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, trainInputs, []float32{1, 10, 2, 20, 3, 30})
	requireMatrixValues(t, trainTargets, []float32{101, 102, 103})
	requireMatrixValues(t, testInputs, []float32{4, 40})
	requireMatrixValues(t, testTargets, []float32{104})
}

func Test_Dataset_SplitUsesSeedAndDoesNotMutateOriginal(t *testing.T) {
	var (
		dataset        *data.Dataset
		trainOne       *data.Dataset
		testOne        *data.Dataset
		trainTwo       *data.Dataset
		testTwo        *data.Dataset
		originalInput  *matrix.Matrix
		trainInputs    *matrix.Matrix
		trainTargets   *matrix.Matrix
		testInputs     *matrix.Matrix
		testTargets    *matrix.Matrix
		currentInput   *matrix.Matrix
		originalValues []float32
		currentValues  []float32
		err            error
	)

	dataset = mustDataset(t,
		5,
		2,
		[]float32{
			1, 10,
			2, 20,
			3, 30,
			4, 40,
			5, 50,
		},
		1,
		[]float32{101, 102, 103, 104, 105},
	)

	originalInput, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	trainOne, testOne, err = dataset.Split(0.4, rand.New(rand.NewSource(19)))
	if err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	trainTwo, testTwo, err = dataset.Split(0.4, rand.New(rand.NewSource(19)))
	if err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	requireDatasetInputsEqual(t, trainOne, trainTwo)
	requireDatasetInputsEqual(t, testOne, testTwo)

	trainInputs, err = trainOne.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	trainTargets, err = trainOne.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	testInputs, err = testOne.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	testTargets, err = testOne.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireAlignedRows(t, trainInputs, trainTargets)
	requireAlignedRows(t, testInputs, testTargets)

	currentInput, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	originalValues, err = originalInput.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	currentValues, err = currentInput.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, currentValues, originalValues, epsilon)
}

func Test_Dataset_SplitRejectsInvalidFraction(t *testing.T) {
	var (
		dataset *data.Dataset
		train   *data.Dataset
		test    *data.Dataset
		err     error
	)

	dataset = mustDatasetWithSamples(t, 2)
	train, test, err = dataset.Split(0, nil)
	if err == nil {
		t.Fatal("Split error = nil, want error")
	}

	if train != nil {
		t.Fatal("Split returned train dataset on error")
	}

	if test != nil {
		t.Fatal("Split returned test dataset on error")
	}
}

func mustDatasetWithSamples(tb testing.TB, samples int) (dataset *data.Dataset) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
	)

	tb.Helper()

	inputValues = make([]float32, samples*2)
	targetValues = make([]float32, samples)
	for row = 0; row < samples; row++ {
		inputValues[row*2] = float32(row + 1)
		inputValues[row*2+1] = float32((row + 1) * 10)
		targetValues[row] = float32(row + 101)
	}

	dataset = mustDataset(tb, samples, 2, inputValues, 1, targetValues)
	return dataset
}

func mustDataset(
	tb testing.TB,
	rows int,
	inputCols int,
	inputValues []float32,
	targetCols int,
	targetValues []float32,
) (dataset *data.Dataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	inputs = mustMatrix(tb, rows, inputCols, inputValues)
	targets = mustMatrix(tb, rows, targetCols, targetValues)

	dataset, err = data.NewDataset(inputs, targets)
	if err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}

	return dataset
}

func mustMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func flattenBatchInputs(tb testing.TB, batches []*data.Batch) (values []float32) {
	var (
		batch       *data.Batch
		inputs      *matrix.Matrix
		inputValues []float32
		err         error
	)

	tb.Helper()

	for _, batch = range batches {
		inputs, err = batch.Inputs()
		if err != nil {
			tb.Fatalf("Inputs returned error: %v", err)
		}

		inputValues, err = inputs.Values()
		if err != nil {
			tb.Fatalf("Values returned error: %v", err)
		}

		values = append(values, inputValues...)
	}

	return values
}

func flattenBatchTargets(tb testing.TB, batches []*data.Batch) (values []float32) {
	var (
		batch        *data.Batch
		targets      *matrix.Matrix
		targetValues []float32
		err          error
	)

	tb.Helper()

	for _, batch = range batches {
		targets, err = batch.Targets()
		if err != nil {
			tb.Fatalf("Targets returned error: %v", err)
		}

		targetValues, err = targets.Values()
		if err != nil {
			tb.Fatalf("Values returned error: %v", err)
		}

		values = append(values, targetValues...)
	}

	return values
}

func requireDatasetInputsEqual(tb testing.TB, got, want *data.Dataset) {
	var (
		gotInputs  *matrix.Matrix
		wantInputs *matrix.Matrix
		gotValues  []float32
		wantValues []float32
		err        error
	)

	tb.Helper()

	gotInputs, err = got.Inputs()
	if err != nil {
		tb.Fatalf("Inputs returned error: %v", err)
	}

	wantInputs, err = want.Inputs()
	if err != nil {
		tb.Fatalf("Inputs returned error: %v", err)
	}

	gotValues, err = gotInputs.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	wantValues, err = wantInputs.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, gotValues, wantValues, epsilon)
}

func requireAlignedRows(tb testing.TB, inputs, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		err          error
	)

	tb.Helper()

	inputValues, err = inputs.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	for row = 0; row < targets.Rows(); row++ {
		testutil.RequireAlmostEqual(tb, inputValues[row*2+1], inputValues[row*2]*10, epsilon)
		testutil.RequireAlmostEqual(tb, targetValues[row], inputValues[row*2]+100, epsilon)
	}
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float32) {
	var (
		values []float32
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}
