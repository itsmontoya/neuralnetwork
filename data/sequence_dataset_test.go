package data_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_NewSequenceDataset_ValidatesAlignedInputs(t *testing.T) {
	type testcase struct {
		name    string
		inputs  func() *matrix.Matrix
		targets func() *matrix.Matrix
		lengths func() *data.SequenceLengths
	}

	var tests []testcase
	tests = []testcase{
		{
			name: "nil inputs",
			inputs: func() (inputs *matrix.Matrix) {
				return nil
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 1, 1, []float32{1})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = mustSequenceLengths(t, 1, []int{1})
				return lengths
			},
		},
		{
			name: "nil targets",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 1, 1, []float32{1})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				return nil
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = mustSequenceLengths(t, 1, []int{1})
				return lengths
			},
		},
		{
			name: "matrix row mismatch",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 1, 1, []float32{1})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = mustSequenceLengths(t, 1, []int{1, 1})
				return lengths
			},
		},
		{
			name: "nil lengths",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 1, 2, []float32{1, 2})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 1, 1, []float32{1})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				return nil
			},
		},
		{
			name: "zero lengths",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 1, 2, []float32{1, 2})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 1, 1, []float32{1})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = &data.SequenceLengths{}
				return lengths
			},
		},
		{
			name: "length count mismatch",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 2, 1, []float32{1, 2})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = mustSequenceLengths(t, 1, []int{1})
				return lengths
			},
		},
		{
			name: "input width not divisible by steps",
			inputs: func() (inputs *matrix.Matrix) {
				inputs = mustMatrix(t, 1, 3, []float32{1, 2, 3})
				return inputs
			},
			targets: func() (targets *matrix.Matrix) {
				targets = mustMatrix(t, 1, 1, []float32{1})
				return targets
			},
			lengths: func() (lengths *data.SequenceLengths) {
				lengths = mustSequenceLengths(t, 2, []int{1})
				return lengths
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dataset *data.SequenceDataset
				err     error
			)

			dataset, err = data.NewSequenceDataset(tt.inputs(), tt.targets(), tt.lengths())
			if err == nil {
				t.Fatal("NewSequenceDataset error = nil, want error")
			}

			if dataset != nil {
				t.Fatal("NewSequenceDataset returned dataset on error")
			}
		})
	}
}

func Test_SequenceDataset_CopiesConstructorArgumentsAndAccessorResults(t *testing.T) {
	var (
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		lengthValues    []int
		lengths         *data.SequenceLengths
		dataset         *data.SequenceDataset
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		returnedLengths *data.SequenceLengths
		values          []int
		err             error
	)

	inputs = mustMatrix(t, 2, 4, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
	})
	targets = mustMatrix(t, 2, 1, []float32{101, 102})
	lengthValues = []int{2, 1}
	lengths = mustSequenceLengths(t, 2, lengthValues)
	dataset, err = data.NewSequenceDataset(inputs, targets, lengths)
	if err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	if err = inputs.Set(0, 0, 99); err != nil {
		t.Fatalf("inputs Set returned error: %v", err)
	}

	if err = targets.Set(0, 0, 99); err != nil {
		t.Fatalf("targets Set returned error: %v", err)
	}
	lengthValues[0] = 1

	returnedInputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	returnedTargets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	returnedLengths, err = dataset.Lengths()
	if err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}

	if err = returnedInputs.Set(0, 0, 88); err != nil {
		t.Fatalf("returned inputs Set returned error: %v", err)
	}

	if err = returnedTargets.Set(0, 0, 88); err != nil {
		t.Fatalf("returned targets Set returned error: %v", err)
	}

	values, err = returnedLengths.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}
	values[0] = 1

	returnedInputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	returnedTargets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	if err = dataset.LengthsInto(values); err != nil {
		t.Fatalf("LengthsInto returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
	})
	requireMatrixValues(t, returnedTargets, []float32{101, 102})
	requireIntValues(t, values, []int{2, 1})

	if dataset.SampleCount() != 2 || dataset.InputSize() != 4 ||
		dataset.TargetSize() != 1 || dataset.Steps() != 2 {
		t.Fatalf(
			"dataset dimensions = samples %d inputs %d targets %d steps %d",
			dataset.SampleCount(),
			dataset.InputSize(),
			dataset.TargetSize(),
			dataset.Steps(),
		)
	}
}

func Test_SequenceDataset_DestinationAccessorsCopyValues(t *testing.T) {
	var (
		dataset *data.SequenceDataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths []int
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 3)
	inputs = mustMatrix(t, 3, 4, make([]float32, 12))
	targets = mustMatrix(t, 3, 1, make([]float32, 3))
	lengths = make([]int, 3)

	if err = dataset.InputsInto(inputs); err != nil {
		t.Fatalf("InputsInto returned error: %v", err)
	}

	if err = dataset.TargetsInto(targets); err != nil {
		t.Fatalf("TargetsInto returned error: %v", err)
	}

	if err = dataset.LengthsInto(lengths); err != nil {
		t.Fatalf("LengthsInto returned error: %v", err)
	}

	requireSequenceAlignment(t, inputs, targets, lengths)
}

func Test_SequenceDataset_SelectRowsIntoPreservesAlignmentAndRepeats(t *testing.T) {
	var (
		dataset *data.SequenceDataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths []int
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 4)
	inputs = mustMatrix(t, 4, 4, make([]float32, 16))
	targets = mustMatrix(t, 4, 1, make([]float32, 4))
	lengths = make([]int, 4)

	if err = dataset.SelectRowsInto([]int{3, 0, 3, 1}, inputs, targets, lengths); err != nil {
		t.Fatalf("SelectRowsInto returned error: %v", err)
	}

	requireMatrixValues(t, inputs, []float32{
		4, 40, 400, 4000,
		1, 10, 100, 1000,
		4, 40, 400, 4000,
		2, 20, 200, 2000,
	})
	requireMatrixValues(t, targets, []float32{104, 101, 104, 102})
	requireIntValues(t, lengths, []int{1, 2, 1, 1})
}

func Test_SequenceDataset_SelectRowsIntoRejectsInvalidArgumentsBeforeWrite(t *testing.T) {
	type testcase struct {
		name        string
		indexes     []int
		inputRows   int
		targetRows  int
		lengthCount int
	}

	var (
		dataset *data.SequenceDataset
		tests   []testcase
	)

	dataset = mustSequenceDatasetWithSamples(t, 3)
	tests = []testcase{
		{name: "empty indexes", indexes: []int{}, inputRows: 1, targetRows: 1, lengthCount: 1},
		{name: "negative index", indexes: []int{-1}, inputRows: 1, targetRows: 1, lengthCount: 1},
		{name: "index too large", indexes: []int{3}, inputRows: 1, targetRows: 1, lengthCount: 1},
		{name: "wrong input rows", indexes: []int{0, 1}, inputRows: 1, targetRows: 2, lengthCount: 2},
		{name: "wrong target rows", indexes: []int{0, 1}, inputRows: 2, targetRows: 1, lengthCount: 2},
		{name: "wrong length count", indexes: []int{0, 1}, inputRows: 2, targetRows: 2, lengthCount: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				inputValues  []float32
				targetValues []float32
				inputs       *matrix.Matrix
				targets      *matrix.Matrix
				lengths      []int
				err          error
			)

			inputValues = filledFloat32(tt.inputRows*4, -1)
			targetValues = filledFloat32(tt.targetRows, -2)
			inputs = mustMatrix(t, tt.inputRows, 4, inputValues)
			targets = mustMatrix(t, tt.targetRows, 1, targetValues)
			lengths = filledInt(tt.lengthCount, -3)

			err = dataset.SelectRowsInto(tt.indexes, inputs, targets, lengths)
			if err == nil {
				t.Fatal("SelectRowsInto error = nil, want error")
			}

			requireMatrixValues(t, inputs, inputValues)
			requireMatrixValues(t, targets, targetValues)
			requireIntValues(t, lengths, filledInt(tt.lengthCount, -3))
		})
	}
}

func Test_SequenceDataset_BatchesPreserveOrderAndPartialBatch(t *testing.T) {
	var (
		dataset *data.SequenceDataset
		batches []*data.SequenceBatch
		ids     []int
		lengths []int
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 5)
	batches, err = dataset.Batches(2, nil)
	if err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	if len(batches) != 3 {
		t.Fatalf("Batches length = %d, want 3", len(batches))
	}

	if batches[0].SampleCount() != 2 || batches[1].SampleCount() != 2 ||
		batches[2].SampleCount() != 1 {
		t.Fatalf(
			"batch sample counts = [%d %d %d], want [2 2 1]",
			batches[0].SampleCount(),
			batches[1].SampleCount(),
			batches[2].SampleCount(),
		)
	}

	ids, lengths = sequenceBatchIDs(t, batches)
	requireIntValues(t, ids, []int{1, 2, 3, 4, 5})
	requireIntValues(t, lengths, []int{2, 1, 2, 1, 2})
}

func Test_SequenceDataset_BatchesShuffleDeterministicallyAndPreserveAlignment(t *testing.T) {
	var (
		dataset       *data.SequenceDataset
		first         []*data.SequenceBatch
		second        []*data.SequenceBatch
		firstIDs      []int
		firstLengths  []int
		secondIDs     []int
		secondLengths []int
		err           error
	)

	dataset = mustSequenceDatasetWithSamples(t, 6)
	first, err = dataset.Batches(4, rand.New(rand.NewSource(17)))
	if err != nil {
		t.Fatalf("first Batches returned error: %v", err)
	}

	second, err = dataset.Batches(4, rand.New(rand.NewSource(17)))
	if err != nil {
		t.Fatalf("second Batches returned error: %v", err)
	}

	firstIDs, firstLengths = sequenceBatchIDs(t, first)
	secondIDs, secondLengths = sequenceBatchIDs(t, second)
	requireIntValues(t, firstIDs, secondIDs)
	requireIntValues(t, firstLengths, secondLengths)
	requireIDLengthAlignment(t, firstIDs, firstLengths)
}

func Test_SequenceDataset_BatchesRejectInvalidSizeWithoutConsumingRandom(t *testing.T) {
	var (
		dataset    *data.SequenceDataset
		random     *rand.Rand
		control    *rand.Rand
		batches    []*data.SequenceBatch
		gotRandom  int
		wantRandom int
		err        error
	)

	dataset = mustSequenceDatasetWithSamples(t, 2)
	random = rand.New(rand.NewSource(19))
	control = rand.New(rand.NewSource(19))
	batches, err = dataset.Batches(0, random)
	if err == nil {
		t.Fatal("Batches error = nil, want error")
	}

	if batches != nil {
		t.Fatal("Batches returned batches on error")
	}

	gotRandom = random.Int()
	wantRandom = control.Int()
	if gotRandom != wantRandom {
		t.Fatalf("random value = %d, want %d", gotRandom, wantRandom)
	}
}

func Test_SequenceDataset_SplitPreservesOrderAndAlignment(t *testing.T) {
	var (
		dataset      *data.SequenceDataset
		train        *data.SequenceDataset
		test         *data.SequenceDataset
		trainIDs     []int
		trainLengths []int
		testIDs      []int
		testLengths  []int
		err          error
	)

	dataset = mustSequenceDatasetWithSamples(t, 5)
	train, test, err = dataset.Split(0.4, nil)
	if err != nil {
		t.Fatalf("Split returned error: %v", err)
	}

	trainIDs, trainLengths = sequenceDatasetIDs(t, train)
	testIDs, testLengths = sequenceDatasetIDs(t, test)
	requireIntValues(t, trainIDs, []int{1, 2, 3})
	requireIntValues(t, trainLengths, []int{2, 1, 2})
	requireIntValues(t, testIDs, []int{4, 5})
	requireIntValues(t, testLengths, []int{1, 2})
}

func Test_SequenceDataset_SplitShufflesDeterministicallyAndPreservesAlignment(t *testing.T) {
	var (
		dataset         *data.SequenceDataset
		firstTrain      *data.SequenceDataset
		firstTest       *data.SequenceDataset
		secondTrain     *data.SequenceDataset
		secondTest      *data.SequenceDataset
		firstTrainIDs   []int
		firstTrainLens  []int
		firstTestIDs    []int
		firstTestLens   []int
		secondTrainIDs  []int
		secondTrainLens []int
		secondTestIDs   []int
		secondTestLens  []int
		err             error
	)

	dataset = mustSequenceDatasetWithSamples(t, 6)
	firstTrain, firstTest, err = dataset.Split(0.5, rand.New(rand.NewSource(23)))
	if err != nil {
		t.Fatalf("first Split returned error: %v", err)
	}

	secondTrain, secondTest, err = dataset.Split(0.5, rand.New(rand.NewSource(23)))
	if err != nil {
		t.Fatalf("second Split returned error: %v", err)
	}

	firstTrainIDs, firstTrainLens = sequenceDatasetIDs(t, firstTrain)
	firstTestIDs, firstTestLens = sequenceDatasetIDs(t, firstTest)
	secondTrainIDs, secondTrainLens = sequenceDatasetIDs(t, secondTrain)
	secondTestIDs, secondTestLens = sequenceDatasetIDs(t, secondTest)

	requireIntValues(t, firstTrainIDs, secondTrainIDs)
	requireIntValues(t, firstTrainLens, secondTrainLens)
	requireIntValues(t, firstTestIDs, secondTestIDs)
	requireIntValues(t, firstTestLens, secondTestLens)
	requireIDLengthAlignment(t, firstTrainIDs, firstTrainLens)
	requireIDLengthAlignment(t, firstTestIDs, firstTestLens)
}

func Test_SequenceDataset_SplitRejectsInvalidFraction(t *testing.T) {
	type testcase struct {
		name     string
		samples  int
		fraction float32
	}

	var tests []testcase
	tests = []testcase{
		{name: "zero", samples: 2, fraction: 0},
		{name: "one", samples: 2, fraction: 1},
		{name: "negative", samples: 2, fraction: -0.5},
		{name: "greater than one", samples: 2, fraction: 1.5},
		{name: "not a number", samples: 2, fraction: float32(math.NaN())},
		{name: "empty test split", samples: 2, fraction: 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				train *data.SequenceDataset
				test  *data.SequenceDataset
				err   error
			)

			train, test, err = mustSequenceDatasetWithSamples(t, tt.samples).Split(tt.fraction, nil)
			if err == nil {
				t.Fatal("Split error = nil, want error")
			}

			if train != nil || test != nil {
				t.Fatal("Split returned a dataset on error")
			}
		})
	}
}

func Test_SequenceDataset_ValidatesNilAndZeroValues(t *testing.T) {
	type testcase struct {
		name    string
		dataset *data.SequenceDataset
	}

	var (
		zero  data.SequenceDataset
		tests []testcase
	)
	tests = []testcase{
		{name: "nil", dataset: nil},
		{name: "zero", dataset: &zero},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			if err = tt.dataset.Validate(); err == nil {
				t.Fatal("Validate error = nil, want error")
			}

			if tt.dataset.SampleCount() != 0 || tt.dataset.InputSize() != 0 ||
				tt.dataset.TargetSize() != 0 || tt.dataset.Steps() != 0 {
				t.Fatal("invalid dataset dimensions are not zero")
			}
		})
	}
}

func mustSequenceLengths(tb testing.TB, steps int, values []int) (lengths *data.SequenceLengths) {
	var err error

	tb.Helper()

	lengths, err = data.NewSequenceLengths(steps, values)
	if err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	return lengths
}

func mustSequenceDatasetWithSamples(tb testing.TB, samples int) (dataset *data.SequenceDataset) {
	var (
		inputValues  []float32
		targetValues []float32
		lengthValues []int
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		row          int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*4)
	targetValues = make([]float32, samples)
	lengthValues = make([]int, samples)
	for row = 0; row < samples; row++ {
		inputValues[row*4] = float32(row + 1)
		inputValues[row*4+1] = float32((row + 1) * 10)
		inputValues[row*4+2] = float32((row + 1) * 100)
		inputValues[row*4+3] = float32((row + 1) * 1000)
		targetValues[row] = float32(row + 101)
		lengthValues[row] = expectedLength(row + 1)
	}

	inputs = mustMatrix(tb, samples, 4, inputValues)
	targets = mustMatrix(tb, samples, 1, targetValues)
	lengths = mustSequenceLengths(tb, 2, lengthValues)
	dataset, err = data.NewSequenceDataset(inputs, targets, lengths)
	if err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}

	return dataset
}

func sequenceBatchIDs(
	tb testing.TB,
	batches []*data.SequenceBatch,
) (ids, lengths []int) {
	var (
		batch        *data.SequenceBatch
		batchIDs     []int
		batchLengths []int
	)

	tb.Helper()

	for _, batch = range batches {
		batchIDs, batchLengths = sequenceBatchValues(tb, batch)
		ids = append(ids, batchIDs...)
		lengths = append(lengths, batchLengths...)
	}

	return ids, lengths
}

func sequenceBatchValues(
	tb testing.TB,
	batch *data.SequenceBatch,
) (ids, lengths []int) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		values  []float32
		row     int
		err     error
	)

	tb.Helper()

	inputs, err = batch.Inputs()
	if err != nil {
		tb.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = batch.Targets()
	if err != nil {
		tb.Fatalf("Targets returned error: %v", err)
	}

	lengths = make([]int, batch.SampleCount())
	if err = batch.LengthsInto(lengths); err != nil {
		tb.Fatalf("LengthsInto returned error: %v", err)
	}

	requireSequenceAlignment(tb, inputs, targets, lengths)
	values, err = inputs.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	ids = make([]int, batch.SampleCount())
	for row = range ids {
		ids[row] = int(values[row*4])
	}

	return ids, lengths
}

func sequenceDatasetIDs(
	tb testing.TB,
	dataset *data.SequenceDataset,
) (ids, lengths []int) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		values  []float32
		row     int
		err     error
	)

	tb.Helper()

	inputs, err = dataset.Inputs()
	if err != nil {
		tb.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = dataset.Targets()
	if err != nil {
		tb.Fatalf("Targets returned error: %v", err)
	}

	lengths = make([]int, dataset.SampleCount())
	if err = dataset.LengthsInto(lengths); err != nil {
		tb.Fatalf("LengthsInto returned error: %v", err)
	}

	requireSequenceAlignment(tb, inputs, targets, lengths)
	values, err = inputs.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	ids = make([]int, dataset.SampleCount())
	for row = range ids {
		ids[row] = int(values[row*4])
	}

	return ids, lengths
}

func requireSequenceAlignment(
	tb testing.TB,
	inputs, targets *matrix.Matrix,
	lengths []int,
) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		id           int
		err          error
	)

	tb.Helper()

	inputValues, err = inputs.Values()
	if err != nil {
		tb.Fatalf("input Values returned error: %v", err)
	}

	targetValues, err = targets.Values()
	if err != nil {
		tb.Fatalf("target Values returned error: %v", err)
	}

	for row = 0; row < inputs.Rows(); row++ {
		id = int(inputValues[row*4])
		if inputValues[row*4+1] != float32(id*10) ||
			inputValues[row*4+2] != float32(id*100) ||
			inputValues[row*4+3] != float32(id*1000) {
			tb.Fatalf("input row %d is not aligned: %v", row, inputValues[row*4:row*4+4])
		}

		if targetValues[row] != float32(id+100) {
			tb.Fatalf("target row %d = %g, want %d", row, targetValues[row], id+100)
		}

		if lengths[row] != expectedLength(id) {
			tb.Fatalf("length row %d = %d, want %d", row, lengths[row], expectedLength(id))
		}
	}
}

func requireIDLengthAlignment(tb testing.TB, ids, lengths []int) {
	var index int

	tb.Helper()

	if len(ids) != len(lengths) {
		tb.Fatalf("id count = %d, length count = %d", len(ids), len(lengths))
	}

	for index = range ids {
		if lengths[index] != expectedLength(ids[index]) {
			tb.Fatalf(
				"length %d = %d, want %d for id %d",
				index,
				lengths[index],
				expectedLength(ids[index]),
				ids[index],
			)
		}
	}
}

func expectedLength(id int) (length int) {
	if id%2 == 0 {
		return 1
	}

	return 2
}

func filledFloat32(count int, value float32) (values []float32) {
	var index int

	values = make([]float32, count)
	for index = range values {
		values[index] = value
	}

	return values
}

func filledInt(count, value int) (values []int) {
	var index int

	values = make([]int, count)
	for index = range values {
		values[index] = value
	}

	return values
}
