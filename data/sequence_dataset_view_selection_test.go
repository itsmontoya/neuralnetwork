package data_test

import (
	"errors"
	"math/rand"
	"reflect"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SequenceDatasetAndBatchWithSelectedRowsApplyViewPolicy(t *testing.T) {
	type testcase struct {
		name        string
		indexes     []int
		policy      data.ViewPolicy
		wantIDs     []int
		wantLengths []int
		wantCopied  bool
		wantError   bool
	}

	var (
		dataset   *data.SequenceDataset
		batches   []*data.SequenceBatch
		batch     *data.SequenceBatch
		inputView *matrix.Matrix
		tests     []testcase
		err       error
	)

	dataset = mustSequenceDatasetWithSamples(t, 5)
	tests = []testcase{
		{
			name:        "contiguous strict",
			indexes:     []int{1, 2, 3},
			policy:      data.ViewOnly,
			wantIDs:     []int{2, 3, 4},
			wantLengths: []int{1, 2, 1},
		},
		{
			name:       "arbitrary strict rejection",
			indexes:    []int{4, 0, 2},
			policy:     data.ViewOnly,
			wantError:  true,
			wantCopied: false,
		},
		{
			name:        "arbitrary copied fallback",
			indexes:     []int{4, 0, 2},
			policy:      data.ViewOrCopy,
			wantIDs:     []int{5, 1, 3},
			wantLengths: []int{2, 2, 2},
			wantCopied:  true,
		},
		{
			name:        "repeated copied fallback",
			indexes:     []int{1, 1, 3},
			policy:      data.ViewOrCopy,
			wantIDs:     []int{2, 2, 4},
			wantLengths: []int{1, 1, 1},
			wantCopied:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called   bool
				retained *data.SequenceDatasetView
				runErr   error
			)

			runErr = dataset.WithSelectedRows(
				tt.indexes,
				tt.policy,
				func(view *data.SequenceDatasetView) (viewErr error) {
					var (
						ids     []int
						lengths []int
					)

					called = true
					retained = view
					if inputView, viewErr = view.Inputs(); viewErr != nil {
						return viewErr
					}
					if view.Copied() != tt.wantCopied {
						t.Fatalf("Copied = %t, want %t", view.Copied(), tt.wantCopied)
					}
					ids, lengths = sequenceDatasetViewIDs(t, view)
					requireIntValues(t, ids, tt.wantIDs)
					requireIntValues(t, lengths, tt.wantLengths)
					return nil
				},
			)
			if tt.wantError {
				if runErr == nil {
					t.Fatal("WithSelectedRows error = nil, want error")
				}
				if called {
					t.Fatal("WithSelectedRows invoked callback on rejected selection")
				}
				return
			}
			if runErr != nil {
				t.Fatalf("WithSelectedRows returned error: %v", runErr)
			}
			if retained.Validate() == nil {
				t.Fatal("retained selected sequence view remains valid")
			}
			if inputView.Validate() == nil {
				t.Fatal("retained selected sequence matrix remains valid")
			}
		})
	}

	if batches, err = dataset.Batches(4, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	err = batch.WithSelectedRows(
		[]int{3, 0},
		data.ViewOnly,
		func(*data.SequenceDatasetView) (viewErr error) {
			return nil
		},
	)
	if err == nil {
		t.Fatal("SequenceBatch strict non-contiguous error = nil, want error")
	}
	err = batch.WithSelectedRows(
		[]int{3, 0, 3},
		data.ViewOrCopy,
		func(view *data.SequenceDatasetView) (viewErr error) {
			var (
				ids     []int
				lengths []int
			)

			ids, lengths = sequenceDatasetViewIDs(t, view)
			requireIntValues(t, ids, []int{4, 1, 4})
			requireIntValues(t, lengths, []int{1, 2, 1})
			if !view.Copied() {
				t.Fatal("SequenceBatch copied selection Copied = false, want true")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("SequenceBatch.WithSelectedRows returned error: %v", err)
	}
}

func Test_SequenceDatasetViewBatchesPreserveOrderedAlignmentAndBoundaries(t *testing.T) {
	type testcase struct {
		name      string
		batchSize int
		wantIDs   [][]int
	}

	var (
		dataset *data.SequenceDataset
		tests   []testcase
	)

	dataset = mustSequenceDatasetWithSamples(t, 6)
	tests = []testcase{
		{name: "single row batches", batchSize: 1, wantIDs: [][]int{{1}, {2}, {3}, {4}, {5}, {6}}},
		{name: "even full batches", batchSize: 2, wantIDs: [][]int{{1, 2}, {3, 4}, {5, 6}}},
		{name: "partial final batch", batchSize: 4, wantIDs: [][]int{{1, 2, 3, 4}, {5, 6}}},
		{name: "larger than dataset", batchSize: 9, wantIDs: [][]int{{1, 2, 3, 4, 5, 6}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got      [][]int
				previous *data.SequenceDatasetView
				runErr   error
			)

			runErr = dataset.ViewBatches(
				tt.batchSize,
				nil,
				data.ViewOnly,
				func(view *data.SequenceDatasetView) (viewErr error) {
					var (
						ids     []int
						lengths []int
					)

					if previous != nil && previous.Validate() == nil {
						t.Fatal("previous sequence batch remains valid after traversal advanced")
					}
					ids, lengths = sequenceDatasetViewIDs(t, view)
					requireIDLengthAlignment(t, ids, lengths)
					got = append(got, ids)
					if view.Copied() {
						t.Fatal("ordered sequence batch Copied = true, want false")
					}
					previous = view
					return nil
				},
			)
			if runErr != nil {
				t.Fatalf("ViewBatches returned error: %v", runErr)
			}
			if !reflect.DeepEqual(got, tt.wantIDs) {
				t.Fatalf("sequence batch ids = %v, want %v", got, tt.wantIDs)
			}
			if previous.Validate() == nil {
				t.Fatal("final sequence batch remains valid after traversal")
			}
		})
	}
}

func Test_SequenceDatasetViewBatchesShuffledFallbackMatchesCopiedBatches(t *testing.T) {
	var (
		dataset     *data.SequenceDataset
		viewRandom  *rand.Rand
		copyRandom  *rand.Rand
		viewIDs     []int
		viewLengths []int
		copyIDs     []int
		copyLengths []int
		batches     []*data.SequenceBatch
		err         error
	)

	dataset = mustSequenceDatasetWithSamples(t, 7)
	viewRandom = rand.New(rand.NewSource(53))
	copyRandom = rand.New(rand.NewSource(53))
	err = dataset.ViewBatches(
		3,
		viewRandom,
		data.ViewOrCopy,
		func(view *data.SequenceDatasetView) (viewErr error) {
			var (
				ids     []int
				lengths []int
			)

			ids, lengths = sequenceDatasetViewIDs(t, view)
			viewIDs = append(viewIDs, ids...)
			viewLengths = append(viewLengths, lengths...)
			if !view.Copied() {
				t.Fatal("shuffled sequence batch Copied = false, want true")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("ViewBatches returned error: %v", err)
	}
	if batches, err = dataset.Batches(3, copyRandom); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	copyIDs, copyLengths = sequenceBatchIDs(t, batches)
	if !reflect.DeepEqual(viewIDs, copyIDs) ||
		!reflect.DeepEqual(viewLengths, copyLengths) {
		t.Fatalf(
			"view rows = %v/%v, copied rows = %v/%v",
			viewIDs,
			viewLengths,
			copyIDs,
			copyLengths,
		)
	}
	if viewRandom.Int63() != copyRandom.Int63() {
		t.Fatal("sequence ViewBatches consumed different randomness than Batches")
	}
}

func Test_SequenceDatasetViewSplitPreservesAlignmentAndRandomParity(t *testing.T) {
	var (
		dataset       *data.SequenceDataset
		retainedTrain *data.SequenceDatasetView
		retainedTest  *data.SequenceDatasetView
		viewRandom    *rand.Rand
		copyRandom    *rand.Rand
		train         *data.SequenceDataset
		test          *data.SequenceDataset
		err           error
	)

	dataset = mustSequenceDatasetWithSamples(t, 5)
	err = dataset.ViewSplit(
		0.4,
		nil,
		data.ViewOnly,
		func(trainView, testView *data.SequenceDatasetView) (viewErr error) {
			var (
				trainIDs     []int
				trainLengths []int
				testIDs      []int
				testLengths  []int
			)

			retainedTrain = trainView
			retainedTest = testView
			trainIDs, trainLengths = sequenceDatasetViewIDs(t, trainView)
			testIDs, testLengths = sequenceDatasetViewIDs(t, testView)
			requireIntValues(t, trainIDs, []int{1, 2, 3})
			requireIntValues(t, trainLengths, []int{2, 1, 2})
			requireIntValues(t, testIDs, []int{4, 5})
			requireIntValues(t, testLengths, []int{1, 2})
			if trainView.Copied() || testView.Copied() {
				t.Fatal("ordered sequence split reports copied storage")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("ordered ViewSplit returned error: %v", err)
	}
	if retainedTrain.Validate() == nil || retainedTest.Validate() == nil {
		t.Fatal("retained ordered sequence split remains valid")
	}

	viewRandom = rand.New(rand.NewSource(59))
	copyRandom = rand.New(rand.NewSource(59))
	err = dataset.ViewSplit(
		0.4,
		viewRandom,
		data.ViewOrCopy,
		func(trainView, testView *data.SequenceDatasetView) (viewErr error) {
			var (
				trainViewIDs     []int
				trainViewLengths []int
				testViewIDs      []int
				testViewLengths  []int
				trainIDs         []int
				trainLengths     []int
				testIDs          []int
				testLengths      []int
			)

			if !trainView.Copied() || !testView.Copied() {
				t.Fatal("shuffled sequence split does not report copied storage")
			}
			if train, test, viewErr = dataset.Split(0.4, copyRandom); viewErr != nil {
				return viewErr
			}
			trainViewIDs, trainViewLengths = sequenceDatasetViewIDs(t, trainView)
			testViewIDs, testViewLengths = sequenceDatasetViewIDs(t, testView)
			trainIDs, trainLengths = sequenceDatasetIDs(t, train)
			testIDs, testLengths = sequenceDatasetIDs(t, test)
			if !reflect.DeepEqual(trainViewIDs, trainIDs) ||
				!reflect.DeepEqual(trainViewLengths, trainLengths) ||
				!reflect.DeepEqual(testViewIDs, testIDs) ||
				!reflect.DeepEqual(testViewLengths, testLengths) {
				t.Fatal("shuffled sequence ViewSplit differs from Split")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("shuffled ViewSplit returned error: %v", err)
	}
	if viewRandom.Int63() != copyRandom.Int63() {
		t.Fatal("sequence ViewSplit consumed different randomness than Split")
	}
}

func Test_SequenceDatasetAndBatchSelectionViewsRejectInvalidOwners(t *testing.T) {
	var (
		nilDataset *data.SequenceDataset
		nilBatch   *data.SequenceBatch
		called     bool
		use        data.SequenceDatasetViewFunc
		splitUse   data.SequenceDatasetSplitViewFunc
		err        error
	)

	use = func(*data.SequenceDatasetView) (viewErr error) {
		called = true
		return nil
	}
	splitUse = func(
		*data.SequenceDatasetView,
		*data.SequenceDatasetView,
	) (viewErr error) {
		called = true
		return nil
	}

	if err = nilDataset.WithSelectedRows([]int{0}, data.ViewOnly, use); err == nil {
		t.Fatal("nil sequence dataset WithSelectedRows error = nil, want error")
	}
	if err = (&data.SequenceDataset{}).ViewBatches(
		1,
		nil,
		data.ViewOnly,
		use,
	); err == nil {
		t.Fatal("zero sequence dataset ViewBatches error = nil, want error")
	}
	if err = nilDataset.ViewSplit(0.5, nil, data.ViewOnly, splitUse); err == nil {
		t.Fatal("nil sequence dataset ViewSplit error = nil, want error")
	}
	if err = nilBatch.WithSelectedRows([]int{0}, data.ViewOnly, use); err == nil {
		t.Fatal("nil sequence batch WithSelectedRows error = nil, want error")
	}
	if err = (&data.SequenceBatch{}).WithSelectedRows(
		[]int{0},
		data.ViewOnly,
		use,
	); err == nil {
		t.Fatal("zero sequence batch WithSelectedRows error = nil, want error")
	}
	if called {
		t.Fatal("invalid sequence owner invoked a view callback")
	}
}

func Test_SequenceSelectionViewsRejectBeforeRandomConsumption(t *testing.T) {
	type testcase struct {
		name string
		run  func(*rand.Rand, data.SequenceDatasetViewFunc, data.SequenceDatasetSplitViewFunc) error
	}

	var (
		dataset *data.SequenceDataset
		tests   []testcase
	)

	dataset = mustSequenceDatasetWithSamples(t, 4)
	tests = []testcase{
		{
			name: "selected rows empty",
			run: func(_ *rand.Rand, use data.SequenceDatasetViewFunc, _ data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows(nil, data.ViewOnly, use)
				return err
			},
		},
		{
			name: "selected row out of range",
			run: func(_ *rand.Rand, use data.SequenceDatasetViewFunc, _ data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows([]int{-1}, data.ViewOrCopy, use)
				return err
			},
		},
		{
			name: "selected invalid policy",
			run: func(_ *rand.Rand, use data.SequenceDatasetViewFunc, _ data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows([]int{0}, data.ViewPolicy(8), use)
				return err
			},
		},
		{
			name: "shuffled strict batches",
			run: func(random *rand.Rand, use data.SequenceDatasetViewFunc, _ data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(2, random, data.ViewOnly, use)
				return err
			},
		},
		{
			name: "invalid batch size",
			run: func(random *rand.Rand, use data.SequenceDatasetViewFunc, _ data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(-1, random, data.ViewOrCopy, use)
				return err
			},
		},
		{
			name: "shuffled strict split",
			run: func(random *rand.Rand, _ data.SequenceDatasetViewFunc, split data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.5, random, data.ViewOnly, split)
				return err
			},
		},
		{
			name: "split empty result",
			run: func(random *rand.Rand, _ data.SequenceDatasetViewFunc, split data.SequenceDatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.1, random, data.ViewOrCopy, split)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				random      *rand.Rand
				control     *rand.Rand
				viewCalled  bool
				splitCalled bool
				runErr      error
			)

			random = rand.New(rand.NewSource(61))
			control = rand.New(rand.NewSource(61))
			runErr = tt.run(
				random,
				func(*data.SequenceDatasetView) (viewErr error) {
					viewCalled = true
					return nil
				},
				func(*data.SequenceDatasetView, *data.SequenceDatasetView) (viewErr error) {
					splitCalled = true
					return nil
				},
			)
			if runErr == nil {
				t.Fatal("invalid sequence view operation error = nil, want error")
			}
			if viewCalled || splitCalled {
				t.Fatal("invalid sequence view operation invoked a callback")
			}
			if random.Int63() != control.Int63() {
				t.Fatal("invalid sequence view operation consumed caller randomness")
			}
		})
	}
}

func Test_SequenceSelectionViewsSupportDistinctOwnerConcurrency(t *testing.T) {
	var (
		first     *data.SequenceDataset
		second    *data.SequenceDataset
		waitGroup sync.WaitGroup
		errorsOut chan error
		run       func(*data.SequenceDataset)
		err       error
	)

	first = mustSequenceDatasetWithSamples(t, 5)
	second = mustSequenceDatasetWithSamples(t, 6)
	errorsOut = make(chan error, 2)
	run = func(dataset *data.SequenceDataset) {
		defer waitGroup.Done()
		errorsOut <- dataset.ViewSplit(
			0.4,
			nil,
			data.ViewOnly,
			func(train, test *data.SequenceDatasetView) (viewErr error) {
				var (
					inputs  *matrix.Matrix
					lengths []int
				)

				if inputs, viewErr = train.Inputs(); viewErr != nil {
					return viewErr
				}
				if lengths, viewErr = test.Lengths(); viewErr != nil {
					return viewErr
				}
				if len(lengths) != test.SampleCount() {
					return errors.New("sequence view lengths are not aligned")
				}
				_, viewErr = inputs.At(0, 0)
				return viewErr
			},
		)
	}

	waitGroup.Add(2)
	go run(first)
	go run(second)
	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("concurrent distinct ViewSplit returned error: %v", err)
		}
	}
}

func sequenceDatasetViewIDs(
	tb testing.TB,
	view *data.SequenceDatasetView,
) (ids, lengths []int) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		values  []float32
		row     int
		err     error
	)

	tb.Helper()
	if inputs, err = view.Inputs(); err != nil {
		tb.Fatalf("sequence view Inputs returned error: %v", err)
	}
	if targets, err = view.Targets(); err != nil {
		tb.Fatalf("sequence view Targets returned error: %v", err)
	}
	if lengths, err = view.Lengths(); err != nil {
		tb.Fatalf("sequence view Lengths returned error: %v", err)
	}
	requireSequenceAlignment(tb, inputs, targets, lengths)
	if values, err = inputs.Values(); err != nil {
		tb.Fatalf("sequence view input Values returned error: %v", err)
	}
	ids = make([]int, inputs.Rows())
	for row = range ids {
		ids[row] = int(values[row*inputs.Cols()])
	}
	return ids, append([]int(nil), lengths...)
}
