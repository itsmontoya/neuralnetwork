package data_test

import (
	"errors"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_DatasetAndBatchWithSelectedRowsApplyViewPolicy(t *testing.T) {
	type testcase struct {
		name       string
		indexes    []int
		policy     data.ViewPolicy
		wantIDs    []int
		wantCopied bool
		wantError  bool
	}

	var (
		dataset   *data.Dataset
		batches   []*data.Batch
		batch     *data.Batch
		retained  *data.DatasetView
		inputView *matrix.Matrix
		tests     []testcase
		err       error
	)

	dataset = mustDatasetWithSamples(t, 5)
	tests = []testcase{
		{
			name:       "contiguous strict",
			indexes:    []int{1, 2, 3},
			policy:     data.ViewOnly,
			wantIDs:    []int{2, 3, 4},
			wantCopied: false,
		},
		{
			name:       "contiguous fallback policy still views",
			indexes:    []int{2, 3},
			policy:     data.ViewOrCopy,
			wantIDs:    []int{3, 4},
			wantCopied: false,
		},
		{
			name:      "arbitrary strict rejection",
			indexes:   []int{4, 1, 3},
			policy:    data.ViewOnly,
			wantError: true,
		},
		{
			name:       "arbitrary copied fallback",
			indexes:    []int{4, 1, 3},
			policy:     data.ViewOrCopy,
			wantIDs:    []int{5, 2, 4},
			wantCopied: true,
		},
		{
			name:       "repeated copied fallback",
			indexes:    []int{2, 2, 0},
			policy:     data.ViewOrCopy,
			wantIDs:    []int{3, 3, 1},
			wantCopied: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				runErr error
			)

			runErr = dataset.WithSelectedRows(
				tt.indexes,
				tt.policy,
				func(view *data.DatasetView) (viewErr error) {
					called = true
					retained = view
					if inputView, viewErr = view.Inputs(); viewErr != nil {
						return viewErr
					}
					if view.Copied() != tt.wantCopied {
						t.Fatalf("Copied = %t, want %t", view.Copied(), tt.wantCopied)
					}
					requireIntValues(t, datasetViewIDs(t, view), tt.wantIDs)
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
			if !called {
				t.Fatal("WithSelectedRows did not invoke callback")
			}
			if err = retained.Validate(); err == nil {
				t.Fatal("retained selected view remains valid after callback")
			}
			if err = inputView.Validate(); err == nil {
				t.Fatal("retained selected matrix remains valid after callback")
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
		func(*data.DatasetView) (viewErr error) {
			return nil
		},
	)
	if err == nil {
		t.Fatal("Batch.WithSelectedRows strict non-contiguous error = nil, want error")
	}
	err = batch.WithSelectedRows(
		[]int{3, 0, 3},
		data.ViewOrCopy,
		func(view *data.DatasetView) (viewErr error) {
			if !view.Copied() {
				t.Fatal("Batch copied selection Copied = false, want true")
			}
			requireIntValues(t, datasetViewIDs(t, view), []int{4, 1, 4})
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Batch.WithSelectedRows returned error: %v", err)
	}
}

func Test_DatasetViewBatchesTraverseOrderedBoundaries(t *testing.T) {
	type testcase struct {
		name      string
		batchSize int
		wantIDs   [][]int
	}

	var (
		dataset *data.Dataset
		tests   []testcase
	)

	dataset = mustDatasetWithSamples(t, 6)
	tests = []testcase{
		{name: "single row batches", batchSize: 1, wantIDs: [][]int{{1}, {2}, {3}, {4}, {5}, {6}}},
		{name: "even full batches", batchSize: 2, wantIDs: [][]int{{1, 2}, {3, 4}, {5, 6}}},
		{name: "partial final batch", batchSize: 4, wantIDs: [][]int{{1, 2, 3, 4}, {5, 6}}},
		{name: "larger than dataset", batchSize: 8, wantIDs: [][]int{{1, 2, 3, 4, 5, 6}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got      [][]int
				previous *data.DatasetView
				runErr   error
			)

			runErr = dataset.ViewBatches(
				tt.batchSize,
				nil,
				data.ViewOnly,
				func(view *data.DatasetView) (viewErr error) {
					if previous != nil && previous.Validate() == nil {
						t.Fatal("previous batch view remains valid after traversal advanced")
					}
					if view.Copied() {
						t.Fatal("ordered batch Copied = true, want false")
					}
					got = append(got, datasetViewIDs(t, view))
					previous = view
					return nil
				},
			)
			if runErr != nil {
				t.Fatalf("ViewBatches returned error: %v", runErr)
			}
			if !reflect.DeepEqual(got, tt.wantIDs) {
				t.Fatalf("batch ids = %v, want %v", got, tt.wantIDs)
			}
			if previous.Validate() == nil {
				t.Fatal("final batch view remains valid after traversal")
			}
		})
	}
}

func Test_DatasetViewBatchesShuffledFallbackMatchesCopiedBatches(t *testing.T) {
	var (
		dataset      *data.Dataset
		viewRandom   *rand.Rand
		copyRandom   *rand.Rand
		viewIDs      []int
		copyIDs      []int
		copiedResult []*data.Batch
		batch        *data.Batch
		err          error
	)

	dataset = mustDatasetWithSamples(t, 7)
	viewRandom = rand.New(rand.NewSource(41))
	copyRandom = rand.New(rand.NewSource(41))
	err = dataset.ViewBatches(
		3,
		viewRandom,
		data.ViewOrCopy,
		func(view *data.DatasetView) (viewErr error) {
			if !view.Copied() {
				t.Fatal("shuffled batch Copied = false, want true")
			}
			viewIDs = append(viewIDs, datasetViewIDs(t, view)...)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("ViewBatches returned error: %v", err)
	}
	if copiedResult, err = dataset.Batches(3, copyRandom); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	for _, batch = range copiedResult {
		copyIDs = append(copyIDs, datasetBatchIDs(t, batch)...)
	}
	if !reflect.DeepEqual(viewIDs, copyIDs) {
		t.Fatalf("view ids = %v, copied ids = %v", viewIDs, copyIDs)
	}
	if viewRandom.Int63() != copyRandom.Int63() {
		t.Fatal("ViewBatches consumed different caller randomness than Batches")
	}
}

func Test_DatasetViewSplitAppliesOrderedAndShuffledPolicies(t *testing.T) {
	var (
		dataset       *data.Dataset
		retainedTrain *data.DatasetView
		retainedTest  *data.DatasetView
		viewRandom    *rand.Rand
		copyRandom    *rand.Rand
		train         *data.Dataset
		test          *data.Dataset
		err           error
	)

	dataset = mustDatasetWithSamples(t, 5)
	err = dataset.ViewSplit(
		0.4,
		nil,
		data.ViewOnly,
		func(trainView, testView *data.DatasetView) (viewErr error) {
			retainedTrain = trainView
			retainedTest = testView
			if trainView.Copied() || testView.Copied() {
				t.Fatal("ordered split reports copied storage")
			}
			requireIntValues(t, datasetViewIDs(t, trainView), []int{1, 2, 3})
			requireIntValues(t, datasetViewIDs(t, testView), []int{4, 5})
			return nil
		},
	)
	if err != nil {
		t.Fatalf("ordered ViewSplit returned error: %v", err)
	}
	if retainedTrain.Validate() == nil || retainedTest.Validate() == nil {
		t.Fatal("retained ordered split view remains valid")
	}

	viewRandom = rand.New(rand.NewSource(43))
	copyRandom = rand.New(rand.NewSource(43))
	err = dataset.ViewSplit(
		0.4,
		viewRandom,
		data.ViewOrCopy,
		func(trainView, testView *data.DatasetView) (viewErr error) {
			if !trainView.Copied() || !testView.Copied() {
				t.Fatal("shuffled split does not report both copied selections")
			}
			if train, test, viewErr = dataset.Split(0.4, copyRandom); viewErr != nil {
				return viewErr
			}
			if !reflect.DeepEqual(datasetViewIDs(t, trainView), datasetIDs(t, train)) {
				t.Fatal("shuffled train view order differs from Split")
			}
			if !reflect.DeepEqual(datasetViewIDs(t, testView), datasetIDs(t, test)) {
				t.Fatal("shuffled test view order differs from Split")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("shuffled ViewSplit returned error: %v", err)
	}
	if viewRandom.Int63() != copyRandom.Int63() {
		t.Fatal("ViewSplit consumed different caller randomness than Split")
	}
}

func Test_DatasetViewSplitFloorsBoundaryFractions(t *testing.T) {
	type testcase struct {
		name      string
		fraction  float32
		wantTrain int
		wantTest  int
	}

	var (
		dataset *data.Dataset
		tests   []testcase
	)

	dataset = mustDatasetWithSamples(t, 5)
	tests = []testcase{
		{name: "fraction just above one row", fraction: 0.21, wantTrain: 4, wantTest: 1},
		{name: "fraction just below all rows", fraction: 0.99, wantTrain: 1, wantTest: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var runErr error

			runErr = dataset.ViewSplit(
				tt.fraction,
				nil,
				data.ViewOnly,
				func(train, test *data.DatasetView) (viewErr error) {
					if train.SampleCount() != tt.wantTrain ||
						test.SampleCount() != tt.wantTest {
						t.Fatalf(
							"split counts = %d/%d, want %d/%d",
							train.SampleCount(),
							test.SampleCount(),
							tt.wantTrain,
							tt.wantTest,
						)
					}
					return nil
				},
			)
			if runErr != nil {
				t.Fatalf("ViewSplit returned error: %v", runErr)
			}
		})
	}
}

func Test_DatasetViewSplitExpiresViewsAfterPanic(t *testing.T) {
	var (
		dataset       *data.Dataset
		retainedTrain *data.DatasetView
		retainedTest  *data.DatasetView
	)

	dataset = mustDatasetWithSamples(t, 4)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("ViewSplit panic = nil, want injected panic")
			}
		}()

		_ = dataset.ViewSplit(
			0.5,
			nil,
			data.ViewOnly,
			func(train, test *data.DatasetView) (viewErr error) {
				retainedTrain = train
				retainedTest = test
				panic("injected split panic")
			},
		)
	}()
	if retainedTrain.Validate() == nil || retainedTest.Validate() == nil {
		t.Fatal("panic-retained split view remains valid")
	}
}

func Test_DatasetAndBatchSelectionViewsRejectInvalidOwners(t *testing.T) {
	var (
		nilDataset *data.Dataset
		nilBatch   *data.Batch
		called     bool
		use        data.DatasetViewFunc
		splitUse   data.DatasetSplitViewFunc
		err        error
	)

	use = func(*data.DatasetView) (viewErr error) {
		called = true
		return nil
	}
	splitUse = func(*data.DatasetView, *data.DatasetView) (viewErr error) {
		called = true
		return nil
	}

	if err = nilDataset.WithSelectedRows([]int{0}, data.ViewOnly, use); err == nil {
		t.Fatal("nil dataset WithSelectedRows error = nil, want error")
	}
	if err = (&data.Dataset{}).ViewBatches(1, nil, data.ViewOnly, use); err == nil {
		t.Fatal("zero dataset ViewBatches error = nil, want error")
	}
	if err = nilDataset.ViewSplit(0.5, nil, data.ViewOnly, splitUse); err == nil {
		t.Fatal("nil dataset ViewSplit error = nil, want error")
	}
	if err = nilBatch.WithSelectedRows([]int{0}, data.ViewOnly, use); err == nil {
		t.Fatal("nil batch WithSelectedRows error = nil, want error")
	}
	if err = (&data.Batch{}).WithSelectedRows([]int{0}, data.ViewOnly, use); err == nil {
		t.Fatal("zero batch WithSelectedRows error = nil, want error")
	}
	if called {
		t.Fatal("invalid owner invoked a view callback")
	}
}

func Test_DatasetSelectionViewsRejectInvalidCallsBeforePublication(t *testing.T) {
	type testcase struct {
		name string
		run  func(*rand.Rand, data.DatasetViewFunc, data.DatasetSplitViewFunc) error
	}

	var (
		dataset *data.Dataset
		tests   []testcase
	)

	dataset = mustDatasetWithSamples(t, 4)
	tests = []testcase{
		{
			name: "selected rows empty",
			run: func(_ *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows(nil, data.ViewOnly, use)
				return err
			},
		},
		{
			name: "selected row out of range",
			run: func(_ *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows([]int{0, 4}, data.ViewOnly, use)
				return err
			},
		},
		{
			name: "selected rows invalid policy",
			run: func(_ *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows([]int{0}, data.ViewPolicy(9), use)
				return err
			},
		},
		{
			name: "selected rows nil callback",
			run: func(_ *rand.Rand, _ data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.WithSelectedRows([]int{0}, data.ViewOnly, nil)
				return err
			},
		},
		{
			name: "batch size zero",
			run: func(random *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(0, random, data.ViewOrCopy, use)
				return err
			},
		},
		{
			name: "shuffled strict batches",
			run: func(random *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(2, random, data.ViewOnly, use)
				return err
			},
		},
		{
			name: "batches invalid policy",
			run: func(random *rand.Rand, use data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(2, random, data.ViewPolicy(7), use)
				return err
			},
		},
		{
			name: "batches nil callback",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewBatches(2, random, data.ViewOrCopy, nil)
				return err
			},
		},
		{
			name: "split invalid fraction",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, split data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0, random, data.ViewOrCopy, split)
				return err
			},
		},
		{
			name: "split empty test",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, split data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.1, random, data.ViewOrCopy, split)
				return err
			},
		},
		{
			name: "shuffled strict split",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, split data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.5, random, data.ViewOnly, split)
				return err
			},
		},
		{
			name: "split invalid policy",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, split data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.5, random, data.ViewPolicy(6), split)
				return err
			},
		},
		{
			name: "split nil callback",
			run: func(random *rand.Rand, _ data.DatasetViewFunc, _ data.DatasetSplitViewFunc) (err error) {
				err = dataset.ViewSplit(0.5, random, data.ViewOrCopy, nil)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				random       *rand.Rand
				control      *rand.Rand
				viewCalled   bool
				splitCalled  bool
				runErr       error
				viewConsumer data.DatasetViewFunc
				splitUse     data.DatasetSplitViewFunc
			)

			random = rand.New(rand.NewSource(47))
			control = rand.New(rand.NewSource(47))
			viewConsumer = func(*data.DatasetView) (viewErr error) {
				viewCalled = true
				return nil
			}
			splitUse = func(*data.DatasetView, *data.DatasetView) (viewErr error) {
				splitCalled = true
				return nil
			}
			runErr = tt.run(random, viewConsumer, splitUse)
			if runErr == nil {
				t.Fatal("invalid view operation error = nil, want error")
			}
			if viewCalled || splitCalled {
				t.Fatal("invalid view operation invoked a callback")
			}
			if random.Int63() != control.Int63() {
				t.Fatal("invalid view operation consumed caller randomness")
			}
		})
	}
}

func Test_DatasetViewBatchesStopsAfterCallbackError(t *testing.T) {
	var (
		injected error
		dataset  *data.Dataset
		retained *data.DatasetView
		calls    int
		err      error
	)

	injected = errors.New("injected batch failure")
	dataset = mustDatasetWithSamples(t, 5)
	err = dataset.ViewBatches(
		2,
		nil,
		data.ViewOnly,
		func(view *data.DatasetView) (viewErr error) {
			retained = view
			calls++
			if calls == 2 {
				return injected
			}
			return nil
		},
	)
	if !errors.Is(err, injected) {
		t.Fatalf("ViewBatches error = %v, want injected error", err)
	}
	if !strings.Contains(err.Error(), "batch=1") {
		t.Fatalf("ViewBatches error = %q, want batch number", err)
	}
	if calls != 2 {
		t.Fatalf("callback calls = %d, want 2", calls)
	}
	if retained.Validate() == nil {
		t.Fatal("error-retained batch view remains valid")
	}
}

func Test_DatasetSelectionViewsSupportDistinctOwnerConcurrency(t *testing.T) {
	var (
		first     *data.Dataset
		second    *data.Dataset
		waitGroup sync.WaitGroup
		errorsOut chan error
		run       func(*data.Dataset)
		err       error
	)

	first = mustDatasetWithSamples(t, 5)
	second = mustDatasetWithSamples(t, 6)
	errorsOut = make(chan error, 2)
	run = func(dataset *data.Dataset) {
		defer waitGroup.Done()
		errorsOut <- dataset.ViewBatches(
			2,
			nil,
			data.ViewOnly,
			func(view *data.DatasetView) (viewErr error) {
				var inputs *matrix.Matrix

				if inputs, viewErr = view.Inputs(); viewErr != nil {
					return viewErr
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
			t.Fatalf("concurrent distinct ViewBatches returned error: %v", err)
		}
	}
}

func datasetViewIDs(tb testing.TB, view *data.DatasetView) (ids []int) {
	var (
		inputs *matrix.Matrix
		values []float32
		row    int
		err    error
	)

	tb.Helper()
	if inputs, err = view.Inputs(); err != nil {
		tb.Fatalf("view Inputs returned error: %v", err)
	}
	if values, err = inputs.Values(); err != nil {
		tb.Fatalf("view input Values returned error: %v", err)
	}
	ids = make([]int, inputs.Rows())
	for row = range ids {
		ids[row] = int(values[row*inputs.Cols()])
	}
	return ids
}

func datasetBatchIDs(tb testing.TB, batch *data.Batch) (ids []int) {
	var (
		inputs *matrix.Matrix
		values []float32
		row    int
		err    error
	)

	tb.Helper()
	if inputs, err = batch.Inputs(); err != nil {
		tb.Fatalf("batch Inputs returned error: %v", err)
	}
	if values, err = inputs.Values(); err != nil {
		tb.Fatalf("batch input Values returned error: %v", err)
	}
	ids = make([]int, batch.SampleCount())
	for row = range ids {
		ids[row] = int(values[row*inputs.Cols()])
	}
	return ids
}

func datasetIDs(tb testing.TB, dataset *data.Dataset) (ids []int) {
	var (
		inputs *matrix.Matrix
		values []float32
		row    int
		err    error
	)

	tb.Helper()
	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("dataset Inputs returned error: %v", err)
	}
	if values, err = inputs.Values(); err != nil {
		tb.Fatalf("dataset input Values returned error: %v", err)
	}
	ids = make([]int, dataset.SampleCount())
	for row = range ids {
		ids[row] = int(values[row*inputs.Cols()])
	}
	return ids
}
