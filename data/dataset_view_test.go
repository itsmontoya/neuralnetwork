package data_test

import (
	"errors"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_DatasetWithViewReturnsPairedWholeView(t *testing.T) {
	var (
		dataset        *data.Dataset
		retained       *data.DatasetView
		retainedCopy   data.DatasetView
		retainedInput  *matrix.Matrix
		retainedTarget *matrix.Matrix
		err            error
	)

	dataset = mustDatasetWithSamples(t, 3)
	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		retained = view
		retainedCopy = *view
		if viewErr = view.Validate(); viewErr != nil {
			return viewErr
		}
		if view.SampleCount() != 3 || view.InputSize() != 2 || view.TargetSize() != 1 {
			t.Fatalf(
				"view dimensions = samples %d inputs %d targets %d, want 3, 2, 1",
				view.SampleCount(),
				view.InputSize(),
				view.TargetSize(),
			)
		}
		if view.Copied() {
			t.Fatal("whole DatasetView Copied = true, want false")
		}
		if retainedInput, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if retainedTarget, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, retainedInput, []float32{1, 10, 2, 20, 3, 30})
		requireMatrixValues(t, retainedTarget, []float32{101, 102, 103})
		return nil
	})
	if err != nil {
		t.Fatalf("WithView returned error: %v", err)
	}

	if err = retained.Validate(); err == nil {
		t.Fatal("retained view Validate error = nil, want expired view error")
	}
	if err = retainedCopy.Validate(); err == nil {
		t.Fatal("copied retained view Validate error = nil, want shared expiry")
	}
	if _, err = retained.Inputs(); err == nil {
		t.Fatal("expired view Inputs error = nil, want error")
	}
	if _, err = retained.Targets(); err == nil {
		t.Fatal("expired view Targets error = nil, want error")
	}
	if retained.SampleCount() != 0 || retained.InputSize() != 0 ||
		retained.TargetSize() != 0 || retained.Copied() {
		t.Fatal("expired view scalar accessors did not return zero values")
	}
	if err = retainedInput.Validate(); err == nil {
		t.Fatal("retained input matrix Validate error = nil, want expired matrix error")
	}
	if err = retainedTarget.Validate(); err == nil {
		t.Fatal("retained target matrix Validate error = nil, want expired matrix error")
	}
}

func Test_DatasetAndBatchWithRowViewReturnPairedWindows(t *testing.T) {
	type testcase struct {
		name        string
		start       int
		end         int
		wantInputs  []float32
		wantTargets []float32
	}

	var (
		dataset *data.Dataset
		batches []*data.Batch
		batch   *data.Batch
		tests   []testcase
		err     error
	)

	dataset = mustDatasetWithSamples(t, 5)
	tests = []testcase{
		{
			name:        "single row",
			start:       2,
			end:         3,
			wantInputs:  []float32{3, 30},
			wantTargets: []float32{103},
		},
		{
			name:        "full window",
			start:       0,
			end:         5,
			wantInputs:  []float32{1, 10, 2, 20, 3, 30, 4, 40, 5, 50},
			wantTargets: []float32{101, 102, 103, 104, 105},
		},
		{
			name:        "partial final window",
			start:       4,
			end:         5,
			wantInputs:  []float32{5, 50},
			wantTargets: []float32{105},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var runErr error

			runErr = dataset.WithRowView(tt.start, tt.end, func(view *data.DatasetView) (viewErr error) {
				var (
					inputs  *matrix.Matrix
					targets *matrix.Matrix
				)

				if inputs, viewErr = view.Inputs(); viewErr != nil {
					return viewErr
				}
				if targets, viewErr = view.Targets(); viewErr != nil {
					return viewErr
				}
				requireMatrixValues(t, inputs, tt.wantInputs)
				requireMatrixValues(t, targets, tt.wantTargets)
				return nil
			})
			if runErr != nil {
				t.Fatalf("WithRowView returned error: %v", runErr)
			}
		})
	}

	if batches, err = dataset.Batches(3, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	err = batch.WithRowView(1, 3, func(view *data.DatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, inputs, []float32{2, 20, 3, 30})
		requireMatrixValues(t, targets, []float32{102, 103})
		if view.Copied() {
			t.Fatal("contiguous Batch view Copied = true, want false")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch WithRowView returned error: %v", err)
	}
}

func Test_DatasetViewExpiresAfterCallbackErrorAndPanic(t *testing.T) {
	var (
		callbackErr error
		dataset     *data.Dataset
		retained    *data.DatasetView
		err         error
	)

	callbackErr = errors.New("injected callback failure")
	dataset = mustDatasetWithSamples(t, 2)
	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		retained = view
		return callbackErr
	})
	if !errors.Is(err, callbackErr) {
		t.Fatalf("WithView error = %v, want %v", err, callbackErr)
	}
	if err = retained.Validate(); err == nil {
		t.Fatal("error-retained view Validate error = nil, want expired view error")
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("WithView panic = nil, want injected panic")
			}
		}()

		err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
			retained = view
			panic("injected panic")
		})
	}()
	if err = retained.Validate(); err == nil {
		t.Fatal("panic-retained view Validate error = nil, want expired view error")
	}
}

func Test_DatasetAndBatchViewsRejectInvalidCallsBeforeCallback(t *testing.T) {
	type testcase struct {
		name       string
		wantPrefix string
		run        func(use data.DatasetViewFunc) error
	}

	var (
		nilDataset *data.Dataset
		nilBatch   *data.Batch
		dataset    *data.Dataset
		batches    []*data.Batch
		batch      *data.Batch
		nilView    *data.DatasetView
		zeroView   data.DatasetView
		tests      []testcase
		err        error
	)

	dataset = mustDatasetWithSamples(t, 2)
	if batches, err = dataset.Batches(2, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	tests = []testcase{
		{name: "nil dataset", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return nilDataset.WithView(use)
		}},
		{name: "zero dataset", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return (&data.Dataset{}).WithView(use)
		}},
		{name: "dataset negative start", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return dataset.WithRowView(-1, 1, use)
		}},
		{name: "dataset empty window", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return dataset.WithRowView(1, 1, use)
		}},
		{name: "dataset reversed window", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return dataset.WithRowView(1, 0, use)
		}},
		{name: "dataset end out of range", wantPrefix: "data: dataset view", run: func(use data.DatasetViewFunc) error {
			return dataset.WithRowView(0, 3, use)
		}},
		{name: "nil batch", wantPrefix: "data: batch view", run: func(use data.DatasetViewFunc) error {
			return nilBatch.WithView(use)
		}},
		{name: "zero batch", wantPrefix: "data: batch view", run: func(use data.DatasetViewFunc) error {
			return (&data.Batch{}).WithView(use)
		}},
		{name: "batch end out of range", wantPrefix: "data: batch view", run: func(use data.DatasetViewFunc) error {
			return batch.WithRowView(0, 3, use)
		}},
		{name: "dataset nil callback", wantPrefix: "data: dataset view", run: func(data.DatasetViewFunc) error {
			return dataset.WithView(nil)
		}},
		{name: "batch nil callback", wantPrefix: "data: batch view", run: func(data.DatasetViewFunc) error {
			return batch.WithView(nil)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				runErr error
			)

			runErr = tt.run(func(*data.DatasetView) (viewErr error) {
				called = true
				return nil
			})
			if runErr == nil {
				t.Fatal("view operation error = nil, want error")
			}
			if !strings.HasPrefix(runErr.Error(), tt.wantPrefix) {
				t.Fatalf("view operation error = %q, want prefix %q", runErr, tt.wantPrefix)
			}
			if called {
				t.Fatal("invalid view operation invoked callback")
			}
		})
	}

	if err = nilView.Validate(); err == nil {
		t.Fatal("nil DatasetView Validate error = nil, want error")
	}
	if _, err = nilView.Inputs(); err == nil {
		t.Fatal("nil DatasetView Inputs error = nil, want error")
	}
	if nilView.SampleCount() != 0 || nilView.InputSize() != 0 ||
		nilView.TargetSize() != 0 || nilView.Copied() {
		t.Fatal("nil DatasetView scalar accessors did not return zero values")
	}

	if err = zeroView.Validate(); err == nil {
		t.Fatal("zero DatasetView Validate error = nil, want error")
	}
	if _, err = zeroView.Targets(); err == nil {
		t.Fatal("zero DatasetView Targets error = nil, want error")
	}
	if zeroView.SampleCount() != 0 || zeroView.InputSize() != 0 ||
		zeroView.TargetSize() != 0 || zeroView.Copied() {
		t.Fatal("zero DatasetView scalar accessors did not return zero values")
	}
}

func Test_DatasetViewPreservesSafeCopyIsolation(t *testing.T) {
	var (
		sourceInputs   *matrix.Matrix
		sourceTargets  *matrix.Matrix
		dataset        *data.Dataset
		accessorInputs *matrix.Matrix
		err            error
	)

	sourceInputs = mustMatrix(t, 2, 2, []float32{1, 10, 2, 20})
	sourceTargets = mustMatrix(t, 2, 1, []float32{101, 102})
	if dataset, err = data.NewDataset(sourceInputs, sourceTargets); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}
	if err = sourceInputs.Set(0, 0, 99); err != nil {
		t.Fatalf("source inputs Set returned error: %v", err)
	}
	if err = sourceTargets.Set(0, 0, 199); err != nil {
		t.Fatalf("source targets Set returned error: %v", err)
	}
	if accessorInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if err = accessorInputs.Set(0, 0, 88); err != nil {
		t.Fatalf("accessor inputs Set returned error: %v", err)
	}

	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, inputs, []float32{1, 10, 2, 20})
		requireMatrixValues(t, targets, []float32{101, 102})
		return nil
	})
	if err != nil {
		t.Fatalf("WithView returned error: %v", err)
	}
}

func Test_DatasetViewKeepsBackingStorageAliveDuringCallback(t *testing.T) {
	var (
		dataset *data.Dataset
		err     error
	)

	dataset = mustDatasetWithSamples(t, 2)
	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		var inputs *matrix.Matrix

		dataset = nil
		runtime.GC()
		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, inputs, []float32{1, 10, 2, 20})
		return nil
	})
	if err != nil {
		t.Fatalf("WithView returned error after owner reference release: %v", err)
	}
}

func Test_DatasetViewMatchesCopiedPredictionLossAndTraining(t *testing.T) {
	var (
		dataset           *data.Dataset
		copiedInputs      *matrix.Matrix
		copiedTargets     *matrix.Matrix
		copiedNetwork     *model.Sequential
		viewedNetwork     *model.Sequential
		copiedPredictions *matrix.Matrix
		viewedPredictions *matrix.Matrix
		copiedOptimizer   *optimizer.SGD
		viewedOptimizer   *optimizer.SGD
		copiedMetrics     model.TrainMetrics
		viewedMetrics     model.TrainMetrics
		copiedLoss        float32
		viewedLoss        float32
		parameterIndex    int
		err               error
	)

	dataset = mustDatasetWithSamples(t, 3)
	if copiedInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if copiedTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}
	copiedNetwork = mustDatasetViewNetwork(t)
	viewedNetwork = mustDatasetViewNetwork(t)
	if copiedPredictions, err = copiedNetwork.Predict(copiedInputs); err != nil {
		t.Fatalf("copied Predict returned error: %v", err)
	}
	if copiedLoss, err = (loss.MeanSquaredError{}).Value(copiedPredictions, copiedTargets); err != nil {
		t.Fatalf("copied loss returned error: %v", err)
	}

	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if viewedPredictions, viewErr = viewedNetwork.Predict(inputs); viewErr != nil {
			return viewErr
		}
		viewedLoss, viewErr = loss.MeanSquaredError{}.Value(viewedPredictions, targets)
		return viewErr
	})
	if err != nil {
		t.Fatalf("viewed prediction returned error: %v", err)
	}
	testutil.RequireSliceAlmostEqual(
		t,
		mustDatasetViewValues(t, viewedPredictions),
		mustDatasetViewValues(t, copiedPredictions),
		epsilon,
	)
	testutil.RequireAlmostEqual(t, viewedLoss, copiedLoss, epsilon)

	if copiedOptimizer, err = optimizer.NewSGD(0.001); err != nil {
		t.Fatalf("NewSGD copied returned error: %v", err)
	}
	if viewedOptimizer, err = optimizer.NewSGD(0.001); err != nil {
		t.Fatalf("NewSGD viewed returned error: %v", err)
	}
	if copiedMetrics, err = copiedNetwork.TrainBatch(
		copiedInputs,
		copiedTargets,
		loss.MeanSquaredError{},
		copiedOptimizer,
	); err != nil {
		t.Fatalf("copied TrainBatch returned error: %v", err)
	}
	err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		viewedMetrics, viewErr = viewedNetwork.TrainBatch(
			inputs,
			targets,
			loss.MeanSquaredError{},
			viewedOptimizer,
		)
		return viewErr
	})
	if err != nil {
		t.Fatalf("viewed TrainBatch returned error: %v", err)
	}
	testutil.RequireAlmostEqual(t, viewedMetrics.Loss, copiedMetrics.Loss, epsilon)
	for parameterIndex = range copiedNetwork.Parameters() {
		testutil.RequireSliceAlmostEqual(
			t,
			mustDatasetViewValues(t, viewedNetwork.Parameters()[parameterIndex].Values()),
			mustDatasetViewValues(t, copiedNetwork.Parameters()[parameterIndex].Values()),
			epsilon,
		)
	}
}

func Test_DatasetViewSupportsDistinctOwnerConcurrency(t *testing.T) {
	var (
		first     *data.Dataset
		second    *data.Dataset
		waitGroup sync.WaitGroup
		errorsOut chan error
		run       func(*data.Dataset)
		err       error
	)

	first = mustDatasetWithSamples(t, 2)
	second = mustDatasetWithSamples(t, 3)
	errorsOut = make(chan error, 2)
	run = func(dataset *data.Dataset) {
		defer waitGroup.Done()
		errorsOut <- dataset.WithView(func(view *data.DatasetView) (viewErr error) {
			var inputs *matrix.Matrix

			if inputs, viewErr = view.Inputs(); viewErr != nil {
				return viewErr
			}
			_, viewErr = inputs.At(view.SampleCount()-1, 0)
			return viewErr
		})
	}

	waitGroup.Add(2)
	go run(first)
	go run(second)
	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("concurrent distinct WithView returned error: %v", err)
		}
	}
}

func mustDatasetViewNetwork(tb testing.TB) (network *model.Sequential) {
	var (
		dense *layer.Dense
		err   error
	)

	tb.Helper()
	if dense, err = layer.NewDense(2, 1, func(_, _ int) (weights *matrix.Matrix, initErr error) {
		weights, initErr = matrix.FromSlice(2, 1, []float32{0.25, -0.5})
		return weights, initErr
	}); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(dense); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func mustDatasetViewValues(tb testing.TB, value *matrix.Matrix) (values []float32) {
	var err error

	tb.Helper()
	if values, err = value.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	return values
}
