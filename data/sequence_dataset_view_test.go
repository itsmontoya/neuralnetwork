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

func Test_SequenceDatasetWithViewReturnsAlignedWholeView(t *testing.T) {
	var (
		dataset         *data.SequenceDataset
		retained        *data.SequenceDatasetView
		retainedCopy    data.SequenceDatasetView
		retainedInput   *matrix.Matrix
		retainedTarget  *matrix.Matrix
		retainedLengths []int
		err             error
	)

	dataset = mustSequenceDatasetWithSamples(t, 3)
	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
		retained = view
		retainedCopy = *view
		if viewErr = view.Validate(); viewErr != nil {
			return viewErr
		}
		if view.SampleCount() != 3 || view.InputSize() != 4 ||
			view.TargetSize() != 1 || view.Steps() != 2 {
			t.Fatalf(
				"view dimensions = samples %d inputs %d targets %d steps %d, want 3, 4, 1, 2",
				view.SampleCount(),
				view.InputSize(),
				view.TargetSize(),
				view.Steps(),
			)
		}
		if view.Copied() {
			t.Fatal("whole SequenceDatasetView Copied = true, want false")
		}
		if retainedInput, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if retainedTarget, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if retainedLengths, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		requireSequenceAlignment(t, retainedInput, retainedTarget, retainedLengths)
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
	if _, err = retained.Lengths(); err == nil {
		t.Fatal("expired view Lengths error = nil, want error")
	}
	if retained.SampleCount() != 0 || retained.InputSize() != 0 ||
		retained.TargetSize() != 0 || retained.Steps() != 0 ||
		retained.Copied() {
		t.Fatal("expired view scalar accessors did not return zero values")
	}
	if err = retainedInput.Validate(); err == nil {
		t.Fatal("retained input matrix Validate error = nil, want expired matrix error")
	}
	if err = retainedTarget.Validate(); err == nil {
		t.Fatal("retained target matrix Validate error = nil, want expired matrix error")
	}
}

func Test_SequenceDatasetAndBatchWithRowViewReturnAlignedWindows(t *testing.T) {
	type testcase struct {
		name        string
		start       int
		end         int
		wantInputs  []float32
		wantTargets []float32
		wantLengths []int
	}

	var (
		dataset *data.SequenceDataset
		batches []*data.SequenceBatch
		batch   *data.SequenceBatch
		tests   []testcase
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 5)
	tests = []testcase{
		{
			name:        "single row",
			start:       2,
			end:         3,
			wantInputs:  []float32{3, 30, 300, 3000},
			wantTargets: []float32{103},
			wantLengths: []int{2},
		},
		{
			name:  "overlapping middle window",
			start: 1,
			end:   4,
			wantInputs: []float32{
				2, 20, 200, 2000,
				3, 30, 300, 3000,
				4, 40, 400, 4000,
			},
			wantTargets: []float32{102, 103, 104},
			wantLengths: []int{1, 2, 1},
		},
		{
			name:  "full window",
			start: 0,
			end:   5,
			wantInputs: []float32{
				1, 10, 100, 1000,
				2, 20, 200, 2000,
				3, 30, 300, 3000,
				4, 40, 400, 4000,
				5, 50, 500, 5000,
			},
			wantTargets: []float32{101, 102, 103, 104, 105},
			wantLengths: []int{2, 1, 2, 1, 2},
		},
		{
			name:        "partial final window",
			start:       4,
			end:         5,
			wantInputs:  []float32{5, 50, 500, 5000},
			wantTargets: []float32{105},
			wantLengths: []int{2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var runErr error

			runErr = dataset.WithRowView(
				tt.start,
				tt.end,
				func(view *data.SequenceDatasetView) (viewErr error) {
					var (
						inputs  *matrix.Matrix
						targets *matrix.Matrix
						lengths []int
					)

					if inputs, viewErr = view.Inputs(); viewErr != nil {
						return viewErr
					}
					if targets, viewErr = view.Targets(); viewErr != nil {
						return viewErr
					}
					if lengths, viewErr = view.Lengths(); viewErr != nil {
						return viewErr
					}
					requireMatrixValues(t, inputs, tt.wantInputs)
					requireMatrixValues(t, targets, tt.wantTargets)
					requireIntValues(t, lengths, tt.wantLengths)
					if inputs.Rows() != len(tt.wantLengths) ||
						targets.Rows() != len(tt.wantLengths) {
						t.Fatal("row view components are not aligned")
					}
					return nil
				},
			)
			if runErr != nil {
				t.Fatalf("WithRowView returned error: %v", runErr)
			}
		})
	}

	if batches, err = dataset.Batches(3, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	err = batch.WithRowView(1, 3, func(view *data.SequenceDatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
			lengths []int
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if lengths, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, inputs, []float32{
			2, 20, 200, 2000,
			3, 30, 300, 3000,
		})
		requireMatrixValues(t, targets, []float32{102, 103})
		requireIntValues(t, lengths, []int{1, 2})
		if view.Copied() {
			t.Fatal("contiguous SequenceBatch view Copied = true, want false")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch WithRowView returned error: %v", err)
	}
}

func Test_SequenceDatasetViewExpiresAfterCallbackErrorAndPanic(t *testing.T) {
	var (
		callbackErr error
		dataset     *data.SequenceDataset
		retained    *data.SequenceDatasetView
		err         error
	)

	callbackErr = errors.New("injected callback failure")
	dataset = mustSequenceDatasetWithSamples(t, 2)
	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
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

		err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
			retained = view
			panic("injected panic")
		})
	}()
	if err = retained.Validate(); err == nil {
		t.Fatal("panic-retained view Validate error = nil, want expired view error")
	}
}

func Test_SequenceDatasetAndBatchViewsRejectInvalidCallsBeforeCallback(t *testing.T) {
	type testcase struct {
		name       string
		wantPrefix string
		run        func(use data.SequenceDatasetViewFunc) error
	}

	var (
		nilDataset *data.SequenceDataset
		nilBatch   *data.SequenceBatch
		dataset    *data.SequenceDataset
		batches    []*data.SequenceBatch
		batch      *data.SequenceBatch
		nilView    *data.SequenceDatasetView
		zeroView   data.SequenceDatasetView
		tests      []testcase
		err        error
	)

	dataset = mustSequenceDatasetWithSamples(t, 2)
	if batches, err = dataset.Batches(2, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	tests = []testcase{
		{
			name:       "nil dataset",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = nilDataset.WithView(use)
				return runErr
			},
		},
		{
			name:       "zero dataset",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = (&data.SequenceDataset{}).WithView(use)
				return runErr
			},
		},
		{
			name:       "dataset negative start",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = dataset.WithRowView(-1, 1, use)
				return runErr
			},
		},
		{
			name:       "dataset empty window",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = dataset.WithRowView(1, 1, use)
				return runErr
			},
		},
		{
			name:       "dataset reversed window",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = dataset.WithRowView(1, 0, use)
				return runErr
			},
		},
		{
			name:       "dataset end out of range",
			wantPrefix: "data: sequence dataset view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = dataset.WithRowView(0, 3, use)
				return runErr
			},
		},
		{
			name:       "nil batch",
			wantPrefix: "data: sequence batch view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = nilBatch.WithView(use)
				return runErr
			},
		},
		{
			name:       "zero batch",
			wantPrefix: "data: sequence batch view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = (&data.SequenceBatch{}).WithView(use)
				return runErr
			},
		},
		{
			name:       "batch end out of range",
			wantPrefix: "data: sequence batch view",
			run: func(use data.SequenceDatasetViewFunc) (runErr error) {
				runErr = batch.WithRowView(0, 3, use)
				return runErr
			},
		},
		{
			name:       "dataset nil callback",
			wantPrefix: "data: sequence dataset view",
			run: func(data.SequenceDatasetViewFunc) (runErr error) {
				runErr = dataset.WithView(nil)
				return runErr
			},
		},
		{
			name:       "batch nil callback",
			wantPrefix: "data: sequence batch view",
			run: func(data.SequenceDatasetViewFunc) (runErr error) {
				runErr = batch.WithView(nil)
				return runErr
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				runErr error
			)

			runErr = tt.run(func(*data.SequenceDatasetView) (viewErr error) {
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
		t.Fatal("nil SequenceDatasetView Validate error = nil, want error")
	}
	if _, err = nilView.Inputs(); err == nil {
		t.Fatal("nil SequenceDatasetView Inputs error = nil, want error")
	}
	if _, err = nilView.Lengths(); err == nil {
		t.Fatal("nil SequenceDatasetView Lengths error = nil, want error")
	}
	if nilView.SampleCount() != 0 || nilView.InputSize() != 0 ||
		nilView.TargetSize() != 0 || nilView.Steps() != 0 ||
		nilView.Copied() {
		t.Fatal("nil SequenceDatasetView scalar accessors did not return zero values")
	}

	if err = zeroView.Validate(); err == nil {
		t.Fatal("zero SequenceDatasetView Validate error = nil, want error")
	}
	if _, err = zeroView.Targets(); err == nil {
		t.Fatal("zero SequenceDatasetView Targets error = nil, want error")
	}
	if zeroView.SampleCount() != 0 || zeroView.InputSize() != 0 ||
		zeroView.TargetSize() != 0 || zeroView.Steps() != 0 ||
		zeroView.Copied() {
		t.Fatal("zero SequenceDatasetView scalar accessors did not return zero values")
	}
}

func Test_SequenceDatasetViewRechecksLogicalLengthRange(t *testing.T) {
	type testcase struct {
		name    string
		invalid int
	}

	var (
		dataset *data.SequenceDataset
		tests   []testcase
	)

	dataset = mustSequenceDatasetWithSamples(t, 2)
	tests = []testcase{
		{name: "zero", invalid: 0},
		{name: "greater than steps", invalid: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
				var (
					lengths  []int
					original int
				)

				if lengths, viewErr = view.Lengths(); viewErr != nil {
					return viewErr
				}
				original = lengths[0]
				lengths[0] = tt.invalid
				defer func() {
					lengths[0] = original
				}()

				if viewErr = view.Validate(); viewErr == nil {
					t.Fatal("Validate error = nil after forbidden length mutation")
				}
				if _, viewErr = view.Inputs(); viewErr == nil {
					t.Fatal("Inputs error = nil after invalid length mutation")
				}
				if _, viewErr = view.Targets(); viewErr == nil {
					t.Fatal("Targets error = nil after invalid length mutation")
				}
				if _, viewErr = view.Lengths(); viewErr == nil {
					t.Fatal("Lengths error = nil after invalid length mutation")
				}
				return nil
			})
			if err != nil {
				t.Fatalf("WithView returned error: %v", err)
			}
		})
	}
}

func Test_SequenceDatasetViewPreservesSafeCopyIsolation(t *testing.T) {
	var (
		sourceInputs    *matrix.Matrix
		sourceTargets   *matrix.Matrix
		sourceValues    []int
		sourceLengths   *data.SequenceLengths
		dataset         *data.SequenceDataset
		accessorInputs  *matrix.Matrix
		accessorTargets *matrix.Matrix
		accessorLengths *data.SequenceLengths
		lengthValues    []int
		err             error
	)

	sourceInputs = mustMatrix(t, 2, 4, []float32{
		1, 10, 100, 1000,
		2, 20, 200, 2000,
	})
	sourceTargets = mustMatrix(t, 2, 1, []float32{101, 102})
	sourceValues = []int{2, 1}
	sourceLengths = mustSequenceLengths(t, 2, sourceValues)
	if dataset, err = data.NewSequenceDataset(
		sourceInputs,
		sourceTargets,
		sourceLengths,
	); err != nil {
		t.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	if err = sourceInputs.Set(0, 0, 99); err != nil {
		t.Fatalf("source inputs Set returned error: %v", err)
	}
	if err = sourceTargets.Set(0, 0, 199); err != nil {
		t.Fatalf("source targets Set returned error: %v", err)
	}
	sourceValues[0] = 1

	if accessorInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if accessorTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}
	if accessorLengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}
	if err = accessorInputs.Set(0, 0, 88); err != nil {
		t.Fatalf("accessor inputs Set returned error: %v", err)
	}
	if err = accessorTargets.Set(0, 0, 188); err != nil {
		t.Fatalf("accessor targets Set returned error: %v", err)
	}
	if lengthValues, err = accessorLengths.Values(); err != nil {
		t.Fatalf("accessor length Values returned error: %v", err)
	}
	lengthValues[0] = 1

	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
			lengths []int
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if lengths, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		requireMatrixValues(t, inputs, []float32{
			1, 10, 100, 1000,
			2, 20, 200, 2000,
		})
		requireMatrixValues(t, targets, []float32{101, 102})
		requireIntValues(t, lengths, []int{2, 1})
		return nil
	})
	if err != nil {
		t.Fatalf("WithView returned error: %v", err)
	}
}

func Test_SequenceDatasetViewKeepsBackingStorageAliveDuringCallback(t *testing.T) {
	var (
		dataset *data.SequenceDataset
		err     error
	)

	dataset = mustSequenceDatasetWithSamples(t, 2)
	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
		var (
			inputs  *matrix.Matrix
			targets *matrix.Matrix
			lengths []int
		)

		dataset = nil
		runtime.GC()
		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if lengths, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		requireSequenceAlignment(t, inputs, targets, lengths)
		return nil
	})
	if err != nil {
		t.Fatalf("WithView returned error after owner reference release: %v", err)
	}
}

func Test_SequenceDatasetViewMatchesCopiedLengthAwareOperations(t *testing.T) {
	var (
		dataset             *data.SequenceDataset
		copiedInputs        *matrix.Matrix
		copiedTargets       *matrix.Matrix
		copiedLengths       *data.SequenceLengths
		copiedNetwork       *model.Sequential
		viewedNetwork       *model.Sequential
		copiedPredictions   *matrix.Matrix
		viewedPredictions   *matrix.Matrix
		outputGradient      *matrix.Matrix
		copiedInputGradient *matrix.Matrix
		viewedInputGradient *matrix.Matrix
		copiedTraining      *model.Sequential
		viewedTraining      *model.Sequential
		copiedClipping      *optimizer.GradientClipping
		viewedClipping      *optimizer.GradientClipping
		copiedMetrics       model.TrainMetrics
		viewedMetrics       model.TrainMetrics
		copiedObservation   optimizer.GradientClippingObservation
		viewedObservation   optimizer.GradientClippingObservation
		copiedAvailable     bool
		viewedAvailable     bool
		parameterIndex      int
		err                 error
	)

	dataset = mustSequenceDatasetWithSamples(t, 3)
	if copiedInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}
	if copiedTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}
	if copiedLengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}

	copiedNetwork = mustSequenceDatasetViewNetwork(t)
	viewedNetwork = mustSequenceDatasetViewNetwork(t)
	if copiedPredictions, err = copiedNetwork.PredictWithLengths(
		copiedInputs,
		copiedLengths,
	); err != nil {
		t.Fatalf("copied PredictWithLengths returned error: %v", err)
	}
	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
		var (
			inputs       *matrix.Matrix
			lengthValues []int
			lengths      *data.SequenceLengths
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if lengthValues, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		if lengths, viewErr = data.NewSequenceLengths(
			view.Steps(),
			lengthValues,
		); viewErr != nil {
			return viewErr
		}
		viewedPredictions, viewErr = viewedNetwork.PredictWithLengths(inputs, lengths)
		return viewErr
	})
	if err != nil {
		t.Fatalf("viewed PredictWithLengths returned error: %v", err)
	}
	testutil.RequireSliceAlmostEqual(
		t,
		mustDatasetViewValues(t, viewedPredictions),
		mustDatasetViewValues(t, copiedPredictions),
		epsilon,
	)

	outputGradient = mustMatrix(t, 3, 1, []float32{0.5, -0.25, 0.75})
	if copiedInputGradient, err = copiedNetwork.BackwardWithLengths(
		outputGradient,
	); err != nil {
		t.Fatalf("copied BackwardWithLengths returned error: %v", err)
	}
	if viewedInputGradient, err = viewedNetwork.BackwardWithLengths(
		outputGradient,
	); err != nil {
		t.Fatalf("viewed BackwardWithLengths returned error: %v", err)
	}
	testutil.RequireSliceAlmostEqual(
		t,
		mustDatasetViewValues(t, viewedInputGradient),
		mustDatasetViewValues(t, copiedInputGradient),
		epsilon,
	)
	for parameterIndex = range copiedNetwork.Parameters() {
		testutil.RequireSliceAlmostEqual(
			t,
			mustDatasetViewValues(t, viewedNetwork.Parameters()[parameterIndex].Gradient()),
			mustDatasetViewValues(t, copiedNetwork.Parameters()[parameterIndex].Gradient()),
			epsilon,
		)
	}

	copiedTraining = mustSequenceDatasetViewNetwork(t)
	viewedTraining = mustSequenceDatasetViewNetwork(t)
	copiedClipping = mustSequenceDatasetViewClipping(t)
	viewedClipping = mustSequenceDatasetViewClipping(t)
	if copiedMetrics, err = copiedTraining.TrainBatchWithLengths(
		copiedInputs,
		copiedTargets,
		copiedLengths,
		loss.MeanSquaredError{},
		copiedClipping,
	); err != nil {
		t.Fatalf("copied TrainBatchWithLengths returned error: %v", err)
	}
	err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
		var (
			inputs       *matrix.Matrix
			targets      *matrix.Matrix
			lengthValues []int
			lengths      *data.SequenceLengths
		)

		if inputs, viewErr = view.Inputs(); viewErr != nil {
			return viewErr
		}
		if targets, viewErr = view.Targets(); viewErr != nil {
			return viewErr
		}
		if lengthValues, viewErr = view.Lengths(); viewErr != nil {
			return viewErr
		}
		if lengths, viewErr = data.NewSequenceLengths(
			view.Steps(),
			lengthValues,
		); viewErr != nil {
			return viewErr
		}
		viewedMetrics, viewErr = viewedTraining.TrainBatchWithLengths(
			inputs,
			targets,
			lengths,
			loss.MeanSquaredError{},
			viewedClipping,
		)
		return viewErr
	})
	if err != nil {
		t.Fatalf("viewed TrainBatchWithLengths returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, viewedMetrics.Loss, copiedMetrics.Loss, epsilon)
	for parameterIndex = range copiedTraining.Parameters() {
		testutil.RequireSliceAlmostEqual(
			t,
			mustDatasetViewValues(t, viewedTraining.Parameters()[parameterIndex].Values()),
			mustDatasetViewValues(t, copiedTraining.Parameters()[parameterIndex].Values()),
			epsilon,
		)
	}
	copiedObservation, copiedAvailable = copiedClipping.Observation()
	viewedObservation, viewedAvailable = viewedClipping.Observation()
	if copiedAvailable != viewedAvailable {
		t.Fatalf(
			"viewed clipping observation available = %t, want %t",
			viewedAvailable,
			copiedAvailable,
		)
	}
	if copiedObservation.ValueClipped != viewedObservation.ValueClipped ||
		copiedObservation.BaseUpdateCompleted != viewedObservation.BaseUpdateCompleted {
		t.Fatal("viewed clipping flags differ from copied clipping flags")
	}
	testutil.RequireAlmostEqual(
		t,
		float32(viewedObservation.GlobalNorm),
		float32(copiedObservation.GlobalNorm),
		epsilon,
	)
	testutil.RequireAlmostEqual(
		t,
		float32(viewedObservation.Scale),
		float32(copiedObservation.Scale),
		epsilon,
	)
}

func Test_SequenceDatasetViewSupportsDistinctOwnerConcurrency(t *testing.T) {
	var (
		first     *data.SequenceDataset
		second    *data.SequenceDataset
		waitGroup sync.WaitGroup
		errorsOut chan error
		run       func(*data.SequenceDataset)
		err       error
	)

	first = mustSequenceDatasetWithSamples(t, 2)
	second = mustSequenceDatasetWithSamples(t, 3)
	errorsOut = make(chan error, 2)
	run = func(dataset *data.SequenceDataset) {
		defer waitGroup.Done()
		errorsOut <- dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
			var (
				inputs  *matrix.Matrix
				lengths []int
			)

			if inputs, viewErr = view.Inputs(); viewErr != nil {
				return viewErr
			}
			if lengths, viewErr = view.Lengths(); viewErr != nil {
				return viewErr
			}
			if len(lengths) != view.SampleCount() {
				return errors.New("lengths are not aligned")
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

func mustSequenceDatasetViewNetwork(tb testing.TB) (network *model.Sequential) {
	var (
		shape  layer.SequenceShape
		gather *layer.GatherLastValid
		dense  *layer.Dense
		err    error
	)

	tb.Helper()
	if shape, err = layer.NewSequenceShape(2, 2); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if gather, err = layer.NewGatherLastValid(shape); err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}
	if dense, err = layer.NewDense(2, 1, func(_, _ int) (weights *matrix.Matrix, initErr error) {
		weights, initErr = matrix.FromSlice(2, 1, []float32{0.25, -0.5})
		return weights, initErr
	}); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(gather, dense); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func mustSequenceDatasetViewClipping(tb testing.TB) (clipping *optimizer.GradientClipping) {
	var (
		base *optimizer.SGD
		err  error
	)

	tb.Helper()
	if base, err = optimizer.NewSGD(0.001); err != nil {
		tb.Fatalf("NewSGD returned error: %v", err)
	}
	if clipping, err = optimizer.NewGradientClipping(
		base,
		optimizer.GradientClippingConfig{MaxNorm: 1},
	); err != nil {
		tb.Fatalf("NewGradientClipping returned error: %v", err)
	}
	return clipping
}
