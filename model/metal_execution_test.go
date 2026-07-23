//go:build darwin && cgo && metal && !purego

package model_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_SequentialPredictBatchesAcrossCustomCPUFallback(t *testing.T) {
	var (
		firstDense  *layer.Dense
		secondDense *layer.Dense
		network     *model.Sequential
		input       *matrix.Matrix
		output      *matrix.Matrix
		counters    metaltest.Counters
		err         error
	)

	requireModelMetal(t)
	if firstDense, err = layer.NewDense(128, 128, metalExecutionInitializer); err != nil {
		t.Fatalf("NewDense first returned error: %v", err)
	}
	if secondDense, err = layer.NewDense(128, 128, metalExecutionInitializer); err != nil {
		t.Fatalf("NewDense second returned error: %v", err)
	}
	if network, err = model.NewSequential(
		firstDense,
		&recordingLayer{forwardDelta: 1},
		secondDense,
	); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	input = metalExecutionMatrix(t, 64, 128, 0.25)

	metaltest.Enable()
	defer metaltest.Disable()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("first Predict returned error: %v", err)
	}
	if output.Rows() != 64 || output.Cols() != 128 {
		t.Fatalf("first output shape = %dx%d, want 64x128", output.Rows(), output.Cols())
	}
	counters = metaltest.Snapshot()
	requireModelMetalCounters(t, counters, 10, 6, 1, 2, 2)
	requireRecordingFallbackOutput(t, output, 0.25)

	metaltest.Reset()
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("second Predict returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 1 {
		t.Fatalf("warmed input uploads = %d, want only the custom CPU output upload", counters.InputUploads)
	}
	if counters.CommandSubmissions != 2 || counters.Waits != 2 {
		t.Fatalf("warmed command counters = %+v, want two submissions and waits", counters)
	}
}

func requireRecordingFallbackOutput(tb testing.TB, output *matrix.Matrix, inputScale float32) {
	tb.Helper()

	var (
		values      []float32
		row         int
		col         int
		inputCol    int
		inputSum    float32
		firstOutput float32
		want        float32
		err         error
	)

	if values, err = output.Values(); err != nil {
		tb.Fatalf("fallback output Values returned error: %v", err)
	}
	for row = 0; row < output.Rows(); row++ {
		inputSum = 0
		for inputCol = 0; inputCol < 128; inputCol++ {
			inputSum += inputScale * float32((row*128+inputCol)%17-8)
		}
		firstOutput = 0.001 * inputSum
		want = 0.128 * (firstOutput + 1)
		for col = 0; col < output.Cols(); col++ {
			if difference := values[row*output.Cols()+col] - want; difference < -1e-5 || difference > 1e-5 {
				tb.Fatalf(
					"fallback output row %d col %d = %g, want %g",
					row,
					col,
					values[row*output.Cols()+col],
					want,
				)
			}
		}
	}
}

func Test_SequentialPredictBelowThresholdDoesNotCreateCommands(t *testing.T) {
	var (
		dense    *layer.Dense
		network  *model.Sequential
		input    *matrix.Matrix
		counters metaltest.Counters
		err      error
	)

	requireModelMetal(t)
	if dense, err = layer.NewDense(4, 4, metalExecutionInitializer); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(dense); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	input = metalExecutionMatrix(t, 2, 4, 0.25)

	metaltest.Enable()
	defer metaltest.Disable()
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireModelMetalCounters(t, counters, 0, 0, 0, 0, 0)
}

func Test_SequentialPredictReusesNestedBoundExecution(t *testing.T) {
	var (
		execution *device.Execution
		bound     *device.Execution
		network   *model.Sequential
		input     *matrix.Matrix
		output    *matrix.Matrix
		available bool
		err       error
	)

	if execution, available, err = device.NewSharedExecution(); err != nil {
		t.Fatalf("NewSharedExecution returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	input = metalExecutionMatrix(t, 2, 2, 0.5)
	if err = execution.Bind(input); err != nil {
		t.Fatalf("Bind input returned error: %v", err)
	}
	if network, err = model.NewSequential(&recordingLayer{forwardDelta: 1}); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if bound, err = device.BoundExecution(output); err != nil {
		t.Fatalf("BoundExecution output returned error: %v", err)
	}
	if bound != execution {
		t.Fatalf("nested output execution = %p, want %p", bound, execution)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish outer execution returned error: %v", err)
	}
	if bound, err = device.BoundExecution(output); err != nil {
		t.Fatalf("BoundExecution after Finish returned error: %v", err)
	}
	if bound != nil {
		t.Fatalf("output remained bound after Finish: %p", bound)
	}
}

func Test_SequentialPredictAbortsPendingWorkOnLayerError(t *testing.T) {
	var (
		runtimeValue *device.Runtime
		dense        *layer.Dense
		failure      *evaluationErrorLayer
		network      *model.Sequential
		input        *matrix.Matrix
		before       device.ResourceSnapshot
		after        device.ResourceSnapshot
		available    bool
		err          error
	)

	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	if dense, err = layer.NewDense(128, 128, metalExecutionInitializer); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	failure = &evaluationErrorLayer{err: errors.New("injected layer failure")}
	if network, err = model.NewSequential(dense, failure); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if err = network.SetTraining(false); err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}
	input = metalExecutionMatrix(t, 64, 128, 0.125)
	before = runtimeValue.ResourceSnapshot()

	if _, err = network.Predict(input); err == nil || !strings.Contains(err.Error(), "injected layer failure") {
		t.Fatalf("Predict error = %v, want injected layer failure", err)
	}
	after = runtimeValue.ResourceSnapshot()
	if after.LiveScopes != before.LiveScopes || after.CreatedScopes-after.ReleasedScopes != before.LiveScopes {
		t.Fatalf("scope resources before=%+v after=%+v, want no leaked scope", before, after)
	}
}

func Test_SequentialPredictReleasesPendingWorkOnPanic(t *testing.T) {
	var (
		runtimeValue *device.Runtime
		dense        *layer.Dense
		network      *model.Sequential
		input        *matrix.Matrix
		before       device.ResourceSnapshot
		after        device.ResourceSnapshot
		panicValue   any
		available    bool
		err          error
	)

	if runtimeValue, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}
	if dense, err = layer.NewDense(128, 128, metalExecutionInitializer); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(dense, panickingLayer{}); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	input = metalExecutionMatrix(t, 64, 128, 0.375)
	before = runtimeValue.ResourceSnapshot()

	func() {
		defer func() {
			panicValue = recover()
		}()
		network.Predict(input)
	}()
	if panicValue == nil {
		t.Fatal("Predict did not propagate layer panic")
	}
	after = runtimeValue.ResourceSnapshot()
	if after.LiveScopes != before.LiveScopes || after.CreatedScopes-after.ReleasedScopes != before.LiveScopes {
		t.Fatalf("scope resources before=%+v after=%+v, want no leaked scope", before, after)
	}
}

type panickingLayer struct{}

func (panickingLayer) Forward(*matrix.Matrix) (output *matrix.Matrix, err error) {
	panic("injected layer panic")
}

func (panickingLayer) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient = outputGradient
	return inputGradient, nil
}

func metalExecutionInitializer(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
	if weights, err = matrix.New(inputSize, outputSize); err != nil {
		return nil, err
	}
	if err = weights.Fill(0.001); err != nil {
		return nil, err
	}
	return weights, nil
}

func metalExecutionMatrix(
	tb testing.TB,
	rows,
	cols int,
	scale float32,
) (value *matrix.Matrix) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = scale * float32(index%17-8)
	}
	if value, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}
	return value
}

func requireModelMetal(tb testing.TB) {
	tb.Helper()

	var (
		execution *device.Execution
		available bool
		err       error
	)

	if execution, available, err = device.NewSharedExecution(); err != nil {
		tb.Fatalf("NewSharedExecution returned error: %v", err)
	}
	if !available {
		tb.Skip("Metal device unavailable")
	}
	if err = execution.Finish(); err != nil {
		tb.Fatalf("Finish availability execution returned error: %v", err)
	}
}

func requireModelMetalCounters(
	tb testing.TB,
	counters metaltest.Counters,
	buffers,
	uploads,
	downloads,
	commands,
	waits uint64,
) {
	tb.Helper()

	if counters.BufferCreations != buffers ||
		counters.InputUploads != uploads ||
		counters.ResultDownloads != downloads ||
		counters.CommandSubmissions != commands ||
		counters.Waits != waits {
		tb.Fatalf(
			"Metal counters = %+v, want buffers=%d uploads=%d downloads=%d commands=%d waits=%d",
			counters,
			buffers,
			uploads,
			downloads,
			commands,
			waits,
		)
	}
}
