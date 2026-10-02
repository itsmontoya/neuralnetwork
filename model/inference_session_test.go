package model_test

import (
	"errors"
	"math/rand"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_InferenceSession_ConcurrentSessionsMatchSerialPrediction(t *testing.T) {
	const sessionCount = 16

	var (
		network         *model.Sequential
		input           *matrix.Matrix
		reference       *matrix.Matrix
		referenceValues []float32
		sessions        []*model.InferenceSession
		wait            sync.WaitGroup
		errorsByIndex   []error
		valuesByIndex   [][]float32
		index           int
		err             error
	)

	network = newInferenceSessionNetwork(t)
	input = mustMatrix(t, 3, 2, []float32{1, 2, -1, 0.5, 3, -2})
	if reference, err = network.Predict(input); err != nil {
		t.Fatalf("reference Predict returned error: %v", err)
	}
	if referenceValues, err = reference.Values(); err != nil {
		t.Fatalf("reference Values returned error: %v", err)
	}
	sessions = make([]*model.InferenceSession, sessionCount)
	errorsByIndex = make([]error, sessionCount)
	valuesByIndex = make([][]float32, sessionCount)
	for index = range sessions {
		if sessions[index], err = network.NewInferenceSession(); err != nil {
			t.Fatalf("NewInferenceSession %d returned error: %v", index, err)
		}
		defer sessions[index].Close()
	}

	wait.Add(sessionCount)
	for index = range sessions {
		go func(sessionIndex int) {
			var output *matrix.Matrix

			defer wait.Done()
			output, errorsByIndex[sessionIndex] = sessions[sessionIndex].Predict(input)
			if errorsByIndex[sessionIndex] != nil {
				return
			}
			valuesByIndex[sessionIndex], errorsByIndex[sessionIndex] = output.Values()
		}(index)
	}
	wait.Wait()

	for index = range sessions {
		if errorsByIndex[index] != nil {
			t.Fatalf("session %d Predict returned error: %v", index, errorsByIndex[index])
		}
		requireFloat32Values(t, valuesByIndex[index], referenceValues)
	}
}

func Test_InferenceSession_OutputOwnershipAndPredictInto(t *testing.T) {
	var (
		network        *model.Sequential
		first          *model.InferenceSession
		second         *model.InferenceSession
		input          *matrix.Matrix
		firstOutput    *matrix.Matrix
		secondOutput   *matrix.Matrix
		destination    *matrix.Matrix
		retainedValues []float32
		err            error
	)

	network = newInferenceSessionNetwork(t)
	input = mustMatrix(t, 2, 2, []float32{1, 2, -1, 0.5})
	if first, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("first NewInferenceSession returned error: %v", err)
	}
	defer first.Close()
	if second, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("second NewInferenceSession returned error: %v", err)
	}
	defer second.Close()
	if firstOutput, err = first.Predict(input); err != nil {
		t.Fatalf("first Predict returned error: %v", err)
	}
	if retainedValues, err = firstOutput.Values(); err != nil {
		t.Fatalf("first Values returned error: %v", err)
	}
	if secondOutput, err = second.Predict(input); err != nil {
		t.Fatalf("second Predict returned error: %v", err)
	}
	if firstOutput == secondOutput {
		t.Fatal("different sessions returned the same output matrix")
	}
	requireMatrixValues(t, firstOutput, retainedValues)

	if destination, err = matrix.New(2, 2); err != nil {
		t.Fatalf("New destination returned error: %v", err)
	}
	if err = first.PredictInto(input, destination); err != nil {
		t.Fatalf("PredictInto returned error: %v", err)
	}
	requireMatrixValues(t, destination, mustValues(t, secondOutput))
}

func Test_InferenceSession_CloseAndOwnerLifecycle(t *testing.T) {
	var (
		network       *model.Sequential
		first         *model.InferenceSession
		second        *model.InferenceSession
		input         *matrix.Matrix
		optimizerRule *optimizer.SGD
		err           error
	)

	network = newInferenceSessionNetwork(t)
	input = mustMatrix(t, 1, 2, []float32{1, 2})
	if first, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("first NewInferenceSession returned error: %v", err)
	}
	if second, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("second NewInferenceSession returned error: %v", err)
	}
	defer second.Close()

	if _, err = network.Predict(input); !errors.Is(err, model.ErrInferenceSessionsActive) {
		t.Fatalf("owner Predict error = %v, want ErrInferenceSessionsActive", err)
	}
	if err = network.Add(mustActivationLayer(t, activation.Linear{})); !errors.Is(err, model.ErrInferenceSessionsActive) {
		t.Fatalf("owner Add error = %v, want ErrInferenceSessionsActive", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(
		input,
		mustMatrix(t, 1, 2, []float32{0, 0}),
		loss.MeanSquaredError{},
		optimizerRule,
	); !errors.Is(err, model.ErrInferenceSessionsActive) {
		t.Fatalf("TrainBatch error = %v, want ErrInferenceSessionsActive", err)
	}

	if err = first.Close(); err != nil {
		t.Fatalf("first Close returned error: %v", err)
	}
	if err = first.Close(); err != nil {
		t.Fatalf("idempotent Close returned error: %v", err)
	}
	if _, err = first.Predict(input); !errors.Is(err, model.ErrInferenceSessionClosed) {
		t.Fatalf("closed Predict error = %v, want ErrInferenceSessionClosed", err)
	}
	if _, err = second.Predict(input); err != nil {
		t.Fatalf("second Predict after first Close returned error: %v", err)
	}
	if err = second.Close(); err != nil {
		t.Fatalf("second Close returned error: %v", err)
	}
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("owner Predict after all sessions closed returned error: %v", err)
	}
}

func Test_InferenceSession_UsesEvaluationModeAndRecoversAfterError(t *testing.T) {
	var (
		dropout *layer.Dropout
		network *model.Sequential
		session *model.InferenceSession
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
	)

	if dropout, err = layer.NewDropout(0.75, rand.New(rand.NewSource(7))); err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}
	if network, err = model.NewSequential(dropout); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if session, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("NewInferenceSession returned error: %v", err)
	}
	defer session.Close()
	if _, err = session.Predict(mustMatrix(t, 1, 2, []float32{1, 2})); err != nil {
		t.Fatalf("initial Predict returned error: %v", err)
	}
	if _, err = session.Predict(nil); err == nil {
		t.Fatal("Predict error = nil for nil input")
	}
	input = mustMatrix(t, 1, 3, []float32{1, 2, 3})
	if output, err = session.Predict(input); err != nil {
		t.Fatalf("Predict after error returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{1, 2, 3})
}

func Test_InferenceSession_PanicDoesNotLeaveSessionInUse(t *testing.T) {
	var (
		function        *panicOnceActivation
		activationLayer *layer.Activation
		network         *model.Sequential
		session         *model.InferenceSession
		input           *matrix.Matrix
		output          *matrix.Matrix
		panicValue      any
		err             error
	)

	function = &panicOnceActivation{}
	if activationLayer, err = layer.NewActivation(function); err != nil {
		t.Fatalf("NewActivation returned error: %v", err)
	}
	if network, err = model.NewSequential(activationLayer); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if session, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("NewInferenceSession returned error: %v", err)
	}
	defer session.Close()
	input = mustMatrix(t, 1, 2, []float32{1, 2})
	func() {
		defer func() {
			panicValue = recover()
		}()
		_, _ = session.Predict(input)
	}()
	if panicValue == nil {
		t.Fatal("Predict did not propagate activation panic")
	}
	if output, err = session.Predict(input); err != nil {
		t.Fatalf("Predict after panic returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{1, 2})
}

func Test_InferenceSession_RepeatedPredictionAllocations(t *testing.T) {
	var (
		network     *model.Sequential
		session     *model.InferenceSession
		input       *matrix.Matrix
		allocations float64
		err         error
	)

	network = newInferenceSessionNetwork(t)
	input = mustMatrix(t, 8, 2, []float32{
		1, 2, 2, 1, -1, 0.5, 0.5, -1,
		3, 2, 2, 3, -2, 1, 1, -2,
	})
	if session, err = network.NewInferenceSession(); err != nil {
		t.Fatalf("NewInferenceSession returned error: %v", err)
	}
	defer session.Close()
	if _, err = session.Predict(input); err != nil {
		t.Fatalf("warm-up Predict returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		if _, err = session.Predict(input); err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("Predict allocations = %g, want 0", allocations)
	}
}

func Benchmark_InferenceSession_Create(b *testing.B) {
	var (
		network *model.Sequential
		session *model.InferenceSession
		err     error
	)

	network = newInferenceSessionNetwork(b)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if session, err = network.NewInferenceSession(); err != nil {
			b.Fatalf("NewInferenceSession returned error: %v", err)
		}
		if err = session.Close(); err != nil {
			b.Fatalf("Close returned error: %v", err)
		}
	}
}

func Benchmark_InferenceSession_Predict(b *testing.B) {
	var (
		network *model.Sequential
		session *model.InferenceSession
		input   *matrix.Matrix
		err     error
	)

	network = newInferenceSessionNetwork(b)
	input = mustMatrix(b, 8, 2, []float32{
		1, 2, 2, 1, -1, 0.5, 0.5, -1,
		3, 2, 2, 3, -2, 1, 1, -2,
	})
	if session, err = network.NewInferenceSession(); err != nil {
		b.Fatalf("NewInferenceSession returned error: %v", err)
	}
	defer session.Close()
	if _, err = session.Predict(input); err != nil {
		b.Fatalf("warm-up Predict returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err = session.Predict(input); err != nil {
			b.Fatalf("Predict returned error: %v", err)
		}
	}
}

func newInferenceSessionNetwork(tb testing.TB) (network *model.Sequential) {
	var (
		dense           *layer.Dense
		batchNorm       *layer.BatchNormalization
		dropout         *layer.Dropout
		activationLayer *layer.Activation
		err             error
	)

	tb.Helper()
	if dense, err = layer.NewDense(2, 2, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if err = dense.Weights().Values().CopyFrom(mustMatrix(tb, 2, 2, []float32{1, 2, 3, 4})); err != nil {
		tb.Fatalf("weight CopyFrom returned error: %v", err)
	}
	if err = dense.Biases().Values().CopyFrom(mustMatrix(tb, 1, 2, []float32{0.5, -0.5})); err != nil {
		tb.Fatalf("bias CopyFrom returned error: %v", err)
	}
	if batchNorm, err = layer.NewBatchNormalizationWithConfig(2, 0.9, 1e-5); err != nil {
		tb.Fatalf("NewBatchNormalizationWithConfig returned error: %v", err)
	}
	if err = batchNorm.RunningMean().CopyFrom(mustMatrix(tb, 1, 2, []float32{1, -1})); err != nil {
		tb.Fatalf("running mean CopyFrom returned error: %v", err)
	}
	if err = batchNorm.RunningVariance().CopyFrom(mustMatrix(tb, 1, 2, []float32{4, 9})); err != nil {
		tb.Fatalf("running variance CopyFrom returned error: %v", err)
	}
	if dropout, err = layer.NewDropout(0.5, rand.New(rand.NewSource(11))); err != nil {
		tb.Fatalf("NewDropout returned error: %v", err)
	}
	if activationLayer, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}
	if network, err = model.NewSequential(dense, batchNorm, dropout, activationLayer); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	if err = network.SetTraining(false); err != nil {
		tb.Fatalf("SetTraining returned error: %v", err)
	}

	return network
}

func requireFloat32Values(tb testing.TB, got, want []float32) {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			tb.Fatalf("value[%d] = %g, want %g", index, got[index], want[index])
		}
	}
}

type panicOnceActivation struct {
	panicked bool
}

func (p *panicOnceActivation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if !p.panicked {
		p.panicked = true
		panic("inference activation panic")
	}

	output, err = input.Clone()
	return output, err
}

func (p *panicOnceActivation) Backward(
	input,
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = outputGradient.Clone()
	return inputGradient, err
}
