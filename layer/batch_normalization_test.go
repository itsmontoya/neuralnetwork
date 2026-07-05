package layer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_BatchNormalization_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.BatchNormalization)(nil)
}

func Test_NewBatchNormalization_InitializesState(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization
		err       error
	)

	batchNorm, err = layer.NewBatchNormalization(3)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	if batchNorm.FeatureSize() != 3 {
		t.Fatalf("FeatureSize = %d, want 3", batchNorm.FeatureSize())
	}

	if batchNorm.Momentum() != 0.9 {
		t.Fatalf("Momentum = %g, want 0.9", batchNorm.Momentum())
	}

	if batchNorm.Epsilon() != 1e-5 {
		t.Fatalf("Epsilon = %g, want 1e-5", batchNorm.Epsilon())
	}

	if !batchNorm.Training() {
		t.Fatal("Training = false, want true")
	}

	requireMatrixValues(t, batchNorm.Gamma().Values(), []float64{1, 1, 1})
	requireMatrixValues(t, batchNorm.Beta().Values(), []float64{0, 0, 0})
	requireMatrixValues(t, batchNorm.RunningMean(), []float64{0, 0, 0})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float64{1, 1, 1})
}

func Test_BatchNormalization_NilReceiverAccessors(t *testing.T) {
	var batchNorm *layer.BatchNormalization

	if batchNorm.FeatureSize() != 0 {
		t.Fatalf("FeatureSize = %d, want 0", batchNorm.FeatureSize())
	}

	if batchNorm.Momentum() != 0 {
		t.Fatalf("Momentum = %g, want 0", batchNorm.Momentum())
	}

	if batchNorm.Epsilon() != 0 {
		t.Fatalf("Epsilon = %g, want 0", batchNorm.Epsilon())
	}

	if batchNorm.Gamma() != nil {
		t.Fatal("Gamma returned value for nil receiver")
	}

	if batchNorm.Beta() != nil {
		t.Fatal("Beta returned value for nil receiver")
	}

	if batchNorm.RunningMean() != nil {
		t.Fatal("RunningMean returned value for nil receiver")
	}

	if batchNorm.RunningVariance() != nil {
		t.Fatal("RunningVariance returned value for nil receiver")
	}

	if batchNorm.Parameters() != nil {
		t.Fatal("Parameters returned value for nil receiver")
	}

	if batchNorm.Training() {
		t.Fatal("Training returned true for nil receiver")
	}

	batchNorm.SetTraining(true)
	if batchNorm.Training() {
		t.Fatal("Training changed after SetTraining on nil receiver")
	}
}

func Test_NewBatchNormalizationWithConfig_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name        string
		featureSize int
		momentum    float64
		epsilon     float64
	}

	tests := []testcase{
		{
			name:        "feature size",
			featureSize: 0,
			momentum:    0.9,
			epsilon:     1e-5,
		},
		{
			name:        "negative momentum",
			featureSize: 1,
			momentum:    -0.1,
			epsilon:     1e-5,
		},
		{
			name:        "one momentum",
			featureSize: 1,
			momentum:    1,
			epsilon:     1e-5,
		},
		{
			name:        "nan momentum",
			featureSize: 1,
			momentum:    math.NaN(),
			epsilon:     1e-5,
		},
		{
			name:        "zero epsilon",
			featureSize: 1,
			momentum:    0.9,
			epsilon:     0,
		},
		{
			name:        "nan epsilon",
			featureSize: 1,
			momentum:    0.9,
			epsilon:     math.NaN(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				batchNorm *layer.BatchNormalization
				err       error
			)

			batchNorm, err = layer.NewBatchNormalizationWithConfig(tt.featureSize, tt.momentum, tt.epsilon)
			if err == nil {
				t.Fatal("NewBatchNormalizationWithConfig error = nil, want error")
			}

			if batchNorm != nil {
				t.Fatal("NewBatchNormalizationWithConfig returned layer on error")
			}
		})
	}
}

func Test_BatchNormalization_ForwardTrainingUpdatesRunningStatistics(t *testing.T) {
	var (
		batchNorm       *layer.BatchNormalization
		input           *matrix.Matrix
		output          *matrix.Matrix
		runningMean     *matrix.Matrix
		runningVariance *matrix.Matrix
		inverse0        float64
		inverse1        float64
		err             error
	)

	batchNorm, err = layer.NewBatchNormalizationWithConfig(2, 0.8, 1e-5)
	if err != nil {
		t.Fatalf("NewBatchNormalizationWithConfig returned error: %v", err)
	}

	err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{2, 3}))
	if err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}

	err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{0.5, -1}))
	if err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 6,
	})

	runningMean = batchNorm.RunningMean()
	runningVariance = batchNorm.RunningVariance()
	output, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inverse0 = 1 / math.Sqrt(1+batchNorm.Epsilon())
	inverse1 = 1 / math.Sqrt(4+batchNorm.Epsilon())
	requireMatrixValues(t, output, []float64{
		-1*inverse0*2 + 0.5, -2*inverse1*3 - 1,
		inverse0*2 + 0.5, 2*inverse1*3 - 1,
	})
	requireMatrixValues(t, batchNorm.RunningMean(), []float64{0.4, 0.8})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float64{1, 1.6})

	if batchNorm.RunningMean() != runningMean {
		t.Fatal("Forward replaced running mean matrix, want CopyFrom into existing matrix")
	}

	if batchNorm.RunningVariance() != runningVariance {
		t.Fatal("Forward replaced running variance matrix, want CopyFrom into existing matrix")
	}
}

func Test_BatchNormalization_ForwardEvaluationUsesRunningStatistics(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization
		input     *matrix.Matrix
		output    *matrix.Matrix
		inverse0  float64
		inverse1  float64
		err       error
	)

	batchNorm, err = layer.NewBatchNormalizationWithConfig(2, 0.8, 1e-5)
	if err != nil {
		t.Fatalf("NewBatchNormalizationWithConfig returned error: %v", err)
	}

	err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{2, 3}))
	if err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}

	err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{0.5, -1}))
	if err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningMean().CopyFrom(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err != nil {
		t.Fatalf("running mean CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningVariance().CopyFrom(mustMatrix(t, 1, 2, []float64{4, 9}))
	if err != nil {
		t.Fatalf("running variance CopyFrom returned error: %v", err)
	}

	batchNorm.SetTraining(false)
	input = mustMatrix(t, 1, 2, []float64{3, 8})
	output, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inverse0 = 1 / math.Sqrt(4+batchNorm.Epsilon())
	inverse1 = 1 / math.Sqrt(9+batchNorm.Epsilon())
	requireMatrixValues(t, output, []float64{
		(3-1)*inverse0*2 + 0.5,
		(8-2)*inverse1*3 - 1,
	})
	requireMatrixValues(t, batchNorm.RunningMean(), []float64{1, 2})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float64{4, 9})
}

func Test_BatchNormalization_BackwardTraining(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		inverse0       float64
		inverse1       float64
		normalized0    float64
		normalized1    float64
		gammaGradient0 float64
		gammaGradient1 float64
		err            error
	)

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 6,
	})
	_, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	inputGradient, err = batchNorm.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	inverse0 = 1 / math.Sqrt(1+batchNorm.Epsilon())
	inverse1 = 1 / math.Sqrt(4+batchNorm.Epsilon())
	normalized0 = inverse0
	normalized1 = 2 * inverse1
	gammaGradient0 = 2 * normalized0
	gammaGradient1 = 2 * normalized1

	requireMatrixValues(t, batchNorm.Beta().Gradient(), []float64{4, 6})
	requireMatrixValues(t, batchNorm.Gamma().Gradient(), []float64{gammaGradient0, gammaGradient1})
	requireMatrixValues(t, inputGradient, []float64{
		inverse0 / 2 * (-2 + normalized0*gammaGradient0),
		inverse1 / 2 * (-2 + normalized1*gammaGradient1),
		inverse0 / 2 * (2 - normalized0*gammaGradient0),
		inverse1 / 2 * (2 - normalized1*gammaGradient1),
	})
}

func Test_BatchNormalization_BackwardEvaluation(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		inverse0       float64
		inverse1       float64
		err            error
	)

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{2, 3}))
	if err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningMean().CopyFrom(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err != nil {
		t.Fatalf("running mean CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningVariance().CopyFrom(mustMatrix(t, 1, 2, []float64{4, 9}))
	if err != nil {
		t.Fatalf("running variance CopyFrom returned error: %v", err)
	}

	batchNorm.SetTraining(false)
	input = mustMatrix(t, 2, 2, []float64{
		3, 8,
		5, -1,
	})
	_, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	inputGradient, err = batchNorm.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	inverse0 = 1 / math.Sqrt(4+batchNorm.Epsilon())
	inverse1 = 1 / math.Sqrt(9+batchNorm.Epsilon())
	requireMatrixValues(t, batchNorm.Beta().Gradient(), []float64{4, 6})
	requireMatrixValues(t, batchNorm.Gamma().Gradient(), []float64{14 * inverse0, 0})
	requireMatrixValues(t, inputGradient, []float64{
		2 * inverse0,
		6 * inverse1,
		6 * inverse0,
		12 * inverse1,
	})
}

func Test_BatchNormalization_ParametersAndResetGradients(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization
		parameters     []*optimizer.Parameter
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	parameters = batchNorm.Parameters()
	if len(parameters) != 2 {
		t.Fatalf("Parameters length = %d, want 2", len(parameters))
	}

	if parameters[0] != batchNorm.Gamma() {
		t.Fatal("Parameters[0] did not match gamma")
	}

	if parameters[1] != batchNorm.Beta() {
		t.Fatal("Parameters[1] did not match beta")
	}

	input = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 6,
	})
	_, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	})
	_, err = batchNorm.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	err = batchNorm.ResetGradients()
	if err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMatrixValues(t, batchNorm.Gamma().Gradient(), []float64{0, 0})
	requireMatrixValues(t, batchNorm.Beta().Gradient(), []float64{0, 0})
}

func Test_BatchNormalization_BackwardRequiresForward(t *testing.T) {
	var (
		batchNorm     *layer.BatchNormalization
		inputGradient *matrix.Matrix
		err           error
	)

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	inputGradient, err = batchNorm.Backward(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err == nil {
		t.Fatalf("Backward returned gradient %v and nil error, want error", inputGradient)
	}
}

func Test_BatchNormalization_BackwardReportsShapeMismatch(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization
		input     *matrix.Matrix
		err       error
	)

	batchNorm, err = layer.NewBatchNormalization(2)
	if err != nil {
		t.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float64{
		1, 2,
		3, 6,
	})
	_, err = batchNorm.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = batchNorm.Backward(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err == nil {
		t.Fatal("Backward error = nil, want shape error")
	}
}
