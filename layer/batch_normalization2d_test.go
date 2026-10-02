package layer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_BatchNormalization2D_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.BatchNormalization2D)(nil)
}

func Test_NewBatchNormalization2D_InitializesStateAndShapes(t *testing.T) {
	var (
		shape     layer.SpatialShape
		batchNorm *layer.BatchNormalization2D
		config    layer.BatchNormalization2DConfig
		err       error
	)

	if shape, err = layer.NewSpatialShape(3, 2, 4); err != nil {
		t.Fatalf("NewSpatialShape returned error: %v", err)
	}
	config = layer.BatchNormalization2DConfig{InputShape: shape, Momentum: 0.8, Epsilon: 1e-4}
	if batchNorm, err = layer.NewBatchNormalization2D(config); err != nil {
		t.Fatalf("NewBatchNormalization2D returned error: %v", err)
	}

	if batchNorm.Config() != config {
		t.Fatalf("Config = %+v, want %+v", batchNorm.Config(), config)
	}
	if batchNorm.InputShape() != shape || batchNorm.OutputShape() != shape {
		t.Fatalf("shapes = %+v/%+v, want %+v", batchNorm.InputShape(), batchNorm.OutputShape(), shape)
	}
	if !batchNorm.Training() {
		t.Fatal("Training = false, want true")
	}
	requireMatrixValues(t, batchNorm.Gamma().Values(), []float32{1, 1, 1})
	requireMatrixValues(t, batchNorm.Beta().Values(), []float32{0, 0, 0})
	requireMatrixValues(t, batchNorm.RunningMean(), []float32{0, 0, 0})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float32{1, 1, 1})
}

func Test_NewBatchNormalization2D_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name     string
		shape    layer.SpatialShape
		momentum float32
		epsilon  float32
	}

	var (
		shape     layer.SpatialShape
		tests     []testcase
		tt        testcase
		batchNorm *layer.BatchNormalization2D
		err       error
	)

	if shape, err = layer.NewSpatialShape(1, 2, 2); err != nil {
		t.Fatalf("NewSpatialShape returned error: %v", err)
	}
	tests = []testcase{
		{name: "shape", momentum: 0.9, epsilon: 1e-5},
		{name: "negative momentum", shape: shape, momentum: -0.1, epsilon: 1e-5},
		{name: "one momentum", shape: shape, momentum: 1, epsilon: 1e-5},
		{name: "nan momentum", shape: shape, momentum: float32(math.NaN()), epsilon: 1e-5},
		{name: "zero epsilon", shape: shape, momentum: 0.9},
		{name: "nan epsilon", shape: shape, momentum: 0.9, epsilon: float32(math.NaN())},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			batchNorm, err = layer.NewBatchNormalization2D(layer.BatchNormalization2DConfig{
				InputShape: tt.shape,
				Momentum:   tt.momentum,
				Epsilon:    tt.epsilon,
			})
			if err == nil {
				t.Fatal("NewBatchNormalization2D error = nil, want error")
			}
			if batchNorm != nil {
				t.Fatal("NewBatchNormalization2D returned layer on error")
			}
		})
	}
}

func Test_BatchNormalization2D_ForwardAggregatesBatchAndSpatialPositions(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization2D
		input     *matrix.Matrix
		output    *matrix.Matrix
		inverse0  float32
		inverse1  float32
		err       error
	)

	batchNorm = mustBatchNormalization2D(t, 2, 1, 2, 0.8, 1e-5)
	if err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{2, 0.5})); err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}
	if err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{1, -1})); err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}
	input = mustMatrix(t, 2, 4, []float32{
		1, 3, 10, 14,
		5, 7, 18, 22,
	})
	if output, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inverse0 = 1 / f32.Sqrt(5+batchNorm.Epsilon())
	inverse1 = 1 / f32.Sqrt(20+batchNorm.Epsilon())
	requireMatrixValues(t, output, []float32{
		(1-4)*inverse0*2 + 1,
		(3-4)*inverse0*2 + 1,
		(10-16)*inverse1*0.5 - 1,
		(14-16)*inverse1*0.5 - 1,
		(5-4)*inverse0*2 + 1,
		(7-4)*inverse0*2 + 1,
		(18-16)*inverse1*0.5 - 1,
		(22-16)*inverse1*0.5 - 1,
	})
	requireMatrixValues(t, batchNorm.RunningMean(), []float32{0.8, 3.2})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float32{1.8, 4.8})
}

func Test_BatchNormalization2D_ForwardEvaluationUsesRunningStatistics(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization2D
		input     *matrix.Matrix
		output    *matrix.Matrix
		inverse0  float32
		inverse1  float32
		err       error
	)

	batchNorm = mustBatchNormalization2D(t, 2, 1, 2, 0.8, 1e-5)
	if err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{2, 3})); err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}
	if err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{0.5, -1})); err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}
	if err = batchNorm.RunningMean().CopyFrom(mustMatrix(t, 1, 2, []float32{1, 2})); err != nil {
		t.Fatalf("running mean CopyFrom returned error: %v", err)
	}
	if err = batchNorm.RunningVariance().CopyFrom(mustMatrix(t, 1, 2, []float32{4, 9})); err != nil {
		t.Fatalf("running variance CopyFrom returned error: %v", err)
	}

	batchNorm.SetTraining(false)
	input = mustMatrix(t, 1, 4, []float32{3, 5, 8, -1})
	if output, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	inverse0 = 1 / f32.Sqrt(4+batchNorm.Epsilon())
	inverse1 = 1 / f32.Sqrt(9+batchNorm.Epsilon())
	requireMatrixValues(t, output, []float32{
		(3-1)*inverse0*2 + 0.5,
		(5-1)*inverse0*2 + 0.5,
		(8-2)*inverse1*3 - 1,
		(-1-2)*inverse1*3 - 1,
	})
	requireMatrixValues(t, batchNorm.RunningMean(), []float32{1, 2})
	requireMatrixValues(t, batchNorm.RunningVariance(), []float32{4, 9})
}

func Test_BatchNormalization2D_BackwardTraining(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		inverse        float32
		normalized     float32
		gammaGradient  float32
		err            error
	)

	batchNorm = mustBatchNormalization2D(t, 1, 1, 2, 0.9, 1e-5)
	input = mustMatrix(t, 1, 2, []float32{1, 3})
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	outputGradient = mustMatrix(t, 1, 2, []float32{1, 3})
	if inputGradient, err = batchNorm.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	inverse = 1 / f32.Sqrt(1+batchNorm.Epsilon())
	normalized = inverse
	gammaGradient = 2 * normalized
	requireMatrixValues(t, batchNorm.Beta().Gradient(), []float32{4})
	requireMatrixValues(t, batchNorm.Gamma().Gradient(), []float32{gammaGradient})
	requireMatrixValues(t, inputGradient, []float32{
		inverse / 2 * (-2 + normalized*gammaGradient),
		inverse / 2 * (2 - normalized*gammaGradient),
	})
}

func Test_BatchNormalization2D_BatchOneSinglePosition(t *testing.T) {
	var (
		batchNorm      *layer.BatchNormalization2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		output         *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	batchNorm = mustBatchNormalization2D(t, 1, 1, 1, 0.9, 1e-5)
	input = mustMatrix(t, 1, 1, []float32{7})
	if output, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{0})
	outputGradient = mustMatrix(t, 1, 1, []float32{3})
	if inputGradient, err = batchNorm.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{0})
}

func Test_BatchNormalization2D_ParametersAndResetGradients(t *testing.T) {
	var (
		batchNorm  *layer.BatchNormalization2D
		parameters []*optimizer.Parameter
		input      *matrix.Matrix
		gradient   *matrix.Matrix
		err        error
	)

	batchNorm = mustBatchNormalization2D(t, 2, 1, 2, 0.9, 1e-5)
	parameters = batchNorm.Parameters()
	if len(parameters) != 2 || parameters[0] != batchNorm.Gamma() || parameters[1] != batchNorm.Beta() {
		t.Fatal("Parameters did not return gamma and beta in order")
	}
	parameters = batchNorm.AppendParameters(make([]*optimizer.Parameter, 0, 2))
	if len(parameters) != 2 || parameters[0] != batchNorm.Gamma() || parameters[1] != batchNorm.Beta() {
		t.Fatal("AppendParameters did not append gamma and beta in order")
	}

	input = mustMatrix(t, 1, 4, []float32{1, 3, 2, 4})
	gradient = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = batchNorm.Backward(gradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if err = batchNorm.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}
	requireMatrixValues(t, batchNorm.Gamma().Gradient(), []float32{0, 0})
	requireMatrixValues(t, batchNorm.Beta().Gradient(), []float32{0, 0})
}

func Test_BatchNormalization2D_ValidatesLifecycleAndShapes(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization2D
		input     *matrix.Matrix
		err       error
	)

	batchNorm = mustBatchNormalization2D(t, 2, 1, 2, 0.9, 1e-5)
	if _, err = batchNorm.Backward(mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})); err == nil {
		t.Fatal("Backward error = nil before Forward")
	}
	if _, err = batchNorm.Forward(nil); err == nil {
		t.Fatal("Forward error = nil for nil input")
	}
	if _, err = batchNorm.Forward(mustMatrix(t, 1, 3, []float32{1, 2, 3})); err == nil {
		t.Fatal("Forward error = nil for wrong column count")
	}
	input = mustMatrix(t, 2, 4, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	if _, err = batchNorm.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if _, err = batchNorm.Backward(nil); err == nil {
		t.Fatal("Backward error = nil for nil gradient")
	}
	if _, err = batchNorm.Backward(mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})); err == nil {
		t.Fatal("Backward error = nil for wrong row count")
	}
}

func Test_BatchNormalization2D_NilReceiverAccessors(t *testing.T) {
	var batchNorm *layer.BatchNormalization2D
	var parameters []*optimizer.Parameter

	if batchNorm.Config() != (layer.BatchNormalization2DConfig{}) {
		t.Fatal("Config returned value for nil receiver")
	}
	if batchNorm.InputShape() != (layer.SpatialShape{}) || batchNorm.OutputShape() != (layer.SpatialShape{}) {
		t.Fatal("shape accessor returned value for nil receiver")
	}
	if batchNorm.Momentum() != 0 || batchNorm.Epsilon() != 0 {
		t.Fatal("numeric accessor returned value for nil receiver")
	}
	if batchNorm.Gamma() != nil || batchNorm.Beta() != nil {
		t.Fatal("parameter accessor returned value for nil receiver")
	}
	if batchNorm.RunningMean() != nil || batchNorm.RunningVariance() != nil {
		t.Fatal("running-statistic accessor returned value for nil receiver")
	}
	if batchNorm.Parameters() != nil || batchNorm.Training() {
		t.Fatal("state accessor returned value for nil receiver")
	}
	parameters = []*optimizer.Parameter{nil}
	if len(batchNorm.AppendParameters(parameters)) != 1 {
		t.Fatal("AppendParameters changed prefix for nil receiver")
	}
	batchNorm.SetTraining(true)
}

func mustBatchNormalization2D(
	tb testing.TB,
	channels,
	height,
	width int,
	momentum,
	epsilon float32,
) (batchNorm *layer.BatchNormalization2D) {
	var (
		shape  layer.SpatialShape
		config layer.BatchNormalization2DConfig
		err    error
	)

	tb.Helper()
	if shape, err = layer.NewSpatialShape(channels, height, width); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	config = layer.BatchNormalization2DConfig{InputShape: shape, Momentum: momentum, Epsilon: epsilon}
	if batchNorm, err = layer.NewBatchNormalization2D(config); err != nil {
		tb.Fatalf("NewBatchNormalization2D returned error: %v", err)
	}

	return batchNorm
}
