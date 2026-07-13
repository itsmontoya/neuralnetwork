package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NewAdam_UsesDefaults(t *testing.T) {
	var (
		adam *optimizer.Adam
		err  error
	)

	adam, err = optimizer.NewAdam(0.001)
	if err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, adam.LearningRate(), 0.001, epsilon)
	testutil.RequireAlmostEqual(t, adam.Beta1(), optimizer.DefaultAdamBeta1, epsilon)
	testutil.RequireAlmostEqual(t, adam.Beta2(), optimizer.DefaultAdamBeta2, epsilon)
	testutil.RequireAlmostEqual(t, adam.Epsilon(), optimizer.DefaultAdamEpsilon, epsilon)
}

func Test_NewAdamWithConfig_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name         string
		learningRate float32
		beta1        float32
		beta2        float32
		epsilon      float32
	}

	tests := []testcase{
		{
			name:         "learning rate",
			learningRate: 0,
			beta1:        0.9,
			beta2:        0.999,
			epsilon:      1e-8,
		},
		{
			name:         "beta1",
			learningRate: 0.1,
			beta1:        1,
			beta2:        0.999,
			epsilon:      1e-8,
		},
		{
			name:         "beta2",
			learningRate: 0.1,
			beta1:        0.9,
			beta2:        float32(math.NaN()),
			epsilon:      1e-8,
		},
		{
			name:         "epsilon",
			learningRate: 0.1,
			beta1:        0.9,
			beta2:        0.999,
			epsilon:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				adam *optimizer.Adam
				err  error
			)

			adam, err = optimizer.NewAdamWithConfig(tt.learningRate, tt.beta1, tt.beta2, tt.epsilon)
			if err == nil {
				t.Fatal("NewAdamWithConfig error = nil, want error")
			}

			if adam != nil {
				t.Fatal("NewAdamWithConfig returned optimizer on error")
			}
		})
	}
}

func Test_Adam_Update_Repeated(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		adam      *optimizer.Adam
		firstWant float32
		want      float32
		err       error
	)

	parameter = mustParameter(t, 1, 1, []float32{1})
	adam, err = optimizer.NewAdamWithConfig(0.1, 0.5, 0.25, 0.1)
	if err != nil {
		t.Fatalf("NewAdamWithConfig returned error: %v", err)
	}

	accumulateGradient(t, parameter, []float32{2})
	err = adam.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	firstWant = 1 - adamStep(0.1, 0.1, 2, 4)
	requireMatrixValues(t, parameter.Values(), []float32{firstWant})
	requireMatrixValues(t, parameter.Gradient(), []float32{0})

	accumulateGradient(t, parameter, []float32{4})
	err = adam.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	want = firstWant - adamStep(0.1, 0.1, 2.5/(1-f32.Pow(0.5, 2)), 12.75/(1-f32.Pow(0.25, 2)))
	requireMatrixValues(t, parameter.Values(), []float32{want})
	requireMatrixValues(t, parameter.Gradient(), []float32{0})
}

func Test_Adam_Update_MultiStepMatrixMatchesReference(t *testing.T) {
	var (
		parameter     *optimizer.Parameter
		adam          *optimizer.Adam
		values        []float32
		firstMoments  []float32
		secondMoments []float32
		gradientSteps [][]float32
		gradients     []float32
		err           error
		step          int
	)

	values = []float32{1, -2, 0.5, 3}
	firstMoments = make([]float32, len(values))
	secondMoments = make([]float32, len(values))
	gradientSteps = [][]float32{
		{0.25, -0.5, 0.75, -1},
		{0.5, 0.25, -0.25, 0.125},
		{-0.75, 1, 0.5, -0.375},
	}

	parameter = mustParameter(t, 2, 2, values)
	adam, err = optimizer.NewAdamWithConfig(0.05, 0.8, 0.9, 1e-6)
	if err != nil {
		t.Fatalf("NewAdamWithConfig returned error: %v", err)
	}

	for step, gradients = range gradientSteps {
		accumulateGradient(t, parameter, gradients)
		err = adam.Update([]*optimizer.Parameter{parameter})
		if err != nil {
			t.Fatalf("Update returned error: %v", err)
		}

		applyAdamReferenceStep(values, firstMoments, secondMoments, gradients, step+1, 0.05, 0.8, 0.9, 1e-6)
		requireMatrixValues(t, parameter.Values(), values)
		requireMatrixValues(t, parameter.Gradient(), []float32{0, 0, 0, 0})
	}
}

func Test_Adam_StateIsolation(t *testing.T) {
	var (
		first      *optimizer.Parameter
		second     *optimizer.Parameter
		adam       *optimizer.Adam
		firstWant  float32
		secondWant float32
		err        error
	)

	first = mustParameter(t, 1, 1, []float32{1})
	second = mustParameter(t, 1, 1, []float32{1})
	adam, err = optimizer.NewAdamWithConfig(0.1, 0.5, 0.25, 0.1)
	if err != nil {
		t.Fatalf("NewAdamWithConfig returned error: %v", err)
	}

	accumulateGradient(t, first, []float32{2})
	err = adam.Update([]*optimizer.Parameter{first})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	accumulateGradient(t, first, []float32{4})
	accumulateGradient(t, second, []float32{4})
	err = adam.Update([]*optimizer.Parameter{first, second})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	firstWant = 1 -
		adamStep(0.1, 0.1, 2, 4) -
		adamStep(0.1, 0.1, 2.5/(1-f32.Pow(0.5, 2)), 12.75/(1-f32.Pow(0.25, 2)))
	secondWant = 1 - adamStep(0.1, 0.1, 4, 16)

	requireMatrixValues(t, first.Values(), []float32{firstWant})
	requireMatrixValues(t, second.Values(), []float32{secondWant})
}

func Test_Adam_Setters(t *testing.T) {
	var (
		adam *optimizer.Adam
		err  error
	)

	adam, err = optimizer.NewAdam(0.001)
	if err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	if err = adam.SetLearningRate(0.01); err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}

	if err = adam.SetBeta1(0.8); err != nil {
		t.Fatalf("SetBeta1 returned error: %v", err)
	}

	if err = adam.SetBeta2(0.95); err != nil {
		t.Fatalf("SetBeta2 returned error: %v", err)
	}

	if err = adam.SetEpsilon(1e-6); err != nil {
		t.Fatalf("SetEpsilon returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, adam.LearningRate(), 0.01, epsilon)
	testutil.RequireAlmostEqual(t, adam.Beta1(), 0.8, epsilon)
	testutil.RequireAlmostEqual(t, adam.Beta2(), 0.95, epsilon)
	testutil.RequireAlmostEqual(t, adam.Epsilon(), 1e-6, epsilon)
}

func adamStep(learningRate, epsilon, firstEstimate, secondEstimate float32) (step float32) {
	step = learningRate * firstEstimate / (f32.Sqrt(secondEstimate) + epsilon)
	return step
}

func applyAdamReferenceStep(
	values []float32,
	firstMoments []float32,
	secondMoments []float32,
	gradients []float32,
	step int,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
) {
	var (
		firstCorrection  float32
		secondCorrection float32
		firstEstimate    float32
		secondEstimate   float32
		gradient         float32
		index            int
	)

	firstCorrection = 1 - f32.Pow(beta1, float32(step))
	secondCorrection = 1 - f32.Pow(beta2, float32(step))
	for index = range values {
		gradient = gradients[index]
		firstMoments[index] = beta1*firstMoments[index] + (1-beta1)*gradient
		secondMoments[index] = beta2*secondMoments[index] + (1-beta2)*gradient*gradient
		firstEstimate = firstMoments[index] / firstCorrection
		secondEstimate = secondMoments[index] / secondCorrection
		values[index] -= learningRate * firstEstimate / (f32.Sqrt(secondEstimate) + epsilon)
	}
}
