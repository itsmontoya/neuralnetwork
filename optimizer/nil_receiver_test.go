package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NilReceivers_ReturnZeroValues(t *testing.T) {
	var (
		sgd                 *optimizer.SGD
		momentum            *optimizer.Momentum
		adam                *optimizer.Adam
		l1                  *optimizer.L1
		decay               *optimizer.L2WeightDecay
		regularized         *optimizer.Regularized
		parameter           *optimizer.Parameter
		constant            *optimizer.ConstantLearningRate
		step                *optimizer.StepDecay
		exponential         *optimizer.ExponentialDecay
		regularizerCount    int
		stepSize            int
		learningRate        float32
		coefficient         float32
		beta                float32
		epsilon             float32
		initialLearningRate float32
		factor              float32
		decayRate           float32
	)

	if learningRate = sgd.LearningRate(); learningRate != 0 {
		t.Fatalf("SGD LearningRate = %g, want 0", learningRate)
	}

	if learningRate = momentum.LearningRate(); learningRate != 0 {
		t.Fatalf("Momentum LearningRate = %g, want 0", learningRate)
	}

	if coefficient = momentum.Coefficient(); coefficient != 0 {
		t.Fatalf("Momentum Coefficient = %g, want 0", coefficient)
	}

	if learningRate = adam.LearningRate(); learningRate != 0 {
		t.Fatalf("Adam LearningRate = %g, want 0", learningRate)
	}

	if beta = adam.Beta1(); beta != 0 {
		t.Fatalf("Adam Beta1 = %g, want 0", beta)
	}

	if beta = adam.Beta2(); beta != 0 {
		t.Fatalf("Adam Beta2 = %g, want 0", beta)
	}

	if epsilon = adam.Epsilon(); epsilon != 0 {
		t.Fatalf("Adam Epsilon = %g, want 0", epsilon)
	}

	if coefficient = l1.Coefficient(); coefficient != 0 {
		t.Fatalf("L1 Coefficient = %g, want 0", coefficient)
	}

	if coefficient = decay.Coefficient(); coefficient != 0 {
		t.Fatalf("L2WeightDecay Coefficient = %g, want 0", coefficient)
	}

	if learningRate = regularized.LearningRate(); learningRate != 0 {
		t.Fatalf("Regularized LearningRate = %g, want 0", learningRate)
	}

	if regularized.Base() != nil {
		t.Fatal("Regularized Base returned non-nil optimizer")
	}

	if regularizerCount = len(regularized.Regularizers()); regularizerCount != 0 {
		t.Fatalf("Regularized Regularizers length = %d, want 0", regularizerCount)
	}

	if parameter.Values() != nil {
		t.Fatal("Parameter Values returned non-nil matrix")
	}

	if parameter.Gradient() != nil {
		t.Fatal("Parameter Gradient returned non-nil matrix")
	}

	if learningRate = constant.Rate(); learningRate != 0 {
		t.Fatalf("ConstantLearningRate Rate = %g, want 0", learningRate)
	}

	if initialLearningRate = step.InitialLearningRate(); initialLearningRate != 0 {
		t.Fatalf("StepDecay InitialLearningRate = %g, want 0", initialLearningRate)
	}

	if factor = step.Factor(); factor != 0 {
		t.Fatalf("StepDecay Factor = %g, want 0", factor)
	}

	if stepSize = step.StepSize(); stepSize != 0 {
		t.Fatalf("StepDecay StepSize = %d, want 0", stepSize)
	}

	if initialLearningRate = exponential.InitialLearningRate(); initialLearningRate != 0 {
		t.Fatalf("ExponentialDecay InitialLearningRate = %g, want 0", initialLearningRate)
	}

	if decayRate = exponential.DecayRate(); decayRate != 0 {
		t.Fatalf("ExponentialDecay DecayRate = %g, want 0", decayRate)
	}
}

func Test_NilReceivers_SettersReturnErrors(t *testing.T) {
	type testcase struct {
		name string
		set  func() (err error)
	}

	tests := []testcase{
		{
			name: "sgd learning rate",
			set: func() (err error) {
				var sgd *optimizer.SGD

				err = sgd.SetLearningRate(0.1)
				return err
			},
		},
		{
			name: "momentum learning rate",
			set: func() (err error) {
				var momentum *optimizer.Momentum

				err = momentum.SetLearningRate(0.1)
				return err
			},
		},
		{
			name: "momentum coefficient",
			set: func() (err error) {
				var momentum *optimizer.Momentum

				err = momentum.SetCoefficient(0.5)
				return err
			},
		},
		{
			name: "adam learning rate",
			set: func() (err error) {
				var adam *optimizer.Adam

				err = adam.SetLearningRate(0.1)
				return err
			},
		},
		{
			name: "adam beta1",
			set: func() (err error) {
				var adam *optimizer.Adam

				err = adam.SetBeta1(0.5)
				return err
			},
		},
		{
			name: "adam beta2",
			set: func() (err error) {
				var adam *optimizer.Adam

				err = adam.SetBeta2(0.5)
				return err
			},
		},
		{
			name: "adam epsilon",
			set: func() (err error) {
				var adam *optimizer.Adam

				err = adam.SetEpsilon(1e-8)
				return err
			},
		},
		{
			name: "l1 coefficient",
			set: func() (err error) {
				var l1 *optimizer.L1

				err = l1.SetCoefficient(0.1)
				return err
			},
		},
		{
			name: "l2 coefficient",
			set: func() (err error) {
				var decay *optimizer.L2WeightDecay

				err = decay.SetCoefficient(0.1)
				return err
			},
		},
		{
			name: "regularized learning rate",
			set: func() (err error) {
				var regularized *optimizer.Regularized

				err = regularized.SetLearningRate(0.1)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			err = tt.set()
			if err == nil {
				t.Fatal("setter error = nil, want error")
			}
		})
	}
}

func Test_NilSchedules_LearningRateReturnsErrors(t *testing.T) {
	type testcase struct {
		name         string
		learningRate func() (rate float32, err error)
	}

	tests := []testcase{
		{
			name: "constant",
			learningRate: func() (rate float32, err error) {
				var schedule *optimizer.ConstantLearningRate

				rate, err = schedule.LearningRate(1)
				return rate, err
			},
		},
		{
			name: "step",
			learningRate: func() (rate float32, err error) {
				var schedule *optimizer.StepDecay

				rate, err = schedule.LearningRate(1)
				return rate, err
			},
		},
		{
			name: "exponential",
			learningRate: func() (rate float32, err error) {
				var schedule *optimizer.ExponentialDecay

				rate, err = schedule.LearningRate(1)
				return rate, err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				rate float32
				err  error
			)

			rate, err = tt.learningRate()
			if err == nil {
				t.Fatal("LearningRate error = nil, want error")
			}

			if rate != 0 {
				t.Fatalf("LearningRate = %g, want 0", rate)
			}
		})
	}
}
