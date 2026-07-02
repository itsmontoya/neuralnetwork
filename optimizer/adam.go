package optimizer

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	// DefaultAdamBeta1 is the default exponential decay rate for first moments.
	DefaultAdamBeta1 = 0.9
	// DefaultAdamBeta2 is the default exponential decay rate for second moments.
	DefaultAdamBeta2 = 0.999
	// DefaultAdamEpsilon is the default denominator stabilizer.
	DefaultAdamEpsilon = 1e-8
)

// NewAdam constructs Adam with default beta and epsilon values.
func NewAdam(learningRate float64) (out *Adam, err error) {
	out, err = NewAdamWithConfig(learningRate, DefaultAdamBeta1, DefaultAdamBeta2, DefaultAdamEpsilon)
	return out, err
}

// NewAdamWithConfig constructs Adam with explicit beta and epsilon values.
func NewAdamWithConfig(learningRate, beta1, beta2, epsilon float64) (out *Adam, err error) {
	if err = validateLearningRate(learningRate); err != nil {
		return nil, err
	}

	if err = validateUnitCoefficient("beta1", beta1); err != nil {
		return nil, err
	}

	if err = validateUnitCoefficient("beta2", beta2); err != nil {
		return nil, err
	}

	if err = validatePositiveFinite("epsilon", epsilon); err != nil {
		return nil, err
	}

	var a Adam
	a.learningRate = learningRate
	a.beta1 = beta1
	a.beta2 = beta2
	a.epsilon = epsilon
	a.states = make(map[*Parameter]*adamState)
	return &a, nil
}

// Adam applies adaptive moment estimation updates with bias correction.
//
// First and second moment state is isolated per parameter. Gradients are reset
// after a successful update.
type Adam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	states       map[*Parameter]*adamState
}

// Update applies one Adam update to each parameter.
func (a *Adam) Update(parameters []*Parameter) (err error) {
	var parameter *Parameter

	if err = a.validate(); err != nil {
		return err
	}

	if err = validateParameters(parameters); err != nil {
		return err
	}

	for _, parameter = range parameters {
		if err = a.updateParameter(parameter); err != nil {
			return err
		}
	}

	err = resetGradients(parameters)
	return err
}

// LearningRate returns the current learning rate.
func (a *Adam) LearningRate() (learningRate float64) {
	if a == nil {
		return 0
	}

	learningRate = a.learningRate
	return learningRate
}

// SetLearningRate updates the learning rate.
func (a *Adam) SetLearningRate(learningRate float64) (err error) {
	if a == nil {
		err = nilOptimizerError("adam")
		return err
	}

	if err = validateLearningRate(learningRate); err != nil {
		return err
	}

	a.learningRate = learningRate
	return nil
}

// Beta1 returns the first moment decay rate.
func (a *Adam) Beta1() (beta1 float64) {
	if a == nil {
		return 0
	}

	beta1 = a.beta1
	return beta1
}

// SetBeta1 updates the first moment decay rate.
func (a *Adam) SetBeta1(beta1 float64) (err error) {
	if a == nil {
		err = nilOptimizerError("adam")
		return err
	}

	if err = validateUnitCoefficient("beta1", beta1); err != nil {
		return err
	}

	a.beta1 = beta1
	return nil
}

// Beta2 returns the second moment decay rate.
func (a *Adam) Beta2() (beta2 float64) {
	if a == nil {
		return 0
	}

	beta2 = a.beta2
	return beta2
}

// SetBeta2 updates the second moment decay rate.
func (a *Adam) SetBeta2(beta2 float64) (err error) {
	if a == nil {
		err = nilOptimizerError("adam")
		return err
	}

	if err = validateUnitCoefficient("beta2", beta2); err != nil {
		return err
	}

	a.beta2 = beta2
	return nil
}

// Epsilon returns the denominator stabilizer.
func (a *Adam) Epsilon() (epsilon float64) {
	if a == nil {
		return 0
	}

	epsilon = a.epsilon
	return epsilon
}

// SetEpsilon updates the denominator stabilizer.
func (a *Adam) SetEpsilon(epsilon float64) (err error) {
	if a == nil {
		err = nilOptimizerError("adam")
		return err
	}

	if err = validatePositiveFinite("epsilon", epsilon); err != nil {
		return err
	}

	a.epsilon = epsilon
	return nil
}

func (a *Adam) updateParameter(parameter *Parameter) (err error) {
	var (
		state              *adamState
		rows               int
		cols               int
		values             []float64
		gradients          []float64
		firstRows          int
		firstCols          int
		firstMomentValues  []float64
		secondRows         int
		secondCols         int
		secondMomentValues []float64
		nextStep           int
		firstCorrection    float64
		secondCorrection   float64
		firstEstimate      float64
		secondEstimate     float64
		index              int
	)

	if state, err = a.stateFor(parameter); err != nil {
		return err
	}

	if rows, cols, values, gradients, err = parameterValues(parameter); err != nil {
		return err
	}

	if firstRows, firstCols, firstMomentValues, err = matrixValues(state.firstMoment); err != nil {
		return err
	}

	if secondRows, secondCols, secondMomentValues, err = matrixValues(state.secondMoment); err != nil {
		return err
	}

	if rows != firstRows || cols != firstCols || rows != secondRows || cols != secondCols {
		err = fmt.Errorf(
			"optimizer: adam state shape mismatch: parameter %dx%d, first moment %dx%d, second moment %dx%d",
			rows,
			cols,
			firstRows,
			firstCols,
			secondRows,
			secondCols,
		)
		return err
	}

	nextStep = state.step + 1
	firstCorrection = 1 - math.Pow(a.beta1, float64(nextStep))
	secondCorrection = 1 - math.Pow(a.beta2, float64(nextStep))

	for index = range values {
		firstMomentValues[index] = a.beta1*firstMomentValues[index] + (1-a.beta1)*gradients[index]
		secondMomentValues[index] = a.beta2*secondMomentValues[index] + (1-a.beta2)*gradients[index]*gradients[index]

		firstEstimate = firstMomentValues[index] / firstCorrection
		secondEstimate = secondMomentValues[index] / secondCorrection
		values[index] -= a.learningRate * firstEstimate / (math.Sqrt(secondEstimate) + a.epsilon)
	}

	if err = copyMatrixValues(state.firstMoment, rows, cols, firstMomentValues); err != nil {
		return err
	}

	if err = copyMatrixValues(state.secondMoment, rows, cols, secondMomentValues); err != nil {
		return err
	}

	if err = copyMatrixValues(parameter.Values(), rows, cols, values); err != nil {
		return err
	}

	state.step = nextStep
	return nil
}

func (a *Adam) stateFor(parameter *Parameter) (state *adamState, err error) {
	var (
		rows int
		cols int
		next adamState
	)

	if state = a.states[parameter]; state != nil {
		return state, nil
	}

	rows, cols = parameter.Values().Shape()
	state = &next
	if state.firstMoment, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	if state.secondMoment, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	a.states[parameter] = state
	return state, nil
}

func (a *Adam) validate() (err error) {
	if a == nil {
		err = nilOptimizerError("adam")
		return err
	}

	if err = validateLearningRate(a.learningRate); err != nil {
		return err
	}

	if err = validateUnitCoefficient("beta1", a.beta1); err != nil {
		return err
	}

	if err = validateUnitCoefficient("beta2", a.beta2); err != nil {
		return err
	}

	if err = validatePositiveFinite("epsilon", a.epsilon); err != nil {
		return err
	}

	if a.states == nil {
		a.states = make(map[*Parameter]*adamState)
	}

	return nil
}
