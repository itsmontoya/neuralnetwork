package optimizer

import (
	"fmt"

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
func NewAdam(learningRate float32) (out *Adam, err error) {
	out, err = NewAdamWithConfig(learningRate, DefaultAdamBeta1, DefaultAdamBeta2, DefaultAdamEpsilon)
	return out, err
}

// NewAdamWithConfig constructs Adam with explicit beta and epsilon values.
func NewAdamWithConfig(learningRate, beta1, beta2, epsilon float32) (out *Adam, err error) {
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
	learningRate float32
	beta1        float32
	beta2        float32
	epsilon      float32
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
func (a *Adam) LearningRate() (learningRate float32) {
	if a == nil {
		return 0
	}

	learningRate = a.learningRate
	return learningRate
}

// SetLearningRate updates the learning rate.
func (a *Adam) SetLearningRate(learningRate float32) (err error) {
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
func (a *Adam) Beta1() (beta1 float32) {
	if a == nil {
		return 0
	}

	beta1 = a.beta1
	return beta1
}

// SetBeta1 updates the first moment decay rate.
func (a *Adam) SetBeta1(beta1 float32) (err error) {
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
func (a *Adam) Beta2() (beta2 float32) {
	if a == nil {
		return 0
	}

	beta2 = a.beta2
	return beta2
}

// SetBeta2 updates the second moment decay rate.
func (a *Adam) SetBeta2(beta2 float32) (err error) {
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
func (a *Adam) Epsilon() (epsilon float32) {
	if a == nil {
		return 0
	}

	epsilon = a.epsilon
	return epsilon
}

// SetEpsilon updates the denominator stabilizer.
func (a *Adam) SetEpsilon(epsilon float32) (err error) {
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
		state            *adamState
		values           *matrix.Matrix
		gradients        *matrix.Matrix
		rows             int
		cols             int
		firstRows        int
		firstCols        int
		secondRows       int
		secondCols       int
		nextStep         int
		firstCorrection  float32
		secondCorrection float32
	)

	if err = parameter.validate(); err != nil {
		return err
	}

	if state, err = a.stateFor(parameter); err != nil {
		return err
	}

	values = parameter.Values()
	gradients = parameter.Gradient()
	rows, cols = values.Shape()
	firstRows, firstCols = state.firstMoment.Shape()
	secondRows, secondCols = state.secondMoment.Shape()
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
	firstCorrection = 1 - powUnitCoefficient(a.beta1, nextStep)
	secondCorrection = 1 - powUnitCoefficient(a.beta2, nextStep)

	err = values.AdamUpdateInPlace(
		gradients,
		state.firstMoment,
		state.secondMoment,
		a.learningRate,
		a.beta1,
		a.beta2,
		a.epsilon,
		firstCorrection,
		secondCorrection,
	)
	if err != nil {
		return err
	}

	state.step = nextStep
	return nil
}

func (a *Adam) stateFor(parameter *Parameter) (state *adamState, err error) {
	var (
		rows int
		cols int
	)

	if state = a.states[parameter]; state != nil {
		return state, nil
	}

	var next adamState
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

func powUnitCoefficient(base float32, exponent int) (result float32) {
	result = 1
	for exponent > 0 {
		if exponent%2 == 1 {
			result *= base
		}

		base *= base
		exponent /= 2
	}

	return result
}
