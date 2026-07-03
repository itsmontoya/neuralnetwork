package layer

import (
	"errors"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	defaultBatchNormalizationEpsilon  = 1e-5
	defaultBatchNormalizationMomentum = 0.9
)

// NewBatchNormalization constructs a per-feature batch normalization layer.
func NewBatchNormalization(featureSize int) (out *BatchNormalization, err error) {
	out, err = NewBatchNormalizationWithConfig(
		featureSize,
		defaultBatchNormalizationMomentum,
		defaultBatchNormalizationEpsilon,
	)
	return out, err
}

// NewBatchNormalizationWithConfig constructs a batch normalization layer with
// explicit running-statistic momentum and numerical epsilon values.
func NewBatchNormalizationWithConfig(featureSize int, momentum, epsilon float64) (out *BatchNormalization, err error) {
	var (
		gammaMatrix     *matrix.Matrix
		betaMatrix      *matrix.Matrix
		runningMean     *matrix.Matrix
		runningVariance *matrix.Matrix
		gamma           *optimizer.Parameter
		beta            *optimizer.Parameter
		b               BatchNormalization
	)

	if featureSize <= 0 {
		err = fmt.Errorf("layer: batch normalization feature size must be positive: featureSize=%d", featureSize)
		return nil, err
	}

	if err = validateBatchNormalizationMomentum(momentum); err != nil {
		return nil, err
	}

	if err = validateBatchNormalizationEpsilon(epsilon); err != nil {
		return nil, err
	}

	if gammaMatrix, err = matrix.New(1, featureSize); err != nil {
		return nil, err
	}

	if err = gammaMatrix.Fill(1); err != nil {
		return nil, err
	}

	if betaMatrix, err = matrix.New(1, featureSize); err != nil {
		return nil, err
	}

	if runningMean, err = matrix.New(1, featureSize); err != nil {
		return nil, err
	}

	if runningVariance, err = matrix.New(1, featureSize); err != nil {
		return nil, err
	}

	if err = runningVariance.Fill(1); err != nil {
		return nil, err
	}

	if gamma, err = optimizer.NewParameter(gammaMatrix); err != nil {
		return nil, err
	}

	if beta, err = optimizer.NewParameter(betaMatrix); err != nil {
		return nil, err
	}

	b.featureSize = featureSize
	b.momentum = momentum
	b.epsilon = epsilon
	b.gamma = gamma
	b.beta = beta
	b.runningMean = runningMean
	b.runningVariance = runningVariance
	b.training = true
	return &b, nil
}

// BatchNormalization normalizes each input feature across a batch.
//
// During training, Forward uses the current batch mean and variance and updates
// running statistics. During evaluation, Forward uses the stored running
// statistics. Gamma and beta are trainable per-feature scale and offset
// parameters.
type BatchNormalization struct {
	featureSize     int
	momentum        float64
	epsilon         float64
	gamma           *optimizer.Parameter
	beta            *optimizer.Parameter
	runningMean     *matrix.Matrix
	runningVariance *matrix.Matrix
	training        bool
	normalizedCache *matrix.Matrix
	inverseStdCache []float64
	forwardRows     int
	forwardCols     int
	forwardCalled   bool
	forwardTraining bool
}

// Forward normalizes input features and applies trainable scale and offset.
func (b *BatchNormalization) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows           int
		cols           int
		inputValues    []float64
		gammaValues    []float64
		betaValues     []float64
		meanValues     []float64
		varianceValues []float64
		inverseStd     []float64
		normalized     []float64
		outputValues   []float64
		index          int
		col            int
	)

	if err = b.validate(); err != nil {
		return nil, err
	}

	if rows, cols, inputValues, err = b.inputValues(input); err != nil {
		return nil, err
	}

	if gammaValues, err = b.gamma.Values().Values(); err != nil {
		return nil, err
	}

	if betaValues, err = b.beta.Values().Values(); err != nil {
		return nil, err
	}

	if b.training {
		meanValues = batchNormalizationMeans(rows, cols, inputValues)
		varianceValues = batchNormalizationVariances(rows, cols, inputValues, meanValues)
		if err = b.updateRunningStatistics(meanValues, varianceValues); err != nil {
			return nil, err
		}
	} else {
		if meanValues, err = b.runningMean.Values(); err != nil {
			return nil, err
		}

		if varianceValues, err = b.runningVariance.Values(); err != nil {
			return nil, err
		}
	}

	inverseStd = batchNormalizationInverseStd(varianceValues, b.epsilon)
	normalized = make([]float64, len(inputValues))
	outputValues = make([]float64, len(inputValues))
	for index = range inputValues {
		col = index % cols
		normalized[index] = (inputValues[index] - meanValues[col]) * inverseStd[col]
		outputValues[index] = normalized[index]*gammaValues[col] + betaValues[col]
	}

	if output, err = matrix.FromSlice(rows, cols, outputValues); err != nil {
		return nil, err
	}

	if b.normalizedCache, err = matrix.FromSlice(rows, cols, normalized); err != nil {
		return nil, err
	}

	b.inverseStdCache = make([]float64, len(inverseStd))
	copy(b.inverseStdCache, inverseStd)
	b.forwardRows = rows
	b.forwardCols = cols
	b.forwardCalled = true
	b.forwardTraining = b.training
	return output, nil
}

// Backward accumulates gamma and beta gradients and returns input gradients.
func (b *BatchNormalization) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		gradientValues   []float64
		normalizedValues []float64
		gammaValues      []float64
		gammaGradient    []float64
		betaGradient     []float64
		inputGradientRaw []float64
		rows             int
		cols             int
		index            int
		col              int
		gammaGradientM   *matrix.Matrix
		betaGradientM    *matrix.Matrix
	)

	if err = b.validate(); err != nil {
		return nil, err
	}

	if !b.forwardCalled {
		err = errors.New("layer: batch normalization backward called before forward")
		return nil, err
	}

	if rows, cols, gradientValues, err = b.outputGradientValues(outputGradient); err != nil {
		return nil, err
	}

	if normalizedValues, err = b.normalizedCache.Values(); err != nil {
		return nil, err
	}

	if gammaValues, err = b.gamma.Values().Values(); err != nil {
		return nil, err
	}

	gammaGradient = make([]float64, cols)
	betaGradient = make([]float64, cols)
	for index = range gradientValues {
		col = index % cols
		betaGradient[col] += gradientValues[index]
		gammaGradient[col] += gradientValues[index] * normalizedValues[index]
	}

	if gammaGradientM, err = matrix.FromSlice(1, cols, gammaGradient); err != nil {
		return nil, err
	}

	if betaGradientM, err = matrix.FromSlice(1, cols, betaGradient); err != nil {
		return nil, err
	}

	if err = b.gamma.AccumulateGradient(gammaGradientM); err != nil {
		return nil, err
	}

	if err = b.beta.AccumulateGradient(betaGradientM); err != nil {
		return nil, err
	}

	if b.forwardTraining {
		inputGradientRaw = b.trainingInputGradient(rows, cols, gradientValues, normalizedValues, gammaValues, betaGradient, gammaGradient)
	} else {
		inputGradientRaw = b.evaluationInputGradient(cols, gradientValues, gammaValues)
	}

	inputGradient, err = matrix.FromSlice(rows, cols, inputGradientRaw)
	return inputGradient, err
}

// FeatureSize returns the number of features normalized by the layer.
func (b *BatchNormalization) FeatureSize() (featureSize int) {
	if b == nil {
		return 0
	}

	featureSize = b.featureSize
	return featureSize
}

// Momentum returns the running-statistic update momentum.
func (b *BatchNormalization) Momentum() (momentum float64) {
	if b == nil {
		return 0
	}

	momentum = b.momentum
	return momentum
}

// Epsilon returns the numerical stability value added to variances.
func (b *BatchNormalization) Epsilon() (epsilon float64) {
	if b == nil {
		return 0
	}

	epsilon = b.epsilon
	return epsilon
}

// Gamma returns the trainable per-feature scale parameter.
func (b *BatchNormalization) Gamma() (gamma *optimizer.Parameter) {
	if b == nil {
		return nil
	}

	gamma = b.gamma
	return gamma
}

// Beta returns the trainable per-feature offset parameter.
func (b *BatchNormalization) Beta() (beta *optimizer.Parameter) {
	if b == nil {
		return nil
	}

	beta = b.beta
	return beta
}

// RunningMean returns the mutable per-feature running mean.
func (b *BatchNormalization) RunningMean() (runningMean *matrix.Matrix) {
	if b == nil {
		return nil
	}

	runningMean = b.runningMean
	return runningMean
}

// RunningVariance returns the mutable per-feature running variance.
func (b *BatchNormalization) RunningVariance() (runningVariance *matrix.Matrix) {
	if b == nil {
		return nil
	}

	runningVariance = b.runningVariance
	return runningVariance
}

// Parameters returns trainable gamma and beta parameters in that order.
func (b *BatchNormalization) Parameters() (parameters []*optimizer.Parameter) {
	if b == nil {
		return nil
	}

	parameters = []*optimizer.Parameter{b.gamma, b.beta}
	return parameters
}

// ResetGradients clears accumulated gamma and beta gradients.
func (b *BatchNormalization) ResetGradients() (err error) {
	if err = b.validate(); err != nil {
		return err
	}

	if err = b.gamma.ResetGradient(); err != nil {
		return err
	}

	err = b.beta.ResetGradient()
	return err
}

// SetTraining updates whether Forward uses batch or running statistics.
func (b *BatchNormalization) SetTraining(training bool) {
	if b == nil {
		return
	}

	b.training = training
}

// Training reports whether Forward uses batch statistics.
func (b *BatchNormalization) Training() (training bool) {
	if b == nil {
		return false
	}

	training = b.training
	return training
}

func (b *BatchNormalization) inputValues(input *matrix.Matrix) (rows, cols int, values []float64, err error) {
	if input == nil {
		err = errors.New("layer: batch normalization input is nil")
		return 0, 0, nil, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization input invalid: %w", err)
		return 0, 0, nil, err
	}

	rows, cols = input.Shape()
	if cols != b.featureSize {
		err = fmt.Errorf("layer: batch normalization input shape mismatch: got %dx%d, want batch rows x %d", rows, cols, b.featureSize)
		return 0, 0, nil, err
	}

	if values, err = input.Values(); err != nil {
		return 0, 0, nil, err
	}

	return rows, cols, values, nil
}

func (b *BatchNormalization) outputGradientValues(outputGradient *matrix.Matrix) (rows, cols int, values []float64, err error) {
	if outputGradient == nil {
		err = errors.New("layer: batch normalization output gradient is nil")
		return 0, 0, nil, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization output gradient invalid: %w", err)
		return 0, 0, nil, err
	}

	rows, cols = outputGradient.Shape()
	if rows != b.forwardRows || cols != b.forwardCols {
		err = fmt.Errorf(
			"layer: batch normalization output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			cols,
			b.forwardRows,
			b.forwardCols,
		)
		return 0, 0, nil, err
	}

	if b.normalizedCache == nil {
		err = errors.New("layer: batch normalization normalized cache is nil")
		return 0, 0, nil, err
	}

	if len(b.inverseStdCache) != cols {
		err = fmt.Errorf("layer: batch normalization inverse std cache length mismatch: got %d, want %d", len(b.inverseStdCache), cols)
		return 0, 0, nil, err
	}

	if values, err = outputGradient.Values(); err != nil {
		return 0, 0, nil, err
	}

	return rows, cols, values, nil
}

func (b *BatchNormalization) trainingInputGradient(rows, cols int, gradientValues, normalizedValues, gammaValues, betaGradient, gammaGradient []float64) (values []float64) {
	var (
		row        int
		col        int
		index      int
		rowCount   float64
		multiplier float64
	)

	values = make([]float64, len(gradientValues))
	rowCount = float64(rows)
	for row = 0; row < rows; row++ {
		for col = 0; col < cols; col++ {
			index = row*cols + col
			multiplier = gammaValues[col] * b.inverseStdCache[col] / rowCount
			values[index] = multiplier * (rowCount*gradientValues[index] - betaGradient[col] - normalizedValues[index]*gammaGradient[col])
		}
	}

	return values
}

func (b *BatchNormalization) evaluationInputGradient(cols int, gradientValues, gammaValues []float64) (values []float64) {
	var (
		index int
		col   int
	)

	values = make([]float64, len(gradientValues))
	for index = range gradientValues {
		col = index % cols
		values[index] = gradientValues[index] * gammaValues[col] * b.inverseStdCache[col]
	}

	return values
}

func (b *BatchNormalization) updateRunningStatistics(meanValues, varianceValues []float64) (err error) {
	var (
		currentMeanValues     []float64
		currentVarianceValues []float64
		nextMeanValues        []float64
		nextVarianceValues    []float64
		nextMean              *matrix.Matrix
		nextVariance          *matrix.Matrix
		index                 int
		updateScale           float64
	)

	if currentMeanValues, err = b.runningMean.Values(); err != nil {
		return err
	}

	if currentVarianceValues, err = b.runningVariance.Values(); err != nil {
		return err
	}

	updateScale = 1 - b.momentum
	nextMeanValues = make([]float64, len(meanValues))
	nextVarianceValues = make([]float64, len(varianceValues))
	for index = range meanValues {
		nextMeanValues[index] = b.momentum*currentMeanValues[index] + updateScale*meanValues[index]
		nextVarianceValues[index] = b.momentum*currentVarianceValues[index] + updateScale*varianceValues[index]
	}

	if nextMean, err = matrix.FromSlice(1, b.featureSize, nextMeanValues); err != nil {
		return err
	}

	if nextVariance, err = matrix.FromSlice(1, b.featureSize, nextVarianceValues); err != nil {
		return err
	}

	if err = b.runningMean.CopyFrom(nextMean); err != nil {
		return err
	}

	err = b.runningVariance.CopyFrom(nextVariance)
	return err
}

func (b *BatchNormalization) validate() (err error) {
	if b == nil {
		err = errors.New("layer: batch normalization layer is nil")
		return err
	}

	if b.featureSize <= 0 {
		err = fmt.Errorf("layer: batch normalization feature size must be positive: featureSize=%d", b.featureSize)
		return err
	}

	if err = validateBatchNormalizationMomentum(b.momentum); err != nil {
		return err
	}

	if err = validateBatchNormalizationEpsilon(b.epsilon); err != nil {
		return err
	}

	if b.gamma == nil {
		err = errors.New("layer: batch normalization gamma parameter is nil")
		return err
	}

	if b.beta == nil {
		err = errors.New("layer: batch normalization beta parameter is nil")
		return err
	}

	if err = validateMatrixShape("batch normalization gamma", b.gamma.Values(), 1, b.featureSize); err != nil {
		return err
	}

	if err = validateMatrixShape("batch normalization gamma gradient", b.gamma.Gradient(), 1, b.featureSize); err != nil {
		return err
	}

	if err = validateMatrixShape("batch normalization beta", b.beta.Values(), 1, b.featureSize); err != nil {
		return err
	}

	if err = validateMatrixShape("batch normalization beta gradient", b.beta.Gradient(), 1, b.featureSize); err != nil {
		return err
	}

	if err = validateMatrixShape("batch normalization running mean", b.runningMean, 1, b.featureSize); err != nil {
		return err
	}

	if err = validateMatrixShape("batch normalization running variance", b.runningVariance, 1, b.featureSize); err != nil {
		return err
	}

	return nil
}

func batchNormalizationMeans(rows, cols int, values []float64) (means []float64) {
	var (
		index int
		col   int
		scale float64
	)

	means = make([]float64, cols)
	for index = range values {
		col = index % cols
		means[col] += values[index]
	}

	scale = 1 / float64(rows)
	for col = range means {
		means[col] *= scale
	}

	return means
}

func batchNormalizationVariances(rows, cols int, values, means []float64) (variances []float64) {
	var (
		index      int
		col        int
		difference float64
		scale      float64
	)

	variances = make([]float64, cols)
	for index = range values {
		col = index % cols
		difference = values[index] - means[col]
		variances[col] += difference * difference
	}

	scale = 1 / float64(rows)
	for col = range variances {
		variances[col] *= scale
	}

	return variances
}

func batchNormalizationInverseStd(variances []float64, epsilon float64) (inverseStd []float64) {
	var index int

	inverseStd = make([]float64, len(variances))
	for index = range variances {
		inverseStd[index] = 1 / math.Sqrt(variances[index]+epsilon)
	}

	return inverseStd
}

func validateBatchNormalizationMomentum(momentum float64) (err error) {
	if momentum < 0 || momentum >= 1 || math.IsNaN(momentum) || math.IsInf(momentum, 0) {
		err = fmt.Errorf("layer: batch normalization momentum must be greater than or equal to 0 and less than 1: momentum=%g", momentum)
		return err
	}

	return nil
}

func validateBatchNormalizationEpsilon(epsilon float64) (err error) {
	if epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		err = fmt.Errorf("layer: batch normalization epsilon must be positive and finite: epsilon=%g", epsilon)
		return err
	}

	return nil
}
