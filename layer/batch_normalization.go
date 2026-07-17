package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
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
func NewBatchNormalizationWithConfig(featureSize int, momentum, epsilon float32) (out *BatchNormalization, err error) {
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
	momentum        float32
	epsilon         float32
	gamma           *optimizer.Parameter
	beta            *optimizer.Parameter
	runningMean     *matrix.Matrix
	runningVariance *matrix.Matrix
	training        bool
	outputScratch   *matrix.Matrix
	inputValues     []float32
	gammaValues     []float32
	betaValues      []float32
	meanValues      []float32
	varianceValues  []float32
	normalizedCache []float32
	inverseStdCache []float32
	outputValues    []float32

	gradientValues        []float32
	gammaGradientValues   []float32
	betaGradientValues    []float32
	inputGradientValues   []float32
	runningMeanValues     []float32
	runningVarianceValues []float32
	inputGradientScratch  *matrix.Matrix
	gammaGradientScratch  *matrix.Matrix
	betaGradientScratch   *matrix.Matrix
	forwardRows           int
	forwardCols           int
	forwardCalled         bool
	forwardTraining       bool
}

// Forward normalizes input features and applies trainable scale and offset.
func (b *BatchNormalization) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows       int
		cols       int
		index      int
		col        int
		valueCount int
	)

	if err = b.validate(); err != nil {
		return nil, err
	}

	if rows, cols, err = b.validateInput(input); err != nil {
		return nil, err
	}

	valueCount = rows * cols
	if err = b.ensureForwardScratch(rows, cols, valueCount); err != nil {
		return nil, err
	}

	if err = input.ValuesInto(b.inputValues); err != nil {
		return nil, err
	}

	if err = b.gamma.Values().ValuesInto(b.gammaValues); err != nil {
		return nil, err
	}

	if err = b.beta.Values().ValuesInto(b.betaValues); err != nil {
		return nil, err
	}

	if b.training {
		batchNormalizationMeansInto(rows, cols, b.inputValues, b.meanValues)
		batchNormalizationVariancesInto(rows, cols, b.inputValues, b.meanValues, b.varianceValues)
		if err = b.updateRunningStatistics(b.meanValues, b.varianceValues); err != nil {
			return nil, err
		}
	} else {
		if err = b.runningMean.ValuesInto(b.meanValues); err != nil {
			return nil, err
		}

		if err = b.runningVariance.ValuesInto(b.varianceValues); err != nil {
			return nil, err
		}
	}

	batchNormalizationInverseStdInto(b.varianceValues, b.epsilon, b.inverseStdCache)
	for index = range b.inputValues {
		col = index % cols
		b.normalizedCache[index] = (b.inputValues[index] - b.meanValues[col]) * b.inverseStdCache[col]
		b.outputValues[index] = b.normalizedCache[index]*b.gammaValues[col] + b.betaValues[col]
	}

	if err = b.outputScratch.CopyValuesFrom(b.outputValues); err != nil {
		return nil, err
	}

	b.forwardRows = rows
	b.forwardCols = cols
	b.forwardCalled = true
	b.forwardTraining = b.training
	output = b.outputScratch
	return output, nil
}

// Backward accumulates gamma and beta gradients and returns input gradients.
func (b *BatchNormalization) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows  int
		cols  int
		index int
		col   int
	)

	if err = b.validate(); err != nil {
		return nil, err
	}

	if !b.forwardCalled {
		err = errors.New("layer: batch normalization backward called before forward")
		return nil, err
	}

	if rows, cols, err = b.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	if err = b.ensureBackwardScratch(rows, cols, rows*cols); err != nil {
		return nil, err
	}

	if err = outputGradient.ValuesInto(b.gradientValues); err != nil {
		return nil, err
	}

	if err = b.gamma.Values().ValuesInto(b.gammaValues); err != nil {
		return nil, err
	}

	for col = range b.gammaGradientValues {
		b.gammaGradientValues[col] = 0
		b.betaGradientValues[col] = 0
	}

	for index = range b.gradientValues {
		col = index % cols
		b.betaGradientValues[col] += b.gradientValues[index]
		b.gammaGradientValues[col] += b.gradientValues[index] * b.normalizedCache[index]
	}

	if err = b.gammaGradientScratch.CopyValuesFrom(b.gammaGradientValues); err != nil {
		return nil, err
	}

	if err = b.betaGradientScratch.CopyValuesFrom(b.betaGradientValues); err != nil {
		return nil, err
	}

	if err = b.gamma.AccumulateGradient(b.gammaGradientScratch); err != nil {
		return nil, err
	}

	if err = b.beta.AccumulateGradient(b.betaGradientScratch); err != nil {
		return nil, err
	}

	if b.forwardTraining {
		b.trainingInputGradientInto(rows, cols)
	} else {
		b.evaluationInputGradientInto(cols)
	}

	if err = b.inputGradientScratch.CopyValuesFrom(b.inputGradientValues); err != nil {
		return nil, err
	}

	inputGradient = b.inputGradientScratch
	return inputGradient, nil
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
func (b *BatchNormalization) Momentum() (momentum float32) {
	if b == nil {
		return 0
	}

	momentum = b.momentum
	return momentum
}

// Epsilon returns the numerical stability value added to variances.
func (b *BatchNormalization) Epsilon() (epsilon float32) {
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

// AppendParameters appends trainable gamma and beta parameters in that order.
// The returned slice is caller-owned, and BatchNormalization does not retain it.
func (b *BatchNormalization) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter) {
	if b == nil {
		return parameters
	}

	out = append(parameters, b.gamma, b.beta)
	return out
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

func (b *BatchNormalization) validateInput(input *matrix.Matrix) (rows, cols int, err error) {
	if input == nil {
		err = errors.New("layer: batch normalization input is nil")
		return 0, 0, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization input invalid: %w", err)
		return 0, 0, err
	}

	rows, cols = input.Shape()
	if cols != b.featureSize {
		err = fmt.Errorf("layer: batch normalization input shape mismatch: got %dx%d, want batch rows x %d", rows, cols, b.featureSize)
		return 0, 0, err
	}

	return rows, cols, nil
}

func (b *BatchNormalization) validateOutputGradient(outputGradient *matrix.Matrix) (rows, cols int, err error) {
	if outputGradient == nil {
		err = errors.New("layer: batch normalization output gradient is nil")
		return 0, 0, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization output gradient invalid: %w", err)
		return 0, 0, err
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
		return 0, 0, err
	}

	if b.normalizedCache == nil {
		err = errors.New("layer: batch normalization normalized cache is nil")
		return 0, 0, err
	}

	if len(b.inverseStdCache) != cols {
		err = fmt.Errorf("layer: batch normalization inverse std cache length mismatch: got %d, want %d", len(b.inverseStdCache), cols)
		return 0, 0, err
	}

	return rows, cols, nil
}

func (b *BatchNormalization) trainingInputGradientInto(rows, cols int) {
	var (
		row        int
		col        int
		index      int
		rowCount   float32
		multiplier float32
	)

	rowCount = float32(rows)
	for row = 0; row < rows; row++ {
		for col = 0; col < cols; col++ {
			index = row*cols + col
			multiplier = b.gammaValues[col] * b.inverseStdCache[col] / rowCount
			b.inputGradientValues[index] = multiplier * (rowCount*b.gradientValues[index] - b.betaGradientValues[col] - b.normalizedCache[index]*b.gammaGradientValues[col])
		}
	}
}

func (b *BatchNormalization) evaluationInputGradientInto(cols int) {
	var (
		index int
		col   int
	)

	for index = range b.gradientValues {
		col = index % cols
		b.inputGradientValues[index] = b.gradientValues[index] * b.gammaValues[col] * b.inverseStdCache[col]
	}
}

func (b *BatchNormalization) updateRunningStatistics(meanValues, varianceValues []float32) (err error) {
	var (
		index       int
		updateScale float32
	)

	b.runningMeanValues = floatScratch(b.runningMeanValues, b.featureSize)
	b.runningVarianceValues = floatScratch(b.runningVarianceValues, b.featureSize)

	if err = b.runningMean.ValuesInto(b.runningMeanValues); err != nil {
		return err
	}

	if err = b.runningVariance.ValuesInto(b.runningVarianceValues); err != nil {
		return err
	}

	updateScale = 1 - b.momentum
	for index = range meanValues {
		b.runningMeanValues[index] = b.momentum*b.runningMeanValues[index] + updateScale*meanValues[index]
		b.runningVarianceValues[index] = b.momentum*b.runningVarianceValues[index] + updateScale*varianceValues[index]
	}

	if err = b.runningMean.CopyValuesFrom(b.runningMeanValues); err != nil {
		return err
	}

	err = b.runningVariance.CopyValuesFrom(b.runningVarianceValues)
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

func (b *BatchNormalization) ensureForwardScratch(rows, cols, valueCount int) (err error) {
	if b.outputScratch, err = matrixScratch(b.outputScratch, rows, cols); err != nil {
		return err
	}

	b.inputValues = floatScratch(b.inputValues, valueCount)
	b.gammaValues = floatScratch(b.gammaValues, cols)
	b.betaValues = floatScratch(b.betaValues, cols)
	b.meanValues = floatScratch(b.meanValues, cols)
	b.varianceValues = floatScratch(b.varianceValues, cols)
	b.normalizedCache = floatScratch(b.normalizedCache, valueCount)
	b.inverseStdCache = floatScratch(b.inverseStdCache, cols)
	b.outputValues = floatScratch(b.outputValues, valueCount)
	return nil
}

func (b *BatchNormalization) ensureBackwardScratch(rows, cols, valueCount int) (err error) {
	b.gradientValues = floatScratch(b.gradientValues, valueCount)
	b.gammaValues = floatScratch(b.gammaValues, cols)
	b.gammaGradientValues = floatScratch(b.gammaGradientValues, cols)
	b.betaGradientValues = floatScratch(b.betaGradientValues, cols)
	b.inputGradientValues = floatScratch(b.inputGradientValues, valueCount)

	if b.inputGradientScratch, err = matrixScratch(b.inputGradientScratch, rows, cols); err != nil {
		return err
	}

	if b.gammaGradientScratch, err = matrixScratch(b.gammaGradientScratch, 1, cols); err != nil {
		return err
	}

	if b.betaGradientScratch, err = matrixScratch(b.betaGradientScratch, 1, cols); err != nil {
		return err
	}

	return nil
}

func batchNormalizationMeansInto(rows, cols int, values, means []float32) {
	var (
		index int
		col   int
		scale float32
	)

	for col = range means {
		means[col] = 0
	}

	for index = range values {
		col = index % cols
		means[col] += values[index]
	}

	scale = 1 / float32(rows)
	for col = range means {
		means[col] *= scale
	}
}

func batchNormalizationVariancesInto(rows, cols int, values, means, variances []float32) {
	var (
		index      int
		col        int
		difference float32
		scale      float32
	)

	for col = range variances {
		variances[col] = 0
	}

	for index = range values {
		col = index % cols
		difference = values[index] - means[col]
		variances[col] += difference * difference
	}

	scale = 1 / float32(rows)
	for col = range variances {
		variances[col] *= scale
	}
}

func batchNormalizationInverseStdInto(variances []float32, epsilon float32, inverseStd []float32) {
	var index int

	for index = range variances {
		inverseStd[index] = 1 / f32.Sqrt(variances[index]+epsilon)
	}
}

func validateBatchNormalizationMomentum(momentum float32) (err error) {
	if momentum < 0 || momentum >= 1 || f32.IsNaN(momentum) || f32.IsInf(momentum, 0) {
		err = fmt.Errorf("layer: batch normalization momentum must be greater than or equal to 0 and less than 1: momentum=%g", momentum)
		return err
	}

	return nil
}

func validateBatchNormalizationEpsilon(epsilon float32) (err error) {
	if epsilon <= 0 || f32.IsNaN(epsilon) || f32.IsInf(epsilon, 0) {
		err = fmt.Errorf("layer: batch normalization epsilon must be positive and finite: epsilon=%g", epsilon)
		return err
	}

	return nil
}
