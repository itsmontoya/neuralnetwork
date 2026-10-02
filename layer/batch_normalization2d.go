package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

// NewBatchNormalization2D constructs spatial batch normalization.
func NewBatchNormalization2D(config BatchNormalization2DConfig) (out *BatchNormalization2D, err error) {
	var (
		channels        int
		gammaMatrix     *matrix.Matrix
		betaMatrix      *matrix.Matrix
		runningMean     *matrix.Matrix
		runningVariance *matrix.Matrix
		gamma           *optimizer.Parameter
		beta            *optimizer.Parameter
		b               BatchNormalization2D
	)

	if err = config.validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization2d configuration invalid: %w", err)
		return nil, err
	}

	channels = config.InputShape.Channels()
	if gammaMatrix, err = matrix.New(1, channels); err != nil {
		return nil, err
	}
	if err = gammaMatrix.Fill(1); err != nil {
		return nil, err
	}
	if betaMatrix, err = matrix.New(1, channels); err != nil {
		return nil, err
	}
	if runningMean, err = matrix.New(1, channels); err != nil {
		return nil, err
	}
	if runningVariance, err = matrix.New(1, channels); err != nil {
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

	b.config = config
	b.gamma = gamma
	b.beta = beta
	b.runningMean = runningMean
	b.runningVariance = runningVariance
	b.training = true
	return &b, nil
}

// BatchNormalization2D normalizes each NCHW channel across batch and spatial
// positions while preserving flattened row layout.
//
// Training uses current N*H*W statistics and updates per-channel running
// statistics. Evaluation uses the stored running statistics. Gamma and beta
// are trainable per-channel scale and offset parameters.
type BatchNormalization2D struct {
	config           BatchNormalization2DConfig
	gamma            *optimizer.Parameter
	beta             *optimizer.Parameter
	runningMean      *matrix.Matrix
	runningVariance  *matrix.Matrix
	training         bool
	outputPool       scratch.MatrixPool
	outputScratch    *matrix.Matrix
	inputValuesPool  scratch.Float32Pool
	inputValues      []float32
	gammaValues      []float32
	betaValues       []float32
	meanValues       []float32
	varianceValues   []float32
	normalizedPool   scratch.Float32Pool
	normalizedCache  []float32
	inverseStdCache  []float32
	outputValuesPool scratch.Float32Pool
	outputValues     []float32

	gradientValuesPool      scratch.Float32Pool
	gradientValues          []float32
	gammaGradientValues     []float32
	betaGradientValues      []float32
	inputGradientValuesPool scratch.Float32Pool
	inputGradientValues     []float32
	runningMeanValues       []float32
	runningVarianceValues   []float32
	inputGradientPool       scratch.MatrixPool
	inputGradientScratch    *matrix.Matrix
	gammaGradientScratch    *matrix.Matrix
	betaGradientScratch     *matrix.Matrix
	forwardRows             int
	forwardCalled           bool
	forwardTraining         bool
}

// Forward normalizes spatial channels and applies trainable scale and offset.
func (b *BatchNormalization2D) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows        int
		columns     int
		channels    int
		spatialSize int
		valueCount  int
		index       int
		channel     int
	)

	if err = b.validate(); err != nil {
		return nil, err
	}
	if rows, err = b.validateInput(input); err != nil {
		return nil, err
	}

	columns = b.config.InputShape.Size()
	channels = b.config.InputShape.Channels()
	spatialSize = b.config.InputShape.Height() * b.config.InputShape.Width()
	valueCount = rows * columns
	if err = b.ensureForwardScratch(rows, columns, valueCount, channels); err != nil {
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
		batchNormalization2DMeansInto(rows, channels, spatialSize, b.inputValues, b.meanValues)
		batchNormalization2DVariancesInto(
			rows,
			channels,
			spatialSize,
			b.inputValues,
			b.meanValues,
			b.varianceValues,
		)
		if err = b.updateRunningStatistics(); err != nil {
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

	batchNormalizationInverseStdInto(b.varianceValues, b.config.Epsilon, b.inverseStdCache)
	for index = range b.inputValues {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		b.normalizedCache[index] = (b.inputValues[index] - b.meanValues[channel]) * b.inverseStdCache[channel]
		b.outputValues[index] = b.normalizedCache[index]*b.gammaValues[channel] + b.betaValues[channel]
	}
	if err = b.outputScratch.CopyValuesFrom(b.outputValues); err != nil {
		return nil, err
	}

	b.forwardRows = rows
	b.forwardCalled = true
	b.forwardTraining = b.training
	output = b.outputScratch
	return output, nil
}

// Backward accumulates per-channel parameter gradients and returns input gradients.
func (b *BatchNormalization2D) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows        int
		columns     int
		channels    int
		spatialSize int
		index       int
		channel     int
	)

	if err = b.validate(); err != nil {
		return nil, err
	}
	if !b.forwardCalled {
		err = errors.New("layer: batch normalization2d backward called before forward")
		return nil, err
	}
	if rows, err = b.validateOutputGradient(outputGradient); err != nil {
		return nil, err
	}

	columns = b.config.InputShape.Size()
	channels = b.config.InputShape.Channels()
	spatialSize = b.config.InputShape.Height() * b.config.InputShape.Width()
	if err = b.ensureBackwardScratch(rows, columns, rows*columns, channels); err != nil {
		return nil, err
	}
	if err = outputGradient.ValuesInto(b.gradientValues); err != nil {
		return nil, err
	}
	if err = b.gamma.Values().ValuesInto(b.gammaValues); err != nil {
		return nil, err
	}

	for channel = range b.gammaGradientValues {
		b.gammaGradientValues[channel] = 0
		b.betaGradientValues[channel] = 0
	}
	for index = range b.gradientValues {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		b.betaGradientValues[channel] += b.gradientValues[index]
		b.gammaGradientValues[channel] += b.gradientValues[index] * b.normalizedCache[index]
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
		b.trainingInputGradientInto(rows, columns, spatialSize)
	} else {
		b.evaluationInputGradientInto(columns, spatialSize)
	}
	if err = b.inputGradientScratch.CopyValuesFrom(b.inputGradientValues); err != nil {
		return nil, err
	}

	inputGradient = b.inputGradientScratch
	return inputGradient, nil
}

// Config returns the immutable spatial batch-normalization configuration.
func (b *BatchNormalization2D) Config() (config BatchNormalization2DConfig) {
	if b == nil {
		return config
	}

	config = b.config
	return config
}

// InputShape returns the configured channels-first input shape.
func (b *BatchNormalization2D) InputShape() (shape SpatialShape) {
	if b == nil {
		return shape
	}

	shape = b.config.InputShape
	return shape
}

// OutputShape returns the unchanged channels-first output shape.
func (b *BatchNormalization2D) OutputShape() (shape SpatialShape) {
	shape = b.InputShape()
	return shape
}

// Momentum returns the running-statistic update momentum.
func (b *BatchNormalization2D) Momentum() (momentum float32) {
	if b == nil {
		return 0
	}

	momentum = b.config.Momentum
	return momentum
}

// Epsilon returns the numerical stability value added to variances.
func (b *BatchNormalization2D) Epsilon() (epsilon float32) {
	if b == nil {
		return 0
	}

	epsilon = b.config.Epsilon
	return epsilon
}

// Gamma returns the trainable per-channel scale parameter.
func (b *BatchNormalization2D) Gamma() (gamma *optimizer.Parameter) {
	if b == nil {
		return nil
	}

	gamma = b.gamma
	return gamma
}

// Beta returns the trainable per-channel offset parameter.
func (b *BatchNormalization2D) Beta() (beta *optimizer.Parameter) {
	if b == nil {
		return nil
	}

	beta = b.beta
	return beta
}

// RunningMean returns the mutable per-channel running mean.
func (b *BatchNormalization2D) RunningMean() (runningMean *matrix.Matrix) {
	if b == nil {
		return nil
	}

	runningMean = b.runningMean
	return runningMean
}

// RunningVariance returns the mutable per-channel running variance.
func (b *BatchNormalization2D) RunningVariance() (runningVariance *matrix.Matrix) {
	if b == nil {
		return nil
	}

	runningVariance = b.runningVariance
	return runningVariance
}

// Parameters returns trainable gamma and beta parameters in that order.
func (b *BatchNormalization2D) Parameters() (parameters []*optimizer.Parameter) {
	if b == nil {
		return nil
	}

	parameters = []*optimizer.Parameter{b.gamma, b.beta}
	return parameters
}

// AppendParameters appends trainable gamma and beta parameters in that order.
// The returned slice is caller-owned, and BatchNormalization2D does not retain
// it. Appending allocates only when the supplied slice lacks capacity.
func (b *BatchNormalization2D) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter) {
	if b == nil {
		return parameters
	}

	out = append(parameters, b.gamma, b.beta)
	return out
}

// ResetGradients clears accumulated gamma and beta gradients.
func (b *BatchNormalization2D) ResetGradients() (err error) {
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
func (b *BatchNormalization2D) SetTraining(training bool) {
	if b == nil {
		return
	}

	b.training = training
}

// Training reports whether Forward uses batch statistics.
func (b *BatchNormalization2D) Training() (training bool) {
	if b == nil {
		return false
	}

	training = b.training
	return training
}

func (b *BatchNormalization2D) validate() (err error) {
	var channels int

	if b == nil {
		err = errors.New("layer: batch normalization2d layer is nil")
		return err
	}
	if err = b.config.validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization2d configuration invalid: %w", err)
		return err
	}

	channels = b.config.InputShape.Channels()
	if b.gamma == nil {
		err = errors.New("layer: batch normalization2d gamma parameter is nil")
		return err
	}
	if b.beta == nil {
		err = errors.New("layer: batch normalization2d beta parameter is nil")
		return err
	}
	if err = validateMatrixShape("batch normalization2d gamma", b.gamma.Values(), 1, channels); err != nil {
		return err
	}
	if err = validateMatrixShape("batch normalization2d gamma gradient", b.gamma.Gradient(), 1, channels); err != nil {
		return err
	}
	if err = validateMatrixShape("batch normalization2d beta", b.beta.Values(), 1, channels); err != nil {
		return err
	}
	if err = validateMatrixShape("batch normalization2d beta gradient", b.beta.Gradient(), 1, channels); err != nil {
		return err
	}
	if err = validateMatrixShape("batch normalization2d running mean", b.runningMean, 1, channels); err != nil {
		return err
	}
	if err = validateMatrixShape("batch normalization2d running variance", b.runningVariance, 1, channels); err != nil {
		return err
	}

	return nil
}

func (b *BatchNormalization2D) validateInput(input *matrix.Matrix) (rows int, err error) {
	var columns int

	if input == nil {
		err = errors.New("layer: batch normalization2d input is nil")
		return 0, err
	}
	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization2d input invalid: %w", err)
		return 0, err
	}

	rows, columns = input.Shape()
	if columns != b.config.InputShape.Size() {
		err = fmt.Errorf(
			"layer: batch normalization2d input shape mismatch: got %dx%d, want batch rows x %d",
			rows,
			columns,
			b.config.InputShape.Size(),
		)
		return 0, err
	}

	return rows, nil
}

func (b *BatchNormalization2D) validateOutputGradient(outputGradient *matrix.Matrix) (rows int, err error) {
	var columns int

	if outputGradient == nil {
		err = errors.New("layer: batch normalization2d output gradient is nil")
		return 0, err
	}
	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization2d output gradient invalid: %w", err)
		return 0, err
	}

	rows, columns = outputGradient.Shape()
	if rows != b.forwardRows || columns != b.config.InputShape.Size() {
		err = fmt.Errorf(
			"layer: batch normalization2d output gradient shape mismatch: got %dx%d, want %dx%d",
			rows,
			columns,
			b.forwardRows,
			b.config.InputShape.Size(),
		)
		return 0, err
	}
	if len(b.normalizedCache) != rows*columns {
		err = fmt.Errorf(
			"layer: batch normalization2d normalized cache length mismatch: got %d, want %d",
			len(b.normalizedCache),
			rows*columns,
		)
		return 0, err
	}
	if len(b.inverseStdCache) != b.config.InputShape.Channels() {
		err = fmt.Errorf(
			"layer: batch normalization2d inverse std cache length mismatch: got %d, want %d",
			len(b.inverseStdCache),
			b.config.InputShape.Channels(),
		)
		return 0, err
	}

	return rows, nil
}

func (b *BatchNormalization2D) ensureForwardScratch(
	rows,
	columns,
	valueCount,
	channels int,
) (err error) {
	if b.outputScratch, _, err = b.outputPool.Get(rows, columns); err != nil {
		return err
	}
	if b.inputValues, _, err = b.inputValuesPool.Get(valueCount); err != nil {
		return err
	}
	if b.gammaValues == nil {
		b.gammaValues = make([]float32, channels)
	}
	if b.betaValues == nil {
		b.betaValues = make([]float32, channels)
	}
	if b.meanValues == nil {
		b.meanValues = make([]float32, channels)
	}
	if b.varianceValues == nil {
		b.varianceValues = make([]float32, channels)
	}
	if b.normalizedCache, _, err = b.normalizedPool.Get(valueCount); err != nil {
		return err
	}
	if b.inverseStdCache == nil {
		b.inverseStdCache = make([]float32, channels)
	}
	if b.outputValues, _, err = b.outputValuesPool.Get(valueCount); err != nil {
		return err
	}

	return nil
}

func (b *BatchNormalization2D) ensureBackwardScratch(
	rows,
	columns,
	valueCount,
	channels int,
) (err error) {
	if b.gradientValues, _, err = b.gradientValuesPool.Get(valueCount); err != nil {
		return err
	}
	if b.gammaValues == nil {
		b.gammaValues = make([]float32, channels)
	}
	if b.gammaGradientValues == nil {
		b.gammaGradientValues = make([]float32, channels)
	}
	if b.betaGradientValues == nil {
		b.betaGradientValues = make([]float32, channels)
	}
	if b.inputGradientValues, _, err = b.inputGradientValuesPool.Get(valueCount); err != nil {
		return err
	}
	if b.inputGradientScratch, _, err = b.inputGradientPool.Get(rows, columns); err != nil {
		return err
	}
	if b.gammaGradientScratch == nil {
		if b.gammaGradientScratch, err = matrix.New(1, channels); err != nil {
			return err
		}
	}
	if b.betaGradientScratch == nil {
		if b.betaGradientScratch, err = matrix.New(1, channels); err != nil {
			return err
		}
	}

	return nil
}

func (b *BatchNormalization2D) trainingInputGradientInto(rows, columns, spatialSize int) {
	var (
		index       int
		channel     int
		sampleCount float32
		multiplier  float32
	)

	sampleCount = float32(rows * spatialSize)
	for index = range b.gradientValues {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		multiplier = b.gammaValues[channel] * b.inverseStdCache[channel] / sampleCount
		b.inputGradientValues[index] = multiplier * (sampleCount*b.gradientValues[index] -
			b.betaGradientValues[channel] -
			b.normalizedCache[index]*b.gammaGradientValues[channel])
	}
}

func (b *BatchNormalization2D) evaluationInputGradientInto(columns, spatialSize int) {
	var (
		index   int
		channel int
	)

	for index = range b.gradientValues {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		b.inputGradientValues[index] = b.gradientValues[index] * b.gammaValues[channel] * b.inverseStdCache[channel]
	}
}

func (b *BatchNormalization2D) updateRunningStatistics() (err error) {
	var (
		channel     int
		updateScale float32
	)

	if b.runningMeanValues == nil {
		b.runningMeanValues = make([]float32, b.config.InputShape.Channels())
	}
	if b.runningVarianceValues == nil {
		b.runningVarianceValues = make([]float32, b.config.InputShape.Channels())
	}
	if err = b.runningMean.ValuesInto(b.runningMeanValues); err != nil {
		return err
	}
	if err = b.runningVariance.ValuesInto(b.runningVarianceValues); err != nil {
		return err
	}

	updateScale = 1 - b.config.Momentum
	for channel = range b.meanValues {
		b.runningMeanValues[channel] = b.config.Momentum*b.runningMeanValues[channel] + updateScale*b.meanValues[channel]
		b.runningVarianceValues[channel] = b.config.Momentum*b.runningVarianceValues[channel] + updateScale*b.varianceValues[channel]
	}
	if err = b.runningMean.CopyValuesFrom(b.runningMeanValues); err != nil {
		return err
	}

	err = b.runningVariance.CopyValuesFrom(b.runningVarianceValues)
	return err
}

func batchNormalization2DChannel(index, columns, spatialSize int) (channel int) {
	channel = (index % columns) / spatialSize
	return channel
}

func batchNormalization2DMeansInto(
	rows,
	channels,
	spatialSize int,
	values,
	means []float32,
) {
	var (
		index   int
		channel int
		columns int
		scale   float32
	)

	columns = channels * spatialSize
	for channel = range means {
		means[channel] = 0
	}
	for index = range values {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		means[channel] += values[index]
	}

	scale = 1 / float32(rows*spatialSize)
	for channel = range means {
		means[channel] *= scale
	}
}

func batchNormalization2DVariancesInto(
	rows,
	channels,
	spatialSize int,
	values,
	means,
	variances []float32,
) {
	var (
		index      int
		channel    int
		columns    int
		difference float32
		scale      float32
	)

	columns = channels * spatialSize
	for channel = range variances {
		variances[channel] = 0
	}
	for index = range values {
		channel = batchNormalization2DChannel(index, columns, spatialSize)
		difference = values[index] - means[channel]
		variances[channel] += difference * difference
	}

	scale = 1 / float32(rows*spatialSize)
	for channel = range variances {
		variances[channel] *= scale
	}
}
