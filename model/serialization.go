package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"

	activationpkg "github.com/itsmontoya/neuralnetwork/activation"
	layerpkg "github.com/itsmontoya/neuralnetwork/layer"
	matrixpkg "github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	serializationFormatSequential        = "neuralnetwork.sequential"
	serializationLayerActivation         = "activation"
	serializationLayerBatchNormalization = "batch_normalization"
	serializationLayerConv2D             = "conv2d"
	serializationLayerDense              = "dense"
	serializationLayerDropout            = "dropout"
	serializationLayerFlatten            = "flatten"
	serializationLayerLastStep           = "last_step"
	serializationLayerMaxPool2D          = "max_pool2d"
	serializationLayerSimpleRNN          = "simple_rnn"
	serializationDropoutSeed             = 1
	serializationVersion                 = 1
)

type sequentialDocument struct {
	Format  string            `json:"format"`
	Version int               `json:"version"`
	Layers  []serializedLayer `json:"layers"`
}

func sequentialDocumentFromModel(s *Sequential) (document sequentialDocument, err error) {
	var (
		index        int
		currentLayer layerpkg.Layer
		serialized   serializedLayer
	)

	if s == nil {
		err = errors.New("model: sequential model is nil")
		return document, err
	}

	document = sequentialDocument{
		Format:  serializationFormatSequential,
		Version: serializationVersion,
		Layers:  make([]serializedLayer, 0, len(s.layers)),
	}

	for index, currentLayer = range s.layers {
		if serialized, err = serializedLayerFromLayer(index, currentLayer); err != nil {
			return document, err
		}

		document.Layers = append(document.Layers, serialized)
	}

	return document, nil
}

func (d sequentialDocument) model() (s *Sequential, err error) {
	var (
		index        int
		serialized   serializedLayer
		currentLayer layerpkg.Layer
	)

	if d.Format != serializationFormatSequential {
		err = fmt.Errorf("model: unsupported serialization format %q", d.Format)
		return nil, err
	}

	if d.Version != serializationVersion {
		err = fmt.Errorf("model: unsupported serialization version %d", d.Version)
		return nil, err
	}

	if s, err = NewSequential(); err != nil {
		return nil, err
	}

	for index, serialized = range d.Layers {
		if currentLayer, err = serialized.layer(index); err != nil {
			return nil, err
		}

		if err = s.Add(currentLayer); err != nil {
			err = fmt.Errorf("model: layer %d add failed: %w", index, err)
			return nil, err
		}
	}

	return s, nil
}

type serializedLayer struct {
	Type             string            `json:"type"`
	InputSize        int               `json:"input_size,omitempty"`
	OutputSize       int               `json:"output_size,omitempty"`
	Steps            int               `json:"steps,omitempty"`
	FeatureSize      int               `json:"feature_size,omitempty"`
	HiddenSize       int               `json:"hidden_size,omitempty"`
	Activation       string            `json:"activation,omitempty"`
	Rate             float32           `json:"rate,omitempty"`
	Momentum         float32           `json:"momentum,omitempty"`
	Epsilon          float32           `json:"epsilon,omitempty"`
	InputWeights     *serializedMatrix `json:"input_weights,omitempty"`
	RecurrentWeights *serializedMatrix `json:"recurrent_weights,omitempty"`
	Weights          *serializedMatrix `json:"weights,omitempty"`
	Biases           *serializedMatrix `json:"biases,omitempty"`
	Gamma            *serializedMatrix `json:"gamma,omitempty"`
	Beta             *serializedMatrix `json:"beta,omitempty"`
	RunningMean      *serializedMatrix `json:"running_mean,omitempty"`
	RunningVariance  *serializedMatrix `json:"running_variance,omitempty"`
	InputChannels    int               `json:"input_channels,omitempty"`
	InputHeight      int               `json:"input_height,omitempty"`
	InputWidth       int               `json:"input_width,omitempty"`
	OutputChannels   int               `json:"output_channels,omitempty"`
	KernelHeight     int               `json:"kernel_height,omitempty"`
	KernelWidth      int               `json:"kernel_width,omitempty"`
	StrideHeight     int               `json:"stride_height,omitempty"`
	StrideWidth      int               `json:"stride_width,omitempty"`
	PaddingHeight    int               `json:"padding_height,omitempty"`
	PaddingWidth     int               `json:"padding_width,omitempty"`
	WindowHeight     int               `json:"window_height,omitempty"`
	WindowWidth      int               `json:"window_width,omitempty"`
}

func serializedLayerFromLayer(index int, currentLayer layerpkg.Layer) (serialized serializedLayer, err error) {
	if currentLayer == nil {
		err = fmt.Errorf("model: layer %d is nil", index)
		return serialized, err
	}

	switch current := currentLayer.(type) {
	case *layerpkg.Activation:
		serialized, err = serializedActivationLayer(index, current)
	case *layerpkg.BatchNormalization:
		serialized, err = serializedBatchNormalizationLayer(index, current)
	case *layerpkg.Conv2D:
		serialized, err = serializedConv2DLayer(index, current)
	case *layerpkg.Dense:
		serialized, err = serializedDenseLayer(index, current)
	case *layerpkg.Dropout:
		serialized, err = serializedDropoutLayer(index, current)
	case *layerpkg.Flatten:
		serialized, err = serializedFlattenLayer(index, current)
	case *layerpkg.LastStep:
		serialized, err = serializedLastStepLayer(index, current)
	case *layerpkg.MaxPool2D:
		serialized, err = serializedMaxPool2DLayer(index, current)
	case *layerpkg.SimpleRNN:
		serialized, err = serializedSimpleRNNLayer(index, current)
	default:
		err = fmt.Errorf("model: layer %d unsupported layer type %T", index, currentLayer)
		return serialized, err
	}

	return serialized, err
}

func (s serializedLayer) layer(index int) (currentLayer layerpkg.Layer, err error) {
	switch s.Type {
	case serializationLayerActivation:
		currentLayer, err = s.activationLayer(index)
	case serializationLayerBatchNormalization:
		currentLayer, err = s.batchNormalizationLayer(index)
	case serializationLayerConv2D:
		currentLayer, err = s.conv2DLayer(index)
	case serializationLayerDense:
		currentLayer, err = s.denseLayer(index)
	case serializationLayerDropout:
		currentLayer, err = s.dropoutLayer(index)
	case serializationLayerFlatten:
		currentLayer, err = s.flattenLayer(index)
	case serializationLayerLastStep:
		currentLayer, err = s.lastStepLayer(index)
	case serializationLayerMaxPool2D:
		currentLayer, err = s.maxPool2DLayer(index)
	case serializationLayerSimpleRNN:
		currentLayer, err = s.simpleRNNLayer(index)
	default:
		err = fmt.Errorf("model: layer %d unknown layer type %q", index, s.Type)
		return nil, err
	}

	return currentLayer, err
}

func serializedActivationLayer(index int, activationLayer *layerpkg.Activation) (serialized serializedLayer, err error) {
	var name string

	if activationLayer == nil {
		err = fmt.Errorf("model: layer %d activation layer is nil", index)
		return serialized, err
	}

	if name, err = activationpkg.Name(activationLayer.Function()); err != nil {
		err = fmt.Errorf("model: layer %d activation name failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:       serializationLayerActivation,
		Activation: name,
	}
	return serialized, nil
}

func (s serializedLayer) activationLayer(index int) (activationLayer *layerpkg.Activation, err error) {
	var function activationpkg.Activation

	if function, err = activationpkg.FromName(s.Activation); err != nil {
		err = fmt.Errorf("model: layer %d activation load failed: %w", index, err)
		return nil, err
	}

	if activationLayer, err = layerpkg.NewActivation(function); err != nil {
		err = fmt.Errorf("model: layer %d activation construct failed: %w", index, err)
		return nil, err
	}

	return activationLayer, nil
}

func serializedBatchNormalizationLayer(index int, batchNormLayer *layerpkg.BatchNormalization) (serialized serializedLayer, err error) {
	var (
		gamma           serializedMatrix
		beta            serializedMatrix
		runningMean     serializedMatrix
		runningVariance serializedMatrix
	)

	if batchNormLayer == nil {
		err = fmt.Errorf("model: layer %d batch normalization layer is nil", index)
		return serialized, err
	}

	if batchNormLayer.Gamma() == nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma parameter is nil", index)
		return serialized, err
	}

	if batchNormLayer.Beta() == nil {
		err = fmt.Errorf("model: layer %d batch normalization beta parameter is nil", index)
		return serialized, err
	}

	if gamma, err = serializedMatrixFromMatrix(batchNormLayer.Gamma().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma serialize failed: %w", index, err)
		return serialized, err
	}

	if beta, err = serializedMatrixFromMatrix(batchNormLayer.Beta().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta serialize failed: %w", index, err)
		return serialized, err
	}

	if runningMean, err = serializedMatrixFromMatrix(batchNormLayer.RunningMean()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean serialize failed: %w", index, err)
		return serialized, err
	}

	if runningVariance, err = serializedMatrixFromMatrix(batchNormLayer.RunningVariance()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:            serializationLayerBatchNormalization,
		FeatureSize:     batchNormLayer.FeatureSize(),
		Momentum:        batchNormLayer.Momentum(),
		Epsilon:         batchNormLayer.Epsilon(),
		Gamma:           &gamma,
		Beta:            &beta,
		RunningMean:     &runningMean,
		RunningVariance: &runningVariance,
	}
	return serialized, nil
}

func (s serializedLayer) batchNormalizationLayer(index int) (batchNormLayer *layerpkg.BatchNormalization, err error) {
	if s.Gamma == nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma is missing", index)
		return nil, err
	}

	if s.Beta == nil {
		err = fmt.Errorf("model: layer %d batch normalization beta is missing", index)
		return nil, err
	}

	if s.RunningMean == nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean is missing", index)
		return nil, err
	}

	if s.RunningVariance == nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance is missing", index)
		return nil, err
	}

	if err = s.Gamma.validate(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma load failed: %w", index, err)
		return nil, err
	}

	if err = s.Beta.validate(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta load failed: %w", index, err)
		return nil, err
	}

	if err = s.RunningMean.validate(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean load failed: %w", index, err)
		return nil, err
	}

	if err = s.RunningVariance.validate(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance load failed: %w", index, err)
		return nil, err
	}

	if batchNormLayer, err = layerpkg.NewBatchNormalizationWithConfig(s.FeatureSize, s.Momentum, s.Epsilon); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization construct failed: %w", index, err)
		return nil, err
	}

	if err = s.Gamma.copyInto(batchNormLayer.Gamma().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma copy failed: %w", index, err)
		return nil, err
	}

	if err = s.Beta.copyInto(batchNormLayer.Beta().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta copy failed: %w", index, err)
		return nil, err
	}

	if err = s.RunningMean.copyInto(batchNormLayer.RunningMean()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean copy failed: %w", index, err)
		return nil, err
	}

	if err = s.RunningVariance.copyInto(batchNormLayer.RunningVariance()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance copy failed: %w", index, err)
		return nil, err
	}

	return batchNormLayer, nil
}

func serializedConv2DLayer(index int, convLayer *layerpkg.Conv2D) (serialized serializedLayer, err error) {
	var (
		config         layerpkg.Conv2DConfig
		expectedConfig layerpkg.Conv2DConfig
		shape          layerpkg.SpatialShape
		weights        serializedMatrix
		biases         serializedMatrix
	)

	if convLayer == nil {
		err = fmt.Errorf("model: layer %d conv2d layer is nil", index)
		return serialized, err
	}

	config = convLayer.Config()
	shape = config.InputShape()
	if expectedConfig, err = layerpkg.NewConv2DConfig(
		shape,
		config.OutputChannels(),
		config.KernelHeight(),
		config.KernelWidth(),
		config.StrideHeight(),
		config.StrideWidth(),
		config.PaddingHeight(),
		config.PaddingWidth(),
	); err != nil {
		err = fmt.Errorf("model: layer %d conv2d configuration serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedConfig != config {
		err = fmt.Errorf("model: layer %d conv2d configuration is inconsistent", index)
		return serialized, err
	}

	if convLayer.Weights() == nil {
		err = fmt.Errorf("model: layer %d conv2d weights parameter is nil", index)
		return serialized, err
	}

	if convLayer.Biases() == nil {
		err = fmt.Errorf("model: layer %d conv2d biases parameter is nil", index)
		return serialized, err
	}

	if weights, err = serializedMatrixFromMatrix(convLayer.Weights().Values()); err != nil {
		err = fmt.Errorf("model: layer %d conv2d weights serialize failed: %w", index, err)
		return serialized, err
	}

	if biases, err = serializedMatrixFromMatrix(convLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d conv2d biases serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:           serializationLayerConv2D,
		Weights:        &weights,
		Biases:         &biases,
		InputChannels:  shape.Channels(),
		InputHeight:    shape.Height(),
		InputWidth:     shape.Width(),
		OutputChannels: config.OutputChannels(),
		KernelHeight:   config.KernelHeight(),
		KernelWidth:    config.KernelWidth(),
		StrideHeight:   config.StrideHeight(),
		StrideWidth:    config.StrideWidth(),
		PaddingHeight:  config.PaddingHeight(),
		PaddingWidth:   config.PaddingWidth(),
	}
	return serialized, nil
}

func (s serializedLayer) conv2DLayer(index int) (convLayer *layerpkg.Conv2D, err error) {
	var (
		inputShape layerpkg.SpatialShape
		config     layerpkg.Conv2DConfig
		weights    *matrixpkg.Matrix
	)

	if s.Weights == nil {
		err = fmt.Errorf("model: layer %d conv2d weights are missing", index)
		return nil, err
	}

	if s.Biases == nil {
		err = fmt.Errorf("model: layer %d conv2d biases are missing", index)
		return nil, err
	}

	if inputShape, err = s.spatialInputShape(index, serializationLayerConv2D); err != nil {
		return nil, err
	}

	if config, err = layerpkg.NewConv2DConfig(
		inputShape,
		s.OutputChannels,
		s.KernelHeight,
		s.KernelWidth,
		s.StrideHeight,
		s.StrideWidth,
		s.PaddingHeight,
		s.PaddingWidth,
	); err != nil {
		err = fmt.Errorf("model: layer %d conv2d configuration load failed: %w", index, err)
		return nil, err
	}

	if weights, err = s.Weights.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d conv2d weights load failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.validate(); err != nil {
		err = fmt.Errorf("model: layer %d conv2d biases load failed: %w", index, err)
		return nil, err
	}

	if convLayer, err = layerpkg.NewConv2D(config, func(inputSize, outputSize int) (initialized *matrixpkg.Matrix, err error) {
		initialized = weights
		return initialized, nil
	}); err != nil {
		err = fmt.Errorf("model: layer %d conv2d construct failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.copyInto(convLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d conv2d biases copy failed: %w", index, err)
		return nil, err
	}

	return convLayer, nil
}

func serializedDenseLayer(index int, denseLayer *layerpkg.Dense) (serialized serializedLayer, err error) {
	var (
		weights serializedMatrix
		biases  serializedMatrix
	)

	if denseLayer == nil {
		err = fmt.Errorf("model: layer %d dense layer is nil", index)
		return serialized, err
	}

	if denseLayer.Weights() == nil {
		err = fmt.Errorf("model: layer %d dense weights parameter is nil", index)
		return serialized, err
	}

	if denseLayer.Biases() == nil {
		err = fmt.Errorf("model: layer %d dense biases parameter is nil", index)
		return serialized, err
	}

	if weights, err = serializedMatrixFromMatrix(denseLayer.Weights().Values()); err != nil {
		err = fmt.Errorf("model: layer %d dense weights serialize failed: %w", index, err)
		return serialized, err
	}

	if biases, err = serializedMatrixFromMatrix(denseLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d dense biases serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:       serializationLayerDense,
		InputSize:  denseLayer.InputSize(),
		OutputSize: denseLayer.OutputSize(),
		Weights:    &weights,
		Biases:     &biases,
	}
	return serialized, nil
}

func (s serializedLayer) denseLayer(index int) (denseLayer *layerpkg.Dense, err error) {
	var (
		weights *matrixpkg.Matrix
	)

	if s.Weights == nil {
		err = fmt.Errorf("model: layer %d dense weights are missing", index)
		return nil, err
	}

	if s.Biases == nil {
		err = fmt.Errorf("model: layer %d dense biases are missing", index)
		return nil, err
	}

	if weights, err = s.Weights.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d dense weights load failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.validate(); err != nil {
		err = fmt.Errorf("model: layer %d dense biases load failed: %w", index, err)
		return nil, err
	}

	if denseLayer, err = layerpkg.NewDense(s.InputSize, s.OutputSize, func(inputSize, outputSize int) (initialized *matrixpkg.Matrix, err error) {
		initialized = weights
		return initialized, nil
	}); err != nil {
		err = fmt.Errorf("model: layer %d dense construct failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.copyInto(denseLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d dense biases copy failed: %w", index, err)
		return nil, err
	}

	return denseLayer, nil
}

func serializedDropoutLayer(index int, dropoutLayer *layerpkg.Dropout) (serialized serializedLayer, err error) {
	if dropoutLayer == nil {
		err = fmt.Errorf("model: layer %d dropout layer is nil", index)
		return serialized, err
	}

	serialized = serializedLayer{
		Type: serializationLayerDropout,
		Rate: dropoutLayer.Rate(),
	}
	return serialized, nil
}

func (s serializedLayer) dropoutLayer(index int) (dropoutLayer *layerpkg.Dropout, err error) {
	var random *rand.Rand

	random = rand.New(rand.NewSource(serializationDropoutSeed + int64(index)))
	if dropoutLayer, err = layerpkg.NewDropout(s.Rate, random); err != nil {
		err = fmt.Errorf("model: layer %d dropout construct failed: %w", index, err)
		return nil, err
	}

	return dropoutLayer, nil
}

func serializedFlattenLayer(index int, flattenLayer *layerpkg.Flatten) (serialized serializedLayer, err error) {
	var (
		shape         layerpkg.SpatialShape
		expectedShape layerpkg.SpatialShape
	)

	if flattenLayer == nil {
		err = fmt.Errorf("model: layer %d flatten layer is nil", index)
		return serialized, err
	}

	shape = flattenLayer.InputShape()
	if expectedShape, err = layerpkg.NewSpatialShape(shape.Channels(), shape.Height(), shape.Width()); err != nil {
		err = fmt.Errorf("model: layer %d flatten input shape serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedShape != shape {
		err = fmt.Errorf("model: layer %d flatten input shape is inconsistent", index)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:          serializationLayerFlatten,
		InputChannels: shape.Channels(),
		InputHeight:   shape.Height(),
		InputWidth:    shape.Width(),
	}
	return serialized, nil
}

func (s serializedLayer) flattenLayer(index int) (flattenLayer *layerpkg.Flatten, err error) {
	var inputShape layerpkg.SpatialShape

	if inputShape, err = s.spatialInputShape(index, serializationLayerFlatten); err != nil {
		return nil, err
	}

	if flattenLayer, err = layerpkg.NewFlatten(inputShape); err != nil {
		err = fmt.Errorf("model: layer %d flatten construct failed: %w", index, err)
		return nil, err
	}

	return flattenLayer, nil
}

func serializedLastStepLayer(index int, lastStepLayer *layerpkg.LastStep) (serialized serializedLayer, err error) {
	var (
		shape         layerpkg.SequenceShape
		expectedShape layerpkg.SequenceShape
	)

	if lastStepLayer == nil {
		err = fmt.Errorf("model: layer %d last step layer is nil", index)
		return serialized, err
	}

	shape = lastStepLayer.InputShape()
	if expectedShape, err = layerpkg.NewSequenceShape(shape.Steps(), shape.FeatureSize()); err != nil {
		err = fmt.Errorf("model: layer %d last step input shape serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedShape != shape {
		err = fmt.Errorf("model: layer %d last step input shape is inconsistent", index)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:        serializationLayerLastStep,
		Steps:       shape.Steps(),
		FeatureSize: shape.FeatureSize(),
	}
	return serialized, nil
}

func (s serializedLayer) lastStepLayer(index int) (lastStepLayer *layerpkg.LastStep, err error) {
	var inputShape layerpkg.SequenceShape

	if inputShape, err = s.sequenceInputShape(index, serializationLayerLastStep); err != nil {
		return nil, err
	}

	if lastStepLayer, err = layerpkg.NewLastStep(inputShape); err != nil {
		err = fmt.Errorf("model: layer %d last step construct failed: %w", index, err)
		return nil, err
	}

	return lastStepLayer, nil
}

func serializedMaxPool2DLayer(index int, poolLayer *layerpkg.MaxPool2D) (serialized serializedLayer, err error) {
	var (
		config         layerpkg.MaxPool2DConfig
		expectedConfig layerpkg.MaxPool2DConfig
		shape          layerpkg.SpatialShape
	)

	if poolLayer == nil {
		err = fmt.Errorf("model: layer %d max pool2d layer is nil", index)
		return serialized, err
	}

	config = poolLayer.Config()
	shape = config.InputShape()
	if expectedConfig, err = layerpkg.NewMaxPool2DConfig(
		shape,
		config.WindowHeight(),
		config.WindowWidth(),
		config.StrideHeight(),
		config.StrideWidth(),
	); err != nil {
		err = fmt.Errorf("model: layer %d max pool2d configuration serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedConfig != config {
		err = fmt.Errorf("model: layer %d max pool2d configuration is inconsistent", index)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:          serializationLayerMaxPool2D,
		InputChannels: shape.Channels(),
		InputHeight:   shape.Height(),
		InputWidth:    shape.Width(),
		StrideHeight:  config.StrideHeight(),
		StrideWidth:   config.StrideWidth(),
		WindowHeight:  config.WindowHeight(),
		WindowWidth:   config.WindowWidth(),
	}
	return serialized, nil
}

func (s serializedLayer) maxPool2DLayer(index int) (poolLayer *layerpkg.MaxPool2D, err error) {
	var (
		inputShape layerpkg.SpatialShape
		config     layerpkg.MaxPool2DConfig
	)

	if inputShape, err = s.spatialInputShape(index, serializationLayerMaxPool2D); err != nil {
		return nil, err
	}

	if config, err = layerpkg.NewMaxPool2DConfig(
		inputShape,
		s.WindowHeight,
		s.WindowWidth,
		s.StrideHeight,
		s.StrideWidth,
	); err != nil {
		err = fmt.Errorf("model: layer %d max pool2d configuration load failed: %w", index, err)
		return nil, err
	}

	if poolLayer, err = layerpkg.NewMaxPool2D(config); err != nil {
		err = fmt.Errorf("model: layer %d max pool2d construct failed: %w", index, err)
		return nil, err
	}

	return poolLayer, nil
}

func serializedSimpleRNNLayer(index int, recurrentLayer *layerpkg.SimpleRNN) (serialized serializedLayer, err error) {
	var (
		config           layerpkg.SimpleRNNConfig
		expectedConfig   layerpkg.SimpleRNNConfig
		inputShape       layerpkg.SequenceShape
		expectedShape    layerpkg.SequenceShape
		inputWeights     serializedMatrix
		recurrentWeights serializedMatrix
		biases           serializedMatrix
	)

	if recurrentLayer == nil {
		err = fmt.Errorf("model: layer %d simple rnn layer is nil", index)
		return serialized, err
	}

	config = recurrentLayer.Config()
	inputShape = config.InputShape()
	if expectedShape, err = layerpkg.NewSequenceShape(inputShape.Steps(), inputShape.FeatureSize()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn input shape serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedShape != inputShape {
		err = fmt.Errorf("model: layer %d simple rnn input shape is inconsistent", index)
		return serialized, err
	}

	if expectedConfig, err = layerpkg.NewSimpleRNNConfig(expectedShape, config.HiddenSize()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn configuration serialize failed: %w", index, err)
		return serialized, err
	}

	if expectedConfig != config {
		err = fmt.Errorf("model: layer %d simple rnn configuration is inconsistent", index)
		return serialized, err
	}

	if recurrentLayer.InputWeights() == nil {
		err = fmt.Errorf("model: layer %d simple rnn input weights parameter is nil", index)
		return serialized, err
	}

	if recurrentLayer.RecurrentWeights() == nil {
		err = fmt.Errorf("model: layer %d simple rnn recurrent weights parameter is nil", index)
		return serialized, err
	}

	if recurrentLayer.Biases() == nil {
		err = fmt.Errorf("model: layer %d simple rnn biases parameter is nil", index)
		return serialized, err
	}

	if inputWeights, err = serializedMatrixFromMatrix(recurrentLayer.InputWeights().Values()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn input weights serialize failed: %w", index, err)
		return serialized, err
	}

	if recurrentWeights, err = serializedMatrixFromMatrix(recurrentLayer.RecurrentWeights().Values()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn recurrent weights serialize failed: %w", index, err)
		return serialized, err
	}

	if biases, err = serializedMatrixFromMatrix(recurrentLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn biases serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:             serializationLayerSimpleRNN,
		Steps:            inputShape.Steps(),
		FeatureSize:      inputShape.FeatureSize(),
		HiddenSize:       config.HiddenSize(),
		InputWeights:     &inputWeights,
		RecurrentWeights: &recurrentWeights,
		Biases:           &biases,
	}
	return serialized, nil
}

func (s serializedLayer) simpleRNNLayer(index int) (recurrentLayer *layerpkg.SimpleRNN, err error) {
	var (
		inputShape       layerpkg.SequenceShape
		config           layerpkg.SimpleRNNConfig
		inputWeights     *matrixpkg.Matrix
		recurrentWeights *matrixpkg.Matrix
	)

	if s.InputWeights == nil {
		err = fmt.Errorf("model: layer %d simple rnn input weights are missing", index)
		return nil, err
	}

	if s.RecurrentWeights == nil {
		err = fmt.Errorf("model: layer %d simple rnn recurrent weights are missing", index)
		return nil, err
	}

	if s.Biases == nil {
		err = fmt.Errorf("model: layer %d simple rnn biases are missing", index)
		return nil, err
	}

	if inputShape, err = s.sequenceInputShape(index, serializationLayerSimpleRNN); err != nil {
		return nil, err
	}

	if config, err = layerpkg.NewSimpleRNNConfig(inputShape, s.HiddenSize); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn configuration load failed: %w", index, err)
		return nil, err
	}

	if inputWeights, err = s.InputWeights.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn input weights load failed: %w", index, err)
		return nil, err
	}

	if recurrentWeights, err = s.RecurrentWeights.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn recurrent weights load failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.validate(); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn biases load failed: %w", index, err)
		return nil, err
	}

	if recurrentLayer, err = layerpkg.NewSimpleRNN(
		config,
		func(inputSize, outputSize int) (initialized *matrixpkg.Matrix, err error) {
			initialized = inputWeights
			return initialized, nil
		},
		func(inputSize, outputSize int) (initialized *matrixpkg.Matrix, err error) {
			initialized = recurrentWeights
			return initialized, nil
		},
	); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn construct failed: %w", index, err)
		return nil, err
	}

	if err = s.Biases.copyInto(recurrentLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d simple rnn biases copy failed: %w", index, err)
		return nil, err
	}

	return recurrentLayer, nil
}

func (s serializedLayer) sequenceInputShape(index int, layerName string) (shape layerpkg.SequenceShape, err error) {
	if s.Steps == 0 && s.FeatureSize == 0 {
		err = fmt.Errorf("model: layer %d %s input shape is missing", index, layerName)
		return shape, err
	}

	if shape, err = layerpkg.NewSequenceShape(s.Steps, s.FeatureSize); err != nil {
		err = fmt.Errorf("model: layer %d %s input shape load failed: %w", index, layerName, err)
		return shape, err
	}

	return shape, nil
}

func (s serializedLayer) spatialInputShape(index int, layerName string) (shape layerpkg.SpatialShape, err error) {
	if s.InputChannels == 0 && s.InputHeight == 0 && s.InputWidth == 0 {
		err = fmt.Errorf("model: layer %d %s input shape is missing", index, layerName)
		return shape, err
	}

	if shape, err = layerpkg.NewSpatialShape(s.InputChannels, s.InputHeight, s.InputWidth); err != nil {
		err = fmt.Errorf("model: layer %d %s input shape load failed: %w", index, layerName, err)
		return shape, err
	}

	return shape, nil
}

type serializedMatrix struct {
	Rows   int       `json:"rows"`
	Cols   int       `json:"cols"`
	Values []float32 `json:"values"`
}

func serializedMatrixFromMatrix(source *matrixpkg.Matrix) (serialized serializedMatrix, err error) {
	var values []float32

	if values, err = source.Values(); err != nil {
		return serialized, err
	}

	serialized = serializedMatrix{
		Rows:   source.Rows(),
		Cols:   source.Cols(),
		Values: values,
	}
	return serialized, nil
}

func (s serializedMatrix) matrix() (m *matrixpkg.Matrix, err error) {
	m, err = matrixpkg.FromSlice(s.Rows, s.Cols, s.Values)
	return m, err
}

func (s serializedMatrix) copyInto(destination *matrixpkg.Matrix) (err error) {
	var (
		rows int
		cols int
	)

	if err = s.validate(); err != nil {
		return err
	}

	if destination == nil {
		err = errors.New("matrix: matrix is nil")
		return err
	}

	if err = destination.Validate(); err != nil {
		return err
	}

	rows, cols = destination.Shape()
	if s.Rows != rows || s.Cols != cols {
		err = fmt.Errorf("matrix: destination shape mismatch: got %dx%d, want %dx%d", s.Rows, s.Cols, rows, cols)
		return err
	}

	err = destination.CopyValuesFrom(s.Values)
	return err
}

func (s serializedMatrix) validate() (err error) {
	var (
		maxInt int
		size   int
	)

	if s.Rows <= 0 || s.Cols <= 0 {
		err = fmt.Errorf("matrix: dimensions must be positive: rows=%d cols=%d", s.Rows, s.Cols)
		return err
	}

	maxInt = int(^uint(0) >> 1)
	if s.Rows > maxInt/s.Cols {
		err = fmt.Errorf("matrix: dimensions are too large: rows=%d cols=%d", s.Rows, s.Cols)
		return err
	}

	size = s.Rows * s.Cols
	if len(s.Values) != size {
		err = fmt.Errorf("matrix: values length mismatch: got %d, want %d", len(s.Values), size)
		return err
	}

	return nil
}

func encodeSequential(writer io.Writer, s *Sequential) (err error) {
	var (
		document sequentialDocument
		encoder  *json.Encoder
	)

	if document, err = sequentialDocumentFromModel(s); err != nil {
		return err
	}

	encoder = json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	if err = encoder.Encode(document); err != nil {
		err = fmt.Errorf("encode JSON: %w", err)
		return err
	}

	return nil
}

func decodeSequential(reader io.Reader) (s *Sequential, err error) {
	var (
		document sequentialDocument
		decoder  *json.Decoder
	)

	decoder = json.NewDecoder(reader)
	if err = decoder.Decode(&document); err != nil {
		err = fmt.Errorf("decode JSON: %w", err)
		return nil, err
	}

	if err = ensureNoTrailingJSON(decoder); err != nil {
		return nil, err
	}

	s, err = document.model()
	return s, err
}

func ensureNoTrailingJSON(decoder *json.Decoder) (err error) {
	var extra interface{}

	if err = decoder.Decode(&extra); err == nil {
		err = errors.New("model: JSON contains multiple values")
		return err
	}

	if errors.Is(err, io.EOF) {
		return nil
	}

	err = fmt.Errorf("decode trailing JSON: %w", err)
	return err
}
