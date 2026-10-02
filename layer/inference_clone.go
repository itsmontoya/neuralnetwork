package layer

import (
	"errors"
	"fmt"
	"math/rand"
)

// CloneForInference creates a layer with independent runtime state and shared
// immutable parameters.
//
// The source layer and clone must not train or mutate shared parameters while
// either is serving inference. The clone starts in evaluation mode when the
// layer supports training and evaluation behavior.
func CloneForInference(source Layer) (out Layer, err error) {
	if source == nil {
		err = errors.New("layer: inference clone source is nil")
		return nil, err
	}

	switch current := source.(type) {
	case *Activation:
		out, err = cloneActivationForInference(current)
	case *BatchNormalization:
		out, err = cloneBatchNormalizationForInference(current)
	case *BatchNormalization2D:
		out, err = cloneBatchNormalization2DForInference(current)
	case *Conv2D:
		out, err = cloneConv2DForInference(current)
	case *Dense:
		out, err = cloneDenseForInference(current)
	case *Dropout:
		out, err = cloneDropoutForInference(current)
	case *Flatten:
		out, err = NewFlatten(current.InputShape())
	case *GatherLastValid:
		out, err = NewGatherLastValid(current.InputShape())
	case *LastStep:
		out, err = NewLastStep(current.InputShape())
	case *MaxPool2D:
		out, err = NewMaxPool2D(current.Config())
	case *SimpleRNN:
		out, err = cloneSimpleRNNForInference(current)
	default:
		err = fmt.Errorf("layer: inference clone does not support %T", source)
		return nil, err
	}
	if err != nil {
		err = fmt.Errorf("layer: inference clone %T: %w", source, err)
		return nil, err
	}

	return out, nil
}

func cloneActivationForInference(source *Activation) (out *Activation, err error) {
	if err = source.validate(); err != nil {
		return nil, err
	}

	out, err = NewActivation(source.function)
	return out, err
}

func cloneBatchNormalizationForInference(
	source *BatchNormalization,
) (out *BatchNormalization, err error) {
	if err = source.validate(); err != nil {
		return nil, err
	}

	var clone BatchNormalization
	clone.featureSize = source.featureSize
	clone.momentum = source.momentum
	clone.epsilon = source.epsilon
	clone.gamma = source.gamma
	clone.beta = source.beta
	clone.runningMean = source.runningMean
	clone.runningVariance = source.runningVariance
	clone.training = false
	return &clone, nil
}

func cloneBatchNormalization2DForInference(
	source *BatchNormalization2D,
) (out *BatchNormalization2D, err error) {
	if err = source.validate(); err != nil {
		return nil, err
	}

	var clone BatchNormalization2D
	clone.config = source.config
	clone.gamma = source.gamma
	clone.beta = source.beta
	clone.runningMean = source.runningMean
	clone.runningVariance = source.runningVariance
	clone.training = false
	return &clone, nil
}

func cloneConv2DForInference(source *Conv2D) (out *Conv2D, err error) {
	var (
		fanIn int
		clone Conv2D
	)

	if err = source.validate(); err != nil {
		return nil, err
	}

	fanIn = source.config.InputShape().Channels() * source.config.KernelHeight() * source.config.KernelWidth()
	clone.config = source.config
	clone.weights = source.weights
	clone.biases = source.biases
	clone.weightValues = make([]float32, fanIn*source.config.OutputChannels())
	clone.biasValues = make([]float32, source.config.OutputChannels())
	return &clone, nil
}

func cloneDenseForInference(source *Dense) (out *Dense, err error) {
	if err = source.validate(); err != nil {
		return nil, err
	}

	var clone Dense
	clone.inputSize = source.inputSize
	clone.outputSize = source.outputSize
	clone.weights = source.weights
	clone.biases = source.biases
	return &clone, nil
}

func cloneDropoutForInference(source *Dropout) (out *Dropout, err error) {
	if err = source.validate(); err != nil {
		return nil, err
	}

	if out, err = NewDropout(source.rate, rand.New(rand.NewSource(1))); err != nil {
		return nil, err
	}
	out.SetTraining(false)
	return out, nil
}

func cloneSimpleRNNForInference(source *SimpleRNN) (out *SimpleRNN, err error) {
	var (
		inputFeatureSize int
		hiddenSize       int
		clone            SimpleRNN
	)

	if err = source.validate(); err != nil {
		return nil, err
	}

	inputFeatureSize = source.config.InputShape().FeatureSize()
	hiddenSize = source.config.HiddenSize()
	clone.config = source.config
	clone.inputWeights = source.inputWeights
	clone.recurrentWeights = source.recurrentWeights
	clone.biases = source.biases
	clone.inputWeightValues = make([]float32, inputFeatureSize*hiddenSize)
	clone.recurrentWeightValues = make([]float32, hiddenSize*hiddenSize)
	clone.biasValues = make([]float32, hiddenSize)
	return &clone, nil
}
