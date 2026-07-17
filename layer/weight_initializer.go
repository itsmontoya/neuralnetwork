package layer

import (
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// WeightInitializer constructs a fan-in by fan-out weight matrix.
type WeightInitializer func(inputSize, outputSize int) (weights *matrix.Matrix, err error)

// ZeroWeights initializes all weights to zero.
func ZeroWeights(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
	weights, err = matrix.New(inputSize, outputSize)
	return weights, err
}

// UniformWeights returns a weight initializer using a uniform distribution.
func UniformWeights(min, max float32, random *rand.Rand) (initializer WeightInitializer) {
	initializer = func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.NewUniform(inputSize, outputSize, min, max, random)
		return weights, err
	}
	return initializer
}

// NormalWeights returns a weight initializer using a normal distribution.
func NormalWeights(mean, stddev float32, random *rand.Rand) (initializer WeightInitializer) {
	initializer = func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.NewNormal(inputSize, outputSize, mean, stddev, random)
		return weights, err
	}
	return initializer
}

// XavierUniformWeights returns a weight initializer using Xavier/Glorot uniform initialization.
func XavierUniformWeights(random *rand.Rand) (initializer WeightInitializer) {
	initializer = func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.NewXavierUniform(inputSize, outputSize, random)
		return weights, err
	}
	return initializer
}

// HeNormalWeights returns a weight initializer using He normal initialization.
func HeNormalWeights(random *rand.Rand) (initializer WeightInitializer) {
	initializer = func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.NewHeNormal(inputSize, outputSize, random)
		return weights, err
	}
	return initializer
}
