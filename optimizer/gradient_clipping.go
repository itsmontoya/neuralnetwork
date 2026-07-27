package optimizer

import (
	"errors"
	"fmt"
	"math"
)

// NewGradientClipping wraps base with opt-in gradient clipping.
func NewGradientClipping(
	base Optimizer,
	config GradientClippingConfig,
) (out *GradientClipping, err error) {
	if base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return nil, err
	}

	if err = validateGradientClippingConfig(config); err != nil {
		return nil, err
	}

	var c GradientClipping
	c.base = base
	c.config = config
	return &c, nil
}

// GradientClipping clips accumulated gradients before delegating to its base.
type GradientClipping struct {
	base                 Optimizer
	config               GradientClippingConfig
	scratch              []float32
	observation          GradientClippingObservation
	observationAvailable bool
}

// Update clips accumulated gradients and delegates one update to the base.
func (c *GradientClipping) Update(parameters []*Parameter) (err error) {
	var (
		elementCount int
		observation  GradientClippingObservation
	)

	if err = c.validate(); err != nil {
		return err
	}
	if elementCount, err = validateGradientClippingParameters(parameters); err != nil {
		return err
	}

	c.prepareScratch(elementCount)
	if err = c.snapshotGradients(parameters); err != nil {
		return err
	}
	if observation, err = c.transformScratch(parameters); err != nil {
		return err
	}
	if observation.ValueClipped || observation.Scale < 1 {
		if err = c.publishGradients(parameters); err != nil {
			return err
		}
	}

	err = c.base.Update(parameters)
	observation.BaseUpdateCompleted = err == nil
	c.observation = observation
	c.observationAvailable = true
	return err
}

// LearningRate returns the wrapped optimizer learning rate.
func (c *GradientClipping) LearningRate() (learningRate float32) {
	if c == nil || c.base == nil {
		return 0
	}

	learningRate = c.base.LearningRate()
	return learningRate
}

// SetLearningRate updates the wrapped optimizer learning rate.
func (c *GradientClipping) SetLearningRate(learningRate float32) (err error) {
	if c == nil {
		err = nilOptimizerError("gradient clipping")
		return err
	}
	if c.base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return err
	}

	err = c.base.SetLearningRate(learningRate)
	return err
}

// Base returns the wrapped optimizer without transferring ownership.
func (c *GradientClipping) Base() (base Optimizer) {
	if c == nil {
		return nil
	}

	base = c.base
	return base
}

// Config returns a copy of the clipping configuration.
func (c *GradientClipping) Config() (config GradientClippingConfig) {
	if c == nil {
		return config
	}

	config = c.config
	return config
}

// Observation returns the most recently published clipping observation.
func (c *GradientClipping) Observation() (
	observation GradientClippingObservation,
	available bool,
) {
	if c == nil {
		return observation, false
	}

	observation = c.observation
	available = c.observationAvailable
	return observation, available
}

func (c *GradientClipping) validate() (err error) {
	if c == nil {
		err = nilOptimizerError("gradient clipping")
		return err
	}
	if c.base == nil {
		err = errors.New("optimizer: base optimizer is nil")
		return err
	}

	err = validateGradientClippingConfig(c.config)
	return err
}

func (c *GradientClipping) prepareScratch(elementCount int) {
	if cap(c.scratch) < elementCount {
		c.scratch = make([]float32, elementCount)
		return
	}

	c.scratch = c.scratch[:elementCount]
}

func (c *GradientClipping) snapshotGradients(parameters []*Parameter) (err error) {
	var (
		index        int
		offset       int
		elementCount int
		parameter    *Parameter
	)

	for index, parameter = range parameters {
		elementCount = parameter.Gradient().Rows() * parameter.Gradient().Cols()
		if err = parameter.Gradient().ValuesInto(c.scratch[offset : offset+elementCount]); err != nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d gradient snapshot failed: %w",
				index,
				err,
			)
			return err
		}
		offset += elementCount
	}

	return nil
}

func (c *GradientClipping) transformScratch(
	parameters []*Parameter,
) (observation GradientClippingObservation, err error) {
	var (
		index        int
		row          int
		col          int
		offset       int
		elementIndex int
		rows         int
		cols         int
		value        float32
		candidate    float32
		parameter    *Parameter
	)

	observation.Scale = 1
	for index, parameter = range parameters {
		rows, cols = parameter.Gradient().Shape()
		for row = 0; row < rows; row++ {
			for col = 0; col < cols; col++ {
				elementIndex = offset + row*cols + col
				value = c.scratch[elementIndex]
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					err = fmt.Errorf(
						"optimizer: gradient clipping non-finite gradient: parameter=%d row=%d col=%d value=%g",
						index,
						row,
						col,
						value,
					)
					return observation, err
				}

				candidate = value
				if c.config.MaxValue != 0 {
					if value < -c.config.MaxValue {
						candidate = -c.config.MaxValue
						observation.ValueClipped = true
					} else if value > c.config.MaxValue {
						candidate = c.config.MaxValue
						observation.ValueClipped = true
					}
					c.scratch[elementIndex] = candidate
				}

				if c.config.MaxNorm != 0 {
					if observation.GlobalNorm, err = foldGradientClippingNorm(
						observation.GlobalNorm,
						float64(candidate),
					); err != nil {
						err = fmt.Errorf(
							"%w: parameter=%d row=%d col=%d",
							err,
							index,
							row,
							col,
						)
						return observation, err
					}
				}
			}
		}
		offset += rows * cols
	}

	if c.config.MaxNorm != 0 {
		if observation.Scale, err = calculateGradientClippingScale(
			observation.GlobalNorm,
			c.config.MaxNorm,
		); err != nil {
			return observation, err
		}
	}
	if observation.Scale < 1 {
		for elementIndex = range c.scratch {
			c.scratch[elementIndex] = float32(
				float64(c.scratch[elementIndex]) * observation.Scale,
			)
		}
	}

	return observation, nil
}

func (c *GradientClipping) publishGradients(parameters []*Parameter) (err error) {
	var (
		index        int
		offset       int
		elementCount int
		parameter    *Parameter
	)

	for index, parameter = range parameters {
		elementCount = parameter.Gradient().Rows() * parameter.Gradient().Cols()
		if err = parameter.Gradient().CopyValuesFrom(c.scratch[offset : offset+elementCount]); err != nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d gradient transformation failed: %w",
				index,
				err,
			)
			return err
		}
		offset += elementCount
	}

	return nil
}

func validateGradientClippingConfig(config GradientClippingConfig) (err error) {
	if config.MaxValue == 0 && config.MaxNorm == 0 {
		err = errors.New("optimizer: gradient clipping requires at least one enabled limit")
		return err
	}
	if config.MaxValue < 0 ||
		math.IsNaN(float64(config.MaxValue)) ||
		math.IsInf(float64(config.MaxValue), 0) {
		err = fmt.Errorf(
			"optimizer: gradient clipping max value must be positive and finite when enabled: maxValue=%g",
			config.MaxValue,
		)
		return err
	}
	if config.MaxNorm < 0 ||
		math.IsNaN(float64(config.MaxNorm)) ||
		math.IsInf(float64(config.MaxNorm), 0) {
		err = fmt.Errorf(
			"optimizer: gradient clipping max norm must be positive and finite when enabled: maxNorm=%g",
			config.MaxNorm,
		)
		return err
	}

	return nil
}

func validateGradientClippingParameters(parameters []*Parameter) (elementCount int, err error) {
	var (
		index          int
		parameterCount int
		parameter      *Parameter
	)

	for index, parameter = range parameters {
		if parameter == nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d invalid: optimizer: parameter is nil",
				index,
			)
			return 0, err
		}
		if err = parameter.Values().Validate(); err != nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d invalid: values: %w",
				index,
				err,
			)
			return 0, err
		}
		if err = parameter.Gradient().Validate(); err != nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d invalid: gradient: %w",
				index,
				err,
			)
			return 0, err
		}
		if err = parameter.validate(); err != nil {
			err = fmt.Errorf(
				"optimizer: gradient clipping parameter %d invalid: %w",
				index,
				err,
			)
			return 0, err
		}

		parameterCount = parameter.Gradient().Rows() * parameter.Gradient().Cols()
		if elementCount, err = addGradientClippingElementCount(
			elementCount,
			parameterCount,
			index,
		); err != nil {
			return 0, err
		}
	}

	return elementCount, nil
}

func addGradientClippingElementCount(total, count, parameterIndex int) (next int, err error) {
	var maxInt int

	maxInt = int(^uint(0) >> 1)
	if count < 0 || total > maxInt-count {
		err = fmt.Errorf(
			"optimizer: gradient clipping element count overflow: parameter=%d",
			parameterIndex,
		)
		return 0, err
	}

	next = total + count
	return next, nil
}

func foldGradientClippingNorm(current, value float64) (norm float64, err error) {
	norm = math.Hypot(current, value)
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		err = errors.New("optimizer: gradient clipping global norm is non-finite")
		return 0, err
	}

	return norm, nil
}

func calculateGradientClippingScale(norm float64, limit float32) (scale float64, err error) {
	scale = 1
	if norm > float64(limit) {
		scale = float64(limit) / norm
	}
	if math.IsNaN(scale) || math.IsInf(scale, 0) || scale <= 0 || scale > 1 {
		err = fmt.Errorf(
			"optimizer: gradient clipping scale is invalid: norm=%g maxNorm=%g scale=%g",
			norm,
			limit,
			scale,
		)
		return 0, err
	}

	return scale, nil
}

var _ Optimizer = (*GradientClipping)(nil)
