package layer

import "fmt"

// BatchNormalization2DConfig configures spatial batch normalization.
type BatchNormalization2DConfig struct {
	InputShape SpatialShape
	Epsilon    float32
	Momentum   float32
}

func (c BatchNormalization2DConfig) validate() (err error) {
	if err = c.InputShape.validate(); err != nil {
		err = fmt.Errorf("layer: batch normalization2d input shape invalid: %w", err)
		return err
	}
	if err = validateBatchNormalizationMomentum(c.Momentum); err != nil {
		return err
	}
	if err = validateBatchNormalizationEpsilon(c.Epsilon); err != nil {
		return err
	}

	return nil
}
