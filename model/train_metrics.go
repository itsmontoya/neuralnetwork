package model

// TrainMetrics reports metrics from one training batch.
type TrainMetrics struct {
	// Loss is the scalar loss computed before the optimizer update.
	Loss float64
}
